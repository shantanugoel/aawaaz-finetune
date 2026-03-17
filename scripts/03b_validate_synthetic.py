#!/usr/bin/env python3
"""LLM-as-Judge quality gate for synthetic transcript-cleanup pairs.

Samples a configurable fraction of synthetic pairs per category, sends each
to an LLM judge, and makes per-category pass/warn/reject decisions.  Rejected
categories are moved to ``data/synthetic/rejected/``.  Individual pairs
failing ``content_preserved`` or ``no_hallucination`` are removed even from
accepted categories.

Outputs:
- ``data/synthetic/validation_report.json`` — full results
- ``data/synthetic/validation_summary.txt`` — human-readable summary
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from common import (
    DATA_SYNTHETIC,
    DATA_SYNTHETIC_REJECTED,
    base_arg_parser,
    ensure_dirs,
    load_config,
    setup_logging,
)
from llm_client import create_client_from_config, LLMClient

logger = logging.getLogger("aawaaz.validate_synthetic")

# ── Constants ──────────────────────────────────────────────────────────────

CRITERIA = [
    "input_realistic",
    "content_preserved",
    "no_hallucination",
    "corrections_applied",
]

# Criteria that cause individual pair removal even in accepted categories
CRITICAL_CRITERIA = {"content_preserved", "no_hallucination"}

REPORT_PATH = DATA_SYNTHETIC / "validation_report.json"
SUMMARY_PATH = DATA_SYNTHETIC / "validation_summary.txt"

# ── Judge prompt ───────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating a synthetic training pair for a speech transcript cleanup model.
Your job is to ensure the training data is high quality. Be strict but fair.

INPUT (raw transcript):
{transcript}

OUTPUT (cleaned text):
{output}

Score this pair on 4 criteria. Answer ONLY with a JSON object, no other text.

1. "input_realistic": Is the input realistic speech-to-text (ASR) output?
   Modern ASR like Whisper MAY produce some punctuation and capitalization — that is normal.
   The key question: does this sound like someone SPEAKING, or like someone carefully TYPING?
   It should have:
   - Natural speech patterns: fillers, run-on sentences, topic changes, hesitations
   - Inconsistent formatting (some punctuation present, some missing — not perfectly formatted)
   - Numbers mostly as spoken words (but some digit usage is acceptable)
   FAIL if it reads like carefully composed written text with a few "um"s mechanically inserted.
   FAIL if it has perfect, consistent formatting throughout that no ASR engine would produce.
   PASS if it sounds like natural speech even if the ASR has added some punctuation/capitalization.

2. "content_preserved": Does the output preserve ALL substantive information from the input?
   Compare carefully — no facts, names, numbers, instructions, or meaning should be dropped.
   Minor rewording is fine. Dropping an entire sentence or fact is a FAIL.
   IMPORTANT — the following are NOT substantive content and may be removed:
   - Filler words and hedging: "um", "uh", "like", "you know", "basically", "honestly", \
"literally", "kind of", "sort of", "I mean", "right", "okay"
   - Discourse markers: "yeah no", "oh my god", "so basically", "I'm not gonna lie"
   - Conversational padding: "that's the thing", "and stuff", "and everything", \
"which is crazy", "and be a mess"
   - Compositional meta-talk: "I want to say", "how do I put this", "what's the word", \
"let me think"
   - Softeners: "kind of", "sort of", "a little bit", "pretty much"
   Removing these is CORRECT cleanup, not content loss. Only flag content_preserved=false \
if actual facts, names, numbers, instructions, or meaningful statements are dropped.

3. "no_hallucination": Does the output contain ONLY information present in the input?
   IMPORTANT — the following are NOT hallucination:
   - Adding structural formatting: headers ("Ingredients:", "Attendees:", "Steps:"), \
bullet points, numbered lists, paragraph breaks — this is formatting, not new content
   - Adding punctuation: periods, commas, dashes, em-dashes, colons, quotation marks
   - Converting numbers: "twenty five" → 25, "two cups" → 2 cups
   - Minor grammatical rewording: "you're gonna" → "you will", "wanna" → "want to", \
"it's like" → "it is"
   - Replacing fillers with proper connectors: removing "so like" and using a comma instead
   - Removing discourse markers without replacement
   What IS hallucination (FAIL for these):
   - Adding facts, data, or details not mentioned by the speaker
   - Adding conclusions, opinions, or interpretations the speaker didn't express
   - Adding clarifying phrases that introduce new meaning (e.g., adding "for safety reasons" \
when the speaker didn't explain why)
   - Inventing specific numbers, names, or dates not in the input

4. "corrections_applied": Is the cleanup done correctly?
   - Fillers (um, uh, basically, actually, like, you know, so, I mean) removed
   - Self-corrections applied correctly (only the corrected version kept)
   - Numbers/dates/currency properly formatted
   - Punctuation and capitalization added consistently
   FAIL if obvious cleanup was missed or done incorrectly.

{{"input_realistic": true/false, "content_preserved": true/false, "no_hallucination": true/false, "corrections_applied": true/false}}"""


# ── Data loading ───────────────────────────────────────────────────────────


def category_file_path(category: str) -> Path:
    """Return the JSONL path for a given category."""
    return DATA_SYNTHETIC / f"synthetic_{category}.jsonl"


def load_category_pairs(path: Path) -> list[dict[str, Any]]:
    """Load all valid JSONL records from a category file.

    Each record must have ``input`` and ``output`` string fields.
    """
    pairs: list[dict[str, Any]] = []
    skipped = 0
    if not path.exists():
        return pairs
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if (
                    isinstance(record, dict)
                    and isinstance(record.get("input"), str)
                    and isinstance(record.get("output"), str)
                    and record["input"].strip()
                    and record["output"].strip()
                ):
                    record["_line_no"] = line_no
                    pairs.append(record)
                else:
                    skipped += 1
                    logger.debug(
                        "Skipping invalid record at line %d in %s "
                        "(missing/non-string input/output)",
                        line_no, path,
                    )
            except json.JSONDecodeError:
                skipped += 1
                logger.debug(
                    "Skipping unparseable line %d in %s", line_no, path
                )

    if skipped > 0:
        logger.warning(
            "%s: skipped %d malformed/invalid lines out of %d total",
            path.name, skipped, skipped + len(pairs),
        )
    return pairs


def discover_categories(
    config_categories: dict[str, float],
    category_filter: str | None = None,
) -> list[str]:
    """Return list of categories with existing synthetic data files.

    Also warns about unexpected files on disk that aren't in config.
    If *category_filter* is given, only that category is returned (if its
    file exists).
    """
    categories: list[str] = []
    for cat in config_categories:
        if category_filter and cat != category_filter:
            continue
        path = category_file_path(cat)
        if path.exists():
            categories.append(cat)
        else:
            logger.debug("No synthetic data file for category '%s', skipping", cat)

    if category_filter and category_filter not in categories:
        logger.warning(
            "Requested category '%s' has no synthetic data file at %s",
            category_filter,
            category_file_path(category_filter),
        )

    # Warn about unexpected files on disk not in config
    if not category_filter:
        for filepath in sorted(DATA_SYNTHETIC.glob("synthetic_*.jsonl")):
            cat_name = filepath.stem.removeprefix("synthetic_")
            if cat_name not in config_categories:
                logger.warning(
                    "Unexpected synthetic file '%s' found on disk but not in "
                    "config categories — it will NOT be validated. Add it to "
                    "config or remove it.",
                    filepath.name,
                )

    return categories


# ── Judge call ─────────────────────────────────────────────────────────────


def judge_pair(
    client: LLMClient,
    pair: dict[str, Any],
) -> dict[str, bool] | None:
    """Send a single pair to the LLM judge and parse the verdict.

    Returns a dict with the four boolean criteria, or ``None`` if parsing
    fails (the pair is treated as if it was not judged).
    """
    prompt = JUDGE_PROMPT.format(
        transcript=pair["input"],
        output=pair["output"],
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        raw = client.generate(
            messages,
            max_tokens=1024,
            temperature=0.0,
            json_mode=True,
        )
    except RuntimeError as exc:
        logger.error("Judge API call failed: %s", exc)
        return None

    return _parse_judge_response(raw)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Find and parse the first top-level JSON object in *text*.

    Uses brace-depth tracking so it handles nested ``{}`` correctly,
    unlike a simple ``\\{[^{}]+\\}`` regex.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try the next '{' in the text
                    start = text.find("{", i + 1)
                    if start == -1:
                        return None
                    depth = 0
    return None


def _parse_judge_response(raw: str) -> dict[str, bool] | None:
    """Parse the JSON verdict from the judge LLM response.

    Handles:
    - ``<think>...</think>`` blocks from reasoning models
    - Markdown code fences anywhere in the text (not just at start)
    - Multi-line / nested JSON objects
    - Reasoning text surrounding the JSON
    """
    import re

    text = raw.strip()

    # Strip <think>...</think> blocks (Qwen3 / reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try to extract JSON from a code fence anywhere in the text
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL
    )
    if fence_match:
        text = fence_match.group(1).strip()

    # 1. Try parsing the (possibly cleaned) text directly
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # 2. Extract the outermost { ... } by tracking brace depth
        parsed = _extract_json_object(text)

    if parsed is None:
        logger.warning(
            "Could not parse judge response as JSON: %.300s", raw
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning(
            "Expected JSON object from judge, got %s", type(parsed).__name__
        )
        return None

    # Validate all criteria are present and boolean
    result: dict[str, bool] = {}
    for criterion in CRITERIA:
        val = parsed.get(criterion)
        if isinstance(val, bool):
            result[criterion] = val
        elif isinstance(val, str) and val.lower() in ("true", "false"):
            result[criterion] = val.lower() == "true"
        else:
            logger.warning(
                "Criterion '%s' missing or invalid in judge response "
                "(got %r, expected bool): %s",
                criterion,
                val,
                parsed,
            )
            return None

    return result


# ── Cost estimation ────────────────────────────────────────────────────────


def estimate_validation_cost(
    categories: list[str],
    sample_rate: float,
    config_categories: dict[str, float],
) -> dict[str, Any]:
    """Estimate API cost for validation judging.

    Uses rough token estimates: ~800 input tokens + ~50 output tokens per
    judge call (the judge prompt + pair is shorter than generation, and the
    response is a small JSON object).
    """
    total_pairs = 0
    category_counts: dict[str, int] = {}
    for cat in categories:
        pairs = load_category_pairs(category_file_path(cat))
        n_pairs = len(pairs)
        n_sample = math.ceil(n_pairs * sample_rate)
        category_counts[cat] = n_sample
        total_pairs += n_sample

    input_tokens_per_call = 800
    output_tokens_per_call = 50

    total_input_tokens = total_pairs * input_tokens_per_call
    total_output_tokens = total_pairs * output_tokens_per_call

    # Pricing: use Anthropic Sonnet as baseline
    cost_per_m_input = 3.0
    cost_per_m_output = 15.0

    input_cost = (total_input_tokens / 1_000_000) * cost_per_m_input
    output_cost = (total_output_tokens / 1_000_000) * cost_per_m_output
    total_cost = input_cost + output_cost

    return {
        "total_pairs_to_judge": total_pairs,
        "sample_rate": sample_rate,
        "category_sample_counts": category_counts,
        "est_input_tokens": total_input_tokens,
        "est_output_tokens": total_output_tokens,
        "est_cost_usd": round(total_cost, 4),
        "pricing_note": (
            f"Estimate based on ~{input_tokens_per_call} input + "
            f"~{output_tokens_per_call} output tokens/call at "
            f"$3/M input, $15/M output (Anthropic Sonnet baseline)"
        ),
    }


# ── Per-category validation ───────────────────────────────────────────────


def validate_category(
    client: LLMClient,
    category: str,
    sample_rate: float,
    verbose: bool = False,
) -> dict[str, Any]:
    """Validate a single category by sampling and judging pairs.

    Returns a result dict with pass rates, verdicts, and failing examples.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    path = category_file_path(category)
    all_pairs = load_category_pairs(path)

    if not all_pairs:
        logger.warning("Category '%s': no pairs found, skipping", category)
        return {
            "category": category,
            "total_pairs": 0,
            "sampled": 0,
            "pass_rate": 0.0,
            "decision": "skip",
            "criterion_pass_rates": {},
            "verdicts": [],
            "failures": {},
        }

    # Sample pairs — use local RNG seeded per category for reproducibility
    n_sample = math.ceil(len(all_pairs) * sample_rate)
    n_sample = min(n_sample, len(all_pairs))

    rng = random.Random(hash((42, category)))
    sampled_indices = sorted(rng.sample(range(len(all_pairs)), n_sample))
    sampled_pairs = [all_pairs[i] for i in sampled_indices]

    logger.info(
        "Category '%s': validating %d/%d pairs (%.0f%% sample rate)",
        category,
        n_sample,
        len(all_pairs),
        sample_rate * 100,
    )

    # Judge each sampled pair
    verdicts: list[dict[str, Any]] = []
    criterion_pass_counts: dict[str, int] = {c: 0 for c in CRITERIA}
    criterion_failures: dict[str, list[dict[str, Any]]] = {c: [] for c in CRITERIA}
    judged_count = 0

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=n_sample,
            desc=f"  Judging {category}",
            unit="pair",
            leave=True,
        )

    for pair in sampled_pairs:
        result = judge_pair(client, pair)

        if result is None:
            # Parse failure — treat as unknown, log and skip
            verdicts.append({
                "pair_index": pair.get("_line_no", -1),
                "input_preview": pair["input"][:100],
                "verdict": None,
                "parse_error": True,
            })
            if pbar:
                pbar.update(1)
            continue

        judged_count += 1
        all_pass = all(result.values())

        verdict_entry: dict[str, Any] = {
            "pair_index": pair.get("_line_no", -1),
            "input_preview": pair["input"][:100],
            "verdict": result,
            "all_pass": all_pass,
        }
        verdicts.append(verdict_entry)

        for criterion in CRITERIA:
            if result[criterion]:
                criterion_pass_counts[criterion] += 1
            else:
                # Store failure example (keep up to 5 per criterion)
                if len(criterion_failures[criterion]) < 5:
                    criterion_failures[criterion].append({
                        "pair_index": pair.get("_line_no", -1),
                        "input_preview": pair["input"][:200],
                        "output_preview": pair["output"][:200],
                    })

        if verbose:
            status = "PASS" if all_pass else "FAIL"
            failing = [c for c, v in result.items() if not v]
            logger.info(
                "  Pair %d: %s%s",
                pair.get("_line_no", -1),
                status,
                f" (failed: {', '.join(failing)})" if failing else "",
            )

        if pbar:
            pbar.update(1)

        # Small delay between judge calls
        time.sleep(0.2)

    if pbar:
        pbar.close()

    # Compute pass rates — parse errors count as failures (fail-closed)
    total_judged_or_errored = judged_count + (n_sample - judged_count)  # == n_sample
    if total_judged_or_errored == 0:
        overall_pass_rate = 0.0
        criterion_pass_rates = {c: 0.0 for c in CRITERIA}
    else:
        # Overall pass rate = fraction of ALL sampled pairs where ALL criteria passed
        # Parse errors count as failures
        all_pass_count = sum(
            1 for v in verdicts
            if v.get("all_pass", False)
        )
        overall_pass_rate = all_pass_count / total_judged_or_errored
        criterion_pass_rates = {
            c: criterion_pass_counts[c] / total_judged_or_errored for c in CRITERIA
        }

    # Identify pairs that individually fail critical criteria
    # (for removal from accepted categories)
    critically_failing_indices: set[int] = set()
    for v in verdicts:
        if v.get("verdict") is None:
            continue
        for criterion in CRITICAL_CRITERIA:
            if not v["verdict"].get(criterion, True):
                critically_failing_indices.add(v["pair_index"])

    return {
        "category": category,
        "total_pairs": len(all_pairs),
        "sampled": n_sample,
        "judged": judged_count,
        "parse_errors": n_sample - judged_count,
        "pass_rate": round(overall_pass_rate, 4),
        "criterion_pass_rates": {
            c: round(r, 4) for c, r in criterion_pass_rates.items()
        },
        "decision": "",  # filled in by caller
        "verdicts": verdicts,
        "failures": {
            c: examples for c, examples in criterion_failures.items() if examples
        },
        "critically_failing_line_nos": sorted(critically_failing_indices),
    }


# ── Category-level decisions ──────────────────────────────────────────────


def apply_decisions(
    results: list[dict[str, Any]],
    pass_threshold: float,
    reject_threshold: float,
) -> list[dict[str, Any]]:
    """Apply pass/warn/reject decisions based on thresholds.

    Mutates each result dict in place, setting the ``decision`` field.
    """
    for result in results:
        if result.get("decision") == "skip":
            continue

        rate = result["pass_rate"]
        if rate >= pass_threshold:
            result["decision"] = "pass"
        elif rate >= reject_threshold:
            result["decision"] = "warn"
        else:
            result["decision"] = "reject"

    return results


def handle_rejected_categories(results: list[dict[str, Any]]) -> int:
    """Move rejected category files to ``data/synthetic/rejected/``.

    Returns count of files moved.  Refuses to overwrite existing rejected
    files — appends a numeric suffix instead.
    """
    DATA_SYNTHETIC_REJECTED.mkdir(parents=True, exist_ok=True)
    moved = 0
    for result in results:
        if result["decision"] != "reject":
            continue
        category = result["category"]
        src = category_file_path(category)
        if not src.exists():
            continue

        dst = DATA_SYNTHETIC_REJECTED / src.name
        # Avoid overwriting existing rejected files
        if dst.exists():
            counter = 1
            while dst.exists():
                dst = DATA_SYNTHETIC_REJECTED / f"{src.stem}_{counter}{src.suffix}"
                counter += 1

        shutil.move(str(src), str(dst))

        # Log detailed error as required by spec
        logger.error(
            "REJECTED category '%s' (pass rate %.1f%%) — moved to %s",
            category,
            result["pass_rate"] * 100,
            dst,
        )
        # Criterion breakdown
        for criterion, rate in sorted(
            result["criterion_pass_rates"].items(), key=lambda x: x[1]
        ):
            if rate < 0.90:
                logger.error(
                    "  ↳ %s: %.1f%% pass (failed %.1f%% of the time)",
                    criterion,
                    rate * 100,
                    (1 - rate) * 100,
                )
        # Example failures (up to 5 per criterion)
        if result.get("failures"):
            for criterion, examples in result["failures"].items():
                if examples:
                    logger.error("  Example failures for %s:", criterion)
                    for ex in examples[:5]:
                        logger.error(
                            "    - Input: %.80s...", ex["input_preview"]
                        )
        # Prompt improvement suggestions
        _log_prompt_suggestions(result)

        moved += 1
    return moved


def _log_prompt_suggestions(result: dict[str, Any]) -> None:
    """Log prompt improvement suggestions based on failing criteria."""
    suggestions: dict[str, str] = {
        "input_realistic": (
            "Make synthetic inputs less polished: remove punctuation, "
            "capitalization, and formatted numbers. Real Whisper output has "
            "NO punctuation and numbers spelled out."
        ),
        "content_preserved": (
            "Explicitly require preserving every fact, instruction, name, "
            "and number from input. Do not allow dropping sentences."
        ),
        "no_hallucination": (
            "Forbid inferred clarifications, added connective text, or "
            "conclusions not present in the input. Only formatting changes "
            "are allowed."
        ),
        "corrections_applied": (
            "Strengthen formatting and self-correction cleanup instructions. "
            "Ensure fillers are removed, numbers formatted, and "
            "self-corrections applied correctly."
        ),
    }
    rates = result.get("criterion_pass_rates", {})
    for criterion, rate in rates.items():
        if rate < 0.90 and criterion in suggestions:
            logger.error(
                "  Suggestion for %s: %s", criterion, suggestions[criterion]
            )


def remove_critical_failures(results: list[dict[str, Any]]) -> dict[str, int]:
    """Remove individually failing pairs from accepted categories.

    For categories with decision ``pass`` or ``warn``, remove any sampled
    pair that failed ``content_preserved`` or ``no_hallucination``.

    Uses atomic write-then-rename to prevent data loss on crash.

    Returns a dict mapping category name → number of pairs removed.
    """
    removed_counts: dict[str, int] = {}

    for result in results:
        if result["decision"] not in ("pass", "warn"):
            continue

        failing_lines = set(result.get("critically_failing_line_nos", []))
        if not failing_lines:
            continue

        category = result["category"]
        path = category_file_path(category)
        if not path.exists():
            continue

        # Read all lines, filter out the failing ones
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()

        original_count = len(lines)
        kept_lines: list[str] = []
        removed = 0
        for line_no, line in enumerate(lines, 1):
            if line_no in failing_lines:
                removed += 1
                logger.debug(
                    "Removing pair at line %d from '%s' (critical failure)",
                    line_no,
                    category,
                )
            else:
                kept_lines.append(line)

        if removed > 0:
            # Atomic write: temp file + os.replace
            temp_fd, temp_path = tempfile.mkstemp(
                dir=path.parent, suffix=".tmp"
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as fh:
                    fh.writelines(kept_lines)
                os.replace(temp_path, path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

            logger.info(
                "Category '%s': removed %d/%d critically failing pairs",
                category,
                removed,
                original_count,
            )
            removed_counts[category] = removed

    return removed_counts


# ── Report generation ─────────────────────────────────────────────────────


def build_report(
    results: list[dict[str, Any]],
    removed_counts: dict[str, int],
) -> dict[str, Any]:
    """Build the full validation report as a JSON-serialisable dict."""
    # Strip internal fields from verdicts for the report
    clean_results = []
    for r in results:
        clean = {k: v for k, v in r.items() if k != "critically_failing_line_nos"}
        # Add removal count
        clean["removed_critical_pairs"] = removed_counts.get(r["category"], 0)
        # Simplify verdicts: remove full verdict details unless they failed
        simplified_verdicts = []
        for v in clean.get("verdicts", []):
            entry: dict[str, Any] = {
                "pair_index": v["pair_index"],
                "input_preview": v["input_preview"],
            }
            if v.get("parse_error"):
                entry["parse_error"] = True
            else:
                entry["verdict"] = v["verdict"]
                entry["all_pass"] = v["all_pass"]
            simplified_verdicts.append(entry)
        clean["verdicts"] = simplified_verdicts
        clean_results.append(clean)

    return {
        "validation_results": clean_results,
        "summary": {
            "total_categories": len(results),
            "passed": sum(1 for r in results if r["decision"] == "pass"),
            "warned": sum(1 for r in results if r["decision"] == "warn"),
            "rejected": sum(1 for r in results if r["decision"] == "reject"),
            "skipped": sum(1 for r in results if r["decision"] == "skip"),
        },
    }


def build_summary_text(
    results: list[dict[str, Any]],
    pass_threshold: float,
    reject_threshold: float,
    removed_counts: dict[str, int],
) -> str:
    """Build a human-readable summary string."""
    lines: list[str] = []
    lines.append("=" * 65)
    lines.append("SYNTHETIC DATA VALIDATION SUMMARY")
    lines.append("=" * 65)
    lines.append("")

    # Per-category results
    lines.append(f"{'Category':<28} {'Pairs':>6} {'Sampled':>8} {'Pass%':>7} {'Decision':>10}")
    lines.append("-" * 65)

    for r in results:
        if r["decision"] == "skip":
            lines.append(f"{r['category']:<28} {'—':>6} {'—':>8} {'—':>7} {'SKIP':>10}")
            continue
        lines.append(
            f"{r['category']:<28} {r['total_pairs']:>6} {r['sampled']:>8} "
            f"{r['pass_rate'] * 100:>6.1f}% {r['decision'].upper():>10}"
        )

    lines.append("-" * 65)
    lines.append("")

    # Criterion breakdown for non-skip categories
    active = [r for r in results if r["decision"] != "skip"]
    if active:
        lines.append("Per-criterion pass rates:")
        lines.append(f"  {'Category':<25} {'input_real':>10} {'content':>10} {'no_halluc':>10} {'correct':>10}")
        lines.append("  " + "-" * 65)
        for r in active:
            rates = r["criterion_pass_rates"]
            lines.append(
                f"  {r['category']:<25} "
                f"{rates.get('input_realistic', 0) * 100:>9.1f}% "
                f"{rates.get('content_preserved', 0) * 100:>9.1f}% "
                f"{rates.get('no_hallucination', 0) * 100:>9.1f}% "
                f"{rates.get('corrections_applied', 0) * 100:>9.1f}%"
            )
        lines.append("")

    # Rejected categories detail
    rejected = [r for r in results if r["decision"] == "reject"]
    if rejected:
        lines.append("REJECTED CATEGORIES (files moved to data/synthetic/rejected/):")
        for r in rejected:
            lines.append(f"  • {r['category']} — pass rate {r['pass_rate'] * 100:.1f}%")
            # Show which criteria failed most
            worst = sorted(
                r["criterion_pass_rates"].items(), key=lambda x: x[1]
            )
            for criterion, rate in worst:
                if rate < pass_threshold:
                    lines.append(
                        f"    ↳ {criterion}: {rate * 100:.1f}% pass "
                        f"(failed {(1 - rate) * 100:.1f}% of the time)"
                    )
            # Show example failures (up to 5 per criterion, per spec)
            if r.get("failures"):
                for criterion, examples in r["failures"].items():
                    if examples:
                        lines.append(f"    Example failures for {criterion}:")
                        for ex in examples[:5]:
                            lines.append(
                                f"      - Input: {ex['input_preview'][:80]}..."
                            )
            # Prompt improvement suggestions
            _suggestions = {
                "input_realistic": "Remove punctuation/capitalization/formatted numbers from inputs",
                "content_preserved": "Require preserving every fact/name/number from input",
                "no_hallucination": "Forbid inferred clarifications or added text",
                "corrections_applied": "Strengthen filler removal and formatting instructions",
            }
            suggested = [
                f"{c}: {_suggestions[c]}"
                for c, rate in r["criterion_pass_rates"].items()
                if rate < pass_threshold and c in _suggestions
            ]
            if suggested:
                lines.append("    Suggested prompt improvements:")
                for s in suggested:
                    lines.append(f"      → {s}")
        lines.append("")

    # Warned categories
    warned = [r for r in results if r["decision"] == "warn"]
    if warned:
        lines.append("WARNED CATEGORIES (accepted but review recommended):")
        for r in warned:
            lines.append(f"  • {r['category']} — pass rate {r['pass_rate'] * 100:.1f}%")
            # Show criterion breakdown for warned categories (per spec)
            for criterion, rate in sorted(
                r["criterion_pass_rates"].items(), key=lambda x: x[1]
            ):
                if rate < pass_threshold:
                    lines.append(
                        f"    ↳ {criterion}: {rate * 100:.1f}% pass"
                    )
            lines.append("    → Consider manual review or prompt revision for next run")
        lines.append("")

    # Individual pair removal
    if removed_counts:
        lines.append("Individual pair removals (content_preserved/no_hallucination failures):")
        for cat, count in removed_counts.items():
            lines.append(f"  • {cat}: {count} pairs removed")
        lines.append("")

    # Overall summary
    n_pass = sum(1 for r in results if r["decision"] == "pass")
    n_warn = sum(1 for r in results if r["decision"] == "warn")
    n_reject = sum(1 for r in results if r["decision"] == "reject")
    n_skip = sum(1 for r in results if r["decision"] == "skip")
    lines.append(
        f"Overall: {n_pass} passed, {n_warn} warned, {n_reject} rejected, {n_skip} skipped"
    )

    # Thresholds reference
    lines.append(
        f"Thresholds: pass >= {pass_threshold * 100:.0f}%, "
        f"warn {reject_threshold * 100:.0f}%-{pass_threshold * 100:.0f}%, "
        f"reject < {reject_threshold * 100:.0f}%"
    )
    lines.append("=" * 65)

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> Any:
    """Parse CLI arguments."""
    parser = base_arg_parser(
        "LLM-as-Judge quality validation of synthetic training data."
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Override the config sample rate (fraction of pairs to judge).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Validate 100%% of pairs (expensive; overrides --sample-rate).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Validate only one specific category.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip cost confirmation prompt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing validation report files.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success, 1 on failure."""
    args = parse_args()
    log = setup_logging(args.verbose)

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Configuration error: %s", exc)
        return 1

    synth = cfg.dataset.synthetic
    val_cfg = synth.validation

    if not synth.enabled:
        log.info("Synthetic data generation is disabled in config. Nothing to validate.")
        return 0

    if not val_cfg.enabled and not args.sample_rate and not args.full:
        log.info("Validation is disabled in config. Use --sample-rate or --full to override.")
        return 0

    # Determine sample rate
    if args.full:
        sample_rate = 1.0
    elif args.sample_rate is not None:
        sample_rate = args.sample_rate
    else:
        sample_rate = val_cfg.sample_rate

    if not (0 < sample_rate <= 1.0):
        log.error("Sample rate must be in (0, 1.0], got %s", sample_rate)
        return 1

    # Discover categories with data
    categories = discover_categories(synth.categories, args.category)
    if not categories:
        log.error("No synthetic data files found to validate.")
        return 1

    log.info(
        "Validation plan: %d categories, %.0f%% sample rate",
        len(categories),
        sample_rate * 100,
    )

    # Check output files
    if not args.force and not args.dry_run:
        if REPORT_PATH.exists() or SUMMARY_PATH.exists():
            log.error(
                "Validation output files already exist: %s, %s. "
                "Use --force to overwrite.",
                REPORT_PATH,
                SUMMARY_PATH,
            )
            return 1

    # Cost estimation
    cost_info = estimate_validation_cost(
        categories, sample_rate, synth.categories
    )

    log.info("Estimated judge calls: %d", cost_info["total_pairs_to_judge"])
    log.info("Estimated cost: $%.4f", cost_info["est_cost_usd"])
    log.info("  %s", cost_info["pricing_note"])

    for cat, count in cost_info["category_sample_counts"].items():
        log.info("  %-25s %d pairs to judge", cat, count)

    # Dry run — stop here
    if args.dry_run:
        log.info("=== DRY RUN — would validate %d pairs across %d categories ===",
                 cost_info["total_pairs_to_judge"], len(categories))
        return 0

    # Confirmation prompt
    if not args.yes:
        log.info("")
        try:
            answer = input(
                f"Proceed with validation? ({cost_info['total_pairs_to_judge']} "
                f"judge calls, est. ${cost_info['est_cost_usd']:.4f}) [y/N]: "
            )
        except (EOFError, KeyboardInterrupt):
            log.info("\nAborted.")
            return 1
        if answer.strip().lower() not in ("y", "yes"):
            log.info("Aborted by user.")
            return 0

    # Ensure directories
    ensure_dirs()

    # Create LLM client for judging
    try:
        client = create_client_from_config(
            provider=val_cfg.provider,
            model=val_cfg.model,
            api_key_env=val_cfg.api_key_env,
            base_url=val_cfg.base_url,
        )
    except (EnvironmentError, ImportError, ValueError) as exc:
        log.error("Failed to create LLM client for validation: %s", exc)
        return 1

    # ── Validate each category ────────────────────────────────────────────
    start_time = time.monotonic()
    results: list[dict[str, Any]] = []

    for category in categories:
        cat_result = validate_category(
            client=client,
            category=category,
            sample_rate=sample_rate,
            verbose=args.verbose,
        )
        results.append(cat_result)

    # ── Apply decisions ───────────────────────────────────────────────────
    apply_decisions(results, val_cfg.pass_threshold, val_cfg.reject_threshold)

    # Log per-category decisions
    for result in results:
        if result["decision"] == "warn":
            log.warning(
                "Category '%s': WARN (pass rate %.1f%%) — accepted but "
                "manual review recommended",
                result["category"],
                result["pass_rate"] * 100,
            )
            for criterion, rate in sorted(
                result["criterion_pass_rates"].items(), key=lambda x: x[1]
            ):
                if rate < val_cfg.pass_threshold:
                    log.warning(
                        "  ↳ %s: %.1f%% pass", criterion, rate * 100
                    )
        elif result["decision"] == "pass":
            log.info(
                "Category '%s': PASS (pass rate %.1f%%)",
                result["category"],
                result["pass_rate"] * 100,
            )

    # ── Generate reports BEFORE destructive mutations ─────────────────────
    # Pre-compute removal counts (which lines would be removed)
    removal_plan: dict[str, set[int]] = {}
    for result in results:
        if result["decision"] in ("pass", "warn"):
            failing = set(result.get("critically_failing_line_nos", []))
            if failing:
                removal_plan[result["category"]] = failing
    removed_counts_planned = {cat: len(lines) for cat, lines in removal_plan.items()}

    report = build_report(results, removed_counts_planned)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    log.info("Full report: %s", REPORT_PATH)

    summary_text = build_summary_text(
        results, val_cfg.pass_threshold, val_cfg.reject_threshold, removed_counts_planned
    )
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        fh.write(summary_text + "\n")
    log.info("Summary: %s", SUMMARY_PATH)

    # ── Now apply destructive mutations ───────────────────────────────────
    # Handle rejected categories (move files)
    moved_count = handle_rejected_categories(results)

    # Remove individually failing pairs from accepted categories
    removed_counts = remove_critical_failures(results)

    # Print summary to console
    for line in summary_text.split("\n"):
        log.info(line)

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    log.info("")
    log.info("Validation complete in %.1fs", elapsed)
    if moved_count > 0:
        log.error(
            "%d category file(s) REJECTED and moved to %s",
            moved_count,
            DATA_SYNTHETIC_REJECTED,
        )

    total_removed = sum(removed_counts.values())
    if total_removed > 0:
        log.info(
            "%d individual pairs removed from accepted categories",
            total_removed,
        )

    # Return non-zero if any categories were rejected
    any_rejected = any(r["decision"] == "reject" for r in results)
    return 1 if any_rejected else 0


if __name__ == "__main__":
    sys.exit(main())
