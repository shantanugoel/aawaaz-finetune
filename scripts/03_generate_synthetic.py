#!/usr/bin/env python3
"""Generate synthetic transcript-cleanup training pairs via LLM API.

Reads category proportions from ``config.yaml``, calls an LLM to produce
realistic Whisper-style transcript / clean-output pairs, validates each
pair, and saves per-category JSONL plus a combined output file.

Supports resume: on restart, existing records are counted and already-done
batches are skipped.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    DATA_SYNTHETIC,
    SyntheticConfig,
    base_arg_parser,
    ensure_dirs,
    load_config,
    setup_logging,
)
from llm_client import create_client_from_config, LLMClient

logger = logging.getLogger("aawaaz.generate_synthetic")

# ── Category-specific generation guidance ──────────────────────────────────

CATEGORY_GUIDANCE: dict[str, str] = {
    "casual_conversation": (
        "Friend-to-friend messages, voice notes, casual updates. "
        "Use contractions, informal language, sometimes trailing off."
    ),
    "email_professional": (
        'Dictating business emails. Include "dear", "regards", salutations. '
        "Formal tone but spoken casually."
    ),
    "technical_code": (
        "Dictating code, CLI commands, error messages, technical docs. "
        "Include function names, file paths, docker commands, SQL queries, "
        "variable names with underscores/camelCase."
    ),
    "medical_clinical": (
        "Patient notes, clinical observations, medication names, dosages, "
        "medical abbreviations."
    ),
    "legal_contract": (
        "Contract clauses, legal terminology, article/section references, "
        "formal language."
    ),
    "meeting_notes": (
        "Action items, attendee names, deadlines, decisions made."
    ),
    "recipe_cooking": (
        "Dictating recipes or cooking instructions. Include ingredient "
        "quantities spoken as words, cooking times, temperatures, step-by-step "
        "directions with natural speech patterns."
    ),
    "academic_research": (
        "Dictating research notes, paper abstracts, citations, methodology "
        "descriptions. Include author names, journal titles, statistical "
        "figures spoken as words, and technical terminology."
    ),
    "creative_writing": (
        "Dictating stories, poems, blog posts, or journal entries. Include "
        "descriptive language, dialogue, and natural pauses or changes of "
        "thought mid-sentence."
    ),
    "financial_business": (
        "Dictating financial reports, budget notes, invoice details, or "
        "business plans. Include currency amounts, percentages, dates, and "
        "company names spoken naturally."
    ),
    "shopping_lists": (
        "Dictating shopping or to-do lists. Items spoken in quick succession, "
        "sometimes with quantities, brands, or notes. Often very short and "
        "list-oriented."
    ),
    "self_corrections_heavy": (
        'Specifically focus on self-correction patterns: "wait", "no", '
        '"scratch that", "actually I meant", "let me rephrase", "correction". '
        "Multiple corrections per example. This category is crucial for quality."
    ),
}

# ── Generation prompt template ─────────────────────────────────────────────

GENERATION_PROMPT = """\
Generate {batch_size} realistic speech-to-text transcript pairs for the category: {category}.

For each pair, create:
1. A "transcript" — what a speech-to-text engine (like Whisper) would output from someone speaking naturally. This should include:
   - Filler words (um, uh, like, basically, actually, you know, so, I mean) placed naturally
   - No punctuation or very minimal/wrong punctuation
   - Numbers, dates, and currency spoken as words (e.g., "two thousand twenty five" not "2025")
   - Occasional self-corrections (e.g., "the meeting is on Tuesday wait no Wednesday")
   - Spoken formatting cues where natural (e.g., "colon", "new line", "bullet point", "dash")
   - Run-on sentences with no clear breaks
   - Some spoken technical terms, code syntax, URLs spelled out
   - Occasional spoken emoji descriptions (e.g., "thumbs up emoji")
   - Realistic speech patterns — not every sentence has fillers, vary the messiness

2. An "output" — the clean, properly formatted version that preserves ALL substantive content but:
   - Removes all fillers and stutters
   - Applies self-corrections (only keeps the corrected version)
   - Has proper punctuation, capitalization, and grammar
   - Numbers, dates, currency in written form ($500, January 15, 2025)
   - Proper formatting (bullet lists, paragraphs, code formatting)
   - Emoji characters where described
   - No content added or removed

Category-specific guidance for "{category}":
{category_guidance}

Respond with a JSON array of objects, each with "transcript" and "output" keys. No other text.
Vary the length: some short (1-2 sentences), some medium (paragraph), some long (multiple paragraphs)."""

# ── Pair validation ────────────────────────────────────────────────────────


def validate_pair(pair: Any, category: str) -> str | None:
    """Validate a single transcript/output pair.

    Returns
    -------
    str | None
        ``None`` if the pair is valid; otherwise a human-readable reason
        for rejection.
    """
    if not isinstance(pair, dict):
        return f"expected dict, got {type(pair).__name__}"

    transcript = pair.get("transcript", "")
    output = pair.get("output", "")

    if not isinstance(transcript, str) or not transcript.strip():
        return "empty or missing 'transcript' field"
    if not isinstance(output, str) or not output.strip():
        return "empty or missing 'output' field"

    # Reject encoding issues: null bytes
    if "\x00" in transcript or "\x00" in output:
        return "contains null bytes"

    # Output should not be excessively longer than input (≤ ~2× input length)
    transcript_len = len(transcript.strip())
    output_len = len(output.strip())
    if transcript_len > 0 and output_len > 2.0 * transcript_len:
        return (
            f"output too long relative to input "
            f"(output={output_len}, input={transcript_len}, "
            f"ratio={output_len / transcript_len:.1f}x)"
        )

    # Reject output that still contains obvious filler words
    # (only standalone fillers, not substrings of real words)
    _filler_pattern = r"\b(?:um|uh|uh huh|you know|I mean|like basically)\b"
    if re.search(_filler_pattern, output, re.IGNORECASE):
        return "output still contains obvious filler words"

    return None


# ── JSON response parsing ──────────────────────────────────────────────────


def parse_llm_response(raw: str) -> list[dict[str, Any]]:
    """Parse the LLM response into a list of pair dicts.

    Handles cases where the LLM wraps the JSON in markdown code fences.

    Raises
    ------
    ValueError
        If the response cannot be parsed as a JSON array.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # Try to find the JSON array within the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not parse LLM response as JSON: {exc}\n"
                    f"Response (first 500 chars): {raw[:500]}"
                ) from exc
        else:
            raise ValueError(
                f"Could not parse LLM response as JSON array: {exc}\n"
                f"Response (first 500 chars): {raw[:500]}"
            ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected JSON array, got {type(parsed).__name__}: "
            f"{raw[:200]}"
        )

    return parsed


# ── File I/O ───────────────────────────────────────────────────────────────


def category_output_path(category: str) -> Path:
    """Return the JSONL path for a given category."""
    return DATA_SYNTHETIC / f"synthetic_{category}.jsonl"


def combined_output_path() -> Path:
    """Return the combined JSONL output path."""
    return DATA_SYNTHETIC / "all_synthetic.jsonl"


def count_existing_records(path: Path) -> int:
    """Count valid JSON records in an existing JSONL file.

    Only lines that parse as valid JSON objects with 'input' and 'output'
    keys are counted, protecting against partial/corrupt lines from crashes.
    """
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if isinstance(record, dict) and "input" in record and "output" in record:
                    count += 1
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSONL line in %s", path)
    return count


def append_pairs(
    path: Path,
    pairs: list[dict[str, str]],
    category: str,
) -> None:
    """Append validated pairs to a JSONL file."""
    with open(path, "a", encoding="utf-8") as fh:
        for pair in pairs:
            record = {
                "input": pair["transcript"],
                "output": pair["output"],
                "category": category,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Cost estimation ────────────────────────────────────────────────────────


def estimate_cost(
    synth: SyntheticConfig,
    total_samples: int,
) -> dict[str, Any]:
    """Estimate API cost for generation.

    Uses rough token estimates: ~1500 input tokens + ~2000 output tokens
    per batch call for Anthropic Claude Sonnet pricing.
    """
    total_batches = math.ceil(total_samples / synth.batch_size)

    input_tokens_per_batch = 1500
    output_tokens_per_batch = 2000

    total_input_tokens = total_batches * input_tokens_per_batch
    total_output_tokens = total_batches * output_tokens_per_batch

    # Pricing varies by provider/model; use Anthropic Sonnet as baseline
    # claude-sonnet-4-20250514: $3/1M input, $15/1M output
    cost_per_m_input = 3.0
    cost_per_m_output = 15.0

    input_cost = (total_input_tokens / 1_000_000) * cost_per_m_input
    output_cost = (total_output_tokens / 1_000_000) * cost_per_m_output
    total_cost = input_cost + output_cost

    return {
        "total_samples": total_samples,
        "batch_size": synth.batch_size,
        "total_batches": total_batches,
        "est_input_tokens": total_input_tokens,
        "est_output_tokens": total_output_tokens,
        "est_cost_usd": round(total_cost, 2),
        "pricing_note": (
            f"Estimate based on ~{input_tokens_per_batch} input + "
            f"~{output_tokens_per_batch} output tokens/batch at "
            f"${cost_per_m_input}/M input, ${cost_per_m_output}/M output "
            f"(Anthropic Sonnet baseline)"
        ),
    }


# ── Category plan ──────────────────────────────────────────────────────────


def build_category_plan(
    synth: SyntheticConfig,
    total_samples: int,
) -> list[dict[str, Any]]:
    """Build a list of per-category generation targets.

    Each entry has: category, target_count, num_batches, existing_count.
    """
    plan: list[dict[str, Any]] = []
    for category, proportion in synth.categories.items():
        target = round(total_samples * proportion)
        if target == 0:
            continue
        existing = count_existing_records(category_output_path(category))
        remaining = max(0, target - existing)
        num_batches = math.ceil(remaining / synth.batch_size) if remaining > 0 else 0
        plan.append(
            {
                "category": category,
                "proportion": proportion,
                "target": target,
                "existing": existing,
                "remaining": remaining,
                "num_batches": num_batches,
            }
        )
    return plan


# ── Batch generation ───────────────────────────────────────────────────────


def generate_batch(
    client: LLMClient,
    category: str,
    batch_size: int,
) -> tuple[list[dict[str, str]], int, int]:
    """Generate one batch of pairs from the LLM.

    Returns
    -------
    tuple
        (valid_pairs, total_parsed, rejected_count)
    """
    guidance = CATEGORY_GUIDANCE.get(category, "General content for this category.")

    prompt = GENERATION_PROMPT.format(
        batch_size=batch_size,
        category=category,
        category_guidance=guidance,
    )

    messages = [{"role": "user", "content": prompt}]

    raw_response = client.generate(
        messages,
        max_tokens=8192,
        temperature=1.0,
    )

    parsed = parse_llm_response(raw_response)

    valid: list[dict[str, str]] = []
    rejected = 0
    for pair in parsed:
        reason = validate_pair(pair, category)
        if reason:
            logger.debug(
                "Rejected pair in '%s': %s (transcript: %.60s...)",
                category,
                reason,
                pair.get("transcript", "")[:60],
            )
            rejected += 1
        else:
            valid.append(pair)

    return valid, len(parsed), rejected


def generate_category(
    client: LLMClient,
    category: str,
    plan_entry: dict[str, Any],
    synth: SyntheticConfig,
    dry_run: bool = False,
) -> dict[str, int]:
    """Generate all pairs for a single category.

    Uses a target-based while loop so rejections/failures don't cause
    silent under-generation.  Caps accepted pairs to ``still_needed``
    so the category doesn't overshoot.

    Returns per-category stats dict.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    target = plan_entry["target"]
    existing = plan_entry["existing"]

    out_path = category_output_path(category)

    stats = {
        "generated": 0,
        "failed_batches": 0,
        "rejected_pairs": 0,
        "total_parsed": 0,
    }

    current_count = existing
    if current_count >= target:
        logger.info(
            "Category '%s': target=%d, existing=%d — already complete, skipping",
            category,
            target,
            existing,
        )
        return stats

    if dry_run:
        remaining = target - current_count
        num_batches = math.ceil(remaining / synth.batch_size)
        logger.info(
            "Category '%s': would generate %d samples in ~%d batches (target=%d, existing=%d)",
            category,
            remaining,
            num_batches,
            target,
            existing,
        )
        return stats

    logger.info(
        "Category '%s': generating up to %d samples (target=%d, existing=%d)",
        category,
        target - current_count,
        target,
        existing,
    )

    # Use tqdm for progress if available
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=target - existing,
            desc=f"  {category}",
            unit="pair",
            leave=True,
        )

    max_consecutive_failures = 5
    consecutive_failures = 0

    while current_count < target:
        still_needed = target - current_count
        batch_size = min(synth.batch_size, still_needed)

        try:
            valid_pairs, total_parsed, rejected = generate_batch(
                client, category, batch_size
            )
            stats["total_parsed"] += total_parsed
            stats["rejected_pairs"] += rejected
            consecutive_failures = 0

            if valid_pairs:
                # Cap to avoid overshooting
                accepted = valid_pairs[:still_needed]
                append_pairs(out_path, accepted, category)
                current_count += len(accepted)
                stats["generated"] += len(accepted)
                if pbar:
                    pbar.update(len(accepted))

        except (RuntimeError, ValueError) as exc:
            logger.error(
                "Batch for '%s' failed: %s",
                category,
                exc,
            )
            stats["failed_batches"] += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    "Category '%s': %d consecutive failures, stopping",
                    category,
                    max_consecutive_failures,
                )
                break
            continue

        # Small delay between batches to respect rate limits
        if current_count < target:
            time.sleep(0.5)

    if pbar:
        pbar.close()

    if current_count < target:
        logger.warning(
            "Category '%s' under target: generated=%d, target=%d (shortfall=%d)",
            category,
            current_count,
            target,
            target - current_count,
        )

    logger.info(
        "Category '%s' done: generated=%d, rejected=%d, failed_batches=%d, total=%d/%d",
        category,
        stats["generated"],
        stats["rejected_pairs"],
        stats["failed_batches"],
        current_count,
        target,
    )

    return stats


# ── Combined output ────────────────────────────────────────────────────────


def build_combined_output(categories: list[str]) -> int:
    """Merge all per-category JSONL files into a single combined file.

    Returns total record count.
    """
    combined_path = combined_output_path()
    total = 0

    with open(combined_path, "w", encoding="utf-8") as out_fh:
        for category in categories:
            cat_path = category_output_path(category)
            if not cat_path.exists():
                continue
            with open(cat_path, encoding="utf-8") as in_fh:
                for line in in_fh:
                    stripped = line.strip()
                    if stripped:
                        out_fh.write(stripped + "\n")
                        total += 1

    logger.info("Combined output: %d records → %s", total, combined_path)
    return total


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> Any:
    """Parse CLI arguments."""
    parser = base_arg_parser(
        "Generate synthetic transcript-cleanup pairs via LLM API."
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=None,
        help="Override num_samples from config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing category files and regenerate from scratch.",
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

    if not synth.enabled:
        log.info("Synthetic data generation is disabled in config. Exiting.")
        return 0

    total_samples = args.synthetic_samples or synth.num_samples
    log.info("Target: %d total synthetic samples", total_samples)

    # Build plan (before ensure_dirs so dry-run has no side effects)
    plan = build_category_plan(synth, total_samples)

    # ── Dry-run output ────────────────────────────────────────────────────
    if args.dry_run:
        cost = estimate_cost(synth, total_samples)
        log.info("=== DRY RUN — Generation Plan ===")
        log.info("Provider: %s | Model: %s", synth.provider, synth.model)
        log.info("Total samples: %d | Batch size: %d", total_samples, synth.batch_size)
        log.info("")
        log.info("%-25s %8s %8s %8s %8s", "Category", "Target", "Existing", "Remaining", "Batches")
        log.info("-" * 65)
        total_remaining = 0
        total_batches = 0
        for entry in plan:
            log.info(
                "%-25s %8d %8d %8d %8d",
                entry["category"],
                entry["target"],
                entry["existing"],
                entry["remaining"],
                entry["num_batches"],
            )
            total_remaining += entry["remaining"]
            total_batches += entry["num_batches"]
        log.info("-" * 65)
        log.info("%-25s %8s %8s %8d %8d", "TOTAL", "", "", total_remaining, total_batches)
        log.info("")
        log.info("Estimated cost: $%.2f", cost["est_cost_usd"])
        log.info("  %s", cost["pricing_note"])
        return 0

    # ── Real run ──────────────────────────────────────────────────────────
    ensure_dirs()

    # Handle --force: remove existing per-category files
    if args.force:
        for category in synth.categories:
            cat_path = category_output_path(category)
            if cat_path.exists():
                log.info("Removing existing file (--force): %s", cat_path)
                cat_path.unlink()
        combined = combined_output_path()
        if combined.exists():
            combined.unlink()
        # Rebuild plan after deleting files
        plan = build_category_plan(synth, total_samples)

    # ── Create LLM client ─────────────────────────────────────────────────
    try:
        client = create_client_from_config(
            provider=synth.provider,
            model=synth.model,
            api_key_env=synth.api_key_env,
            base_url=synth.base_url,
        )
    except (EnvironmentError, ImportError, ValueError) as exc:
        log.error("Failed to create LLM client: %s", exc)
        return 1

    # ── Generate per category ─────────────────────────────────────────────
    start_time = time.monotonic()
    all_stats: dict[str, dict[str, int]] = {}

    for entry in plan:
        category = entry["category"]
        cat_stats = generate_category(
            client=client,
            category=category,
            plan_entry=entry,
            synth=synth,
            dry_run=False,
        )
        all_stats[category] = cat_stats

    # ── Build combined output ─────────────────────────────────────────────
    all_categories = list(synth.categories.keys())
    combined_count = build_combined_output(all_categories)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    total_generated = sum(s["generated"] for s in all_stats.values())
    total_rejected = sum(s["rejected_pairs"] for s in all_stats.values())
    total_failed = sum(s["failed_batches"] for s in all_stats.values())

    log.info("")
    log.info("=== Generation Summary ===")
    log.info("Elapsed: %.1fs", elapsed)
    log.info("Total generated: %d", total_generated)
    log.info("Total rejected pairs: %d", total_rejected)
    log.info("Total failed batches: %d", total_failed)
    log.info("Combined file: %d records", combined_count)
    log.info("")
    log.info("%-25s %10s %10s %10s", "Category", "Generated", "Rejected", "Failed")
    log.info("-" * 60)
    for category in all_categories:
        s = all_stats.get(category, {"generated": 0, "rejected_pairs": 0, "failed_batches": 0})
        log.info(
            "%-25s %10d %10d %10d",
            category,
            s["generated"],
            s["rejected_pairs"],
            s["failed_batches"],
        )
    log.info("-" * 60)

    if total_failed > 0:
        log.warning("Some batches failed — generated data may be below target.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
