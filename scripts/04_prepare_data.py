#!/usr/bin/env python3
"""Combine raw + synthetic data, validate, deduplicate, split, and save.

Reads:
- ``data/raw/*.jsonl``          — pairs from ``02_pull_datasets.py``
- ``data/synthetic/*.jsonl``    — pairs from ``03_generate_synthetic.py`` (validated by 03b)

Writes:
- ``data/combined/train.jsonl``
- ``data/combined/valid.jsonl``
- ``data/combined/test.jsonl``

Each output line is a chat-messages JSON object:

    {"messages": [
        {"role": "system", "content": "<system prompt + /no_think>"},
        {"role": "user", "content": "<raw transcript>"},
        {"role": "assistant", "content": "<cleaned output>"}
    ]}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Allow running as ``python scripts/04_prepare_data.py`` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DATA_COMBINED,
    DATA_RAW,
    DATA_SYNTHETIC,
    DATA_SYNTHETIC_REJECTED,
    base_arg_parser,
    ensure_dirs,
    load_config,
    load_system_prompt,
    setup_logging,
)

logger = logging.getLogger("aawaaz.prepare_data")

# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class RawPair:
    """A single input/output pair with provenance metadata."""

    input: str
    output: str
    source: str  # e.g. "raw/whisper_transcripts" or "synthetic/casual_conversation"
    category: str | None = None  # only for synthetic data


@dataclass
class RejectionRecord:
    """Tracks why a pair was rejected during quality validation."""

    pair: RawPair
    reasons: list[str]


# ── Quality validation ──────────────────────────────────────────────────────

# Unambiguous filler words to detect in output.  "like", "basically",
# "actually" are legitimate English words that appear in cleaned text — only
# flag the truly obvious speech fillers the spec calls out ("um", "uh").
_FILLER_PATTERN = re.compile(
    r"\b(um|uh|uhh|umm|uh huh)\b",
    re.IGNORECASE,
)

# Patterns for quoted speech — content inside quotes is exempt from filler checks.
_QUOTED_SPEECH = re.compile(r'"[^"]*"')

# Null byte check
_NULL_BYTE = re.compile(r"\x00")

# Mojibake heuristic: common replacement-character sequences
_MOJIBAKE_PATTERN = re.compile(r"[\ufffd]|Ã[\x80-\xbf]|â\x80|Ã©|Ã¨|Ã¢|Ã®")


# Detect <think> blocks that should not appear in training outputs
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def _has_fillers_outside_quotes(text: str) -> list[str]:
    """Return list of filler words found outside quoted speech."""
    # Remove quoted portions so fillers inside quotes are exempt
    stripped = _QUOTED_SPEECH.sub("", text)
    matches = _FILLER_PATTERN.findall(stripped)
    return [m.lower() for m in matches]


def _detect_encoding_issues(text: str) -> list[str]:
    """Detect null bytes and mojibake patterns."""
    issues: list[str] = []
    if _NULL_BYTE.search(text):
        issues.append("null bytes detected")
    if _MOJIBAKE_PATTERN.search(text):
        issues.append("possible mojibake encoding issues")
    return issues


def _detect_added_content(input_text: str, output_text: str) -> bool:
    """Heuristic: check if output adds substantial content not in input.

    Uses a simple word-overlap approach — if >20% of output words don't
    appear in the input, flag it.  This is a WARNING, not a rejection.
    """
    if not input_text.strip() or not output_text.strip():
        return False

    input_words = set(input_text.lower().split())
    output_words = output_text.lower().split()
    if not output_words:
        return False

    novel = sum(1 for w in output_words if w not in input_words)
    return (novel / len(output_words)) > 0.20


def validate_pair(pair: RawPair) -> tuple[bool, list[str], list[str]]:
    """Validate a single pair. Returns (accept, rejection_reasons, warnings)."""
    reasons: list[str] = []
    warnings: list[str] = []

    input_text = pair.input.strip()
    output_text = pair.output.strip()

    # Reject: output > 2× input length (likely hallucinated)
    if input_text and output_text:
        if len(output_text) > 2 * len(input_text):
            reasons.append(
                f"output ({len(output_text)} chars) > 2× input ({len(input_text)} chars)"
            )

    # Reject: empty output with substantive input
    # "Substantive" = more than just fillers/whitespace
    if not output_text and input_text:
        cleaned_input = _FILLER_PATTERN.sub("", input_text).strip()
        if len(cleaned_input) > 10:
            reasons.append("empty output with substantive input")

    # Reject: encoding issues in either field
    for label, text in [("input", pair.input), ("output", pair.output)]:
        enc_issues = _detect_encoding_issues(text)
        for issue in enc_issues:
            reasons.append(f"{label}: {issue}")

    # Reject: fillers remaining in output (outside quoted speech)
    if output_text:
        fillers = _has_fillers_outside_quotes(output_text)
        if fillers:
            reasons.append(f"fillers in output: {', '.join(set(fillers))}")

    # Reject: <think> blocks in output (Qwen3 thinking mode leak)
    if output_text and _THINK_TAG.search(output_text):
        reasons.append("output contains <think> tags (thinking mode leak)")

    # Warn (don't reject): output adds content not in input
    if input_text and output_text:
        if _detect_added_content(input_text, output_text):
            warnings.append("output may add content not present in input")

    accept = len(reasons) == 0
    return accept, reasons, warnings


# ── Data loading ────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts. Skips bad lines."""
    records: list[dict] = []
    try:
        fh = open(path, encoding="utf-8")
    except UnicodeDecodeError as exc:
        logger.error(
            "Cannot read %s — encoding error: %s. "
            "Ensure the file is valid UTF-8.",
            path, exc,
        )
        return records

    with fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON on line %d in %s: %s",
                    line_no, path.name, exc,
                )
                continue
            if not isinstance(obj, dict):
                logger.warning(
                    "Skipping non-object JSON on line %d in %s (got %s)",
                    line_no, path.name, type(obj).__name__,
                )
                continue
            # Ensure input/output are strings
            inp = obj.get("input")
            out = obj.get("output")
            if not isinstance(inp, str) or not isinstance(out, str):
                logger.warning(
                    "Skipping line %d in %s: 'input' and 'output' must be strings",
                    line_no, path.name,
                )
                continue
            records.append(obj)
    return records


def load_raw_data() -> list[RawPair]:
    """Load all JSONL files from ``data/raw/``."""
    pairs: list[RawPair] = []
    if not DATA_RAW.exists():
        logger.warning("Raw data directory does not exist: %s", DATA_RAW)
        return pairs

    jsonl_files = sorted(DATA_RAW.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL files found in %s", DATA_RAW)
        return pairs

    for path in jsonl_files:
        source_name = f"raw/{path.stem}"
        records = _load_jsonl(path)
        for rec in records:
            inp = rec.get("input", "")
            out = rec.get("output", "")
            pairs.append(RawPair(input=inp, output=out, source=source_name))
        logger.info("Loaded %d pairs from %s", len(records), path.name)

    return pairs


def load_synthetic_data(
    validation_enabled: bool,
    configured_categories: dict[str, float] | None = None,
) -> list[RawPair]:
    """Load accepted synthetic JSONL files from ``data/synthetic/``.

    If validation was run (``validation_report.json`` exists), only loads files
    NOT in the ``rejected/`` directory.

    If validation was NOT run but is enabled in config, warns loudly.
    If validation is disabled in config, warns and loads all synthetic data.
    """
    pairs: list[RawPair] = []
    if not DATA_SYNTHETIC.exists():
        logger.warning("Synthetic data directory does not exist: %s", DATA_SYNTHETIC)
        return pairs

    report_path = DATA_SYNTHETIC / "validation_report.json"

    if not validation_enabled:
        logger.warning(
            "Synthetic data validation is DISABLED in config. "
            "Loading all synthetic data without quality gate. "
            "Consider enabling validation for production use."
        )
    elif not report_path.exists():
        logger.warning(
            "Validation is enabled in config but validation_report.json not found. "
            "Loading all synthetic data WITHOUT validation. "
            "Run 03b_validate_synthetic.py first for quality assurance."
        )

    # Collect rejected filenames to skip
    rejected_names: set[str] = set()
    if DATA_SYNTHETIC_REJECTED.exists():
        for rpath in DATA_SYNTHETIC_REJECTED.glob("*.jsonl"):
            rejected_names.add(rpath.name)

    # Build set of expected category filenames from config
    expected_names: set[str] | None = None
    if configured_categories:
        expected_names = {
            f"synthetic_{cat}.jsonl" for cat in configured_categories
        }

    jsonl_files = sorted(DATA_SYNTHETIC.glob("synthetic_*.jsonl"))
    if not jsonl_files:
        logger.warning("No synthetic JSONL files found in %s", DATA_SYNTHETIC)
        return pairs

    for path in jsonl_files:
        # Skip if a copy exists in rejected/ (category was rejected)
        if path.name in rejected_names:
            logger.info(
                "Skipping rejected synthetic file: %s (found in rejected/)", path.name
            )
            continue

        # Skip unexpected files not in config categories
        if expected_names is not None and path.name not in expected_names:
            logger.warning(
                "Skipping unexpected synthetic file: %s (not in config categories)",
                path.name,
            )
            continue

        # Extract category from filename: synthetic_casual_conversation.jsonl -> casual_conversation
        category = path.stem.removeprefix("synthetic_")
        source_name = f"synthetic/{category}"
        records = _load_jsonl(path)
        for rec in records:
            inp = rec.get("input", "")
            out = rec.get("output", "")
            pairs.append(
                RawPair(
                    input=inp, output=out, source=source_name, category=category
                )
            )
        logger.info("Loaded %d pairs from %s", len(records), path.name)

    return pairs


# ── Deduplication ───────────────────────────────────────────────────────────


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for deduplication: lowercase, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate(pairs: list[RawPair]) -> tuple[list[RawPair], int]:
    """Remove duplicates by normalized input text. Keeps first occurrence."""
    seen: set[str] = set()
    unique: list[RawPair] = []
    dup_count = 0

    for pair in pairs:
        key = _normalize_for_dedup(pair.input)
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        unique.append(pair)

    return unique, dup_count


# ── Chat-messages formatting ───────────────────────────────────────────────


def format_as_chat_messages(
    pair: RawPair, system_prompt: str
) -> dict:
    """Convert a RawPair to chat-messages format."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair.input},
            {"role": "assistant", "content": pair.output},
        ]
    }


# ── File writing ────────────────────────────────────────────────────────────


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Write records to JSONL atomically (write-to-temp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Statistics ──────────────────────────────────────────────────────────────


def _compute_stats(
    pairs: list[RawPair],
) -> dict:
    """Compute summary statistics for a list of pairs."""
    if not pairs:
        return {"count": 0}

    input_lengths = [len(p.input) for p in pairs]
    output_lengths = [len(p.output) for p in pairs]
    source_counts = Counter(p.source for p in pairs)

    return {
        "count": len(pairs),
        "avg_input_length": sum(input_lengths) / len(input_lengths),
        "avg_output_length": sum(output_lengths) / len(output_lengths),
        "min_input_length": min(input_lengths),
        "max_input_length": max(input_lengths),
        "min_output_length": min(output_lengths),
        "max_output_length": max(output_lengths),
        "per_source": dict(source_counts.most_common()),
    }


def _log_stats(label: str, stats: dict) -> None:
    """Log computed statistics."""
    logger.info(
        "%s — %d samples, avg input: %.0f chars, avg output: %.0f chars",
        label,
        stats.get("count", 0),
        stats.get("avg_input_length", 0),
        stats.get("avg_output_length", 0),
    )
    per_source = stats.get("per_source", {})
    if per_source:
        for source, count in per_source.items():
            logger.info("  %s: %d", source, count)


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = base_arg_parser(
        description="Combine, validate, deduplicate, format, and split training data."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files without prompting.",
    )
    return parser


# ── Main ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    log = setup_logging(verbose=args.verbose)
    start_time = time.monotonic()

    # ── Load config ──
    cfg = load_config(args.config)
    log.info("Loaded config from %s", args.config)

    # ── Dry-run plan ──
    if args.dry_run:
        log.info("[DRY-RUN] Would load raw data from %s", DATA_RAW)
        log.info("[DRY-RUN] Would load synthetic data from %s", DATA_SYNTHETIC)
        log.info(
            "[DRY-RUN] Split ratios: train=%.0f%%, valid=%.0f%%, test=%.0f%%",
            cfg.dataset.train_ratio * 100,
            cfg.dataset.valid_ratio * 100,
            cfg.dataset.test_ratio * 100,
        )
        log.info("[DRY-RUN] Output directory: %s", DATA_COMBINED)
        log.info("[DRY-RUN] Shuffle seed: %d", cfg.dataset.shuffle_seed)
        raw_files = sorted(DATA_RAW.glob("*.jsonl")) if DATA_RAW.exists() else []
        synth_files = (
            sorted(DATA_SYNTHETIC.glob("synthetic_*.jsonl"))
            if DATA_SYNTHETIC.exists()
            else []
        )
        log.info("[DRY-RUN] Raw files: %s", [f.name for f in raw_files])
        log.info("[DRY-RUN] Synthetic files: %s", [f.name for f in synth_files])
        return 0

    # ── Check for existing output ──
    output_files = [
        DATA_COMBINED / "train.jsonl",
        DATA_COMBINED / "valid.jsonl",
        DATA_COMBINED / "test.jsonl",
    ]
    existing = [f for f in output_files if f.exists() and f.stat().st_size > 0]
    if existing and not args.force:
        log.error(
            "Output files already exist: %s. Use --force to overwrite.",
            [f.name for f in existing],
        )
        return 1

    # ── Ensure output directory ──
    ensure_dirs()

    # ── Load system prompt ──
    system_prompt = load_system_prompt(with_no_think=True)
    log.info("System prompt loaded (%d chars, with /no_think)", len(system_prompt))

    # ── Load data ──
    log.info("Loading raw data...")
    raw_pairs = load_raw_data()
    log.info("Total raw pairs: %d", len(raw_pairs))

    log.info("Loading synthetic data...")
    synthetic_pairs = load_synthetic_data(
        validation_enabled=cfg.dataset.synthetic.validation.enabled,
        configured_categories=cfg.dataset.synthetic.categories or None,
    )
    log.info("Total synthetic pairs: %d", len(synthetic_pairs))

    all_pairs = raw_pairs + synthetic_pairs
    log.info("Total combined pairs before validation: %d", len(all_pairs))

    if not all_pairs:
        log.error("No data loaded. Nothing to process.")
        return 1

    # ── Quality validation ──
    log.info("Running data quality validation...")
    accepted: list[RawPair] = []
    rejections: list[RejectionRecord] = []
    warning_count = 0
    rejection_reason_counts: Counter[str] = Counter()
    max_rejection_examples = 20  # Cap per-pair rejection log lines

    for pair in all_pairs:
        ok, reasons, warnings = validate_pair(pair)
        if ok:
            accepted.append(pair)
            if warnings:
                warning_count += 1
                for w in warnings:
                    logger.debug(
                        "WARNING [%s]: %s (input: %.60s...)",
                        pair.source,
                        w,
                        pair.input,
                    )
        else:
            rejections.append(RejectionRecord(pair=pair, reasons=reasons))
            for r in reasons:
                # Normalize reason keys for counting
                key = r.split(":")[0] if ":" in r else r
                rejection_reason_counts[key] += 1
            # Log individual rejections at WARNING, capped to avoid flooding
            if len(rejections) <= max_rejection_examples:
                logger.warning(
                    "REJECTED [%s]: %s (input: %.80s...)",
                    pair.source,
                    "; ".join(reasons),
                    pair.input,
                )

    log.info(
        "Quality validation: %d accepted, %d rejected, %d warnings",
        len(accepted),
        len(rejections),
        warning_count,
    )
    if len(rejections) > max_rejection_examples:
        log.info(
            "(Showed first %d rejection details; use --verbose for all)",
            max_rejection_examples,
        )
    if rejection_reason_counts:
        log.info("Rejection breakdown:")
        for reason, count in rejection_reason_counts.most_common():
            log.info("  %s: %d", reason, count)

    if not accepted:
        log.error("All pairs were rejected. Nothing to save.")
        return 1

    # ── Deduplication ──
    log.info("Deduplicating by normalized input text...")
    unique_pairs, dup_count = deduplicate(accepted)
    log.info(
        "Deduplication: %d unique, %d duplicates removed",
        len(unique_pairs),
        dup_count,
    )

    # ── Per-source breakdown (overall) ──
    overall_stats = _compute_stats(unique_pairs)
    _log_stats("Overall accepted", overall_stats)

    # ── Shuffle ──
    seed = cfg.dataset.shuffle_seed
    log.info("Shuffling with seed %d...", seed)
    random.seed(seed)
    random.shuffle(unique_pairs)

    # ── Split ──
    total = len(unique_pairs)
    train_end = int(total * cfg.dataset.train_ratio)
    valid_end = train_end + int(total * cfg.dataset.valid_ratio)

    train_pairs = unique_pairs[:train_end]
    valid_pairs = unique_pairs[train_end:valid_end]
    test_pairs = unique_pairs[valid_end:]

    log.info(
        "Split sizes — train: %d (%.1f%%), valid: %d (%.1f%%), test: %d (%.1f%%)",
        len(train_pairs),
        len(train_pairs) / total * 100 if total else 0,
        len(valid_pairs),
        len(valid_pairs) / total * 100 if total else 0,
        len(test_pairs),
        len(test_pairs) / total * 100 if total else 0,
    )

    # ── Format as chat messages ──
    log.info("Formatting as chat messages...")
    train_records = [format_as_chat_messages(p, system_prompt) for p in train_pairs]
    valid_records = [format_as_chat_messages(p, system_prompt) for p in valid_pairs]
    test_records = [format_as_chat_messages(p, system_prompt) for p in test_pairs]

    # ── Save ──
    log.info("Writing output files to %s...", DATA_COMBINED)
    _write_jsonl_atomic(DATA_COMBINED / "train.jsonl", train_records)
    log.info("  train.jsonl: %d records", len(train_records))
    _write_jsonl_atomic(DATA_COMBINED / "valid.jsonl", valid_records)
    log.info("  valid.jsonl: %d records", len(valid_records))
    _write_jsonl_atomic(DATA_COMBINED / "test.jsonl", test_records)
    log.info("  test.jsonl: %d records", len(test_records))

    # ── Summary stats ──
    log.info("── Final Summary ──")
    log.info("Total loaded: %d", len(all_pairs))
    log.info("Quality rejections: %d", len(rejections))
    log.info("Duplicates removed: %d", dup_count)
    log.info("Unique accepted: %d", len(unique_pairs))

    for label, pairs_list in [
        ("Train", train_pairs),
        ("Valid", valid_pairs),
        ("Test", test_pairs),
    ]:
        stats = _compute_stats(pairs_list)
        _log_stats(label, stats)

    elapsed = time.monotonic() - start_time
    log.info("Done in %.1f seconds.", elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
