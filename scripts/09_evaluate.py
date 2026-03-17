#!/usr/bin/env python3
"""Evaluate quantized MLX models on the test set.

Loads a quantized model produced by step 08, runs inference on samples from
``data/combined/test.jsonl``, computes metrics (exact match, CER, BLEU,
format accuracy, latency), and writes reports to ``eval_results/{model_name}/``.

This script MUST be run on a Mac with Apple Silicon (MLX is required).
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common import (
    DATA_COMBINED,
    EVAL_RESULTS,
    MODELS_QUANTIZED,
    ModelConfig,
    PipelineConfig,
    add_model_arg,
    base_arg_parser,
    ensure_dirs,
    load_config,
    load_system_prompt,
    resolve_models,
    setup_logging,
)

logger = logging.getLogger("aawaaz.evaluate")

# ── Filler / format-check patterns ─────────────────────────────────────────

# Fillers that are almost never legitimate in cleaned text
STRONG_FILLERS = re.compile(
    r"\b(?:um|uh|umm|uhh|hmm|hm|er|erm)\b", re.IGNORECASE
)

# Self-correction phrases that should have been resolved
SELF_CORRECTION_PATTERNS = re.compile(
    r"\b(?:wait\s+no|scratch\s+that|no\s+wait|I\s+mean(?:t)?)\b",
    re.IGNORECASE,
)

# Spoken number words — only checked when expected output has digits
SPOKEN_NUMBERS = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten"
    r"|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen"
    r"|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy"
    r"|eighty|ninety|hundred|thousand|million|billion)\b",
    re.IGNORECASE,
)

# Emoji detection (broad Unicode range)
EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\u2600-\u26ff"  # misc symbols
    "\u2700-\u27bf"  # dingbats
    "]+",
    flags=re.UNICODE,
)


# ── Data classes ───────────────────────────────────────────────────────────


@dataclass
class FormatSubChecks:
    """Results of individual format-accuracy sub-checks for one sample."""

    number_format_ok: bool | None = None  # None = not applicable
    filler_removal_ok: bool | None = None
    self_correction_ok: bool | None = None
    punctuation_ok: bool | None = None
    emoji_ok: bool | None = None


@dataclass
class SampleResult:
    """Evaluation result for a single test sample."""

    index: int
    input_text: str
    expected: str
    raw_generated: str
    generated: str  # after think-tag stripping
    had_think_tags: bool
    exact_match: bool
    cer: float
    sentence_bleu: float
    format_sub_checks: FormatSubChecks
    format_accuracy: float  # mean of applicable sub-checks
    generation_seconds: float
    generated_token_count: int


# ── Argument parsing ──────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> "argparse.Namespace":
    """Parse command-line arguments."""
    import argparse

    parser = base_arg_parser(
        description="Evaluate quantized MLX models on the test set.",
    )
    add_model_arg(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing evaluation results without prompting.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override num_samples from config.yaml (must be > 0).",
    )
    args = parser.parse_args(argv)
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer.")
    return args


# ── Platform & prerequisite checks ────────────────────────────────────────


def _check_platform(dry_run: bool) -> None:
    """Verify we're running on a Mac with Apple Silicon."""
    import platform as plat

    if sys.platform != "darwin":
        msg = (
            "This script requires macOS with Apple Silicon (MLX). "
            f"Detected platform: {sys.platform}"
        )
        if dry_run:
            logger.warning("[DRY-RUN] %s — continuing in dry-run mode.", msg)
        else:
            raise RuntimeError(msg)
        return

    machine = plat.machine()
    if machine != "arm64":
        logger.warning(
            "Expected Apple Silicon (arm64) but detected %s. "
            "MLX performance may be degraded or unsupported.",
            machine,
        )


def _check_mlx_available() -> None:
    """Verify that mlx_lm is importable."""
    try:
        __import__("mlx_lm")
    except ImportError:
        raise RuntimeError(
            "mlx-lm is required but not installed. "
            "Install it: pip install mlx-lm"
        )


def _check_metric_dependencies() -> None:
    """Verify metric libraries are available before starting inference."""
    missing: list[str] = []
    try:
        __import__("editdistance")
    except ImportError:
        missing.append("editdistance")
    try:
        __import__("nltk")
    except ImportError:
        missing.append("nltk")

    if missing:
        raise RuntimeError(
            f"Required metric libraries not installed: {', '.join(missing)}. "
            "Install them: pip install " + " ".join(missing)
        )


# ── Data loading ─────────────────────────────────────────────────────────


def _load_test_data(
    num_samples: int, seed: int
) -> list[dict[str, str]]:
    """Load test examples from test.jsonl.

    Returns a list of dicts with 'input' and 'expected' keys, sampled
    deterministically using the given seed.
    """
    test_path = DATA_COMBINED / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}. Run 04_prepare_data.py first."
        )

    examples: list[dict[str, str]] = []
    with open(test_path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed line %d in test.jsonl: %s", line_num, exc
                )
                continue

            messages = record.get("messages", [])
            input_text = ""
            expected_text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                if role == "user":
                    input_text = content
                elif role == "assistant":
                    expected_text = content

            if not input_text:
                logger.warning(
                    "Line %d: no user message found, skipping.", line_num
                )
                continue

            if not expected_text:
                logger.warning(
                    "Line %d: no assistant message (empty reference), skipping.",
                    line_num,
                )
                continue

            examples.append({"input": input_text, "expected": expected_text})

    if not examples:
        raise ValueError(f"No valid test examples found in {test_path}")

    logger.info("Loaded %d test examples from %s", len(examples), test_path)

    if num_samples >= len(examples):
        logger.info(
            "Requested %d samples but only %d available — using all.",
            num_samples,
            len(examples),
        )
        return examples

    rng = random.Random(seed)
    sampled = rng.sample(examples, num_samples)
    logger.info("Sampled %d/%d examples (seed=%d)", num_samples, len(examples), seed)
    return sampled


# ── Model validation ─────────────────────────────────────────────────────


def _verify_model_dir(model_path: Path) -> None:
    """Verify a model directory looks like a valid quantized MLX model.

    Warns on missing files but only raises on obviously invalid dirs.
    """
    if not model_path.is_dir():
        raise RuntimeError(f"Model path is not a directory: {model_path}")

    files = {f.name for f in model_path.iterdir() if f.is_file()}
    safetensors = [f for f in files if f.endswith(".safetensors")]

    if "config.json" not in files:
        raise RuntimeError(
            f"Model directory missing config.json: {model_path}"
        )
    if not safetensors:
        raise RuntimeError(
            f"Model directory has no .safetensors files: {model_path}"
        )
    if "tokenizer.json" not in files and "tokenizer_config.json" not in files:
        logger.warning(
            "Model directory may be missing tokenizer files: %s", model_path
        )


# ── Inference ────────────────────────────────────────────────────────────


def _strip_think_tags(text: str) -> tuple[str, bool]:
    """Strip <think>...</think> blocks from generated text.

    Returns (cleaned_text, had_think_tags).
    """
    if "<think>" not in text:
        return text, False

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned, True


def _run_inference(
    model_path: Path,
    examples: list[dict[str, str]],
    system_prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> list[SampleResult]:
    """Run inference on all examples and return per-sample results."""
    import mlx_lm

    logger.info("Loading model from %s ...", model_path)
    model, tokenizer = mlx_lm.load(str(model_path))

    results: list[SampleResult] = []
    think_tag_count = 0

    for i, example in enumerate(examples):
        input_text = example["input"]
        expected = example["expected"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ]

        # Apply chat template with thinking mode disabled
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            if i == 0:
                logger.warning(
                    "Tokenizer does not support enable_thinking; "
                    "proceeding without it."
                )
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Generate with timing
        start_time = time.perf_counter()
        from mlx_lm.sample_utils import make_sampler
        raw_generated = mlx_lm.generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        )
        elapsed = time.perf_counter() - start_time

        # Count tokens from raw output (before think-tag stripping)
        token_count = len(tokenizer.encode(raw_generated))

        # Strip think tags
        generated, had_think = _strip_think_tags(raw_generated)
        if had_think:
            think_tag_count += 1
            logger.warning(
                "Sample %d: output contained <think> tags — stripped.", i
            )

        # Compute per-sample metrics
        exact = generated.strip() == expected.strip()
        cer = _compute_cer(generated, expected)
        sbleu = _compute_sentence_bleu(generated, expected)
        fmt_checks = _compute_format_sub_checks(input_text, expected, generated)
        fmt_accuracy = _format_accuracy_score(fmt_checks)

        results.append(
            SampleResult(
                index=i,
                input_text=input_text,
                expected=expected,
                raw_generated=raw_generated,
                generated=generated,
                had_think_tags=had_think,
                exact_match=exact,
                cer=cer,
                sentence_bleu=sbleu,
                format_sub_checks=fmt_checks,
                format_accuracy=fmt_accuracy,
                generation_seconds=elapsed,
                generated_token_count=token_count,
            )
        )

        if (i + 1) % 50 == 0 or i == 0:
            logger.info(
                "  Processed %d/%d samples (latest: %.1f tok/s)",
                i + 1,
                len(examples),
                token_count / max(elapsed, 1e-6),
            )

    if think_tag_count > 0:
        logger.warning(
            "Total samples with <think> tags: %d/%d",
            think_tag_count,
            len(examples),
        )

    # Release model memory
    del model
    del tokenizer

    return results


# ── Metric computation ───────────────────────────────────────────────────


def _compute_cer(generated: str, expected: str) -> float:
    """Compute Character Error Rate using Levenshtein distance."""
    import editdistance

    if not expected:
        return 0.0 if not generated else 1.0

    distance = editdistance.eval(generated, expected)
    return distance / len(expected)


def _compute_sentence_bleu(generated: str, expected: str) -> float:
    """Compute smoothed sentence-level BLEU score."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    if not expected.strip():
        return 1.0 if not generated.strip() else 0.0

    ref_tokens = expected.strip().split()
    hyp_tokens = generated.strip().split()

    if not hyp_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu(
            [ref_tokens], hyp_tokens, smoothing_function=smoothing
        )
    except (ValueError, ZeroDivisionError):
        return 0.0


def _compute_corpus_bleu(
    all_generated: list[str], all_expected: list[str]
) -> float:
    """Compute corpus-level BLEU score."""
    from nltk.translate.bleu_score import corpus_bleu

    refs = [[exp.strip().split()] for exp in all_expected]
    hyps = [gen.strip().split() for gen in all_generated]

    try:
        return corpus_bleu(refs, hyps)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _compute_format_sub_checks(
    input_text: str, expected: str, generated: str
) -> FormatSubChecks:
    """Compute format accuracy sub-checks for a single sample.

    Each sub-check is None when not applicable (the check doesn't make sense
    for this particular sample), True when the generated output passes, or
    False when it fails.
    """
    checks = FormatSubChecks()

    # ── Number format ────────────────────────────────────────────────
    # Only check when the expected output contains digits (suggesting the
    # model should have converted spoken numbers).
    has_digits_in_expected = bool(re.search(r"\d", expected))
    if has_digits_in_expected:
        # Fail if the generated output still contains spoken number words
        # that do NOT appear in the expected output (false-positive guard).
        spoken_in_gen = set(
            w.lower() for w in SPOKEN_NUMBERS.findall(generated)
        )
        spoken_in_exp = set(
            w.lower() for w in SPOKEN_NUMBERS.findall(expected)
        )
        # Numbers in generated but not expected → likely unconverted
        unconverted = spoken_in_gen - spoken_in_exp
        checks.number_format_ok = len(unconverted) == 0

    # ── Filler removal ───────────────────────────────────────────────
    # Only check strong fillers (um, uh, etc.) — words like "like" and
    # "actually" are often legitimate.
    gen_fillers = STRONG_FILLERS.findall(generated)
    exp_fillers = STRONG_FILLERS.findall(expected)
    if STRONG_FILLERS.search(input_text):
        # Input had fillers, so we expect them to be removed
        # Allow fillers that also appear in expected (some may be intentional)
        extra_fillers = len(gen_fillers) - len(exp_fillers)
        checks.filler_removal_ok = extra_fillers <= 0
    elif gen_fillers and not exp_fillers:
        # No fillers in input but model introduced some — that's wrong
        checks.filler_removal_ok = False

    # ── Self-correction ──────────────────────────────────────────────
    if SELF_CORRECTION_PATTERNS.search(input_text):
        # Input had self-corrections; they shouldn't appear in output
        checks.self_correction_ok = not SELF_CORRECTION_PATTERNS.search(
            generated
        )

    # ── Punctuation ──────────────────────────────────────────────────
    # Only check if expected has sentence-ending punctuation
    exp_has_terminal = bool(re.search(r"[.!?]$", expected.strip()))
    if exp_has_terminal and expected.strip():
        gen_has_terminal = bool(re.search(r"[.!?]$", generated.strip()))
        checks.punctuation_ok = gen_has_terminal

    # ── Emoji ────────────────────────────────────────────────────────
    if EMOJI_RE.search(expected):
        checks.emoji_ok = bool(EMOJI_RE.search(generated))

    return checks


def _format_accuracy_score(checks: FormatSubChecks) -> float:
    """Compute the mean of all applicable (non-None) sub-checks.

    Returns 1.0 if no sub-checks are applicable.
    """
    values: list[float] = []
    for val in [
        checks.number_format_ok,
        checks.filler_removal_ok,
        checks.self_correction_ok,
        checks.punctuation_ok,
        checks.emoji_ok,
    ]:
        if val is not None:
            values.append(1.0 if val else 0.0)

    return sum(values) / len(values) if values else float("nan")


def _median(values: list[float]) -> float:
    """Compute the statistical median of a list of floats."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


# ── Aggregate metrics ────────────────────────────────────────────────────


def _aggregate_metrics(
    results: list[SampleResult],
) -> dict[str, Any]:
    """Compute aggregate metrics from per-sample results."""
    n = len(results)
    if n == 0:
        return {}

    # Exact match rate
    exact_matches = sum(1 for r in results if r.exact_match)

    # CER
    cers = [r.cer for r in results]

    # Sentence BLEU (per-sample average)
    sbleus = [r.sentence_bleu for r in results]

    # Corpus BLEU
    corpus_bleu_score = _compute_corpus_bleu(
        [r.generated for r in results],
        [r.expected for r in results],
    )

    # Format accuracy — exclude samples where no sub-checks applied (NaN)
    fmt_scores_valid = [r.format_accuracy for r in results if not math.isnan(r.format_accuracy)]

    # Format sub-check aggregates
    sub_check_aggs: dict[str, dict[str, int]] = {}
    for field_name in [
        "number_format_ok",
        "filler_removal_ok",
        "self_correction_ok",
        "punctuation_ok",
        "emoji_ok",
    ]:
        applicable = [
            r
            for r in results
            if getattr(r.format_sub_checks, field_name) is not None
        ]
        if applicable:
            passed = sum(
                1
                for r in applicable
                if getattr(r.format_sub_checks, field_name) is True
            )
            sub_check_aggs[field_name] = {
                "applicable": len(applicable),
                "passed": passed,
                "rate": round(passed / len(applicable), 4),
            }

    # Latency
    total_tokens = sum(r.generated_token_count for r in results)
    total_gen_time = sum(r.generation_seconds for r in results)
    tokens_per_sec = total_tokens / max(total_gen_time, 1e-6)

    # Think-tag incidents
    think_tag_count = sum(1 for r in results if r.had_think_tags)

    return {
        "num_samples": n,
        "exact_match": {
            "count": exact_matches,
            "rate": round(exact_matches / n, 4),
        },
        "cer": {
            "mean": round(sum(cers) / n, 4),
            "median": round(_median(cers), 4),
            "min": round(min(cers), 4),
            "max": round(max(cers), 4),
        },
        "bleu": {
            "corpus_bleu": round(corpus_bleu_score, 4),
            "sentence_bleu_mean": round(sum(sbleus) / n, 4),
        },
        "format_accuracy": {
            "mean": round(
                sum(fmt_scores_valid) / len(fmt_scores_valid), 4
            ) if fmt_scores_valid else None,
            "applicable_samples": len(fmt_scores_valid),
            "total_samples": n,
            "sub_checks": sub_check_aggs,
        },
        "latency": {
            "total_tokens_generated": total_tokens,
            "total_generation_seconds": round(total_gen_time, 2),
            "tokens_per_second": round(tokens_per_sec, 2),
            "avg_seconds_per_sample": round(total_gen_time / n, 3),
        },
        "think_tag_incidents": think_tag_count,
    }


# ── Report generation ────────────────────────────────────────────────────


def _build_best_worst(
    results: list[SampleResult], k: int = 20
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the best and worst k samples ranked by CER.

    Worst = highest CER (ties broken by lower exact match, lower BLEU).
    Best = lowest CER (ties broken by exact match, higher BLEU).
    """
    sorted_by_cer = sorted(
        results,
        key=lambda r: (r.cer, -int(r.exact_match), -r.sentence_bleu),
    )

    def _sample_to_dict(r: SampleResult) -> dict[str, Any]:
        return {
            "index": r.index,
            "input": r.input_text[:500],
            "expected": r.expected[:500],
            "generated": r.generated[:500],
            "exact_match": r.exact_match,
            "cer": round(r.cer, 4),
            "sentence_bleu": round(r.sentence_bleu, 4),
            "format_accuracy": round(r.format_accuracy, 4) if not math.isnan(r.format_accuracy) else None,
        }

    best = [_sample_to_dict(r) for r in sorted_by_cer[:k]]
    worst = [_sample_to_dict(r) for r in reversed(sorted_by_cer[-k:])]
    return best, worst


def _save_json_report(
    output_dir: Path,
    model_name: str,
    model_path: str,
    cfg: PipelineConfig,
    effective_num_samples: int,
    metrics: dict[str, Any],
    best: list[dict[str, Any]],
    worst: list[dict[str, Any]],
    results: list[SampleResult],
) -> Path:
    """Save the full evaluation report as JSON."""
    report = {
        "model_name": model_name,
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "num_samples_requested": effective_num_samples,
            "num_samples_evaluated": len(results),
            "temperature": cfg.evaluation.temperature,
            "max_tokens": cfg.evaluation.max_tokens,
            "metrics_enabled": cfg.evaluation.metrics,
        },
        "metrics": metrics,
        "best_20": best,
        "worst_20": worst,
        "per_sample": [
            {
                "index": r.index,
                "input": r.input_text,
                "expected": r.expected,
                "generated": r.generated,
                "raw_generated": r.raw_generated,
                "had_think_tags": r.had_think_tags,
                "exact_match": r.exact_match,
                "cer": round(r.cer, 4),
                "sentence_bleu": round(r.sentence_bleu, 4),
                "format_accuracy": round(r.format_accuracy, 4) if not math.isnan(r.format_accuracy) else None,
                "format_sub_checks": asdict(r.format_sub_checks),
                "generation_seconds": round(r.generation_seconds, 3),
                "generated_token_count": r.generated_token_count,
            }
            for r in results
        ],
    }

    path = output_dir / "eval_summary.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("JSON report saved: %s", path)
    return path


def _save_text_report(
    output_dir: Path,
    model_name: str,
    metrics: dict[str, Any],
    best: list[dict[str, Any]],
    worst: list[dict[str, Any]],
) -> Path:
    """Save a human-readable evaluation report."""
    lines: list[str] = []
    sep = "═" * 70

    lines.append(sep)
    lines.append(f"  Evaluation Report: {model_name}")
    lines.append(sep)
    lines.append("")

    # Headline metrics
    lines.append("Headline Metrics")
    lines.append("─" * 40)
    em = metrics.get("exact_match", {})
    lines.append(
        f"  Exact Match:    {em.get('count', '?')}/{metrics.get('num_samples', '?')} "
        f"({em.get('rate', 0):.1%})"
    )
    cer = metrics.get("cer", {})
    lines.append(
        f"  CER (mean):     {cer.get('mean', 0):.4f} "
        f"(median: {cer.get('median', 0):.4f}, "
        f"range: {cer.get('min', 0):.4f}–{cer.get('max', 0):.4f})"
    )
    bleu = metrics.get("bleu", {})
    lines.append(f"  BLEU (corpus):  {bleu.get('corpus_bleu', 0):.4f}")
    lines.append(
        f"  BLEU (sent avg):{bleu.get('sentence_bleu_mean', 0):.4f}"
    )
    fmt = metrics.get("format_accuracy", {})
    fmt_mean = fmt.get("mean")
    if fmt_mean is not None:
        lines.append(f"  Format Accuracy:{fmt_mean:.4f}")
        lines.append(
            f"    ({fmt.get('applicable_samples', '?')}/{fmt.get('total_samples', '?')} "
            "samples had applicable checks)"
        )
    else:
        lines.append("  Format Accuracy: N/A (no applicable checks)")
    lines.append("")

    # Format sub-checks
    sub = fmt.get("sub_checks", {})
    if sub:
        lines.append("Format Sub-Check Breakdown")
        lines.append("─" * 40)
        for name, vals in sub.items():
            label = name.replace("_ok", "").replace("_", " ").title()
            lines.append(
                f"  {label:20s} {vals['passed']}/{vals['applicable']} "
                f"({vals['rate']:.1%})"
            )
        lines.append("")

    # Latency
    lat = metrics.get("latency", {})
    lines.append("Latency")
    lines.append("─" * 40)
    lines.append(f"  Tokens/sec:          {lat.get('tokens_per_second', 0):.1f}")
    lines.append(
        f"  Avg sec/sample:      {lat.get('avg_seconds_per_sample', 0):.3f}"
    )
    lines.append(
        f"  Total gen time:      {lat.get('total_generation_seconds', 0):.1f}s"
    )
    lines.append(
        f"  Total tokens:        {lat.get('total_tokens_generated', 0)}"
    )
    lines.append("")

    # Think-tag incidents
    think = metrics.get("think_tag_incidents", 0)
    if think > 0:
        lines.append(f"⚠ Think-tag incidents: {think}/{metrics['num_samples']}")
        lines.append("")

    # Per-category note
    lines.append("Per-Category Breakdown: N/A")
    lines.append("  (test.jsonl does not include category metadata)")
    lines.append("")

    # Worst 20
    lines.append(sep)
    lines.append(f"  Worst {len(worst)} Samples (highest CER)")
    lines.append(sep)
    for rank, s in enumerate(worst, 1):
        _append_sample_block(lines, rank, s)

    # Best 20
    lines.append(sep)
    lines.append(f"  Best {len(best)} Samples (lowest CER)")
    lines.append(sep)
    for rank, s in enumerate(best, 1):
        _append_sample_block(lines, rank, s)

    text = "\n".join(lines) + "\n"
    path = output_dir / "eval_report.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)

    logger.info("Text report saved: %s", path)
    return path


def _append_sample_block(
    lines: list[str], rank: int, s: dict[str, Any]
) -> None:
    """Append a formatted sample block to the report lines."""
    fmt_val = s.get("format_accuracy")
    fmt_str = f"{fmt_val:.2f}" if fmt_val is not None else "N/A"
    lines.append(
        f"\n  #{rank}  (CER: {s['cer']:.4f} | BLEU: {s['sentence_bleu']:.4f} | "
        f"Exact: {'✓' if s['exact_match'] else '✗'} | "
        f"Format: {fmt_str})"
    )
    lines.append(f"  INPUT:    {s['input'][:200]}")
    lines.append(f"  EXPECTED: {s['expected'][:200]}")
    lines.append(f"  GENERATED:{s['generated'][:200]}")
    lines.append("  " + "·" * 60)


# ── Core evaluation logic ───────────────────────────────────────────────


def _evaluate_model(
    mcfg: ModelConfig,
    cfg: PipelineConfig,
    *,
    num_samples_override: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate a single model."""
    bits = cfg.quantization.bits
    model_dir_name = f"{mcfg.name}-{bits}bit"
    model_path = MODELS_QUANTIZED / model_dir_name
    output_dir = EVAL_RESULTS / mcfg.name

    num_samples = num_samples_override or cfg.evaluation.num_samples

    logger.info("Evaluating model: %s", mcfg.name)
    logger.info("  Quantized model:  %s", model_path)
    logger.info("  Output dir:       %s", output_dir)
    logger.info("  Num samples:      %d", num_samples)
    logger.info("  Temperature:      %.1f", cfg.evaluation.temperature)
    logger.info("  Max tokens:       %d", cfg.evaluation.max_tokens)

    # Handle dry-run early (before requiring model to exist)
    if dry_run:
        if not model_path.exists():
            logger.info("[DRY-RUN] Quantized model not yet present: %s", model_path)
        logger.info("[DRY-RUN] Would evaluate %s with %d samples.", mcfg.name, num_samples)
        return {"status": "dry-run", "model": mcfg.name}

    # Check model exists and looks valid
    if not model_path.exists():
        raise FileNotFoundError(
            f"Quantized model not found: {model_path}. "
            "Run 08_quantize.py first."
        )
    _verify_model_dir(model_path)

    # Handle existing output
    if output_dir.exists() and any(output_dir.iterdir()):
        if force:
            import shutil

            logger.info("--force: removing existing results in %s", output_dir)
            shutil.rmtree(output_dir)
        else:
            logger.warning(
                "Evaluation results already exist: %s. "
                "Use --force to overwrite.",
                output_dir,
            )
            return {"status": "skipped", "reason": "output exists", "model": mcfg.name}

    # Ensure output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load system prompt and test data
    system_prompt = load_system_prompt(with_no_think=True)
    examples = _load_test_data(num_samples, seed=cfg.dataset.shuffle_seed)

    # Run inference
    eval_start = time.time()
    results = _run_inference(
        model_path,
        examples,
        system_prompt,
        max_tokens=cfg.evaluation.max_tokens,
        temperature=cfg.evaluation.temperature,
    )
    eval_elapsed = time.time() - eval_start

    # Compute aggregate metrics
    metrics = _aggregate_metrics(results)

    # Build best/worst lists
    best, worst = _build_best_worst(results, k=20)

    # Save reports
    _save_json_report(
        output_dir, mcfg.name, str(model_path), cfg, num_samples,
        metrics, best, worst, results,
    )
    _save_text_report(output_dir, mcfg.name, metrics, best, worst)

    logger.info(
        "Evaluation complete for %s in %.1f seconds.", mcfg.name, eval_elapsed
    )

    return {
        "status": "completed",
        "model": mcfg.name,
        "output_dir": str(output_dir),
        "num_samples": len(results),
        "eval_seconds": round(eval_elapsed, 1),
        "headline_metrics": {
            "exact_match_rate": metrics["exact_match"]["rate"],
            "cer_mean": metrics["cer"]["mean"],
            "corpus_bleu": metrics["bleu"]["corpus_bleu"],
            "format_accuracy": metrics["format_accuracy"]["mean"],
            "tokens_per_second": metrics["latency"]["tokens_per_second"],
        },
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the evaluate script."""
    args = parse_args()
    log = setup_logging(verbose=args.verbose)

    cfg = load_config(args.config)
    models = resolve_models(cfg, args.model)

    log.info(
        "Models: %s | Samples: %s | Temp: %.1f",
        ", ".join(m.name for m in models),
        args.num_samples or cfg.evaluation.num_samples,
        cfg.evaluation.temperature,
    )

    # Platform & prerequisite checks
    _check_platform(dry_run=args.dry_run)
    if not args.dry_run:
        _check_mlx_available()
        _check_metric_dependencies()
        ensure_dirs()

    # Process each model
    start = time.time()
    all_results: list[dict[str, Any]] = []
    errors: list[str] = []

    for mcfg in models:
        log.info("═" * 60)
        log.info("Evaluating: %s", mcfg.name)
        log.info("═" * 60)

        try:
            result = _evaluate_model(
                mcfg,
                cfg,
                num_samples_override=args.num_samples,
                force=args.force,
                dry_run=args.dry_run,
            )
            all_results.append(result)
        except Exception as exc:
            log.error("Failed to evaluate %s: %s", mcfg.name, exc, exc_info=True)
            errors.append(f"{mcfg.name}: {exc}")

    elapsed = time.time() - start

    # Final summary
    log.info("═" * 60)
    log.info("Evaluation Summary")
    log.info("═" * 60)
    log.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    for r in all_results:
        status = r.get("status", "unknown")
        model_name = r.get("model", "?")

        if status == "dry-run":
            log.info("  %s: [DRY-RUN]", model_name)
        elif status == "skipped":
            log.info("  %s: skipped (%s)", model_name, r.get("reason", ""))
        elif status == "completed":
            hm = r.get("headline_metrics", {})
            fmt_str = f"{hm['format_accuracy']:.2f}" if hm.get("format_accuracy") is not None else "N/A"
            log.info(
                "  %s: EM=%.1f%% CER=%.4f BLEU=%.4f Fmt=%s Tok/s=%.1f (%d samples, %.1fs)",
                model_name,
                hm.get("exact_match_rate", 0) * 100,
                hm.get("cer_mean", 0),
                hm.get("corpus_bleu", 0),
                fmt_str,
                hm.get("tokens_per_second", 0),
                r.get("num_samples", 0),
                r.get("eval_seconds", 0),
            )
        else:
            log.info("  %s: %s", model_name, status)

    if errors:
        log.error("Errors encountered:")
        for e in errors:
            log.error("  • %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
