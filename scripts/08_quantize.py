#!/usr/bin/env python3
"""Quantize fused models to 4-bit MLX format.

Takes a full-precision fused model (from step 07) and produces a quantized
MLX model suitable for on-device inference via ``mlx-swift-lm``.

Input path resolution (in priority order):
  1. ``models/mlx/{model_name}``  — Linux→Mac converted MLX model
  2. ``models/fused/{model_name}`` — Mac-native fused model

Output:
  ``models/quantized/{model_name}-{bits}bit/``

This step MUST be run on a Mac with Apple Silicon (MLX is required).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    DATA_COMBINED,
    MODELS_FUSED,
    MODELS_MLX,
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

logger = logging.getLogger("aawaaz.quantize")

# Files we expect in a valid quantized model directory.
EXPECTED_OUTPUT_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json"}
SAFETENSOR_GLOB = "*.safetensors"

# Fallback sanity-check prompts if test.jsonl is unavailable.
FALLBACK_SANITY_PROMPTS = [
    "um so i went to the store and i bought like five hundred dollars worth of groceries",
    "the meeting is at two thirty pm on january fifteenth two thousand twenty six",
    "wait no i meant the deadline is friday not thursday so yeah lets plan for that",
    "hey can you send me the file at h t t p colon slash slash example dot com slash report",
    "basically the function takes two arguments x and y and returns x plus y",
]


# ── Argument parsing ───────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> "argparse.Namespace":
    """Parse command-line arguments."""
    import argparse

    parser = base_arg_parser(
        description="Quantize fused models to 4-bit MLX format.",
    )
    add_model_arg(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing quantized output without prompting.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip the post-quantization sanity check (generation test).",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the full-precision vs quantized output comparison.",
    )
    return parser.parse_args(argv)


# ── Platform checks ───────────────────────────────────────────────────────


def _check_platform(dry_run: bool) -> None:
    """Verify we're running on a Mac with Apple Silicon.

    Raises RuntimeError on non-Darwin platforms unless in dry-run mode.
    """
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


# ── Path resolution ──────────────────────────────────────────────────────


def _resolve_source_path(mcfg: ModelConfig) -> Path:
    """Find the full-precision source model to quantize.

    Prefers ``models/mlx/{model_name}`` (Linux→Mac converted), then falls
    back to ``models/fused/{model_name}`` (Mac-native fused).

    Raises FileNotFoundError if neither exists.
    """
    mlx_path = MODELS_MLX / mcfg.name
    fused_path = MODELS_FUSED / mcfg.name

    if mlx_path.exists():
        logger.info("Using MLX-format source: %s", mlx_path)
        return mlx_path

    if fused_path.exists():
        logger.info("Using fused model source: %s", fused_path)
        return fused_path

    raise FileNotFoundError(
        f"No source model found for '{mcfg.name}'. "
        f"Checked:\n  {mlx_path}\n  {fused_path}\n"
        "Run 07_fuse_and_convert.py first."
    )


def _output_dir_name(model_name: str, bits: int) -> str:
    """Build the quantized output directory name."""
    return f"{model_name}-{bits}bit"


# ── Validation helpers ────────────────────────────────────────────────────


def _check_not_already_quantized(source_dir: Path) -> None:
    """Abort if the source model is already quantized.

    Parses ``config.json`` and looks for quantization metadata.
    """
    config_path = source_dir / "config.json"
    if not config_path.exists():
        logger.warning(
            "No config.json in source dir %s — cannot verify quantization status.",
            source_dir,
        )
        return

    try:
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Failed to parse source config.json at %s: %s — "
            "cannot verify quantization status.",
            config_path,
            exc,
        )
        return

    if "quantization" in config or "quantization_config" in config:
        raise RuntimeError(
            f"Source model at {source_dir} appears to already be quantized "
            "(config.json contains quantization metadata). "
            "Always quantize from the full-precision fused model."
        )


def _verify_source_dir(source_dir: Path) -> bool:
    """Check that a source directory looks like a valid model."""
    if not source_dir.exists():
        logger.error("Source directory does not exist: %s", source_dir)
        return False

    files = {f.name for f in source_dir.iterdir() if f.is_file()}
    safetensors = list(source_dir.glob(SAFETENSOR_GLOB))

    if "config.json" not in files:
        logger.warning("Source is missing config.json: %s", source_dir)
        return False

    if not safetensors:
        logger.warning("Source has no .safetensors weight files: %s", source_dir)
        return False

    logger.info(
        "Source verified: %d files, %d safetensors",
        len(files),
        len(safetensors),
    )
    return True


def _verify_quantized_output(output_dir: Path) -> bool:
    """Verify the quantized output directory has all expected files.

    Also checks that config.json contains quantization metadata.
    Returns False if critical files or metadata are missing.
    """
    if not output_dir.exists():
        logger.error("Output directory does not exist: %s", output_dir)
        return False

    files = {f.name for f in output_dir.iterdir() if f.is_file()}
    safetensors = list(output_dir.glob(SAFETENSOR_GLOB))

    missing = EXPECTED_OUTPUT_FILES - files
    if missing:
        logger.warning("Output is missing expected files: %s", missing)
        return False

    if not safetensors:
        logger.warning("Output has no .safetensors weight files.")
        return False

    # Verify quantization metadata in config.json
    config_path = output_dir / "config.json"
    try:
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse output config.json: %s", exc)
        return False

    if "quantization" not in config and "quantization_config" not in config:
        logger.warning(
            "Output config.json does not contain quantization metadata. "
            "The model may not have been quantized correctly."
        )
        return False

    logger.info(
        "Output verified: %d files, %d safetensors",
        len(files),
        len(safetensors),
    )
    return True


def _dir_size_mb(path: Path) -> float:
    """Calculate total size of files in a directory, in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


# ── Sanity check (generation test) ───────────────────────────────────────


def _load_test_prompts(max_prompts: int = 5) -> list[str]:
    """Load test prompts from test.jsonl, falling back to built-in prompts.

    Attempts to read user-turn content from ``data/combined/test.jsonl``.
    If the file is unavailable or malformed, uses hardcoded fallback prompts.
    """
    test_path = DATA_COMBINED / "test.jsonl"
    if test_path.exists():
        try:
            prompts: list[str] = []
            with open(test_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    messages = record.get("messages", [])
                    # Find the user message (the transcript to clean)
                    for msg in messages:
                        if msg.get("role") == "user":
                            content = msg.get("content", "").strip()
                            if content:
                                prompts.append(content)
                                break
                    if len(prompts) >= max_prompts:
                        break

            if prompts:
                # Top up from fallback if we have fewer than max_prompts
                if len(prompts) < max_prompts:
                    shortfall = max_prompts - len(prompts)
                    prompts.extend(FALLBACK_SANITY_PROMPTS[:shortfall])
                    logger.info(
                        "Loaded %d prompts from %s + %d fallback prompts",
                        len(prompts) - shortfall,
                        test_path,
                        shortfall,
                    )
                else:
                    logger.info(
                        "Loaded %d test prompts from %s", len(prompts), test_path
                    )
                return prompts
            logger.warning(
                "test.jsonl exists but no valid prompts found; using fallback."
            )
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning(
                "Failed to load test prompts from %s: %s — using fallback.",
                test_path,
                exc,
            )

    logger.info("Using %d built-in fallback sanity-check prompts.", len(FALLBACK_SANITY_PROMPTS))
    return FALLBACK_SANITY_PROMPTS[:max_prompts]


def _build_chat_messages(
    system_prompt: str, user_text: str
) -> list[dict[str, str]]:
    """Build a chat message list for inference."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def _generate_outputs(
    model_path: Path,
    prompts: list[str],
    system_prompt: str,
    *,
    label: str = "model",
) -> list[str]:
    """Load a model and generate outputs for the given prompts.

    Returns a list of generated text strings.
    """
    import mlx_lm

    logger.info("Loading %s from %s for generation...", label, model_path)
    model, tokenizer = mlx_lm.load(str(model_path))

    outputs: list[str] = []
    for i, prompt_text in enumerate(prompts, 1):
        messages = _build_chat_messages(system_prompt, prompt_text)

        # Apply chat template with thinking mode disabled
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizer version may not support enable_thinking
            logger.warning(
                "Tokenizer does not support enable_thinking; "
                "proceeding without it."
            )
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        from mlx_lm.sample_utils import make_sampler
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=1024,
            sampler=make_sampler(temp=0.0),
        )

        # Strip <think> tags if present
        if "<think>" in response:
            import re

            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            logger.warning(
                "[%s] Prompt %d: output contained <think> tags — stripped.",
                label,
                i,
            )
            response = cleaned

        outputs.append(response)

    # Release model from memory
    del model
    del tokenizer

    return outputs


def _run_sanity_check(
    quantized_dir: Path,
    source_dir: Path,
    *,
    skip_comparison: bool = False,
) -> dict[str, Any]:
    """Run post-quantization sanity checks.

    1. Generate 5 test outputs from the quantized model and print them.
    2. Optionally compare outputs between full-precision and quantized models.
    """
    system_prompt = load_system_prompt(with_no_think=True)
    prompts = _load_test_prompts(max_prompts=5)

    result: dict[str, Any] = {"prompts_used": len(prompts)}

    # ── 1. Quantized model smoke test ──────────────────────────────────
    logger.info("Running sanity check: generating %d outputs from quantized model...", len(prompts))
    quant_outputs = _generate_outputs(
        quantized_dir, prompts, system_prompt, label="quantized"
    )

    issues: list[str] = []
    logger.info("─" * 50)
    logger.info("Sanity check outputs (quantized model):")
    logger.info("─" * 50)
    for i, (prompt, output) in enumerate(zip(prompts, quant_outputs), 1):
        logger.info("  Prompt %d: %s", i, prompt[:80] + ("..." if len(prompt) > 80 else ""))
        logger.info("  Output %d: %s", i, output[:200] + ("..." if len(output) > 200 else ""))
        logger.info("")

        # Basic quality checks
        if not output.strip():
            issues.append(f"Prompt {i}: empty output")
        if "<|im_start|>" in output or "<|im_end|>" in output:
            issues.append(f"Prompt {i}: output contains raw ChatML markers")

    if issues:
        logger.warning("Sanity check issues found:")
        for issue in issues:
            logger.warning("  • %s", issue)
    else:
        logger.info("All sanity check outputs look reasonable.")

    result["smoke_test"] = {
        "passed": len(issues) == 0,
        "issues": issues,
        "outputs": [o[:500] for o in quant_outputs],
    }

    # ── 2. Full-precision vs quantized comparison ──────────────────────
    if skip_comparison:
        logger.info("Skipping full-precision vs quantized comparison (--skip-comparison).")
        result["comparison"] = {"skipped": True}
        return result

    comparison_prompts = prompts[:3]
    logger.info(
        "Comparing full-precision vs quantized outputs on %d prompts...",
        len(comparison_prompts),
    )

    fp_outputs = _generate_outputs(
        source_dir, comparison_prompts, system_prompt, label="full-precision"
    )

    # Use quantized outputs we already generated (first 3)
    quant_comparison = quant_outputs[: len(comparison_prompts)]

    logger.info("─" * 50)
    logger.info("Full-precision vs Quantized comparison:")
    logger.info("─" * 50)
    identical_count = 0
    degradation_signals: list[str] = []
    for i, (prompt, fp_out, q_out) in enumerate(
        zip(comparison_prompts, fp_outputs, quant_comparison), 1
    ):
        match = fp_out.strip() == q_out.strip()
        if match:
            identical_count += 1
        marker = "✓ IDENTICAL" if match else "≠ DIFFERENT"
        logger.info("  Prompt %d: %s", i, prompt[:60] + ("..." if len(prompt) > 60 else ""))
        logger.info("    FP:   %s", fp_out[:150] + ("..." if len(fp_out) > 150 else ""))
        logger.info("    Quant: %s", q_out[:150] + ("..." if len(q_out) > 150 else ""))
        logger.info("    %s", marker)
        logger.info("")

        # Heuristic degradation checks
        if fp_out.strip() and not q_out.strip():
            degradation_signals.append(f"Prompt {i}: quantized output is empty but FP was not")
        fp_len = len(fp_out.strip())
        q_len = len(q_out.strip())
        if fp_len > 0 and q_len > 0:
            ratio = q_len / fp_len
            if ratio < 0.3 or ratio > 3.0:
                degradation_signals.append(
                    f"Prompt {i}: length ratio {ratio:.1f}x "
                    f"(FP={fp_len}, Quant={q_len})"
                )

    all_identical = identical_count == len(comparison_prompts)
    if all_identical:
        logger.info("All comparison outputs are identical — quantization looks clean.")
    elif identical_count > 0:
        logger.info(
            "%d/%d outputs identical. Some variation is normal with quantization.",
            identical_count,
            len(comparison_prompts),
        )
    else:
        logger.warning(
            "No outputs are identical between full-precision and quantized models."
        )

    if degradation_signals:
        logger.warning(
            "Potential quality degradation detected after quantization:"
        )
        for sig in degradation_signals:
            logger.warning("  • %s", sig)
        logger.warning(
            "If quality is significantly degraded, consider trying q-bits=8."
        )

    result["comparison"] = {
        "skipped": False,
        "identical_count": identical_count,
        "total": len(comparison_prompts),
        "all_identical": all_identical,
        "degradation_signals": degradation_signals,
    }

    return result


# ── Quantization ─────────────────────────────────────────────────────────


def _quantize_model(
    mcfg: ModelConfig,
    cfg: PipelineConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
    skip_sanity_check: bool = False,
    skip_comparison: bool = False,
) -> dict[str, Any]:
    """Quantize a single model.

    Resolves the source path, runs ``mlx_lm.convert --quantize``, verifies
    the output, and optionally runs sanity checks.
    """
    bits = cfg.quantization.bits
    group_size = cfg.quantization.group_size
    out_name = _output_dir_name(mcfg.name, bits)
    output_dir = MODELS_QUANTIZED / out_name

    # ── Resolve source ────────────────────────────────────────────────
    if not dry_run:
        source_dir = _resolve_source_path(mcfg)
        if not _verify_source_dir(source_dir):
            raise RuntimeError(f"Source model verification failed: {source_dir}")
        _check_not_already_quantized(source_dir)
    else:
        # In dry-run, show what we'd look for
        mlx_path = MODELS_MLX / mcfg.name
        fused_path = MODELS_FUSED / mcfg.name
        source_dir = mlx_path if mlx_path.exists() else fused_path
        logger.info("[DRY-RUN] Would use source: %s", source_dir)

    source_size_mb = _dir_size_mb(source_dir) if source_dir.exists() else 0.0

    # ── Handle existing output ────────────────────────────────────────
    if output_dir.exists():
        if force:
            logger.info("--force: removing existing quantized directory %s", output_dir)
            if not dry_run:
                shutil.rmtree(output_dir)
        elif _verify_quantized_output(output_dir):
            logger.warning(
                "Quantized model already exists and looks valid: %s. "
                "Use --force to overwrite.",
                output_dir,
            )
            return {"status": "skipped", "reason": "output exists", "model": mcfg.name}
        else:
            raise RuntimeError(
                f"Quantized model directory exists but appears incomplete: {output_dir}. "
                "Use --force to remove and redo, or delete it manually."
            )

    # ── Build command ─────────────────────────────────────────────────
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm", "convert",
        "--hf-path",
        str(source_dir),
        "--mlx-path",
        str(output_dir),
        "--quantize",
        "--q-bits",
        str(bits),
        "--q-group-size",
        str(group_size),
    ]

    logger.info("Quantizing %s", mcfg.name)
    logger.info("  Source:     %s (%.1f MB)", source_dir, source_size_mb)
    logger.info("  Output:     %s", output_dir)
    logger.info("  Bits:       %d", bits)
    logger.info("  Group size: %d", group_size)
    logger.info("  Command:\n    %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY-RUN] Would run the above command.")
        return {"status": "dry-run", "model": mcfg.name}

    # ── Run quantization ──────────────────────────────────────────────
    # Ensure parent dir exists but NOT the output dir itself
    # (mlx_lm.convert creates it)
    MODELS_QUANTIZED.mkdir(parents=True, exist_ok=True)

    quant_start = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info("  [mlx_lm.convert] %s", line)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            logger.info("  [mlx_lm.convert] %s", line)

    if result.returncode != 0:
        # Clean up partial output on failure
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise RuntimeError(
            f"mlx_lm.convert --quantize failed (exit code {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    quant_elapsed = time.time() - quant_start

    # ── Verify output ─────────────────────────────────────────────────
    if not _verify_quantized_output(output_dir):
        raise RuntimeError(f"Quantized model verification failed: {output_dir}")

    output_size_mb = _dir_size_mb(output_dir)
    compression_ratio = source_size_mb / output_size_mb if output_size_mb > 0 else 0.0

    logger.info(
        "Quantization complete: %s → %s", mcfg.name, output_dir
    )
    logger.info(
        "  Source size:      %.1f MB", source_size_mb,
    )
    logger.info(
        "  Quantized size:   %.1f MB", output_size_mb,
    )
    logger.info(
        "  Compression:      %.2fx", compression_ratio,
    )
    logger.info(
        "  Time:             %.1f sec", quant_elapsed,
    )

    summary: dict[str, Any] = {
        "status": "completed",
        "model": mcfg.name,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "bits": bits,
        "group_size": group_size,
        "source_size_mb": round(source_size_mb, 1),
        "output_size_mb": round(output_size_mb, 1),
        "compression_ratio": round(compression_ratio, 2),
        "time_seconds": round(quant_elapsed, 1),
    }

    # ── Sanity check ──────────────────────────────────────────────────
    if skip_sanity_check:
        logger.info("Skipping sanity check (--skip-sanity-check).")
        summary["sanity_check"] = {"skipped": True}
    else:
        try:
            sanity_result = _run_sanity_check(
                output_dir,
                source_dir,
                skip_comparison=skip_comparison,
            )
            summary["sanity_check"] = sanity_result
        except Exception as exc:
            logger.error(
                "Sanity check failed: %s. The quantized model may still be usable.",
                exc,
                exc_info=True,
            )
            summary["sanity_check"] = {"error": str(exc)}

    return summary


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the quantize script."""
    args = parse_args()
    log = setup_logging(verbose=args.verbose)

    cfg = load_config(args.config)
    models = resolve_models(cfg, args.model)

    log.info(
        "Models: %s | Bits: %d | Group size: %d",
        ", ".join(m.name for m in models),
        cfg.quantization.bits,
        cfg.quantization.group_size,
    )

    # ── Platform & prerequisite checks ────────────────────────────────
    _check_platform(dry_run=args.dry_run)
    if not args.dry_run:
        _check_mlx_available()
        ensure_dirs()

    # ── Process each model ────────────────────────────────────────────
    start = time.time()
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for mcfg in models:
        log.info("═" * 60)
        log.info("Processing model: %s", mcfg.name)
        log.info("═" * 60)

        try:
            result = _quantize_model(
                mcfg,
                cfg,
                force=args.force,
                dry_run=args.dry_run,
                skip_sanity_check=args.skip_sanity_check,
                skip_comparison=args.skip_comparison,
            )
            results.append(result)
        except Exception as exc:
            log.error("Failed to quantize %s: %s", mcfg.name, exc, exc_info=True)
            errors.append(f"{mcfg.name}: {exc}")

    elapsed = time.time() - start

    # ── Final summary ─────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Quantization Summary")
    log.info("═" * 60)
    log.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    for r in results:
        status = r.get("status", "unknown")
        model_name = r.get("model", "?")

        if status == "dry-run":
            log.info("  %s: [DRY-RUN]", model_name)
        elif status == "skipped":
            log.info("  %s: skipped (%s)", model_name, r.get("reason", ""))
        elif status == "completed":
            src_mb = r.get("source_size_mb", "?")
            out_mb = r.get("output_size_mb", "?")
            ratio = r.get("compression_ratio", "?")
            t = r.get("time_seconds", "?")
            out = r.get("output_dir", "?")
            log.info(
                "  %s: completed (%.1f MB → %.1f MB, %.2fx compression, %.1f sec) → %s",
                model_name,
                src_mb,
                out_mb,
                ratio,
                t,
                out,
            )
        else:
            log.info("  %s: %s", model_name, status)

    if errors:
        log.error("Errors encountered:")
        for e in errors:
            log.error("  • %s", e)
        sys.exit(1)

    # ── Save summary JSON per model ───────────────────────────────────
    for r in results:
        if r.get("status") == "completed":
            out_dir_str = r.get("output_dir", "")
            out_dir = Path(out_dir_str) if out_dir_str else None
            if out_dir and out_dir.exists():
                summary_path = out_dir / "quantize_summary.json"
                with open(summary_path, "w", encoding="utf-8") as fh:
                    json.dump(r, fh, indent=2)
                log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
