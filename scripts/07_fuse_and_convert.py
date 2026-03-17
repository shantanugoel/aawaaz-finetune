#!/usr/bin/env python3
"""Fuse LoRA adapters into the base model and convert to MLX format.

Supports two platform paths:

- **Linux (after Unsloth training):** Loads the Unsloth model + LoRA adapters,
  merges them via ``save_pretrained_merged`` at 16-bit, producing a standard
  HuggingFace-format model in ``models/fused/{model_name}/``.  Optionally
  converts that HF model to MLX format (``models/mlx/{model_name}/``) if run
  on a Mac afterwards.

- **Mac (after MLX training):** Runs ``mlx_lm.fuse`` to merge the base model
  with the LoRA adapters, producing an MLX-format model directly in
  ``models/fused/{model_name}/``.

Cross-platform workflow: if you fine-tuned on Linux, transfer the fused HF
model to Mac and run this script with ``--convert-only`` to convert it to MLX
format.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    MODELS_ADAPTERS,
    MODELS_BASE,
    MODELS_FUSED,
    MODELS_MLX,
    ModelConfig,
    PipelineConfig,
    add_model_arg,
    base_arg_parser,
    ensure_dirs,
    load_config,
    resolve_models,
    setup_logging,
)

logger = logging.getLogger("aawaaz.fuse")

# Files we expect to find in a valid model output directory.
EXPECTED_OUTPUT_FILES = {"config.json"}
SAFETENSOR_GLOB = "*.safetensors"


# ── Argument parsing ───────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = base_arg_parser(
        description="Fuse LoRA adapters into the base model and convert to MLX format.",
    )
    add_model_arg(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing fused/converted output without prompting.",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help=(
            "Skip the fuse step; only convert an existing HF-format fused "
            "model (models/fused/{model}) to MLX format (models/mlx/{model}). "
            "Use this after transferring a Linux-fused model to Mac."
        ),
    )
    parser.add_argument(
        "--de-quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --de-quantize to mlx_lm.fuse (default: True). "
            "Only relevant for Mac path when the base model was quantized. "
            "Use --no-de-quantize to skip."
        ),
    )
    return parser.parse_args(argv)


# ── Validation helpers ─────────────────────────────────────────────────────


def _verify_output_dir(output_dir: Path, label: str) -> bool:
    """Check that an output directory contains expected model files.

    Returns True if the directory looks like a valid model, False otherwise.
    Logs warnings for any missing expected files.
    """
    if not output_dir.exists():
        logger.error("%s directory does not exist: %s", label, output_dir)
        return False

    files = {f.name for f in output_dir.iterdir() if f.is_file()}
    safetensors = list(output_dir.glob(SAFETENSOR_GLOB))

    missing = EXPECTED_OUTPUT_FILES - files
    if missing:
        logger.warning("%s is missing expected files: %s", label, missing)

    if not safetensors:
        logger.warning("%s has no .safetensors weight files.", label)
        return False

    has_tokenizer = any(
        f.name.startswith("tokenizer") for f in output_dir.iterdir()
    )
    if not has_tokenizer:
        logger.warning("%s has no tokenizer files.", label)

    logger.info(
        "%s verified: %d files, %d safetensors",
        label, len(files), len(safetensors),
    )
    return True


def _dir_size_mb(path: Path) -> float:
    """Calculate total size of files in a directory, in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _check_adapter_exists(adapter_dir: Path, platform: str) -> None:
    """Verify that the adapter directory exists and has the right files."""
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}. "
            "Run 06_finetune.py first."
        )

    if platform == "mac":
        adapter_file = adapter_dir / "adapters.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"MLX adapter file not found: {adapter_file}. "
                "Run 06_finetune.py first."
            )
    else:
        # Linux/Unsloth saves adapter_model.safetensors or similar
        safetensors = list(adapter_dir.glob("*.safetensors"))
        if not safetensors:
            raise FileNotFoundError(
                f"No .safetensors adapter files in {adapter_dir}. "
                "Run 06_finetune.py first."
            )


def _resolve_base_model_path(mcfg: ModelConfig, platform: str) -> str:
    """Resolve the base model path (local or remote) for fusing.

    Mirrors the resolution order in ``06_finetune.py._resolve_model_source``.
    """
    local_base = MODELS_BASE / mcfg.name
    if platform == "linux":
        # Match 06_finetune.py: prefer unsloth variant first on Linux
        local_unsloth = MODELS_BASE / f"{mcfg.name}-unsloth"
        if local_unsloth.exists():
            return str(local_unsloth)
        if local_base.exists():
            return str(local_base)
        return mcfg.unsloth_model or mcfg.base_model
    else:
        if local_base.exists():
            return str(local_base)
        return mcfg.base_model


def _check_import(module_name: str, package_label: str) -> None:
    """Check that a Python module is importable, raise RuntimeError if not."""
    try:
        __import__(module_name)
    except ImportError:
        raise RuntimeError(
            f"{package_label} is required but not installed. "
            f"Install it and try again."
        )


def _check_prerequisites(platform: str, convert_only: bool) -> None:
    """Verify required packages/tools are available before doing any work."""
    if convert_only:
        _check_import("mlx_lm", "mlx-lm (pip install mlx-lm)")
        return

    if platform == "linux":
        _check_import("unsloth", "Unsloth (pip install unsloth)")
    else:
        _check_import("mlx_lm", "mlx-lm (pip install mlx-lm)")


# ── Linux fuse (Unsloth) ──────────────────────────────────────────────────


def _fuse_linux(
    mcfg: ModelConfig,
    cfg: PipelineConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fuse LoRA adapters into the base model using Unsloth (Linux/CUDA).

    Produces a standard HuggingFace-format model at
    ``models/fused/{model_name}/``.
    """
    adapter_dir = MODELS_ADAPTERS / mcfg.name
    fused_dir = MODELS_FUSED / mcfg.name

    if not dry_run:
        _check_adapter_exists(adapter_dir, "linux")

    # Handle existing output
    if fused_dir.exists():
        if force:
            logger.info("--force: removing existing fused directory %s", fused_dir)
            if not dry_run:
                shutil.rmtree(fused_dir)
        elif _verify_output_dir(fused_dir, "Existing fused model"):
            logger.warning(
                "Fused model already exists and looks valid: %s. Use --force to overwrite.",
                fused_dir,
            )
            return {"status": "skipped", "reason": "output exists", "model": mcfg.name}
        else:
            raise RuntimeError(
                f"Fused model directory exists but appears incomplete: {fused_dir}. "
                "Use --force to remove and redo, or delete it manually."
            )

    model_source = _resolve_base_model_path(mcfg, "linux")
    logger.info("Fusing LoRA adapters for %s (Linux/Unsloth)", mcfg.name)
    logger.info("  Base model: %s", model_source)
    logger.info("  Adapters:   %s", adapter_dir)
    logger.info("  Output:     %s", fused_dir)

    if dry_run:
        logger.info("[DRY-RUN] Would fuse model via Unsloth save_pretrained_merged.")
        return {"status": "dry-run", "model": mcfg.name}

    fuse_start = time.time()

    # Import Unsloth and load model + adapters
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth is required for Linux fuse path. "
            "Install it: pip install unsloth"
        ) from exc

    # Unsloth's from_pretrained with the adapter dir loads the base model
    # referenced in the adapter's config and applies the LoRA weights.
    # This ensures the exact base model used during training is reused.
    logger.info("Loading model and adapters via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=cfg.training.max_seq_length,
        load_in_4bit=cfg.training.linux.load_in_4bit,
    )

    # Merge LoRA and save at 16-bit
    logger.info("Merging LoRA adapters (save_method='merged_16bit')...")
    fused_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.save_pretrained_merged(
            str(fused_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    except Exception:
        # Clean up partial output on failure
        if fused_dir.exists():
            shutil.rmtree(fused_dir)
        raise

    fuse_elapsed = time.time() - fuse_start

    # Verify output
    if not _verify_output_dir(fused_dir, "Fused model"):
        raise RuntimeError(f"Fused model verification failed: {fused_dir}")

    size_mb = _dir_size_mb(fused_dir)
    logger.info(
        "Fuse complete: %s → %s (%.1f MB, %.1f sec)",
        mcfg.name, fused_dir, size_mb, fuse_elapsed,
    )

    # Clean up GPU memory
    del model
    del tokenizer
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "status": "completed",
        "model": mcfg.name,
        "platform": "linux",
        "fused_dir": str(fused_dir),
        "size_mb": round(size_mb, 1),
        "time_seconds": round(fuse_elapsed, 1),
    }


# ── Mac fuse (MLX) ────────────────────────────────────────────────────────


def _fuse_mac(
    mcfg: ModelConfig,
    cfg: PipelineConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
    de_quantize: bool = True,
) -> dict[str, Any]:
    """Fuse LoRA adapters into the base model using mlx_lm.fuse (Mac).

    Produces an MLX-format model directly at ``models/fused/{model_name}/``.
    """
    adapter_dir = MODELS_ADAPTERS / mcfg.name
    fused_dir = MODELS_FUSED / mcfg.name

    if not dry_run:
        _check_adapter_exists(adapter_dir, "mac")

    # Handle existing output
    if fused_dir.exists():
        if force:
            logger.info("--force: removing existing fused directory %s", fused_dir)
            if not dry_run:
                shutil.rmtree(fused_dir)
        elif _verify_output_dir(fused_dir, "Existing fused model"):
            logger.warning(
                "Fused model already exists and looks valid: %s. Use --force to overwrite.",
                fused_dir,
            )
            return {"status": "skipped", "reason": "output exists", "model": mcfg.name}
        else:
            raise RuntimeError(
                f"Fused model directory exists but appears incomplete: {fused_dir}. "
                "Use --force to remove and redo, or delete it manually."
            )

    model_source = _resolve_base_model_path(mcfg, "mac")
    logger.info("Fusing LoRA adapters for %s (Mac/MLX)", mcfg.name)
    logger.info("  Base model: %s", model_source)
    logger.info("  Adapters:   %s", adapter_dir)
    logger.info("  Output:     %s", fused_dir)

    # Build mlx_lm.fuse command
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_source,
        "--adapter-path", str(adapter_dir),
        "--save-path", str(fused_dir),
    ]

    if de_quantize:
        cmd.append("--dequantize")

    logger.info("mlx_lm.fuse command:\n  %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY-RUN] Would run the above command.")
        return {"status": "dry-run", "model": mcfg.name}

    fuse_start = time.time()

    fused_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info("  [mlx_lm.fuse] %s", line)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            logger.info("  [mlx_lm.fuse] %s", line)

    if result.returncode != 0:
        # Clean up partial output on failure
        if fused_dir.exists():
            shutil.rmtree(fused_dir)
        raise RuntimeError(
            f"mlx_lm.fuse failed (exit code {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    fuse_elapsed = time.time() - fuse_start

    # Verify output
    if not _verify_output_dir(fused_dir, "Fused model"):
        raise RuntimeError(f"Fused model verification failed: {fused_dir}")

    size_mb = _dir_size_mb(fused_dir)
    logger.info(
        "Fuse complete: %s → %s (%.1f MB, %.1f sec)",
        mcfg.name, fused_dir, size_mb, fuse_elapsed,
    )

    return {
        "status": "completed",
        "model": mcfg.name,
        "platform": "mac",
        "fused_dir": str(fused_dir),
        "size_mb": round(size_mb, 1),
        "time_seconds": round(fuse_elapsed, 1),
    }


# ── HF → MLX conversion ──────────────────────────────────────────────────


def _convert_hf_to_mlx(
    mcfg: ModelConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Convert a HuggingFace-format fused model to MLX format.

    Reads from ``models/fused/{model_name}/`` and writes to
    ``models/mlx/{model_name}/``.  Typically used after Linux fine-tuning
    when the fused HF model has been transferred to Mac.
    """
    fused_dir = MODELS_FUSED / mcfg.name
    mlx_dir = MODELS_MLX / mcfg.name

    # Verify source exists (skip in dry-run)
    if not dry_run:
        if not fused_dir.exists():
            raise FileNotFoundError(
                f"Fused HF model not found: {fused_dir}. "
                "Run the fuse step first, or transfer the fused model from Linux."
            )

        if not _verify_output_dir(fused_dir, "Source (fused HF model)"):
            raise RuntimeError(
                f"Source fused model does not appear valid: {fused_dir}"
            )

    # Handle existing output
    if mlx_dir.exists():
        if force:
            logger.info("--force: removing existing MLX directory %s", mlx_dir)
            if not dry_run:
                shutil.rmtree(mlx_dir)
        elif _verify_output_dir(mlx_dir, "Existing MLX model"):
            logger.warning(
                "MLX model already exists and looks valid: %s. Use --force to overwrite.",
                mlx_dir,
            )
            return {"status": "skipped", "reason": "output exists", "model": mcfg.name}
        else:
            raise RuntimeError(
                f"MLX model directory exists but appears incomplete: {mlx_dir}. "
                "Use --force to remove and redo, or delete it manually."
            )

    logger.info("Converting HF → MLX for %s", mcfg.name)
    logger.info("  Source: %s", fused_dir)
    logger.info("  Output: %s", mlx_dir)

    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", str(fused_dir),
        "--mlx-path", str(mlx_dir),
    ]

    logger.info("mlx_lm.convert command:\n  %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY-RUN] Would run the above command.")
        return {"status": "dry-run", "model": mcfg.name}

    convert_start = time.time()

    mlx_dir.mkdir(parents=True, exist_ok=True)

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
        if mlx_dir.exists():
            shutil.rmtree(mlx_dir)
        raise RuntimeError(
            f"mlx_lm.convert failed (exit code {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    convert_elapsed = time.time() - convert_start

    # Verify output
    if not _verify_output_dir(mlx_dir, "MLX model"):
        raise RuntimeError(f"MLX model verification failed: {mlx_dir}")

    size_mb = _dir_size_mb(mlx_dir)
    logger.info(
        "Conversion complete: %s → %s (%.1f MB, %.1f sec)",
        mcfg.name, mlx_dir, size_mb, convert_elapsed,
    )

    return {
        "status": "completed",
        "model": mcfg.name,
        "mlx_dir": str(mlx_dir),
        "size_mb": round(size_mb, 1),
        "time_seconds": round(convert_elapsed, 1),
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the fuse-and-convert script."""
    args = parse_args()
    log = setup_logging(verbose=args.verbose)

    cfg = load_config(args.config)
    models = resolve_models(cfg, args.model)
    platform = cfg.platform

    log.info(
        "Platform: %s | Models: %s | convert-only: %s",
        platform,
        ", ".join(m.name for m in models),
        args.convert_only,
    )

    if not args.dry_run:
        ensure_dirs()
        _check_prerequisites(platform, args.convert_only)

    # ── Process each model ──────────────────────────────────────────────
    start = time.time()
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for mcfg in models:
        log.info("═" * 60)
        log.info("Processing model: %s", mcfg.name)
        log.info("═" * 60)

        try:
            if args.convert_only:
                # Convert-only mode: skip fuse, just convert HF → MLX
                result = _convert_hf_to_mlx(
                    mcfg, force=args.force, dry_run=args.dry_run,
                )
                results.append(result)
                continue

            # Platform-specific fuse
            if platform == "linux":
                fuse_result = _fuse_linux(
                    mcfg, cfg,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                results.append(fuse_result)

                if fuse_result.get("status") == "completed":
                    # On Linux, also print instructions for Mac conversion
                    fused_dir = MODELS_FUSED / mcfg.name
                    log.info(
                        "\n  ╔══════════════════════════════════════════════════════╗\n"
                        "  ║  NEXT STEP: Transfer to Mac and convert to MLX      ║\n"
                        "  ╠══════════════════════════════════════════════════════╣\n"
                        "  ║  1. Copy %s to Mac            ║\n"
                        "  ║  2. Run:                                            ║\n"
                        "  ║     python scripts/07_fuse_and_convert.py \\         ║\n"
                        "  ║       --convert-only --model %s              ║\n"
                        "  ╚══════════════════════════════════════════════════════╝",
                        fused_dir, mcfg.name,
                    )
            else:
                fuse_result = _fuse_mac(
                    mcfg, cfg,
                    force=args.force,
                    dry_run=args.dry_run,
                    de_quantize=args.de_quantize,
                )
                results.append(fuse_result)

        except Exception as exc:
            log.error("Failed to process %s: %s", mcfg.name, exc, exc_info=True)
            errors.append(f"{mcfg.name}: {exc}")

    elapsed = time.time() - start

    # ── Final summary ───────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Fuse & Convert Summary")
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
            size = r.get("size_mb", "?")
            t = r.get("time_seconds", "?")
            out = r.get("fused_dir") or r.get("mlx_dir") or "?"
            log.info("  %s: completed (%.1f MB, %.1f sec) → %s", model_name, size, t, out)
        else:
            log.info("  %s: %s", model_name, status)

    if errors:
        log.error("Errors encountered:")
        for e in errors:
            log.error("  • %s", e)
        sys.exit(1)

    # Save summary JSON per model
    for r in results:
        if r.get("status") == "completed":
            out_dir_str = r.get("fused_dir") or r.get("mlx_dir", "")
            out_dir = Path(out_dir_str) if out_dir_str else None
            if out_dir and out_dir.exists():
                summary = {
                    "platform": platform,
                    "convert_only": args.convert_only,
                    "result": r,
                    "total_time_seconds": round(elapsed, 1),
                }
                summary_path = out_dir / "fuse_summary.json"
                with open(summary_path, "w", encoding="utf-8") as fh:
                    json.dump(summary, fh, indent=2)
                log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
