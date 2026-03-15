#!/usr/bin/env python3
"""Download base HuggingFace models for fine-tuning.

Downloads Qwen3 models specified in config.yaml to models/base/{model_name}/.
On Linux, also downloads the Unsloth variant for efficient fine-tuning.
Verifies each download by loading the tokenizer and running a quick test
generation to ensure the model is functional.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from common import (
    MODELS_BASE,
    ModelConfig,
    PipelineConfig,
    add_model_arg,
    base_arg_parser,
    ensure_dirs,
    load_config,
    resolve_models,
    setup_logging,
)

logger = logging.getLogger("aawaaz.download_models")


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the model download step."""
    parser = base_arg_parser(
        "Download base HuggingFace models to models/base/."
    )
    add_model_arg(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download models even if they already exist locally.",
    )
    return parser.parse_args()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _model_dir(model_name: str) -> Path:
    """Return the local directory path for a given model name."""
    return MODELS_BASE / model_name


def _is_downloaded(model_dir: Path) -> bool:
    """Check whether a model directory looks like a valid download.

    A valid download should contain at minimum a tokenizer config and at
    least one safetensors or bin weight file.
    """
    if not model_dir.is_dir():
        return False
    has_tokenizer = (
        (model_dir / "tokenizer_config.json").exists()
        or (model_dir / "tokenizer.json").exists()
    )
    has_weights = (
        list(model_dir.glob("*.safetensors"))
        or list(model_dir.glob("*.bin"))
    )
    return has_tokenizer and bool(has_weights)


def _dir_size_mb(path: Path) -> float:
    """Compute total size of a directory in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _download_model(
    repo_id: str,
    local_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Download a model from HuggingFace Hub to the local directory.

    Parameters
    ----------
    repo_id:
        HuggingFace model repository id (e.g. ``Qwen/Qwen3-0.6B``).
    local_dir:
        Target directory for the download.
    dry_run:
        If *True*, log what would happen without downloading.
    """
    if dry_run:
        logger.info("[DRY-RUN] Would download %s → %s", repo_id, local_dir)
        return

    from huggingface_hub import snapshot_download

    logger.info("Downloading %s → %s", repo_id, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    logger.info("Download complete: %s (%.1f MB)", repo_id, _dir_size_mb(local_dir))


def _detect_runtime() -> str:
    """Detect which ML runtime is available: 'mlx', 'torch', or 'none'."""
    try:
        import mlx_lm  # noqa: F401
        return "mlx"
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        return "torch"
    except ImportError:
        pass
    return "none"


def _verify_model_torch(
    model_dir: Path,
    repo_id: str,
) -> dict[str, object]:
    """Verify a model using PyTorch/transformers (Linux path)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        device_map="cpu",
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "  Model: %s | Parameters: %s (%.2fB) | Trainable: %s",
        repo_id,
        f"{total_params:,}",
        total_params / 1e9,
        f"{trainable_params:,}",
    )

    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
    ]
    try:
        input_text = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        logger.warning(
            "  tokenizer.apply_chat_template does not accept 'enable_thinking' — "
            "falling back without it. Verify Qwen3 thinking mode is handled."
        )
        input_text = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    logger.info("  Test generation: %r", generated_text[:100])

    if "<think>" in generated_text:
        logger.warning(
            "  ⚠ Test output contains <think> tags — thinking mode may not be "
            "properly disabled for %s",
            repo_id,
        )

    del model
    del tokenizer

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "test_output": generated_text[:200],
        "size_mb": _dir_size_mb(model_dir),
    }


def _verify_model_mlx(
    model_dir: Path,
    repo_id: str,
) -> dict[str, object]:
    """Verify a model using mlx-lm (Mac path)."""
    from mlx_lm import generate, load

    model, tokenizer = load(str(model_dir))

    # Count parameters from MLX model weights
    import mlx.utils

    total_params = sum(
        v.size for _, v in mlx.utils.tree_flatten(model.parameters())
    )
    logger.info(
        "  Model: %s | Parameters: %s (%.2fB)",
        repo_id,
        f"{total_params:,}",
        total_params / 1e9,
    )

    # Quick test generation
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        logger.warning(
            "  tokenizer.apply_chat_template does not accept 'enable_thinking' — "
            "falling back without it. Verify Qwen3 thinking mode is handled."
        )
        prompt = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    generated_text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=20,
    )
    logger.info("  Test generation: %r", generated_text[:100])

    if "<think>" in generated_text:
        logger.warning(
            "  ⚠ Test output contains <think> tags — thinking mode may not be "
            "properly disabled for %s",
            repo_id,
        )

    del model
    del tokenizer

    return {
        "total_params": total_params,
        "trainable_params": total_params,  # all params trainable for full model
        "test_output": generated_text[:200],
        "size_mb": _dir_size_mb(model_dir),
    }


def _verify_model_tokenizer_only(
    model_dir: Path,
    repo_id: str,
) -> dict[str, object]:
    """Minimal verification using only the tokenizer (no ML runtime available)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )

    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        logger.warning(
            "  tokenizer.apply_chat_template does not accept 'enable_thinking' — "
            "falling back without it."
        )
        formatted = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    logger.info("  Tokenizer loaded OK. Chat template sample:")
    logger.info("  %s", formatted[:200])

    # Verify ChatML markers are present
    if "<|im_start|>" not in formatted:
        logger.warning(
            "  ⚠ Chat template does not contain expected ChatML markers "
            "(<|im_start|>). This may indicate a template issue."
        )

    del tokenizer

    return {
        "total_params": "unknown (no ML runtime)",
        "trainable_params": "unknown",
        "test_output": f"tokenizer-only verification: {formatted[:100]}",
        "size_mb": _dir_size_mb(model_dir),
    }


def _verify_model(
    model_dir: Path,
    repo_id: str,
) -> dict[str, object]:
    """Verify a downloaded model by loading and running test generation.

    Automatically selects the appropriate ML runtime (MLX on Mac, PyTorch
    on Linux). Falls back to tokenizer-only verification if neither is
    available.

    Returns a dict with verification results.

    Raises
    ------
    RuntimeError
        If the model fails verification.
    """
    logger.info("Verifying model at %s ...", model_dir)

    runtime = _detect_runtime()
    logger.info("  Using %s runtime for verification", runtime)

    try:
        if runtime == "mlx":
            return _verify_model_mlx(model_dir, repo_id)
        elif runtime == "torch":
            return _verify_model_torch(model_dir, repo_id)
        else:
            logger.warning(
                "  Neither MLX nor PyTorch available — "
                "performing tokenizer-only verification."
            )
            return _verify_model_tokenizer_only(model_dir, repo_id)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to verify model at {model_dir}: {exc}"
        ) from exc


def _safe_download(
    repo_id: str,
    target_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Download a model to *target_dir*, with safe replacement on --force.

    If ``force`` is True and *target_dir* already has a valid download,
    the new model is downloaded to a staging directory first. The old
    directory is replaced only after the new download completes
    successfully.

    Returns True if a download was performed, False if skipped.
    """
    already_valid = _is_downloaded(target_dir)

    if already_valid and not force:
        logger.info(
            "Skipping %s — already downloaded at %s", repo_id, target_dir
        )
        return False

    # Clean up a partial/corrupt directory (exists but fails validation)
    if target_dir.exists() and not already_valid and not dry_run:
        logger.info("Removing partial/corrupt download: %s", target_dir)
        shutil.rmtree(target_dir)

    if force and already_valid:
        # Safe replacement: download to staging dir, then swap
        staging_dir = target_dir.parent / f".{target_dir.name}.staging"
        if dry_run:
            logger.info(
                "[DRY-RUN] Would re-download %s → %s (replacing existing)",
                repo_id,
                target_dir,
            )
            return True

        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        _download_model(repo_id, staging_dir)

        logger.info("Replacing %s with fresh download", target_dir)
        shutil.rmtree(target_dir)
        staging_dir.rename(target_dir)
    else:
        _download_model(repo_id, target_dir, dry_run=dry_run)

    return True


def _process_model(
    mcfg: ModelConfig,
    platform: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Download and verify a single model (and its Unsloth variant on Linux).

    Returns a summary dict for the model.
    """
    result: dict[str, object] = {"model": mcfg.name}

    # ── Base HF model ───────────────────────────────────────────────────
    base_dir = _model_dir(mcfg.name)
    downloaded = _safe_download(
        mcfg.base_model, base_dir, force=force, dry_run=dry_run
    )
    result["base_skipped"] = not downloaded

    # Verify (unless dry-run)
    if not dry_run:
        info = _verify_model(base_dir, mcfg.base_model)
        result["base_info"] = info

    # ── Unsloth variant (Linux only) ────────────────────────────────────
    if platform == "linux" and mcfg.unsloth_model:
        unsloth_dir = _model_dir(f"{mcfg.name}-unsloth")
        u_downloaded = _safe_download(
            mcfg.unsloth_model, unsloth_dir, force=force, dry_run=dry_run
        )
        result["unsloth_skipped"] = not u_downloaded

        if not dry_run:
            unsloth_info = _verify_model(unsloth_dir, mcfg.unsloth_model)
            result["unsloth_info"] = unsloth_info

    return result


def _check_prerequisites(platform: str, dry_run: bool) -> None:
    """Verify required packages are available before downloading.

    On non-dry-run invocations, the verification step needs an ML runtime.
    Checks that the expected one is available and warns early rather than
    after downloading gigabytes of model weights.
    """
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. "
            "Run scripts/01_setup.sh or install the appropriate requirements file."
        )
        raise SystemExit(1)

    if dry_run:
        return

    runtime = _detect_runtime()
    if runtime == "none":
        logger.error(
            "Neither MLX nor PyTorch is available. Cannot verify downloaded "
            "models. Install the appropriate requirements file for your platform."
        )
        raise SystemExit(1)

    if platform == "mac" and runtime != "mlx":
        logger.warning(
            "Platform is 'mac' but MLX is not available — "
            "will use %s for verification.",
            runtime,
        )
    elif platform == "linux" and runtime != "torch":
        logger.warning(
            "Platform is 'linux' but PyTorch is not available — "
            "will use %s for verification.",
            runtime,
        )


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the model download script."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    cfg: PipelineConfig = load_config(args.config)
    models = resolve_models(cfg, args.model)

    logger.info(
        "Platform: %s | Models to download: %s",
        cfg.platform,
        ", ".join(m.name for m in models),
    )

    _check_prerequisites(cfg.platform, args.dry_run)

    if not args.dry_run:
        ensure_dirs()

    start = time.time()
    summaries: list[dict[str, object]] = []
    errors: list[str] = []

    for mcfg in models:
        logger.info("─" * 60)
        logger.info("Processing model: %s", mcfg.name)
        try:
            summary = _process_model(
                mcfg,
                cfg.platform,
                force=args.force,
                dry_run=args.dry_run,
            )
            summaries.append(summary)
        except Exception as exc:
            logger.error("Failed to process model %s: %s", mcfg.name, exc)
            errors.append(f"{mcfg.name}: {exc}")

    elapsed = time.time() - start

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Download Summary")
    logger.info("═" * 60)
    logger.info("Time elapsed: %.1f seconds", elapsed)

    for s in summaries:
        name = s["model"]
        if args.dry_run:
            base_action = "skip (exists)" if s.get("base_skipped") else "download"
            logger.info("  %s: [DRY-RUN] would %s", name, base_action)
            continue
        base_info = s.get("base_info", {})
        params = base_info.get("total_params", "?")  # type: ignore[union-attr]
        size = base_info.get("size_mb", "?")  # type: ignore[union-attr]
        skipped = "skipped" if s.get("base_skipped") else "downloaded"
        logger.info(
            "  %s: %s | params=%s | size=%.1f MB",
            name,
            skipped,
            f"{params:,}" if isinstance(params, int) else params,
            size if isinstance(size, (int, float)) else 0,
        )
        if "unsloth_info" in s:
            u_info = s["unsloth_info"]
            u_skipped = "skipped" if s.get("unsloth_skipped") else "downloaded"
            logger.info(
                "    └─ unsloth: %s | size=%.1f MB",
                u_skipped,
                u_info.get("size_mb", 0),  # type: ignore[union-attr]
            )

    if errors:
        logger.error("Errors encountered:")
        for e in errors:
            logger.error("  • %s", e)
        raise SystemExit(1)

    logger.info("All models ready.")


if __name__ == "__main__":
    main()
