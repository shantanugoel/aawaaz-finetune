#!/usr/bin/env python3
"""Upload quantized models to HuggingFace Hub.

Creates/updates HuggingFace repositories for each quantized model, uploads
the model directory along with a generated model card (README.md) and the
system prompt file.

Reads training summaries from ``models/adapters/{model_name}/training_summary.json``
and eval results from ``eval_results/{model_name}/eval_summary.json`` to populate
the model card with training details and evaluation metrics.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from common import (
    DATA_COMBINED,
    EVAL_RESULTS,
    MODELS_ADAPTERS,
    MODELS_QUANTIZED,
    SYSTEM_PROMPT_PATH,
    ModelConfig,
    PipelineConfig,
    add_model_arg,
    base_arg_parser,
    load_config,
    load_system_prompt,
    resolve_models,
    setup_logging,
)

logger = logging.getLogger("aawaaz.upload")

# Files expected in a valid quantized model directory
REQUIRED_MODEL_FILES = {"config.json"}
EXPECTED_MODEL_FILES = {"tokenizer.json", "tokenizer_config.json"}


# ── Argument parsing ───────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> "argparse.Namespace":
    """Parse command-line arguments."""
    import argparse

    parser = base_arg_parser(
        description="Upload quantized models to HuggingFace Hub.",
    )
    add_model_arg(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if upload.enabled is false in config.",
    )
    return parser.parse_args(argv)


# ── Helpers ────────────────────────────────────────────────────────────────


def _repo_name(cfg: PipelineConfig, mcfg: ModelConfig) -> str:
    """Derive the HuggingFace repo ID for a model."""
    bits = cfg.quantization.bits
    return f"{cfg.hf_username}/{cfg.upload.repo_prefix}-{mcfg.name}-transcriber-{bits}bit"


def _count_jsonl_lines(path: Path) -> int:
    """Count lines in a JSONL file. Returns 0 if file doesn't exist."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as fh:
        for _ in fh:
            count += 1
    return count


def _load_json_optional(path: Path, label: str) -> dict[str, Any] | None:
    """Load a JSON file, returning None with a warning if unavailable."""
    if not path.exists():
        logger.warning("%s not found: %s", label, path)
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.debug("Loaded %s from %s", label, path)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s (%s): %s", label, path, exc)
        return None


def _validate_model_dir(model_dir: Path) -> list[str]:
    """Validate a quantized model directory. Returns list of warnings."""
    warnings: list[str] = []

    if not model_dir.is_dir():
        warnings.append(f"Model directory does not exist: {model_dir}")
        return warnings

    existing_files = {f.name for f in model_dir.iterdir() if f.is_file()}

    for req in REQUIRED_MODEL_FILES:
        if req not in existing_files:
            warnings.append(f"Required file missing: {req}")

    safetensors = [f for f in existing_files if f.endswith(".safetensors")]
    if not safetensors:
        warnings.append("No .safetensors files found")

    for exp in EXPECTED_MODEL_FILES:
        if exp not in existing_files:
            warnings.append(f"Expected file missing (non-fatal): {exp}")

    return warnings


def _normalize_training_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize Linux/Mac training summary keys into a common format."""
    return {
        "final_train_loss": raw.get("final_train_loss"),
        "final_val_loss": raw.get("final_eval_loss") or raw.get("final_val_loss"),
        "best_val_loss": raw.get("best_eval_loss") or raw.get("best_val_loss"),
        "train_samples": raw.get("train_samples"),
        "valid_samples": raw.get("valid_samples"),
        "train_time_seconds": raw.get("train_time_seconds"),
        "total_steps": raw.get("total_steps") or raw.get("total_iters"),
    }


def _check_dependencies(dry_run: bool) -> None:
    """Verify that required packages are available."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required but not installed. "
            "Install it with: pip install huggingface_hub"
        )
    if not dry_run:
        logger.debug("huggingface_hub version: %s", huggingface_hub.__version__)


def _check_hf_auth(cfg: PipelineConfig, dry_run: bool) -> None:
    """Verify HuggingFace authentication before uploading.

    Raises
    ------
    RuntimeError
        If the token is missing, invalid, or for the wrong user.
    """
    if dry_run:
        logger.info("[DRY-RUN] Would verify HuggingFace authentication")
        return

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    try:
        user_info = api.whoami()
    except HfHubHTTPError as exc:
        raise RuntimeError(
            "HuggingFace authentication failed. Make sure you have a valid "
            "write token configured. Run `huggingface-cli login` to set up "
            f"authentication. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to verify HuggingFace authentication. "
            f"Run `huggingface-cli login` first. Error: {exc}"
        ) from exc

    authenticated_user = user_info.get("name", "")
    if authenticated_user != cfg.hf_username:
        logger.warning(
            "Authenticated as '%s' but config.hf_username is '%s'. "
            "Repo IDs use config.hf_username, so uploads may fail if you "
            "don't have write access to that namespace.",
            authenticated_user,
            cfg.hf_username,
        )


# ── Model card generation ──────────────────────────────────────────────────


def _generate_model_card(
    cfg: PipelineConfig,
    mcfg: ModelConfig,
    repo_id: str,
    system_prompt: str | None,
    training_summary: dict[str, Any] | None,
    eval_summary: dict[str, Any] | None,
    dataset_counts: dict[str, int],
) -> str:
    """Generate a HuggingFace model card (README.md) as a string."""
    bits = cfg.quantization.bits
    norm_train = (
        _normalize_training_summary(training_summary) if training_summary else {}
    )

    # ── YAML front matter ──────────────────────────────────────────────
    lines = [
        "---",
        f"license: apache-2.0",
        f"base_model: {mcfg.base_model}",
        "tags:",
        "  - transcription",
        "  - text-cleanup",
        "  - mlx",
        "  - lora",
        f"  - {bits}bit",
        "  - qwen3",
        "language:",
        "  - en",
        "library_name: mlx",
        "pipeline_tag: text-generation",
        "---",
        "",
    ]

    # ── Title and description ──────────────────────────────────────────
    lines.extend([
        f"# {repo_id.split('/')[-1]}",
        "",
        f"A {bits}-bit quantized [MLX](https://github.com/ml-explore/mlx) model "
        f"fine-tuned from [{mcfg.base_model}](https://huggingface.co/{mcfg.base_model}) "
        f"for **transcription cleanup**.",
        "",
        "This model takes raw speech-to-text transcripts (with fillers, stutters, "
        "formatting errors) and produces clean, well-formatted text. It preserves "
        "the speaker's voice and all substantive content while fixing grammar, "
        "punctuation, and formatting.",
        "",
    ])

    # ── System prompt ──────────────────────────────────────────────────
    if system_prompt:
        lines.extend([
            "## System Prompt",
            "",
            "```",
            system_prompt,
            "```",
            "",
        ])

    # ── Recommended inference settings ─────────────────────────────────
    lines.extend([
        "## Recommended Inference Settings",
        "",
        "| Parameter | Value | Rationale |",
        "|-----------|-------|-----------|",
        "| `temperature` | `0.0` | Deterministic — formatting task, not creative |",
        "| `top_p` | `1.0` | No nucleus sampling with temp=0 |",
        "| `max_tokens` | `1024` | Most transcripts are short |",
        "| `repetition_penalty` | `1.1` | Slight penalty to avoid degenerate loops |",
        "",
        "> **Note:** This model uses Qwen3's chat format. Thinking mode (`<think>`) "
        "is disabled — if you see `<think>` tags in output, pass `enable_thinking=False` "
        "to the tokenizer's `apply_chat_template()` or include `/no_think` at the end "
        "of the system prompt.",
        "",
    ])

    # ── Training details ───────────────────────────────────────────────
    lines.extend([
        "## Training Details",
        "",
    ])

    total_samples = sum(dataset_counts.values())
    if total_samples > 0:
        lines.append(f"- **Dataset size:** {total_samples:,} samples "
                      f"(train: {dataset_counts.get('train', 0):,}, "
                      f"valid: {dataset_counts.get('valid', 0):,}, "
                      f"test: {dataset_counts.get('test', 0):,})")

    lines.extend([
        f"- **Base model:** [{mcfg.base_model}](https://huggingface.co/{mcfg.base_model})",
        f"- **Method:** LoRA (rank={cfg.training.lora.rank}, alpha={cfg.training.lora.alpha})",
        f"- **Max sequence length:** {cfg.training.max_seq_length}",
        f"- **Quantization:** {bits}-bit, group size {cfg.quantization.group_size}",
    ])

    if norm_train.get("total_steps"):
        lines.append(f"- **Training steps:** {norm_train['total_steps']}")
    if norm_train.get("train_time_seconds"):
        minutes = norm_train["train_time_seconds"] / 60
        lines.append(f"- **Training time:** {minutes:.1f} minutes")
    if norm_train.get("final_train_loss") is not None:
        lines.append(f"- **Final training loss:** {norm_train['final_train_loss']:.4f}")
    if norm_train.get("best_val_loss") is not None:
        lines.append(f"- **Best validation loss:** {norm_train['best_val_loss']:.4f}")

    if not training_summary:
        lines.append("")
        lines.append("_Training details not available — run step 06 (finetune) first._")

    lines.append("")

    # ── Evaluation results ─────────────────────────────────────────────
    lines.extend([
        "## Evaluation Results",
        "",
    ])

    if eval_summary:
        metrics = eval_summary.get("metrics", {})
        eval_cfg = eval_summary.get("config", {})
        n_samples = eval_cfg.get("num_samples_evaluated", "?")

        lines.append(f"Evaluated on **{n_samples}** test samples:")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        em = metrics.get("exact_match", {})
        if "rate" in em:
            lines.append(f"| Exact Match | {em['rate'] * 100:.1f}% |")

        cer = metrics.get("cer", {})
        if "mean" in cer:
            lines.append(f"| Character Error Rate (CER) | {cer['mean']:.4f} |")

        bleu = metrics.get("bleu", {})
        if "corpus_bleu" in bleu:
            lines.append(f"| BLEU | {bleu['corpus_bleu']:.4f} |")

        fmt = metrics.get("format_accuracy", {})
        if "mean" in fmt and fmt["mean"] is not None:
            lines.append(f"| Format Accuracy | {fmt['mean']:.2f} |")

        latency = metrics.get("latency", {})
        if "tokens_per_second" in latency:
            lines.append(f"| Tokens/second | {latency['tokens_per_second']:.1f} |")

        lines.append("")
    else:
        lines.extend([
            "_Evaluation results not available — run step 09 (evaluate) first._",
            "",
        ])

    # ── Usage: Python (mlx_lm) ─────────────────────────────────────────
    lines.extend([
        "## Usage",
        "",
        "### Python (mlx_lm)",
        "",
        "```python",
        "from mlx_lm import load, generate",
        "",
        f'model, tokenizer = load("{repo_id}")',
        "",
        "system_prompt = open(\"system_prompt.txt\").read()  # or paste from above",
        "",
        "raw_transcript = \"um so I was thinking uh we should probably like meet tomorrow\"",
        "",
        "messages = [",
        '    {"role": "system", "content": system_prompt},',
        '    {"role": "user", "content": raw_transcript},',
        "]",
        "",
        "prompt = tokenizer.apply_chat_template(",
        "    messages,",
        "    tokenize=False,",
        "    add_generation_prompt=True,",
        "    enable_thinking=False,",
        ")",
        "",
        "response = generate(",
        "    model,",
        "    tokenizer,",
        "    prompt=prompt,",
        "    max_tokens=1024,",
        "    temp=0.0,",
        "    repetition_penalty=1.1,",
        ")",
        "print(response)",
        "```",
        "",
    ])

    # ── Usage: Swift (mlx-swift-lm) ────────────────────────────────────
    lines.extend([
        "### Swift (mlx-swift-lm)",
        "",
        "This model is compatible with "
        "[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-examples) "
        "for on-device inference on Apple Silicon:",
        "",
        "```swift",
        "import LLM",
        "",
        f'let model = try await loadModel(id: "{repo_id}")',
        "let session = ChatSession(model)",
        "let cleaned = try await session.respond(to: rawTranscript)",
        "```",
        "",
        "> See the [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) "
        "repo for full integration details.",
        "",
    ])

    # ── License ────────────────────────────────────────────────────────
    lines.extend([
        "## License",
        "",
        f"This model is a fine-tune of [{mcfg.base_model}](https://huggingface.co/{mcfg.base_model}). "
        "Please refer to the base model's license for usage terms.",
        "",
    ])

    return "\n".join(lines)


# ── Upload logic ───────────────────────────────────────────────────────────


def _upload_model(
    mcfg: ModelConfig,
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upload a single model to HuggingFace Hub.

    Returns a summary dict with upload status and details.
    """
    repo_id = _repo_name(cfg, mcfg)
    bits = cfg.quantization.bits
    model_dir = MODELS_QUANTIZED / f"{mcfg.name}-{bits}bit"

    logger.info("Model: %s → %s", mcfg.name, repo_id)
    logger.info("Quantized model dir: %s", model_dir)

    # ── Validate model directory ───────────────────────────────────────
    validation_warnings = _validate_model_dir(model_dir)
    fatal_warnings = [w for w in validation_warnings if "does not exist" in w
                      or "Required file missing" in w
                      or "No .safetensors" in w]
    for w in validation_warnings:
        if w in fatal_warnings:
            logger.error("  FATAL: %s", w)
        else:
            logger.warning("  WARN: %s", w)

    if fatal_warnings and not dry_run:
        raise FileNotFoundError(
            f"Quantized model directory is invalid or incomplete: {model_dir}\n"
            f"Errors: {'; '.join(fatal_warnings)}\n"
            "Run step 08 (quantize) first."
        )

    # ── Load metadata ──────────────────────────────────────────────────
    training_summary_path = MODELS_ADAPTERS / mcfg.name / "training_summary.json"
    training_summary = _load_json_optional(training_summary_path, "Training summary")

    eval_summary_path = EVAL_RESULTS / mcfg.name / "eval_summary.json"
    eval_summary = _load_json_optional(eval_summary_path, "Eval summary")

    dataset_counts = {
        "train": _count_jsonl_lines(DATA_COMBINED / "train.jsonl"),
        "valid": _count_jsonl_lines(DATA_COMBINED / "valid.jsonl"),
        "test": _count_jsonl_lines(DATA_COMBINED / "test.jsonl"),
    }

    try:
        system_prompt = load_system_prompt(with_no_think=False)
    except FileNotFoundError:
        logger.warning("System prompt not found at %s — model card will omit it",
                        SYSTEM_PROMPT_PATH)
        system_prompt = None

    # ── Generate model card ────────────────────────────────────────────
    model_card = _generate_model_card(
        cfg=cfg,
        mcfg=mcfg,
        repo_id=repo_id,
        system_prompt=system_prompt,
        training_summary=training_summary,
        eval_summary=eval_summary,
        dataset_counts=dataset_counts,
    )

    logger.debug("Generated model card (%d chars)", len(model_card))

    # ── Dry-run reporting ──────────────────────────────────────────────
    if dry_run:
        logger.info("[DRY-RUN] Would create/update repo: %s (private=%s)",
                     repo_id, cfg.upload.private)
        logger.info("[DRY-RUN] Would upload model folder: %s", model_dir)
        logger.info("[DRY-RUN] Would upload generated README.md (%d chars)",
                     len(model_card))
        logger.info("[DRY-RUN] Would upload system_prompt.txt from %s",
                     SYSTEM_PROMPT_PATH)
        logger.info("[DRY-RUN] Training summary: %s",
                     "found" if training_summary else "NOT FOUND")
        logger.info("[DRY-RUN] Eval summary: %s",
                     "found" if eval_summary else "NOT FOUND")
        logger.info("[DRY-RUN] Dataset counts: %s", dataset_counts)
        return {
            "status": "dry-run",
            "model": mcfg.name,
            "repo_id": repo_id,
        }

    # ── Upload ─────────────────────────────────────────────────────────
    from huggingface_hub import HfApi

    api = HfApi()

    # Create / ensure repo exists
    logger.info("Creating/verifying repo: %s", repo_id)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=cfg.upload.private,
        exist_ok=True,
    )

    # Upload quantized model directory
    logger.info("Uploading model folder: %s", model_dir)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        commit_message=f"Upload {mcfg.name} quantized model ({bits}-bit)",
        ignore_patterns=[".*", "__pycache__", "*.pyc"],
    )
    logger.info("Model folder uploaded successfully")

    # Upload model card (README.md)
    logger.info("Uploading model card (README.md)")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(model_card)
        tmp_path = tmp.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add/update model card",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    logger.info("Model card uploaded successfully")

    # Upload system prompt
    if SYSTEM_PROMPT_PATH.exists():
        logger.info("Uploading system_prompt.txt")
        api.upload_file(
            path_or_fileobj=str(SYSTEM_PROMPT_PATH),
            path_in_repo="system_prompt.txt",
            repo_id=repo_id,
            commit_message="Add/update system prompt",
        )
        logger.info("System prompt uploaded successfully")
    else:
        logger.warning("System prompt file not found: %s", SYSTEM_PROMPT_PATH)

    return {
        "status": "completed",
        "model": mcfg.name,
        "repo_id": repo_id,
        "model_dir": str(model_dir),
        "has_training_summary": training_summary is not None,
        "has_eval_summary": eval_summary is not None,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the upload script."""
    args = parse_args()
    log = setup_logging(verbose=args.verbose)

    cfg = load_config(args.config)
    models = resolve_models(cfg, args.model)

    # ── Upload enabled check ───────────────────────────────────────────
    if not cfg.upload.enabled and not args.force and not args.dry_run:
        log.warning(
            "Upload is disabled in config (upload.enabled=false). "
            "Use --force to override, --dry-run to preview, "
            "or set upload.enabled=true in config.yaml."
        )
        sys.exit(0)

    if not cfg.upload.enabled and args.force:
        log.info("Upload disabled in config but --force specified, proceeding.")
    elif not cfg.upload.enabled and args.dry_run:
        log.info("Upload disabled in config — showing dry-run preview anyway.")

    log.info(
        "Models: %s | Repo prefix: %s | Private: %s",
        ", ".join(m.name for m in models),
        cfg.upload.repo_prefix,
        cfg.upload.private,
    )

    # ── Prerequisite checks ────────────────────────────────────────────
    _check_dependencies(dry_run=args.dry_run)
    _check_hf_auth(cfg, dry_run=args.dry_run)

    # ── Process each model ─────────────────────────────────────────────
    start = time.time()
    all_results: list[dict[str, Any]] = []
    errors: list[str] = []

    for mcfg in models:
        log.info("═" * 60)
        log.info("Uploading: %s", mcfg.name)
        log.info("═" * 60)

        try:
            result = _upload_model(
                mcfg,
                cfg,
                dry_run=args.dry_run,
            )
            all_results.append(result)
        except Exception as exc:
            log.error("Failed to upload %s: %s", mcfg.name, exc, exc_info=True)
            errors.append(f"{mcfg.name}: {exc}")

    elapsed = time.time() - start

    # ── Final summary ──────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Upload Summary")
    log.info("═" * 60)
    log.info("Total time: %.1f seconds", elapsed)

    for r in all_results:
        status = r.get("status", "unknown")
        model_name = r.get("model", "?")
        repo_id = r.get("repo_id", "?")

        if status == "dry-run":
            log.info("  %s: [DRY-RUN] → %s", model_name, repo_id)
        elif status == "completed":
            log.info("  %s: ✓ uploaded → %s", model_name, repo_id)
        else:
            log.info("  %s: %s", model_name, status)

    if errors:
        log.error("Errors encountered:")
        for e in errors:
            log.error("  • %s", e)
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
