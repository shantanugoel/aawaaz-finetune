#!/usr/bin/env python3
"""Fine-tune Qwen3 models for transcript cleanup.

Supports two training backends:
  - **Linux/CUDA**: Unsloth + TRL SFTTrainer (full-precision LoRA)
  - **Mac/Apple Silicon**: MLX LoRA via ``mlx_lm.lora`` CLI

Reads hyperparameters from ``config.yaml``, trains LoRA adapters, and saves
them to ``models/adapters/{model_name}/``.  Supports checkpoint resumption,
best-checkpoint tracking, and chat-template validation before training.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    DATA_COMBINED,
    MODELS_ADAPTERS,
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

logger = logging.getLogger("aawaaz.finetune")

# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the fine-tuning step."""
    parser = base_arg_parser(
        "Fine-tune Qwen3 models with LoRA for transcript cleanup."
    )
    add_model_arg(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing adapter output without prompting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint (if available).",
    )
    return parser.parse_args()


# ── Prerequisites ───────────────────────────────────────────────────────────


def _check_training_data() -> None:
    """Verify that training data files exist and are non-empty."""
    for split in ("train", "valid"):
        path = DATA_COMBINED / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Training data not found: {path}. Run 04_prepare_data.py first."
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Training data file is empty: {path}")
        # Quick parse check on the first line
        with open(path, encoding="utf-8") as fh:
            first = fh.readline().strip()
        try:
            obj = json.loads(first)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Training data has invalid JSON on first line: {path}"
            ) from exc
        if "messages" not in obj:
            raise ValueError(
                f"Training data missing 'messages' key: {path}. "
                "Expected {{\"messages\": [...]}} format."
            )


def _count_jsonl_lines(path: Path) -> int:
    """Count the number of lines in a JSONL file."""
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _check_prerequisites_linux() -> None:
    """Check that Unsloth, TRL, and related packages are available."""
    missing: list[str] = []
    try:
        import unsloth  # noqa: F401
    except ImportError:
        missing.append("unsloth")
    try:
        import trl  # noqa: F401
    except ImportError:
        missing.append("trl")
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    if missing:
        raise ImportError(
            f"Missing required packages for Linux training: {', '.join(missing)}. "
            "Install with: pip install -r requirements-linux.txt"
        )


def _check_prerequisites_mac() -> None:
    """Check that mlx-lm is available."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        raise ImportError(
            "mlx-lm is not installed. "
            "Install with: pip install -r requirements-mac.txt"
        )
    # Also verify that mlx_lm.lora CLI entry point is available
    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm.lora", "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "mlx_lm.lora CLI is not working. Check your mlx-lm installation."
        )


# ── Model source resolution ─────────────────────────────────────────────────


def _resolve_model_source(mcfg: ModelConfig, platform: str) -> str:
    """Determine the actual model source (local path or repo ID) for training.

    On Linux, prefers Unsloth variant → local base → remote Unsloth repo.
    On Mac, prefers local base → remote base repo.
    """
    local_base = MODELS_BASE / mcfg.name
    local_unsloth = MODELS_BASE / f"{mcfg.name}-unsloth"

    if platform == "linux":
        if local_unsloth.exists():
            return str(local_unsloth)
        if local_base.exists():
            return str(local_base)
        return mcfg.unsloth_model or mcfg.base_model
    else:
        if local_base.exists():
            return str(local_base)
        return mcfg.base_model


# ── Chat template validation ───────────────────────────────────────────────


def _validate_chat_template(model_source: str, platform: str) -> None:
    """Round-trip a sample through the tokenizer to verify chat template.

    Loads the tokenizer from the actual model source (local path or repo ID),
    applies the chat template to a sample message, tokenizes the result, then
    decodes it back.  Checks that ChatML markers are present and the round-trip
    is lossless.

    Parameters
    ----------
    model_source:
        Local path or HuggingFace repo ID of the model to validate.
    platform:
        ``"linux"`` or ``"mac"``.
    """
    logger.info("Validating chat template for %s ...", model_source)

    # Always use AutoTokenizer for validation — lightweight and platform-agnostic
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_source, trust_remote_code=True
    )

    sample_messages = [
        {"role": "system", "content": "You are an AI transcriber.\n/no_think"},
        {"role": "user", "content": "so um basically I wanted to say hello"},
        {"role": "assistant", "content": "I wanted to say hello."},
    ]

    # Apply chat template
    try:
        formatted = tokenizer.apply_chat_template(
            sample_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        logger.error(
            "tokenizer.apply_chat_template does not accept 'enable_thinking'. "
            "This Qwen3 tokenizer version may not support thinking mode control. "
            "Proceeding with /no_think in system prompt as fallback, but verify "
            "that training data does not contain <think> blocks."
        )
        formatted = tokenizer.apply_chat_template(
            sample_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    logger.debug("Chat template output:\n%s", formatted)

    # Verify ChatML markers
    if "<|im_start|>" not in formatted:
        raise ValueError(
            "Chat template does not contain expected ChatML markers "
            "(<|im_start|>). This model may have an incompatible template."
        )

    # Verify all three roles are present
    for role in ("system", "user", "assistant"):
        if f"<|im_start|>{role}" not in formatted:
            raise ValueError(
                f"Chat template missing <|im_start|>{role} marker. "
                "Template may be misconfigured."
            )

    # Verify thinking mode is not triggered.
    # Qwen3 tokenizers may insert empty <think>\n\n</think> blocks even with
    # enable_thinking=False — that's harmless.  Only fail if the block has
    # actual content.
    import re as _re

    think_blocks = _re.findall(r"<think>(.*?)</think>", formatted, _re.DOTALL)
    non_empty = [b for b in think_blocks if b.strip()]
    if non_empty:
        raise ValueError(
            "Chat template output contains non-empty <think> blocks even with "
            "enable_thinking=False. Thinking mode is not properly disabled. "
            f"Content: {non_empty[0][:200]!r}"
        )
    if think_blocks:
        logger.info(
            "Chat template contains empty <think></think> tags (expected "
            "for Qwen3 with enable_thinking=False). These are harmless."
        )

    # Round-trip: tokenize and decode
    token_ids = tokenizer.encode(formatted)
    decoded = tokenizer.decode(token_ids)
    if "I wanted to say hello." not in decoded:
        logger.warning(
            "Round-trip decode does not contain expected assistant content. "
            "Decoded text: %r",
            decoded[:300],
        )
    else:
        logger.info("Chat template validation passed ✓")

    del tokenizer


# ── Linux/Unsloth training ──────────────────────────────────────────────────


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest HF Trainer checkpoint directory in output_dir."""
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    return checkpoints[-1] if checkpoints else None


def _train_linux(
    mcfg: ModelConfig,
    cfg: PipelineConfig,
    *,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train a model using Unsloth + TRL SFTTrainer (Linux/CUDA)."""
    tr = cfg.training
    lx = tr.linux
    adapter_dir = MODELS_ADAPTERS / mcfg.name

    # Check for existing output — clean on --force
    if adapter_dir.exists() and force and not resume:
        logger.info("--force: removing existing adapter directory %s", adapter_dir)
        if not dry_run:
            shutil.rmtree(adapter_dir)
    elif adapter_dir.exists() and not force and not resume:
        logger.warning(
            "Adapter directory already exists: %s. "
            "Use --force to overwrite or --resume to continue.",
            adapter_dir,
        )
        return {"status": "skipped", "reason": "output exists"}

    # Determine model source
    model_source = _resolve_model_source(mcfg, "linux")
    logger.info("Model source: %s", model_source)

    if dry_run:
        logger.info("[DRY-RUN] Would fine-tune %s", model_source)
        logger.info("[DRY-RUN]   LoRA: rank=%d, alpha=%d", tr.lora.rank, tr.lora.alpha)
        logger.info("[DRY-RUN]   LR=%s, batch=%d, epochs=%d", lx.learning_rate, lx.batch_size, tr.num_epochs)
        logger.info("[DRY-RUN]   Output: %s", adapter_dir)
        return {"status": "dry-run", "model": mcfg.name}

    # Heavy imports deferred until after dry-run check
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
    from unsloth import FastLanguageModel

    # ── Load model ──────────────────────────────────────────────────────
    logger.info("Loading model: %s (load_in_4bit=%s)", model_source, lx.load_in_4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=tr.max_seq_length,
        load_in_4bit=lx.load_in_4bit,
        dtype=torch.bfloat16 if lx.bf16 else None,
    )

    # ── Apply LoRA ──────────────────────────────────────────────────────
    logger.info(
        "Applying LoRA: rank=%d, alpha=%d, dropout=%.2f, targets=%s",
        tr.lora.rank, tr.lora.alpha, tr.lora.dropout, tr.lora.target_modules,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=tr.lora.rank,
        target_modules=tr.lora.target_modules,
        lora_alpha=tr.lora.alpha,
        lora_dropout=tr.lora.dropout,
        use_gradient_checkpointing="unsloth",
    )

    # ── Load dataset ────────────────────────────────────────────────────
    logger.info("Loading training data from %s", DATA_COMBINED)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_COMBINED / "train.jsonl"),
            "validation": str(DATA_COMBINED / "valid.jsonl"),
        },
    )

    train_count = len(dataset["train"])
    valid_count = len(dataset["validation"])
    logger.info("Dataset: %d train, %d validation examples", train_count, valid_count)

    # ── Format data with chat template ──────────────────────────────────
    logger.info("Applying chat template (enable_thinking=False)")

    def format_chat(example: dict[str, Any]) -> dict[str, str]:
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        return {"text": text}

    dataset = dataset.map(format_chat, desc="Formatting chat templates")

    # Spot-check a formatted sample
    sample_text = dataset["train"][0]["text"]
    logger.debug("Sample formatted text:\n%s", sample_text[:500])
    if "<think>" in sample_text:
        logger.warning(
            "Formatted training sample contains <think> tags! "
            "Thinking mode may not be properly disabled."
        )

    # ── Training arguments ──────────────────────────────────────────────
    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=lx.batch_size,
        gradient_accumulation_steps=lx.gradient_accumulation_steps,
        warmup_steps=lx.warmup_steps,
        learning_rate=lx.learning_rate,
        num_train_epochs=tr.num_epochs,
        bf16=lx.bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=tr.eval_every,
        save_steps=tr.save_every,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=lx.weight_decay,
        optim=lx.optimizer,
        seed=42,
        report_to="none",
    )

    # ── Prompt masking: only compute loss on assistant response ─────────
    # The assistant's response starts after "<|im_start|>assistant\n" in
    # ChatML format. DataCollatorForCompletionOnlyLM masks everything
    # before this marker so loss is only computed on the assistant tokens.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ── SFTTrainer ──────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=tr.max_seq_length,
        data_collator=collator,
        args=training_args,
    )

    # ── Sanity check: initial eval loss ─────────────────────────────────
    logger.info("Running initial evaluation to sanity-check loss...")
    initial_metrics = trainer.evaluate()
    initial_loss = initial_metrics.get("eval_loss", -1.0)
    logger.info("Initial eval loss: %.4f", initial_loss)
    if initial_loss < 0.1:
        logger.warning(
            "Initial eval loss is suspiciously low (%.4f). "
            "Prompt masking may not be working correctly — the model might be "
            "memorizing system/user prompts instead of learning the cleanup task.",
            initial_loss,
        )

    # ── Train ───────────────────────────────────────────────────────────
    resume_checkpoint = None
    if resume:
        resume_checkpoint = _find_latest_checkpoint(adapter_dir)
        if resume_checkpoint:
            logger.info("Resuming from checkpoint: %s", resume_checkpoint)
        else:
            logger.info("No checkpoint found to resume from, starting fresh.")

    logger.info("Starting training...")
    train_start = time.time()
    train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    train_elapsed = time.time() - train_start

    # ── Save final adapters ─────────────────────────────────────────────
    logger.info("Saving adapters to %s", adapter_dir)
    # Trainer with load_best_model_at_end will have the best model loaded
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # ── Training summary ────────────────────────────────────────────────
    final_metrics = trainer.evaluate()
    final_train_loss = train_result.metrics.get("train_loss", -1.0)
    final_eval_loss = final_metrics.get("eval_loss", -1.0)

    # Extract best eval loss from trainer log history
    best_eval_loss = final_eval_loss
    if hasattr(trainer, "state") and trainer.state.log_history:
        eval_losses = [
            entry["eval_loss"]
            for entry in trainer.state.log_history
            if "eval_loss" in entry
        ]
        if eval_losses:
            best_eval_loss = min(eval_losses)

    summary = {
        "status": "completed",
        "model": mcfg.name,
        "model_source": model_source,
        "train_samples": train_count,
        "valid_samples": valid_count,
        "initial_eval_loss": initial_loss,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "best_eval_loss": best_eval_loss,
        "train_time_seconds": train_elapsed,
        "total_steps": train_result.metrics.get("train_steps", train_result.global_step),
        "adapter_dir": str(adapter_dir),
    }

    # Save training summary
    summary_path = adapter_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Training summary saved to %s", summary_path)

    # Clean up GPU memory
    del model
    del tokenizer
    del trainer
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return summary


# ── Mac/MLX training ────────────────────────────────────────────────────────


def _parse_mlx_log_line(line: str) -> dict[str, Any] | None:
    """Parse an MLX training log line for loss/val_loss metrics.

    MLX lora output lines look like:
      Iter 10: Train loss 3.456, Learning Rate 1.000e-05, It/sec 2.34, ...
      Iter 100: Val loss 2.789, Val took 1.23s
    """
    result: dict[str, Any] = {}

    iter_match = re.search(r"Iter\s+(\d+)", line)
    if not iter_match:
        return None
    result["iter"] = int(iter_match.group(1))

    train_loss_match = re.search(r"Train loss\s+([\d.]+)", line)
    if train_loss_match:
        result["train_loss"] = float(train_loss_match.group(1))

    val_loss_match = re.search(r"Val loss\s+([\d.]+)", line)
    if val_loss_match:
        result["val_loss"] = float(val_loss_match.group(1))

    lr_match = re.search(r"Learning Rate\s+([\d.eE+-]+)", line)
    if lr_match:
        result["learning_rate"] = float(lr_match.group(1))

    it_sec_match = re.search(r"It/sec\s+([\d.]+)", line)
    if it_sec_match:
        result["it_per_sec"] = float(it_sec_match.group(1))

    tokens_sec_match = re.search(r"Tokens/sec\s+([\d.]+)", line)
    if tokens_sec_match:
        result["tokens_per_sec"] = float(tokens_sec_match.group(1))

    return result


def _find_latest_mlx_checkpoint(adapter_dir: Path) -> Path | None:
    """Find the latest MLX adapter checkpoint file for resume.

    MLX saves checkpoints as ``adapters-{iter:09d}.safetensors``.
    Falls back to ``adapters.safetensors`` if no numbered checkpoints exist.
    """
    checkpoints = sorted(adapter_dir.glob("adapters-*.safetensors"))
    if checkpoints:
        return checkpoints[-1]
    final = adapter_dir / "adapters.safetensors"
    return final if final.exists() else None


def _find_best_mlx_checkpoint(adapter_dir: Path, log_entries: list[dict[str, Any]]) -> Path | None:
    """Find the MLX checkpoint with the lowest validation loss.

    MLX saves adapters as adapters-{iter}.safetensors. We match these
    against the parsed log entries to find the one with the lowest val_loss.
    """
    val_entries = [e for e in log_entries if "val_loss" in e]
    if not val_entries:
        return None

    best = min(val_entries, key=lambda e: e["val_loss"])
    best_iter = best["iter"]

    # MLX adapter checkpoint file
    checkpoint = adapter_dir / f"adapters-{best_iter:09d}.safetensors"
    if not checkpoint.exists():
        # Try without zero-padding
        checkpoint = adapter_dir / f"adapters-{best_iter}.safetensors"
    if not checkpoint.exists():
        # Try the pattern MLX uses
        for candidate in adapter_dir.glob(f"adapters*{best_iter}*"):
            checkpoint = candidate
            break

    if checkpoint.exists():
        logger.info(
            "Best MLX checkpoint: iter %d with val_loss=%.4f → %s",
            best_iter, best["val_loss"], checkpoint.name,
        )
        return checkpoint
    else:
        logger.warning(
            "Best checkpoint should be at iter %d (val_loss=%.4f) but file not found.",
            best_iter, best["val_loss"],
        )
        return None


def _train_mac(
    mcfg: ModelConfig,
    cfg: PipelineConfig,
    *,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train a model using MLX LoRA (Mac/Apple Silicon)."""
    tr = cfg.training
    mac = tr.mac
    adapter_dir = MODELS_ADAPTERS / mcfg.name

    # Check for existing output — clean on --force
    final_adapter = adapter_dir / "adapters.safetensors"
    if adapter_dir.exists() and force and not resume:
        logger.info("--force: removing existing adapter directory %s", adapter_dir)
        if not dry_run:
            shutil.rmtree(adapter_dir)
    elif final_adapter.exists() and not force and not resume:
        logger.warning(
            "Adapter already exists: %s. "
            "Use --force to overwrite or --resume to continue.",
            final_adapter,
        )
        return {"status": "skipped", "reason": "output exists"}

    # Determine model source
    model_source = _resolve_model_source(mcfg, "mac")
    logger.info("Model source: %s", model_source)

    adapter_dir.mkdir(parents=True, exist_ok=True)

    # ── Build mlx_lm.lora command ───────────────────────────────────────
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_source,
        "--data", str(DATA_COMBINED),
        "--train",
        "--batch-size", str(mac.batch_size),
        "--lora-layers", str(mac.lora_layers),
        "--iters", str(mac.iters),
        "--learning-rate", str(mac.learning_rate),
        "--adapter-path", str(adapter_dir),
        "--steps-per-eval", str(tr.eval_every),
        "--save-every", str(tr.save_every),
    ]

    if tr.mask_prompt:
        cmd.append("--mask-prompt")
    if mac.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if mac.grad_accumulation_steps > 1:
        cmd.extend(["--grad-accumulation-steps", str(mac.grad_accumulation_steps)])

    # Resume support: find latest checkpoint adapter file
    if resume:
        resume_file = _find_latest_mlx_checkpoint(adapter_dir)
        if resume_file:
            cmd.extend(["--resume-adapter-file", str(resume_file)])
            logger.info("Resuming from adapter: %s", resume_file)
        else:
            logger.info("No checkpoint found to resume from, starting fresh.")

    logger.info("MLX LoRA command:\n  %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY-RUN] Would run the above command.")
        return {"status": "dry-run", "model": mcfg.name}

    # ── Run training ────────────────────────────────────────────────────
    logger.info("Starting MLX LoRA training...")
    train_start = time.time()

    log_entries: list[dict[str, Any]] = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            logger.info("  [MLX] %s", line)

            parsed = _parse_mlx_log_line(line)
            if parsed:
                log_entries.append(parsed)

                # Log key metrics
                if "val_loss" in parsed:
                    logger.info(
                        "  → Iter %d: val_loss=%.4f",
                        parsed["iter"], parsed["val_loss"],
                    )

    process.wait()
    train_elapsed = time.time() - train_start

    if process.returncode != 0:
        raise RuntimeError(
            f"MLX LoRA training failed with return code {process.returncode}. "
            "Check the log output above for details."
        )

    # ── Best checkpoint handling ────────────────────────────────────────
    best_checkpoint = _find_best_mlx_checkpoint(adapter_dir, log_entries)
    if best_checkpoint and best_checkpoint.name != "adapters.safetensors":
        # Copy best checkpoint as the primary adapter
        best_dest = adapter_dir / "adapters.safetensors"
        shutil.copy2(best_checkpoint, best_dest)
        logger.info("Copied best checkpoint to %s", best_dest)

    # ── Training summary ────────────────────────────────────────────────
    train_losses = [e["train_loss"] for e in log_entries if "train_loss" in e]
    val_losses = [e["val_loss"] for e in log_entries if "val_loss" in e]

    summary: dict[str, Any] = {
        "status": "completed",
        "model": mcfg.name,
        "model_source": model_source,
        "train_time_seconds": train_elapsed,
        "total_iters": mac.iters,
        "adapter_dir": str(adapter_dir),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "log_entries_count": len(log_entries),
    }

    # Save training summary
    summary_path = adapter_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Training summary saved to %s", summary_path)

    return summary


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the fine-tuning script."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    cfg: PipelineConfig = load_config(args.config)
    models = resolve_models(cfg, args.model)

    logger.info(
        "Platform: %s | Models to fine-tune: %s",
        cfg.platform,
        ", ".join(m.name for m in models),
    )

    # ── Prerequisites ───────────────────────────────────────────────────
    _check_training_data()
    train_count = _count_jsonl_lines(DATA_COMBINED / "train.jsonl")
    valid_count = _count_jsonl_lines(DATA_COMBINED / "valid.jsonl")
    logger.info("Training data: %d train, %d validation examples", train_count, valid_count)

    if not args.dry_run:
        if cfg.platform == "linux":
            _check_prerequisites_linux()
        else:
            _check_prerequisites_mac()
        ensure_dirs()

    # ── Train each model ────────────────────────────────────────────────
    start = time.time()
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    for mcfg in models:
        logger.info("═" * 60)
        logger.info("Fine-tuning model: %s", mcfg.name)
        logger.info("═" * 60)

        try:
            # Validate chat template before training (skip on dry-run)
            if not args.dry_run:
                model_source = _resolve_model_source(mcfg, cfg.platform)
                _validate_chat_template(model_source, cfg.platform)

            # Run platform-specific training
            if cfg.platform == "linux":
                summary = _train_linux(
                    mcfg, cfg,
                    force=args.force,
                    resume=args.resume,
                    dry_run=args.dry_run,
                )
            else:
                summary = _train_mac(
                    mcfg, cfg,
                    force=args.force,
                    resume=args.resume,
                    dry_run=args.dry_run,
                )

            summaries.append(summary)

        except Exception as exc:
            logger.error("Failed to fine-tune %s: %s", mcfg.name, exc, exc_info=True)
            errors.append(f"{mcfg.name}: {exc}")

    elapsed = time.time() - start

    # ── Final summary ───────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Fine-tuning Summary")
    logger.info("═" * 60)
    logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    for s in summaries:
        status = s.get("status", "unknown")
        model_name = s.get("model", "?")

        if status == "dry-run":
            logger.info("  %s: [DRY-RUN] would train", model_name)
        elif status == "skipped":
            logger.info("  %s: skipped — %s", model_name, s.get("reason", ""))
        elif status == "completed":
            train_loss = s.get("final_train_loss")
            eval_loss = s.get("final_eval_loss") or s.get("final_val_loss")
            best_loss = s.get("best_val_loss")
            train_time = s.get("train_time_seconds", 0)
            logger.info(
                "  %s: completed in %.1f min | "
                "train_loss=%.4f | eval_loss=%.4f%s",
                model_name,
                train_time / 60,
                train_loss if train_loss is not None else -1,
                eval_loss if eval_loss is not None else -1,
                f" | best_val_loss={best_loss:.4f}" if best_loss is not None else "",
            )
            logger.info("    → Adapters: %s", s.get("adapter_dir", "?"))
        else:
            logger.info("  %s: %s", model_name, status)

    if errors:
        logger.error("Errors encountered:")
        for e in errors:
            logger.error("  • %s", e)
        raise SystemExit(1)

    logger.info("All models fine-tuned successfully.")


if __name__ == "__main__":
    main()
