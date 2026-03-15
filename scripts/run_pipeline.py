#!/usr/bin/env python3
"""Aawaaz fine-tuning pipeline orchestrator.

Runs pipeline steps in order, tracks state, validates dependencies,
and provides clear progress reporting.

Usage examples::

    # Run everything end to end (excludes step 10 — upload)
    python scripts/run_pipeline.py --all

    # Run specific steps
    python scripts/run_pipeline.py --steps 2,3,3b,4
    python scripts/run_pipeline.py --steps 6 --model qwen3-0.6b

    # Resume from where it left off
    python scripts/run_pipeline.py --all --resume

    # Dry run
    python scripts/run_pipeline.py --all --dry-run

    # Upload (must be run standalone)
    python scripts/run_pipeline.py --steps 10
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Allow importing common from the scripts package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CONFIG_PATH,
    PIPELINE_STATE_PATH,
    PROJECT_ROOT,
    base_arg_parser,
    load_config,
    setup_logging,
)

# ── Step registry ───────────────────────────────────────────────────────────


@dataclass
class StepDef:
    """Metadata for a single pipeline step."""

    step_id: str
    script: str
    name: str
    deps: list[str]
    forwards: set[str] = field(default_factory=set)
    include_in_all: bool = True


# Steps in canonical pipeline order.
STEPS: dict[str, StepDef] = {}

_step_defs = [
    StepDef(
        step_id="2",
        script="02_pull_datasets.py",
        name="Pull datasets",
        deps=[],
        forwards={"force"},
    ),
    StepDef(
        step_id="3",
        script="03_generate_synthetic.py",
        name="Generate synthetic data",
        deps=[],
        forwards={"synthetic_samples", "force"},
    ),
    StepDef(
        step_id="3b",
        script="03b_validate_synthetic.py",
        name="Validate synthetic data",
        deps=["3"],
        forwards={"sample_rate", "force", "yes"},
    ),
    StepDef(
        step_id="4",
        script="04_prepare_data.py",
        name="Prepare training data",
        deps=["2"],  # deps adjusted at runtime if validation is enabled
        forwards=set(),
    ),
    StepDef(
        step_id="5",
        script="05_download_models.py",
        name="Download models",
        deps=[],
        forwards={"model", "force"},
    ),
    StepDef(
        step_id="6",
        script="06_finetune.py",
        name="Fine-tune models",
        deps=["4", "5"],
        forwards={"model", "force", "resume"},
    ),
    StepDef(
        step_id="7",
        script="07_fuse_and_convert.py",
        name="Fuse & convert models",
        deps=["6"],
        forwards={"model", "force"},
    ),
    StepDef(
        step_id="8",
        script="08_quantize.py",
        name="Quantize models",
        deps=["7"],
        forwards={"model", "force"},
    ),
    StepDef(
        step_id="9",
        script="09_evaluate.py",
        name="Evaluate models",
        deps=["8"],
        forwards={"model", "force"},
    ),
    StepDef(
        step_id="10",
        script="10_upload.py",
        name="Upload to Hugging Face Hub",
        deps=["9"],
        forwards={"model", "force"},
        include_in_all=False,
    ),
]

for _sd in _step_defs:
    STEPS[_sd.step_id] = _sd

# Canonical order for iteration.
STEP_ORDER: list[str] = [sd.step_id for sd in _step_defs]

ALL_STEPS: list[str] = [sid for sid in STEP_ORDER if STEPS[sid].include_in_all]


# ── Pipeline state management ──────────────────────────────────────────────


def _load_state() -> dict[str, Any]:
    """Load pipeline state from disk, or return an empty state dict."""
    if PIPELINE_STATE_PATH.exists():
        try:
            with open(PIPELINE_STATE_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {"steps": {}}
        # Validate shape: must be a dict with a "steps" dict-of-dicts.
        if not isinstance(data, dict) or not isinstance(data.get("steps"), dict):
            return {"steps": {}}
        return data
    return {"steps": {}}


def _save_state(state: dict[str, Any]) -> None:
    """Atomically write pipeline state to disk."""
    tmp = PIPELINE_STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
        fh.write("\n")
    tmp.replace(PIPELINE_STATE_PATH)


def _mark_step(
    state: dict[str, Any],
    step_id: str,
    status: str,
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
    duration_seconds: float | None = None,
    error: str | None = None,
) -> None:
    """Update a step's status in the state dict and persist.

    Clears stale fields on status transitions to prevent misleading state:
    - ``running``: clears completed_at, duration_seconds, error
    - ``done``: clears error
    """
    entry = state["steps"].setdefault(step_id, {})
    entry["status"] = status

    if status == "running":
        entry.pop("completed_at", None)
        entry.pop("duration_seconds", None)
        entry.pop("error", None)
    elif status == "done":
        entry.pop("error", None)

    if started_at is not None:
        entry["started_at"] = started_at
    if completed_at is not None:
        entry["completed_at"] = completed_at
    if duration_seconds is not None:
        entry["duration_seconds"] = round(duration_seconds, 1)
    if error is not None:
        entry["error"] = error
    _save_state(state)


def _is_step_done(state: dict[str, Any], step_id: str) -> bool:
    """Check if a step is recorded as done in the state."""
    return state.get("steps", {}).get(step_id, {}).get("status") == "done"


def _clear_state() -> None:
    """Remove the pipeline state file."""
    if PIPELINE_STATE_PATH.exists():
        PIPELINE_STATE_PATH.unlink()


# ── Dependency resolution ──────────────────────────────────────────────────


def _resolve_deps(requested_steps: list[str], state: dict[str, Any]) -> list[str]:
    """Validate that dependencies are satisfied for each requested step.

    Returns the list of unsatisfied dependencies (empty = all good).
    A dependency is satisfied if it's either in the requested steps
    (will run before the dependent step) or already done in the state.
    """
    missing: list[str] = []
    requested_set = set(requested_steps)

    for step_id in requested_steps:
        step_def = STEPS[step_id]
        for dep in step_def.deps:
            dep_in_request = dep in requested_set
            dep_done = _is_step_done(state, dep)
            if not dep_in_request and not dep_done:
                missing.append(
                    f"Step {step_id} ({step_def.name}) requires step {dep} "
                    f"({STEPS[dep].name}), which is neither requested nor "
                    f"previously completed."
                )
    return missing


# ── Command building ───────────────────────────────────────────────────────

# Mapping from orchestrator CLI arg names to child script flag names.
_ARG_TO_FLAG: dict[str, str] = {
    "model": "--model",
    "force": "--force",
    "resume": "--resume",
    "synthetic_samples": "--synthetic-samples",
    "sample_rate": "--sample-rate",
    "yes": "--yes",
}


def _build_command(
    step_def: StepDef,
    *,
    config_path: Path,
    verbose: bool,
    dry_run: bool,
    overrides: dict[str, Any],
) -> list[str]:
    """Build the subprocess command list for a pipeline step."""
    script_path = Path(__file__).resolve().parent / step_def.script
    cmd: list[str] = [sys.executable, str(script_path)]

    # Always pass config and common flags.
    cmd.extend(["--config", str(config_path)])
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")

    # Forward allowed overrides.
    for arg_name, value in overrides.items():
        if arg_name not in step_def.forwards:
            continue
        flag = _ARG_TO_FLAG.get(arg_name)
        if flag is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    return cmd


# ── Temp config with overrides ─────────────────────────────────────────────


def _create_override_config(base_config_path: Path, overrides: dict[str, Any]) -> Path:
    """Create a temporary config file with platform/other top-level overrides applied.

    Returns the path to the temp file.  Caller is responsible for cleanup.

    Raises
    ------
    yaml.YAMLError
        If the base config file contains invalid YAML.
    """
    with open(base_config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if "platform" in overrides:
        raw["platform"] = overrides["platform"]

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="aawaaz_pipeline_",
        delete=False,
        encoding="utf-8",
    )
    yaml.safe_dump(raw, tmp, default_flow_style=False)
    tmp.close()
    return Path(tmp.name)


# ── Error suggestions ──────────────────────────────────────────────────────

_STEP_ERROR_HINTS: dict[str, str] = {
    "2": (
        "Check your internet connection and Hugging Face authentication.\n"
        "  Run: huggingface-cli login"
    ),
    "3": (
        "Check that the LLM provider is reachable and API key is set.\n"
        "  Verify the provider/model/base_url in config.yaml under dataset.synthetic."
    ),
    "3b": (
        "Validation may have failed due to provider issues.\n"
        "  Check API key and provider config under dataset.synthetic.validation.\n"
        "  You can also try: --sample-rate 0.05 to reduce validation cost."
    ),
    "4": (
        "Ensure steps 2 and 3/3b completed successfully.\n"
        "  Check that data/raw/ and data/synthetic/ contain valid JSONL files."
    ),
    "5": (
        "Check internet connection and Hugging Face authentication.\n"
        "  Ensure sufficient disk space for model downloads."
    ),
    "6": (
        "Fine-tuning failures are often caused by:\n"
        "  - Insufficient GPU memory → reduce training.linux.batch_size in config.yaml\n"
        "  - Missing dependencies → run 01_setup.sh to reinstall\n"
        "  - Corrupted training data → re-run step 4"
    ),
    "7": (
        "Fuse/convert failures may indicate:\n"
        "  - Missing adapter files → re-run step 6\n"
        "  - Incompatible mlx-lm version → check requirements"
    ),
    "8": (
        "Quantization failures may indicate:\n"
        "  - Missing fused model → re-run step 7\n"
        "  - Insufficient memory → close other applications"
    ),
    "9": (
        "Evaluation failures may indicate:\n"
        "  - Missing quantized model → re-run step 8\n"
        "  - Missing test data → re-run step 4"
    ),
    "10": (
        "Upload failures may indicate:\n"
        "  - Missing Hugging Face token → run: huggingface-cli login\n"
        "  - upload.enabled is false in config.yaml (use --force to override)"
    ),
}


# ── Formatting helpers ─────────────────────────────────────────────────────


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _iso_now() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _step_header(step_id: str, step_def: StepDef, idx: int, total: int) -> str:
    """Format a step header banner."""
    return (
        f"\n{'=' * 70}\n"
        f"  Step {step_id}: {step_def.name}  [{idx}/{total}]\n"
        f"{'=' * 70}"
    )


# ── CLI argument parsing ───────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> Any:
    """Parse orchestrator CLI arguments."""
    parser = base_arg_parser(
        description="Aawaaz fine-tuning pipeline orchestrator.\n\n"
        "Run all steps, specific steps, or resume from a previous run."
    )

    # Step selection (mutually exclusive).
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all pipeline steps (2 through 9). Step 10 (upload) is excluded.",
    )
    step_group.add_argument(
        "--steps",
        type=str,
        default=None,
        help=(
            "Comma-separated list of step IDs to run. "
            f"Valid steps: {', '.join(STEP_ORDER)}. "
            "Example: --steps 2,3,3b,4"
        ),
    )

    # Resume mode.
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps already completed (from .pipeline_state.json). "
        "Re-run failed or interrupted steps.",
    )

    # Config overrides.
    parser.add_argument(
        "--platform",
        type=str,
        choices=["linux", "mac"],
        default=None,
        help="Override the platform setting in config.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to operate on (e.g. 'qwen3-0.6b'). "
        "Forwarded to steps that support --model.",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=None,
        help="Override synthetic sample count (forwarded to step 3).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Override validation sample rate (forwarded to step 3b).",
    )
    parser.add_argument(
        "--clear-state",
        action="store_true",
        help="Clear pipeline state (.pipeline_state.json) and exit.",
    )

    args = parser.parse_args(argv)
    return args


# ── Main orchestration logic ───────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline orchestrator.  Returns 0 on success, 1 on failure."""
    args = parse_args(argv)
    log = setup_logging(args.verbose)

    # Handle --clear-state.
    if args.clear_state:
        _clear_state()
        log.info("Pipeline state cleared.")
        return 0

    # Require --all or --steps when not using --clear-state.
    if not args.run_all and args.steps is None:
        log.error("One of --all or --steps is required (or use --clear-state).")
        return 1

    # Validate override arguments early.
    if args.synthetic_samples is not None and args.synthetic_samples <= 0:
        log.error("--synthetic-samples must be a positive integer.")
        return 1
    if args.sample_rate is not None and not (0 < args.sample_rate <= 1.0):
        log.error("--sample-rate must be in (0, 1.0].")
        return 1

    # Load config for dependency adjustments and summary.
    try:
        effective_config_path = Path(args.config)
        cfg = load_config(effective_config_path)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        log.error("Failed to load config: %s", exc)
        return 1

    # Adjust step 4 deps based on validation config.
    step4 = STEPS["4"]
    if cfg.dataset.synthetic.enabled and cfg.dataset.synthetic.validation.enabled:
        step4.deps = ["2", "3b"]
    elif cfg.dataset.synthetic.enabled:
        step4.deps = ["2", "3"]
    else:
        # Synthetic disabled entirely — step 4 only needs pulled datasets.
        step4.deps = ["2"]

    # Determine which steps to run.
    if args.run_all:
        # Build --all list dynamically based on config.
        requested_steps = []
        for sid in STEP_ORDER:
            if not STEPS[sid].include_in_all:
                continue
            # Skip synthetic steps when synthetic is disabled.
            if sid in ("3", "3b") and not cfg.dataset.synthetic.enabled:
                continue
            # Skip validation step when validation is disabled.
            if sid == "3b" and not cfg.dataset.synthetic.validation.enabled:
                continue
            requested_steps.append(sid)
    else:
        raw_steps = [s.strip() for s in args.steps.split(",") if s.strip()]
        # Validate step IDs.
        invalid = [s for s in raw_steps if s not in STEPS]
        if invalid:
            log.error(
                "Unknown step(s): %s. Valid steps: %s",
                ", ".join(invalid),
                ", ".join(STEP_ORDER),
            )
            return 1

        # Step 10 must be standalone.
        if "10" in raw_steps and len(raw_steps) > 1:
            log.error(
                "Step 10 (Upload to HF Hub) must be run as a standalone step "
                "to prevent accidental uploads.\n"
                "  Use: python scripts/run_pipeline.py --steps 10"
            )
            return 1

        # De-duplicate and sort in canonical pipeline order.
        seen: set[str] = set()
        requested_steps = []
        for sid in STEP_ORDER:
            if sid in raw_steps and sid not in seen:
                requested_steps.append(sid)
                seen.add(sid)

    # Load pipeline state.
    state = _load_state()

    # If --resume, filter out completed steps.
    skipped_steps: list[str] = []
    if args.resume:
        run_steps: list[str] = []
        for sid in requested_steps:
            if _is_step_done(state, sid):
                skipped_steps.append(sid)
            else:
                run_steps.append(sid)
        requested_steps = run_steps

    if not requested_steps:
        if skipped_steps:
            log.info(
                "All requested steps already completed: %s. "
                "Use --clear-state to reset.",
                ", ".join(skipped_steps),
            )
        else:
            log.info("No steps to run.")
        return 0

    # Validate dependencies.
    dep_errors = _resolve_deps(requested_steps, state)
    if dep_errors:
        log.error("Dependency errors:")
        for err in dep_errors:
            log.error("  • %s", err)
        log.error(
            "\nInclude the missing steps in --steps, or run them first.\n"
            "If they were previously completed, use --resume to acknowledge them."
        )
        return 1

    # Build overrides dict for forwarding to child scripts.
    overrides: dict[str, Any] = {}
    if args.model is not None:
        overrides["model"] = args.model
    if args.synthetic_samples is not None:
        overrides["synthetic_samples"] = args.synthetic_samples
    if args.sample_rate is not None:
        overrides["sample_rate"] = args.sample_rate
    if args.resume:
        overrides["resume"] = True
    # When orchestrating, auto-confirm prompts in child scripts.
    overrides["yes"] = True

    # If --platform override, create a temp config.
    temp_config_path: Path | None = None
    if args.platform is not None:
        try:
            temp_config_path = _create_override_config(
                effective_config_path, {"platform": args.platform}
            )
        except (yaml.YAMLError, OSError) as exc:
            log.error("Failed to create platform-override config: %s", exc)
            return 1
        effective_config_path = temp_config_path
        log.info("Platform override: using '%s'", args.platform)

    # Print execution plan.
    platform_label = args.platform or cfg.platform
    log.info("Pipeline plan (%s):", platform_label)
    if skipped_steps:
        log.info(
            "  Skipping (already done): %s",
            ", ".join(f"{s} ({STEPS[s].name})" for s in skipped_steps),
        )
    for i, sid in enumerate(requested_steps, 1):
        log.info("  %d. Step %s — %s", i, sid, STEPS[sid].name)

    if args.dry_run:
        log.info("\n[DRY RUN] Showing commands that would be executed:\n")
        for i, sid in enumerate(requested_steps, 1):
            step_def = STEPS[sid]
            cmd = _build_command(
                step_def,
                config_path=effective_config_path,
                verbose=args.verbose,
                dry_run=True,
                overrides=overrides,
            )
            log.info("  %d. %s", i, " ".join(cmd))
        _cleanup_temp_config(temp_config_path)
        return 0

    # Execute steps.
    pipeline_start = time.monotonic()
    completed: list[str] = []
    failed_step: str | None = None

    for i, sid in enumerate(requested_steps, 1):
        step_def = STEPS[sid]
        total = len(requested_steps)

        print(_step_header(sid, step_def, i, total), flush=True)

        cmd = _build_command(
            step_def,
            config_path=effective_config_path,
            verbose=args.verbose,
            dry_run=False,
            overrides=overrides,
        )

        log.info("Running: %s", " ".join(cmd))

        # Mark as running.
        _mark_step(state, sid, "running", started_at=_iso_now())

        step_start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                check=False,
            )
            step_duration = time.monotonic() - step_start

            if result.returncode == 0:
                _mark_step(
                    state,
                    sid,
                    "done",
                    completed_at=_iso_now(),
                    duration_seconds=step_duration,
                )
                completed.append(sid)
                log.info(
                    "✓ Step %s completed in %s",
                    sid,
                    _format_duration(step_duration),
                )
            else:
                _mark_step(
                    state,
                    sid,
                    "failed",
                    completed_at=_iso_now(),
                    duration_seconds=step_duration,
                    error=f"Exit code {result.returncode}",
                )
                failed_step = sid
                log.error(
                    "✗ Step %s failed (exit code %d) after %s",
                    sid,
                    result.returncode,
                    _format_duration(step_duration),
                )
                hint = _STEP_ERROR_HINTS.get(sid, "")
                if hint:
                    log.error("  Suggestions:\n  %s", hint)
                break

        except OSError as exc:
            step_duration = time.monotonic() - step_start
            _mark_step(
                state,
                sid,
                "failed",
                completed_at=_iso_now(),
                duration_seconds=step_duration,
                error=str(exc),
            )
            failed_step = sid
            log.error("✗ Step %s could not be started: %s", sid, exc)
            break

    # Final summary.
    pipeline_duration = time.monotonic() - pipeline_start
    print(f"\n{'=' * 70}", flush=True)

    if failed_step:
        log.info(
            "Pipeline stopped at step %s (%s).",
            failed_step,
            STEPS[failed_step].name,
        )
        if completed:
            log.info(
                "Completed before failure: %s",
                ", ".join(f"{s} ({STEPS[s].name})" for s in completed),
            )
        remaining = requested_steps[requested_steps.index(failed_step) + 1 :]
        if remaining:
            log.info(
                "Skipped due to failure: %s",
                ", ".join(f"{s} ({STEPS[s].name})" for s in remaining),
            )
        log.info(
            "Fix the issue and re-run with --resume to continue "
            "from step %s.",
            failed_step,
        )
    else:
        log.info("All %d steps completed successfully!", len(completed))
        for sid in completed:
            dur = state["steps"].get(sid, {}).get("duration_seconds")
            dur_str = f" ({_format_duration(dur)})" if dur else ""
            log.info("  ✓ Step %s — %s%s", sid, STEPS[sid].name, dur_str)

        # Remind about step 10 if it wasn't run.
        if "10" not in completed and args.run_all:
            log.info(
                "\n  Note: Step 10 (Upload to HF Hub) was not included.\n"
                "  To upload, run: python scripts/run_pipeline.py --steps 10"
            )

    log.info("Total time: %s", _format_duration(pipeline_duration))
    print(f"{'=' * 70}\n", flush=True)

    _cleanup_temp_config(temp_config_path)

    return 1 if failed_step else 0


def _cleanup_temp_config(temp_config_path: Path | None) -> None:
    """Remove the temporary config file if one was created."""
    if temp_config_path is not None:
        try:
            temp_config_path.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
