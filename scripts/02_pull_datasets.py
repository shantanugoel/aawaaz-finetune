#!/usr/bin/env python3
"""Download and normalize upstream transcript-cleanup datasets."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable

from common import DATA_RAW, HFDatasetConfig, base_arg_parser, ensure_dirs, load_config, setup_logging


WHISPER_TRANSCRIPTS_REPO = "bingbangboom/whisper-transcripts"
TRANSCRIPTION_CLEANUP_REPO = "danielrosehill/Transcription-Cleanup-Trainer"

OUTPUT_FILE_NAMES: dict[str, str] = {
    WHISPER_TRANSCRIPTS_REPO: "whisper_transcripts.jsonl",
    TRANSCRIPTION_CLEANUP_REPO: "transcription_cleanup_trainer.jsonl",
}
DEPENDENCY_INSTALL_HINT = (
    "run scripts/01_setup.sh or install the appropriate requirements file "
    "for your platform"
)


@dataclass(frozen=True)
class TrainingPair:
    """Normalized training pair written to the raw JSONL output."""

    input: str
    output: str


@dataclass(frozen=True)
class PullResult:
    """Summary for a single upstream source."""

    repo: str
    output_path: Path
    pair_count: int | None
    skipped_existing: bool = False


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the dataset pull step."""
    parser = base_arg_parser(
        "Pull existing datasets and normalize them into data/raw/*.jsonl."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    """Return a clean UTF-8-safe string representation for dataset text."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def slugify_repo_name(repo: str) -> str:
    """Convert an upstream repo id into a filesystem-safe stem."""
    stem = repo.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()


def output_path_for_repo(repo: str) -> Path:
    """Return the JSONL output path for an upstream repo."""
    filename = OUTPUT_FILE_NAMES.get(repo, f"{slugify_repo_name(repo)}.jsonl")
    return DATA_RAW / filename


def count_jsonl_records(path: Path) -> int:
    """Count JSONL records in an existing output file."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_jsonl(path: Path, pairs: Iterable[TrainingPair]) -> int:
    """Write normalized pairs to JSONL atomically and return the row count."""
    count = 0
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
        ) as handle:
            temp_path = Path(handle.name)
            for pair in pairs:
                handle.write(
                    json.dumps(
                        {"input": pair.input, "output": pair.output},
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")
                count += 1

        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return count


def require_dependency(
    import_name: str,
    install_hint: str,
) -> Any:
    """Import a dependency lazily with a helpful failure message."""
    try:
        return __import__(import_name, fromlist=["*"])
    except ImportError as exc:  # pragma: no cover - exercised by runtime environment
        raise RuntimeError(
            f"Missing required dependency '{import_name}'. "
            f"Install project dependencies first ({install_hint})."
        ) from exc


def load_tabular_hf_pairs(
    dataset_cfg: HFDatasetConfig,
    logger: logging.Logger,
) -> list[TrainingPair]:
    """Load a Hugging Face dataset with explicit input/output columns."""
    if not dataset_cfg.input_col or not dataset_cfg.output_col:
        raise ValueError(
            f"Dataset '{dataset_cfg.repo}' requires both input_col and output_col."
        )

    datasets_module = require_dependency("datasets", DEPENDENCY_INSTALL_HINT)
    load_dataset: Callable[..., Any] = datasets_module.load_dataset
    datasets_exceptions = getattr(datasets_module, "exceptions", None)
    handled_errors: list[type[BaseException]] = [ConnectionError, OSError]
    for name in ("DatasetNotFoundError", "DataFilesNotFoundError", "DatasetGenerationError"):
        error_type = getattr(datasets_exceptions, name, None)
        if isinstance(error_type, type) and issubclass(error_type, Exception):
            handled_errors.append(error_type)

    logger.info("Loading Hugging Face dataset: %s", dataset_cfg.repo)
    try:
        loaded = load_dataset(dataset_cfg.repo)
    except tuple(handled_errors) as exc:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_cfg.repo}' from Hugging Face. "
            "Check your network connection, dataset availability, and local auth if needed."
        ) from exc

    if hasattr(loaded, "items"):
        splits = list(loaded.items())
    else:
        splits = [("train", loaded)]

    pairs: list[TrainingPair] = []
    skipped_invalid = 0

    for split_name, split_dataset in splits:
        column_names = set(getattr(split_dataset, "column_names", []))
        missing_columns = {
            dataset_cfg.input_col,
            dataset_cfg.output_col,
        } - column_names
        if missing_columns:
            raise ValueError(
                f"Dataset '{dataset_cfg.repo}' split '{split_name}' is missing "
                f"required columns: {sorted(missing_columns)}. "
                f"Available columns: {sorted(column_names)}"
            )

        for row in split_dataset:
            input_text = normalize_text(row[dataset_cfg.input_col])
            output_text = normalize_text(row[dataset_cfg.output_col])
            if not input_text or not output_text:
                skipped_invalid += 1
                continue
            pairs.append(TrainingPair(input=input_text, output=output_text))

    if skipped_invalid:
        logger.warning(
            "Skipped %s invalid/empty rows while processing %s.",
            skipped_invalid,
            dataset_cfg.repo,
        )

    if not pairs:
        raise ValueError(f"No valid rows were loaded from dataset '{dataset_cfg.repo}'.")

    return pairs


def resolve_cleanup_data_root(snapshot_root: Path) -> Path:
    """Find the directory that contains the paired cleanup source files."""
    candidates = [
        snapshot_root / "data",
        snapshot_root / "dataset" / "data",
    ]
    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "whisper-transcripts").is_dir()
            and (candidate / "manual-cleanups").is_dir()
        ):
            return candidate

    matches = [
        path.parent
        for path in snapshot_root.rglob("whisper-transcripts")
        if path.is_dir() and (path.parent / "manual-cleanups").is_dir()
    ]
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Could not find paired data directories in the "
        f"snapshot at {snapshot_root}. Expected 'data/' or 'dataset/data/' "
        "containing both 'whisper-transcripts/' and 'manual-cleanups/'."
    )


def collect_text_files(root: Path) -> dict[str, Path]:
    """Collect all .txt files under a directory, keyed by relative stem."""
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        key = path.relative_to(root).with_suffix("").as_posix()
        files[key] = path
    return files


def load_cleanup_trainer_pairs(logger: logging.Logger) -> list[TrainingPair]:
    """Download the cleanup-trainer dataset snapshot and pair text files."""
    hub_module = require_dependency(
        "huggingface_hub",
        DEPENDENCY_INSTALL_HINT,
    )
    snapshot_download: Callable[..., str] = hub_module.snapshot_download
    hub_errors = getattr(hub_module, "errors", None)
    handled_errors: list[type[BaseException]] = [ConnectionError, OSError]
    for name in ("HfHubHTTPError", "RepositoryNotFoundError", "LocalEntryNotFoundError"):
        error_type = getattr(hub_errors, name, None)
        if isinstance(error_type, type) and issubclass(error_type, Exception):
            handled_errors.append(error_type)

    logger.info("Downloading dataset snapshot: %s", TRANSCRIPTION_CLEANUP_REPO)
    try:
        snapshot_root = Path(
            snapshot_download(repo_id=TRANSCRIPTION_CLEANUP_REPO, repo_type="dataset")
        )
    except tuple(handled_errors) as exc:
        raise RuntimeError(
            f"Failed to download dataset snapshot '{TRANSCRIPTION_CLEANUP_REPO}'. "
            "Check your network connection and Hugging Face availability."
        ) from exc
    data_root = resolve_cleanup_data_root(snapshot_root)

    transcript_root = data_root / "whisper-transcripts"
    cleanup_root = data_root / "manual-cleanups"
    if not transcript_root.is_dir() or not cleanup_root.is_dir():
        raise FileNotFoundError(
            f"Expected directories '{transcript_root}' and '{cleanup_root}' to exist."
        )

    transcript_files = collect_text_files(transcript_root)
    cleanup_files = collect_text_files(cleanup_root)

    if not transcript_files:
        raise ValueError(f"No transcript files were found in {transcript_root}.")
    if not cleanup_files:
        raise ValueError(f"No cleanup files were found in {cleanup_root}.")

    transcript_keys = set(transcript_files)
    cleanup_keys = set(cleanup_files)
    shared_keys = sorted(transcript_keys & cleanup_keys)
    missing_cleanups = sorted(transcript_keys - cleanup_keys)
    missing_transcripts = sorted(cleanup_keys - transcript_keys)

    if missing_cleanups:
        logger.warning(
            "%s transcript files had no matching manual cleanup. Example keys: %s",
            len(missing_cleanups),
            ", ".join(missing_cleanups[:5]),
        )
    if missing_transcripts:
        logger.warning(
            "%s manual cleanup files had no matching transcript. Example keys: %s",
            len(missing_transcripts),
            ", ".join(missing_transcripts[:5]),
        )
    if not shared_keys:
        raise ValueError(
            "No matching whisper-transcript/manual-cleanup file pairs were found."
        )

    pairs: list[TrainingPair] = []
    skipped_invalid = 0
    for key in shared_keys:
        input_text = normalize_text(transcript_files[key].read_text(encoding="utf-8"))
        output_text = normalize_text(cleanup_files[key].read_text(encoding="utf-8"))
        if not input_text or not output_text:
            skipped_invalid += 1
            continue
        pairs.append(TrainingPair(input=input_text, output=output_text))

    if skipped_invalid:
        logger.warning(
            "Skipped %s empty file pairs while processing %s.",
            skipped_invalid,
            TRANSCRIPTION_CLEANUP_REPO,
        )
    if not pairs:
        raise ValueError(
            f"No valid paired records were found in {TRANSCRIPTION_CLEANUP_REPO}."
        )

    return pairs


def pull_source(
    dataset_cfg: HFDatasetConfig,
    force: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> PullResult:
    """Pull one configured source and normalize it into JSONL."""
    output_path = output_path_for_repo(dataset_cfg.repo)
    if output_path.exists() and not force:
        existing_count = count_jsonl_records(output_path)
        logger.info(
            "Skipping %s because %s already exists with %s rows. Use --force to overwrite.",
            dataset_cfg.repo,
            output_path,
            existing_count,
        )
        return PullResult(
            repo=dataset_cfg.repo,
            output_path=output_path,
            pair_count=existing_count,
            skipped_existing=True,
        )

    if dry_run:
        action = "overwrite" if output_path.exists() else "create"
        logger.info(
            "[dry-run] Would %s %s from source %s.",
            action,
            output_path,
            dataset_cfg.repo,
        )
        return PullResult(repo=dataset_cfg.repo, output_path=output_path, pair_count=None)

    if dataset_cfg.repo == TRANSCRIPTION_CLEANUP_REPO:
        pairs = load_cleanup_trainer_pairs(logger)
    else:
        pairs = load_tabular_hf_pairs(dataset_cfg, logger)

    row_count = write_jsonl(output_path, pairs)
    logger.info(
        "Saved %s normalized pairs from %s to %s.",
        row_count,
        dataset_cfg.repo,
        output_path,
    )
    return PullResult(repo=dataset_cfg.repo, output_path=output_path, pair_count=row_count)


def configured_sources(config_path: Path) -> list[HFDatasetConfig]:
    """Load config and return enabled dataset sources."""
    config = load_config(config_path)
    return [dataset for dataset in config.dataset.hf_datasets if dataset.enabled]


def log_summary(results: list[PullResult], dry_run: bool, logger: logging.Logger) -> None:
    """Log final per-source and total stats."""
    if not results:
        logger.warning("No dataset sources were processed.")
        return

    mode_label = "planned" if dry_run else "pulled"
    logger.info("Dataset pull summary (%s):", mode_label)
    for result in results:
        status = "skipped existing" if result.skipped_existing else mode_label
        pair_count = (
            str(result.pair_count)
            if result.pair_count is not None
            else "unknown until executed"
        )
        logger.info(
            "  - %s: %s pairs (%s) -> %s",
            result.repo,
            pair_count,
            status,
            result.output_path,
        )
    known_counts = [result.pair_count for result in results if result.pair_count is not None]
    if len(known_counts) == len(results):
        logger.info(
            "Total pairs %s across all sources: %s",
            mode_label,
            sum(known_counts),
        )
    else:
        logger.info(
            "Total pairs %s across all sources: unknown until execution",
            mode_label,
        )


def main() -> int:
    """Run the dataset pull workflow."""
    args = parse_args()
    logger = setup_logging(args.verbose)

    try:
        if not args.dry_run:
            ensure_dirs()
        sources = configured_sources(args.config)
        if not sources:
            logger.warning("No enabled datasets found in config. Nothing to do.")
            return 0

        results: list[PullResult] = []
        for dataset_cfg in sources:
            results.append(
                pull_source(
                    dataset_cfg=dataset_cfg,
                    force=args.force,
                    dry_run=args.dry_run,
                    logger=logger,
                )
            )

        log_summary(results, args.dry_run, logger)
        return 0
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        logger.error("Dataset pull failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
