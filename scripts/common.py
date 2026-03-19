"""Shared helpers for the aawaaz fine-tuning pipeline.

Provides:
- Config loading and validation
- Model config resolution (--model flag handling)
- Standard argparse base with --verbose, --dry-run, --config
- Logging setup
- Path constants
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── Path constants ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_SYNTHETIC = PROJECT_ROOT / "data" / "synthetic"
DATA_SYNTHETIC_REJECTED = DATA_SYNTHETIC / "rejected"
DATA_PREPARED = PROJECT_ROOT / "data" / "prepared"
DATA_COMBINED = PROJECT_ROOT / "data" / "combined"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"

MODELS_BASE = PROJECT_ROOT / "models" / "base"
MODELS_ADAPTERS = PROJECT_ROOT / "models" / "adapters"
MODELS_FUSED = PROJECT_ROOT / "models" / "fused"
MODELS_MLX = PROJECT_ROOT / "models" / "mlx"
MODELS_QUANTIZED = PROJECT_ROOT / "models" / "quantized"

EVAL_RESULTS = PROJECT_ROOT / "eval_results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PIPELINE_STATE_PATH = PROJECT_ROOT / ".pipeline_state.json"

ALL_DATA_DIRS = [
    DATA_RAW,
    DATA_SYNTHETIC,
    DATA_SYNTHETIC_REJECTED,
    DATA_PREPARED,
    DATA_COMBINED,
    DATA_EVAL,
]
ALL_MODEL_DIRS = [
    MODELS_BASE,
    MODELS_ADAPTERS,
    MODELS_FUSED,
    MODELS_MLX,
    MODELS_QUANTIZED,
]
ALL_OUTPUT_DIRS = ALL_DATA_DIRS + ALL_MODEL_DIRS + [EVAL_RESULTS]


# ── Data classes for typed config access ────────────────────────────────────


@dataclass
class ModelConfig:
    name: str
    base_model: str
    unsloth_model: str
    enabled: bool = True


@dataclass
class HFDatasetConfig:
    repo: str
    enabled: bool = True
    input_col: str | None = None
    output_col: str | None = None


@dataclass
class ValidationConfig:
    enabled: bool = True
    sample_rate: float = 0.12
    pass_threshold: float = 0.90
    reject_threshold: float = 0.70
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str | None = None


@dataclass
class GenerationModesConfig:
    generate_both: bool = True
    clean_to_messy: bool = True


@dataclass
class CleanTextSourceConfig:
    dataset: str = "wikimedia/wikipedia"
    subset: str = "20231101.en"
    text_column: str = "text"
    max_samples: int = 2500
    min_text_length: int = 100
    max_text_length: int = 2000


@dataclass
class SyntheticConfig:
    enabled: bool = True
    num_samples: int = 5000
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str | None = None
    batch_size: int = 25
    categories: dict[str, float] = field(default_factory=dict)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    generation_modes: GenerationModesConfig = field(
        default_factory=GenerationModesConfig
    )
    clean_text_source: CleanTextSourceConfig = field(
        default_factory=CleanTextSourceConfig
    )


@dataclass
class DatasetConfig:
    hf_datasets: list[HFDatasetConfig] = field(default_factory=list)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    train_ratio: float = 0.90
    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    shuffle_seed: int = 42


@dataclass
class LoRAConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class LinuxTrainingConfig:
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 50
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"
    bf16: bool = True
    load_in_4bit: bool = False


@dataclass
class MacTrainingConfig:
    learning_rate: float = 1e-5
    batch_size: int = 4
    lora_layers: int = 16
    iters: int = 1500
    grad_accumulation_steps: int = 4
    grad_checkpoint: bool = True


@dataclass
class TrainingConfig:
    max_seq_length: int = 2048
    mask_prompt: bool = True
    num_epochs: int = 3
    save_every: int = 200
    eval_every: int = 100
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    linux: LinuxTrainingConfig = field(default_factory=LinuxTrainingConfig)
    mac: MacTrainingConfig = field(default_factory=MacTrainingConfig)


@dataclass
class QuantizationConfig:
    bits: int = 4
    group_size: int = 64


@dataclass
class EvaluationConfig:
    num_samples: int = 200
    metrics: list[str] = field(
        default_factory=lambda: [
            "exact_match",
            "char_error_rate",
            "bleu",
            "format_accuracy",
        ]
    )
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class UploadConfig:
    enabled: bool = False
    repo_prefix: str = "aawaaz"
    private: bool = False


@dataclass
class PipelineConfig:
    project_name: str = "aawaaz-transcriber"
    hf_username: str = "shantanugoel"
    models: list[ModelConfig] = field(default_factory=list)
    platform: str = "linux"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)


# ── Config loading ──────────────────────────────────────────────────────────


def _build_nested(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, ignoring unknown keys.

    Nested dataclass fields that are already constructed (not raw dicts) are
    passed through unchanged.  Raw dicts are recursively converted using the
    field's type annotation.
    """
    if not isinstance(data, dict):
        return data
    import dataclasses
    import typing

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}

    # Resolve stringified annotations (from `from __future__ import annotations`)
    hints = typing.get_type_hints(cls)

    for f in dataclasses.fields(cls):
        resolved_type = hints.get(f.name, f.type)
        if (
            f.name in filtered
            and isinstance(filtered[f.name], dict)
            and isinstance(resolved_type, type)
            and dataclasses.is_dataclass(resolved_type)
        ):
            filtered[f.name] = _build_nested(resolved_type, filtered[f.name])
    return cls(**filtered)


def load_config(path: Path | str | None = None) -> PipelineConfig:
    """Load and validate ``config.yaml``, returning a typed ``PipelineConfig``.

    Parameters
    ----------
    path:
        Explicit path to the YAML config file. Falls back to
        ``CONFIG_PATH`` (project root) when *None*.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required fields are missing or invalid.
    """
    config_path = Path(path) if path else CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Build model list
    models = [
        _build_nested(ModelConfig, m) for m in raw.get("models", [])
    ]

    # Build dataset config
    ds_raw = raw.get("dataset", {})
    hf_datasets = [
        _build_nested(HFDatasetConfig, d) for d in ds_raw.get("hf_datasets", [])
    ]
    synth_raw = dict(ds_raw.get("synthetic", {}))  # copy to avoid mutating raw
    val_raw = synth_raw.pop("validation", {})
    validation = _build_nested(ValidationConfig, val_raw)
    gen_modes_raw = synth_raw.pop("generation_modes", {})
    gen_modes = _build_nested(GenerationModesConfig, gen_modes_raw)
    clean_src_raw = synth_raw.pop("clean_text_source", {})
    clean_src = _build_nested(CleanTextSourceConfig, clean_src_raw)
    synthetic = _build_nested(
        SyntheticConfig,
        {
            **synth_raw,
            "validation": validation,
            "generation_modes": gen_modes,
            "clean_text_source": clean_src,
        },
    )
    dataset = _build_nested(
        DatasetConfig,
        {**ds_raw, "hf_datasets": hf_datasets, "synthetic": synthetic},
    )

    # Build training config
    tr_raw = raw.get("training", {})
    lora = _build_nested(LoRAConfig, tr_raw.get("lora", {}))
    linux = _build_nested(LinuxTrainingConfig, tr_raw.get("linux", {}))
    mac = _build_nested(MacTrainingConfig, tr_raw.get("mac", {}))
    training = _build_nested(
        TrainingConfig, {**tr_raw, "lora": lora, "linux": linux, "mac": mac}
    )

    # Build remaining configs
    quantization = _build_nested(QuantizationConfig, raw.get("quantization", {}))
    evaluation = _build_nested(EvaluationConfig, raw.get("evaluation", {}))
    upload = _build_nested(UploadConfig, raw.get("upload", {}))

    cfg = PipelineConfig(
        project_name=raw.get("project_name", "aawaaz-transcriber"),
        hf_username=raw.get("hf_username", "shantanugoel"),
        models=models,
        platform=raw.get("platform", "linux"),
        dataset=dataset,
        training=training,
        quantization=quantization,
        evaluation=evaluation,
        upload=upload,
    )

    _validate_config(cfg)
    return cfg


def _validate_provider(provider: str, label: str, base_url: str | None) -> None:
    """Validate a provider field value."""
    valid = ("anthropic", "openai", "openai_compatible")
    if provider not in valid:
        raise ValueError(
            f"Invalid {label} provider '{provider}'. Must be one of {valid}."
        )
    if provider == "openai_compatible" and not base_url:
        raise ValueError(
            f"{label} provider is 'openai_compatible' but 'base_url' is not set."
        )


def _validate_config(cfg: PipelineConfig) -> None:
    """Raise ``ValueError`` on invalid config values."""
    if cfg.platform not in ("linux", "mac"):
        raise ValueError(
            f"Invalid platform '{cfg.platform}'. Must be 'linux' or 'mac'."
        )

    if not cfg.models:
        raise ValueError("At least one model must be configured.")

    enabled_models = [m for m in cfg.models if m.enabled]
    if not enabled_models:
        raise ValueError("At least one model must be enabled.")

    for m in cfg.models:
        if not m.name or not m.base_model:
            raise ValueError(f"Model entry missing 'name' or 'base_model': {m}")

    ds = cfg.dataset
    total = ds.train_ratio + ds.valid_ratio + ds.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.4f} "
            f"({ds.train_ratio} + {ds.valid_ratio} + {ds.test_ratio})"
        )

    synth = ds.synthetic
    if synth.enabled:
        if synth.num_samples <= 0:
            raise ValueError("synthetic.num_samples must be positive.")
        if synth.batch_size <= 0:
            raise ValueError("synthetic.batch_size must be positive.")
        if not synth.categories:
            raise ValueError(
                "synthetic.categories must be non-empty when synthetic is enabled."
            )
        cat_sum = sum(synth.categories.values())
        if abs(cat_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Synthetic category proportions must sum to 1.0, got {cat_sum:.4f}"
            )
        _validate_provider(
            synth.provider, "synthetic", synth.base_url
        )

        val = synth.validation
        if val.enabled:
            if not (0 < val.sample_rate <= 1.0):
                raise ValueError("validation.sample_rate must be in (0, 1].")
            if not (0 <= val.reject_threshold <= val.pass_threshold <= 1.0):
                raise ValueError(
                    "validation thresholds must satisfy: "
                    "0 <= reject_threshold <= pass_threshold <= 1."
                )
            _validate_provider(
                val.provider, "validation", val.base_url
            )

    tr = cfg.training
    if tr.max_seq_length <= 0:
        raise ValueError("training.max_seq_length must be positive.")
    if tr.lora.rank <= 0 or tr.lora.alpha <= 0:
        raise ValueError("training.lora.rank and alpha must be positive.")

    q = cfg.quantization
    if q.bits <= 0 or q.group_size <= 0:
        raise ValueError("quantization.bits and group_size must be positive.")


# ── Model resolution ───────────────────────────────────────────────────────


def resolve_models(
    cfg: PipelineConfig, model_filter: str | None = None
) -> list[ModelConfig]:
    """Return the list of enabled models, optionally filtered by ``--model``.

    Parameters
    ----------
    cfg:
        The loaded pipeline config.
    model_filter:
        ``None`` → all enabled models. A model name like ``"qwen3-0.6b"``
        → just that one. ``"all"`` → all enabled models (same as *None*).

    Raises
    ------
    ValueError
        If the requested model name is not found in the config.
    """
    enabled = [m for m in cfg.models if m.enabled]

    if model_filter is None or model_filter == "all":
        if not enabled:
            raise ValueError("No enabled models configured.")
        return enabled

    matches = [m for m in enabled if m.name == model_filter]
    if not matches:
        available = ", ".join(m.name for m in enabled)
        raise ValueError(
            f"Model '{model_filter}' not found. Available: {available}"
        )
    return matches


# ── Argparse base ──────────────────────────────────────────────────────────


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    """Create an ``ArgumentParser`` with the standard flags every script shares.

    Includes: ``--config``, ``--verbose``, ``--dry-run``.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Path to config.yaml (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes.",
    )
    return parser


def add_model_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--model`` argument to *parser*."""
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to operate on (e.g. 'qwen3-0.6b'). "
        "Use 'all' or omit for all enabled models.",
    )


# ── Logging setup ──────────────────────────────────────────────────────────


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure the root logger and return the ``aawaaz`` logger.

    Logs go to *stderr* so stdout remains available for data output.
    When *verbose* is True, only the ``aawaaz`` hierarchy is set to DEBUG;
    third-party libraries (httpx, openai, urllib3, etc.) stay at WARNING
    to avoid noisy HTTP-level debug output.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    app_logger = logging.getLogger("aawaaz")
    if verbose:
        app_logger.setLevel(logging.DEBUG)

    # Silence chatty third-party loggers even in verbose mode
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return app_logger


# ── System prompt helper ───────────────────────────────────────────────────


def load_system_prompt(with_no_think: bool = False) -> str:
    """Load the system prompt from ``prompts/system_prompt.txt``.

    Parameters
    ----------
    with_no_think:
        If *True*, append ``\\n/no_think`` to the prompt text.  Used when
        building training data (plan decision (b) — keep the source file
        clean, append at data-prep time).
    """
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"System prompt not found: {SYSTEM_PROMPT_PATH}"
        )
    text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if with_no_think:
        text += "\n/no_think"
    return text


# ── Directory helpers ──────────────────────────────────────────────────────


def ensure_dirs() -> None:
    """Create all output directories if they don't already exist."""
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
