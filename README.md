# Aawaaz Fine-Tuning Pipeline

Fine-tune Qwen3 models (0.6B and 1.7B) for speech transcript cleanup, producing 4-bit quantized MLX models for on-device inference in [aawaaz](https://github.com/shantanugoel/aawaaz).

The pipeline cleans messy ASR (automatic speech recognition) output — removing filler words, fixing grammar, correcting self-corrections, and formatting — while preserving the speaker's original meaning.

## Quickstart

### Prerequisites

- **Python 3.11+** and [**uv**](https://docs.astral.sh/uv/) (package manager)
- **Linux**: NVIDIA GPU with CUDA support
- **Mac**: Apple Silicon (M1+)
- **HuggingFace account** (for dataset download and model upload)
- **LLM API key** for synthetic data generation (OpenAI, Anthropic, or any OpenAI-compatible provider like OpenRouter)

### 1. Setup

```bash
git clone <this-repo>
cd aawaaz-finetune

# Run setup — auto-detects platform, creates venv, installs deps
chmod +x scripts/01_setup.sh
./scripts/01_setup.sh              # auto-detect
./scripts/01_setup.sh --platform mac   # or force platform

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Configure

Edit `config.yaml`:

```yaml
platform: "mac"                          # "linux" or "mac"
hf_username: "your-username"             # HuggingFace username

dataset:
  synthetic:
    provider: "openai_compatible"        # "anthropic", "openai", or "openai_compatible"
    model: "google/gemini-2.5-flash"     # Model to use for generation
    api_key_env: "OPENROUTER_API_KEY"    # Env var holding your API key
    base_url: "https://openrouter.ai/api/v1"  # Only for openai_compatible
```

Set the API key in your environment:
```bash
export OPENROUTER_API_KEY="sk-..."       # Or whichever provider you chose
```

See `config.yaml` for all available options (model toggles, hyperparameters, training settings, etc.).

### 3. Run the Pipeline

```bash
# Run everything end-to-end (steps 2–9, excludes upload)
python scripts/run_pipeline.py --all

# Run specific steps
python scripts/run_pipeline.py --steps 2,3,3b,4       # Data prep only
python scripts/run_pipeline.py --steps 6               # Fine-tuning only
python scripts/run_pipeline.py --steps 7,8,9           # Fuse → quantize → eval

# Target a specific model
python scripts/run_pipeline.py --steps 6 --model qwen3-0.6b

# Override config values
python scripts/run_pipeline.py --steps 3 --synthetic-samples 10000

# Preview what would run (no side effects)
python scripts/run_pipeline.py --all --dry-run

# Resume after interruption (skips completed steps)
python scripts/run_pipeline.py --all --resume

# Upload to HuggingFace (must be run explicitly)
python scripts/run_pipeline.py --steps 10
```

### 4. Run Scripts Individually

Every script can also be run standalone:

```bash
python scripts/02_pull_datasets.py --verbose
python scripts/03_generate_synthetic.py --dry-run --synthetic-samples 100
python scripts/06_finetune.py --model qwen3-0.6b --verbose
```

All scripts support `--verbose`, `--dry-run`, and `--config <path>` flags.

## Pipeline Steps

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `01_setup.sh` | Create venv, install dependencies (run manually) |
| 2 | `02_pull_datasets.py` | Download transcript cleanup datasets from HuggingFace |
| 3 | `03_generate_synthetic.py` | Generate synthetic training pairs via LLM API |
| 3b | `03b_validate_synthetic.py` | LLM-as-judge quality gate on synthetic data |
| 4 | `04_prepare_data.py` | Combine (raw + synthetic + prepared), deduplicate, format, and split into train/valid/test |
| 5 | `05_download_models.py` | Download base Qwen3 models from HuggingFace |
| 6 | `06_finetune.py` | LoRA fine-tuning (Unsloth on Linux, MLX on Mac) |
| 7 | `07_fuse_and_convert.py` | Fuse LoRA adapters into base model |
| 8 | `08_quantize.py` | Quantize to 4-bit MLX format |
| 9 | `09_evaluate.py` | Evaluate on test set (CER, BLEU, exact match, format accuracy) |
| 10 | `10_upload.py` | Upload quantized model to HuggingFace Hub |

## Project Structure

```
aawaaz-finetune/
├── config.yaml                  # All pipeline configuration
├── prompts/system_prompt.txt    # System prompt used in training & inference
├── scripts/
│   ├── 01_setup.sh              # Environment setup
│   ├── 02–10_*.py               # Pipeline step scripts
│   ├── run_pipeline.py          # Orchestrator
│   ├── common.py                # Shared utilities (config, logging, CLI)
│   └── llm_client.py            # Multi-provider LLM client
├── data/
│   ├── raw/                     # Downloaded HF datasets
│   ├── synthetic/               # Generated + validated synthetic data (script-based)
│   ├── prepared/                # Agent-generated, pre-validated training pairs
│   └── combined/                # Final train/valid/test splits
├── models/
│   ├── base/                    # Downloaded base models
│   ├── adapters/                # LoRA adapters from fine-tuning
│   ├── fused/                   # Merged full models (Mac)
│   ├── mlx/                     # Converted MLX models (Linux→Mac)
│   └── quantized/               # Final 4-bit quantized models
├── eval_results/                # Evaluation reports and metrics
└── docs/
    └── aawaaz-finetune-spec.md  # Full specification
```

## Platforms

| | Linux | Mac |
|---|---|---|
| **Fine-tuning** | Unsloth + HuggingFace Transformers (CUDA) | MLX-LM LoRA |
| **Quantization** | — (transfer to Mac) | MLX-LM convert |
| **Final output** | 4-bit quantized MLX model | 4-bit quantized MLX model |

**Cross-platform workflow**: Fine-tune on Linux (faster with NVIDIA GPU), then transfer the fused model to Mac for quantization and evaluation. Use `--convert-only` in step 7 on Mac to convert a Linux-fused model to MLX format.

## Uploading to HuggingFace

Dataset and model are uploaded to separate HuggingFace repos.

### Upload training dataset

Upload prepared training data **before** fine-tuning so it's versioned and preserved regardless of training outcome.

```bash
# Login (one-time)
hf auth login

# Upload prepared data as a dataset repo
hf upload <hf_username>/aawaaz-transcript-cleanup data/prepared/ \
  --type dataset --exclude ".gitkeep"

# To update after adding more data, just re-run the same command
```

Once uploaded, others can download it by adding to `config.yaml`:
```yaml
dataset:
  hf_datasets:
    - repo: "<hf_username>/aawaaz-transcript-cleanup"
      input_col: "input"
      output_col: "output"
      enabled: true
```

### Upload fine-tuned model

After training, fusing, quantizing, and evaluating (steps 6–9):

```bash
# Preview what would be uploaded
python scripts/10_upload.py --model qwen3-0.6b --dry-run

# Upload quantized model with auto-generated model card
python scripts/10_upload.py --model qwen3-0.6b --verbose
```

This creates a repo like `<hf_username>/aawaaz-qwen3-0.6b-transcriber-4bit` with the quantized weights, model card, eval metrics, and system prompt.

## Key Design Decisions

- **Qwen3 thinking mode disabled** — All scripts pass `enable_thinking=False` and the training data includes `/no_think` in the system prompt. Any `<think>` tags in output are stripped and logged as warnings.
- **Prompt masking** — Loss is computed only on the assistant's response, not the system prompt or user input.
- **Idempotent** — Re-running any script skips already-completed work. Use `--force` to override.
- **Incremental saves** — Long-running steps (synthetic generation, training) save progress so crashes don't lose work.

## Details

See [`docs/aawaaz-finetune-spec.md`](docs/aawaaz-finetune-spec.md) for the full specification and [`docs/plan.md`](docs/plan.md) for implementation notes.
