# Aawaaz Fine-Tuning Pipeline

Fine-tune Qwen3 models (0.6B and 1.7B) for speech transcript cleanup, producing 4-bit quantized MLX models for on-device inference in [aawaaz](https://github.com/shantanugoel/aawaaz).

## Quickstart

### 1. Setup

```bash
# Clone and enter the project
git clone <this-repo>
cd aawaaz-finetune

# Run setup (auto-detects platform, creates venv, installs deps)
chmod +x scripts/01_setup.sh
./scripts/01_setup.sh            # or: ./scripts/01_setup.sh --platform mac

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Configure

Edit `config.yaml` to set:
- `platform`: `"linux"` (Unsloth/CUDA) or `"mac"` (MLX)
- `hf_username`: your HuggingFace username
- `dataset.synthetic.api_key_env`: name of the env var holding your API key
- Model toggles, hyperparameters, etc.

### 3. Run the pipeline

```bash
# Run everything end-to-end
python scripts/run_pipeline.py --all

# Or run specific steps
python scripts/run_pipeline.py --steps 2,3,3b,4       # Data prep only
python scripts/run_pipeline.py --steps 6               # Fine-tuning only
python scripts/run_pipeline.py --steps 7,8,9           # Fuse → quantize → eval

# Override config values
python scripts/run_pipeline.py --steps 6 --model qwen3-0.6b
python scripts/run_pipeline.py --steps 3 --synthetic-samples 10000

# Dry run
python scripts/run_pipeline.py --all --dry-run
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 2 | `02_pull_datasets.py` | Download existing datasets from HuggingFace |
| 3 | `03_generate_synthetic.py` | Generate synthetic training data via LLM API |
| 3b | `03b_validate_synthetic.py` | LLM-as-judge quality validation |
| 4 | `04_prepare_data.py` | Combine, format, and split into train/valid/test |
| 5 | `05_download_models.py` | Download base Qwen3 models |
| 6 | `06_finetune.py` | LoRA fine-tuning (MLX or Unsloth/HF) |
| 7 | `07_fuse_and_convert.py` | Fuse LoRA adapters into base model |
| 8 | `08_quantize.py` | Quantize to 4-bit MLX format |
| 9 | `09_evaluate.py` | Evaluate on test set |
| 10 | `10_upload.py` | Upload to HuggingFace Hub |

## Project Structure

```
aawaaz-finetune/
├── config.yaml                  # Central configuration
├── prompts/system_prompt.txt    # System prompt for training & inference
├── scripts/                     # All pipeline scripts
├── data/                        # Raw, synthetic, and prepared datasets
├── models/                      # Base, adapter, fused, and quantized models
└── eval_results/                # Evaluation outputs
```

## Platforms

- **Linux (Unsloth/CUDA)**: Faster fine-tuning with NVIDIA GPU. Uses Unsloth + HF Transformers.
- **Mac (MLX)**: Fine-tuning on Apple Silicon. Uses MLX-LM.

Both produce the same final output: a 4-bit quantized MLX model.

## Details

See [`docs/aawaaz-finetune-spec.md`](docs/aawaaz-finetune-spec.md) for the full specification.
