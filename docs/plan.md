# Aawaaz Fine-Tuning Pipeline — Implementation Plan

> **Spec:** `docs/aawaaz-finetune-spec.md`

---

## Work Units

### WU-0: Project Scaffolding & Shared Utilities ✅
**Spec sections:** Project Structure, `config.yaml`, `prompts/system_prompt.txt`, Dependencies  
**Deliverables:**
- [x] `config.yaml` (verbatim from spec)
- [x] `prompts/system_prompt.txt` (verbatim from spec, with `/no_think` appended)
- [x] `requirements-linux.txt` / `requirements-mac.txt`
- [x] `scripts/__init__.py` (empty, so scripts can share modules)
- [x] `scripts/common.py` — shared helpers used by every script:
  - Load & validate `config.yaml` (returns typed dict/dataclass)
  - Resolve model configs (filter by `--model` flag, handle `--model all`)
  - Standard `argparse` base with `--verbose`, `--dry-run`, `--config`
  - Logging setup (to stderr, `logging` module)
  - Path constants (`DATA_RAW`, `DATA_SYNTHETIC`, etc.)
- [x] Create empty directory structure (`data/raw/`, `data/synthetic/rejected/`, `data/combined/`, `data/eval/`, `models/base/`, `models/adapters/`, `models/fused/`, `models/mlx/`, `models/quantized/`, `eval_results/`)
- [x] `.gitignore` (ignore `models/`, `data/`, `.pipeline_state.json`, `__pycache__`, `.venv`)
- [x] `01_setup.sh` — platform-detect, create venv with `uv`, install from correct requirements file (run manually, NOT called by orchestrator)
- [x] `README.md` — concise: what the project is, quickstart (setup → run pipeline), link to spec for details

**Dependencies:** None  
**Open questions / flags:** None

---

### WU-1: `02_pull_datasets.py` ✅
**Spec sections:** `02_pull_datasets.py`  
**Deliverables:**
- [x] Download `bingbangboom/whisper-transcripts` via HF `datasets` lib, map `Transcript` → `input`, `Output` → `output`
- [x] Download `danielrosehill/Transcription-Cleanup-Trainer` — clone/download repo, pair files from `whisper-transcripts/` with `manual-cleanups/` by filename
- [x] Save each source as `data/raw/{source_name}.jsonl` with schema `{"input": "...", "output": "..."}`
- [x] Skip re-download if output file exists (idempotent), support `--force`
- [x] Log stats: pairs per source, total

**Dependencies:** WU-0  
**Open questions / flags:**
1. **`danielrosehill/Transcription-Cleanup-Trainer` structure:** The spec says it has `audio/`, `whisper-transcripts/`, `auto-cleanup/`, and `manual-cleanups/` directories. I'll need to inspect the actual repo to confirm the directory names and file-matching strategy (likely by filename stem). If the structure differs, I'll adapt and document.
2. **`bingbangboom/whisper-transcripts` column names:** The config specifies `input_col: "Transcript"` and `output_col: "Output"`. I'll verify these exist when the dataset is loaded and fail clearly if they don't.

---

### WU-2: `03_generate_synthetic.py` ✅
**Spec sections:** `03_generate_synthetic.py` (lines 246–314)  
**Deliverables:**
- [x] **Multi-provider LLM client** (shared with WU-3, lives in `scripts/llm_client.py`):
  - `anthropic` — native Anthropic SDK (with system message extraction)
  - `openai` — native OpenAI SDK
  - `openai_compatible` — OpenAI SDK with custom `base_url` (covers OpenRouter, Gemini, local servers, any OpenAI-compatible endpoint)
  - Config fields: `provider`, `model`, `api_key_env`, and optional `base_url` (only needed for `openai_compatible`)
  - Uniform interface: `generate(messages, **kwargs) → str` so all scripts are provider-agnostic
- [x] Generation prompt exactly as spec, with per-category guidance
- [x] Batch generation: send `batch_size` pairs per API call, parse JSON response
- [x] Exponential backoff retry (3 retries) on API errors
- [x] Incremental save: per-category JSONL (`data/synthetic/synthetic_{category}.jsonl`)
- [x] Resume support: on restart, count existing valid JSONL records per category, skip already-done batches
- [x] Combined output: `data/synthetic/all_synthetic.jsonl`
- [x] Progress bar with ETA (tqdm)
- [x] Validate each pair: both fields non-empty, output ≤ ~2× input length, no null bytes, no filler words in output
- [x] `--dry-run`: print plan (samples per category, estimated API calls & cost) without calling API or creating directories
- [x] `--synthetic-samples N`: override `num_samples` from config
- [x] Log: total generated, failed, rejected, per-category counts
- [x] Target-based generation loop (while current < target) with consecutive-failure safety valve
- [x] Inter-batch delay for rate limiting
- [x] Clean error handling with exit codes

**Dependencies:** WU-0  
**Open questions / flags:**
1. **Category proportions don't sum to 1.0.** The 12 categories in the spec sum to **1.00** — I verified: 0.15+0.12+0.12+0.08+0.08+0.10+0.05+0.08+0.05+0.07+0.03+0.07 = 1.00. ✓ No issue.
2. **Missing categories in guidance.** The spec provides category-specific guidance for 7 of 12 categories (casual_conversation, email_professional, technical_code, medical_clinical, legal_contract, meeting_notes, self_corrections_heavy). Missing: recipe_cooking, academic_research, creative_writing, financial_business, shopping_lists. **I'll write reasonable guidance for these 5 based on their names.** Flag if you'd prefer to provide them.
3. **API cost estimation in `--dry-run`.** I'll estimate based on ~1500 input tokens + ~2000 output tokens per batch call for Anthropic. These are rough — want me to calibrate differently?

---

### WU-3: `03b_validate_synthetic.py` ✅
**Spec sections:** `03b_validate_synthetic.py` — LLM-as-Judge Quality Gate (lines 316–399)  
**Deliverables:**
- [x] Sample `sample_rate` fraction of pairs per category
- [x] Send each sampled pair to LLM judge with the exact judge prompt from spec
- [x] Parse binary pass/fail on 4 criteria: `input_realistic`, `content_preserved`, `no_hallucination`, `corrections_applied`
- [x] Per-category decision: pass (≥90%), warn (70–90%), reject (<70%)
- [x] Rejected categories: move file to `data/synthetic/rejected/`
- [x] Individual pair rejection: remove pairs failing `content_preserved` or `no_hallucination` from accepted categories
- [x] Output: `validation_report.json` (full results), `validation_summary.txt` (human-readable)
- [x] Cost estimation + confirmation prompt (skippable with `--yes`)
- [x] Flags: `--sample-rate`, `--full`, `--category`, `--yes`, `--verbose`
- [x] Fail-closed: parse/API errors count as failures in pass rate denominator
- [x] Atomic file writes (write-to-temp + `os.replace`) to prevent data loss
- [x] Rejected file move guards against overwriting existing files
- [x] Reports written before destructive file mutations
- [x] Per-criterion logging and prompt improvement suggestions for rejected/warned categories
- [x] Warns about unexpected synthetic files on disk not in config

**Dependencies:** WU-2 (needs synthetic data files to exist)  
**Open questions / flags:** None — spec is very detailed here.

---

### WU-4: `04_prepare_data.py` ✅
**Spec sections:** `04_prepare_data.py` (lines 400–424), Data Quality Validation (lines 689–696), Chat Template Handling (lines 676–681)  
**Deliverables:**
- [x] Load all JSONL from `data/raw/` and accepted files from `data/synthetic/` (skip `rejected/`)
- [x] If validation wasn't run (no `validation_report.json`), load all synthetic with warning
- [x] Data quality validation (reject + log):
  - Output > 2× input length
  - Empty output with substantive input
  - Encoding issues (null bytes, mojibake detection)
  - Fillers remaining in output (unless in quoted speech)
  - `<think>` tags in output (Qwen3 thinking mode leak)
  - Flag (don't reject): output adds content not in input
- [x] Convert to chat-messages format with system prompt (+ `/no_think`)
- [x] Deduplicate by normalized input text
- [x] Shuffle (seed from config), split train/valid/test
- [x] Save to `data/combined/{train,valid,test}.jsonl`
- [x] Log stats: total, per-source, split sizes, avg lengths, rejection counts
- [x] Robust JSONL loading: handles non-dict, non-string fields, malformed lines
- [x] Only loads synthetic categories defined in config (skips unexpected files)
- [x] Warns when validation is disabled in config

**Dependencies:** WU-1 and WU-3 (or WU-2 if validation disabled)  
**Open questions / flags:**
1. **`/no_think` placement.** ✅ Resolved: system prompt text kept clean in `prompts/system_prompt.txt`; `04_prepare_data.py` appends `/no_think` via `load_system_prompt(with_no_think=True)`.
2. **Chat template validation round-trip.** ✅ Deferred to WU-6 as planned — requires model/tokenizer from WU-5. The `04_prepare_data.py` output is the canonical `messages` format that both MLX and HF training paths consume.

---

### WU-5: `05_download_models.py` ✅
**Spec sections:** `05_download_models.py` (lines 425–432)  
**Deliverables:**
- [x] Download base HF models to `models/base/{model_name}/`
- [x] On Linux: also download Unsloth variant
- [x] Verify by loading tokenizer + quick test generation
- [x] Log model sizes, parameter counts
- [x] Skip if already downloaded (idempotent)
- [x] `--model` flag to download specific model

**Dependencies:** WU-0  
**Open questions / flags:** None

---

### WU-6: `06_finetune.py` ✅
**Spec sections:** `06_finetune.py` (lines 433–536), mask_prompt (lines 683–687), Qwen3 Thinking Mode (lines 668–675), Model-Specific Considerations (lines 706–709)  
**Deliverables:**
- [x] **Linux/Unsloth path:**
  - Load model with `FastLanguageModel.from_pretrained` (full precision)
  - Apply LoRA via `get_peft_model`
  - Format data with `apply_chat_template(enable_thinking=False)`
  - `SFTTrainer` with all config params + `DataCollatorForCompletionOnlyLM` for prompt masking
  - Save adapters (best model via `load_best_model_at_end=True`)
- [x] **Mac/MLX path:**
  - Shell out to `mlx_lm.lora` with all flags from config
  - Parse and log training output (loss, val loss)
  - Best checkpoint detection and copy to `adapters.safetensors`
- [x] Platform detection from config
- [x] Chat template validation: round-trip a sample through tokenizer before training starts (validates actual model source)
- [x] Save best checkpoint (lowest val loss) — Linux via HF Trainer, Mac via log parsing
- [x] Resume from last checkpoint if interrupted — Linux via `checkpoint-*`, Mac via `adapters-*.safetensors`
- [x] Training summary: final losses, total time, best eval loss
- [x] `--model` flag to train specific model
- [x] `--force` cleans adapter dir before retrain
- [x] `--dry-run` works without heavy ML imports

**Dependencies:** WU-4 (training data), WU-5 (base models)  
**Decisions made:**
1. **Unsloth variant used** — as recommended, Unsloth provides optimized kernels even at full precision.
2. **Prompt masking** — used `DataCollatorForCompletionOnlyLM` with `<|im_start|>assistant\n` response template to mask prompt tokens. Also includes initial eval loss sanity check (warns if < 0.1).
3. **Best checkpoint** — Linux: `load_best_model_at_end=True` + `metric_for_best_model="eval_loss"`. Mac: parse logs for lowest val_loss iteration and copy checkpoint.
4. **Model source resolution** — centralized `_resolve_model_source()` helper used by both validation and training.

---

### WU-7: `07_fuse_and_convert.py` ✅
**Spec sections:** `07_fuse_and_convert.py` (lines 538–566)  
**Deliverables:**
- [x] **Linux path:** `model.save_pretrained_merged(save_method="merged_16bit")`
  - Then `mlx_lm.convert` HF → MLX format (to `models/mlx/{model_name}`) via `--convert-only`
- [x] **Mac path:** `mlx_lm.fuse` (produces MLX format directly to `models/fused/{model_name}`)
- [x] Verify output directory has expected files (config.json, safetensors, tokenizer files)
- [x] Log: input/output paths, model size
- [x] `--convert-only` flag for cross-platform workflow (Linux→Mac)
- [x] `--de-quantize` / `--no-de-quantize` flag (BooleanOptionalAction, default True)
- [x] Preflight checks for required packages (unsloth/mlx_lm)
- [x] Partial output cleanup on failure (prevents false skips on reruns)
- [x] Validates existing output before skipping (not just dir existence)

**Dependencies:** WU-6  
**Open questions / flags:**
1. ~~**Linux→MLX conversion on Mac.** If fine-tuned on Linux, the fused HF model must be transferred to Mac for `mlx_lm.convert`. The script should detect this and either: (a) run the conversion if on Mac, or (b) print instructions to transfer and convert. I'll handle both cases.~~ **Resolved:** Both paths implemented — `--convert-only` for Mac conversion, and instructions printed after Linux fuse.
2. ~~**`--de-quantize` flag on `mlx_lm.fuse`.** This is only needed if the base model was quantized. Since we train at full precision on both platforms, this flag may not be needed (or may be harmless). I'll include it as shown in the spec.~~ **Resolved:** Included with `--no-de-quantize` opt-out.

---

### WU-8: `08_quantize.py` ✅
**Spec sections:** `08_quantize.py` (lines 568–581), Quantization notes (lines 711–715)  
**Deliverables:**
- [x] Run `mlx_lm.convert` with `--quantize`, `--q-bits`, `--q-group-size`
- [x] Detect input path: `models/mlx/` (Linux→Mac path, preferred) or `models/fused/` (Mac path)
- [x] Post-quantization sanity check: generate 5 test outputs, print them
- [x] Warning if not running on Mac (MLX required) — fails on non-Darwin unless --dry-run
- [x] Log: model sizes before/after, compression ratio
- [x] Verify source is not already quantized (config.json metadata check)
- [x] Full-precision vs quantized output comparison with degradation heuristics
- [x] Verify output has expected files (config.json, tokenizer.json, tokenizer_config.json, *.safetensors, quantization metadata)
- [x] Dynamic output naming (`{model}-{bits}bit`) to support non-4-bit configs
- [x] Test prompts from test.jsonl with fallback to built-in prompts (always guarantees 5)
- [x] `--force`, `--skip-sanity-check`, `--skip-comparison` flags
- [x] Summary JSON saved per model (`quantize_summary.json`)
- [x] `enable_thinking=False` with TypeError fallback, `<think>` tag stripping

**Dependencies:** WU-7  
**Open questions / flags:** None

---

### WU-9: `09_evaluate.py`
**Spec sections:** `09_evaluate.py` (lines 583–608), Evaluation config  
**Status:** ✅ COMPLETE  
**Deliverables:**
- [x] Load quantized MLX model via `mlx_lm`
- [x] Run inference on `num_samples` from test set (temperature=0)
- [x] Strip `<think>` tags from output if present (+ log warning)
- [x] Compute metrics:
  - Exact match rate
  - Character Error Rate (CER) via `editdistance`
  - BLEU score (corpus-level headline + smoothed sentence-level per-sample) via `nltk`
  - Format accuracy (custom): number conversion, filler removal, self-correction, emoji, punctuation — each as separate sub-metrics with composite average
  - Latency (tokens/sec + avg sec/sample)
- [x] Generate report in `eval_results/{model_name}/`:
  - `eval_summary.json` — full machine-readable results with per-sample data
  - `eval_report.txt` — human-readable summary
  - Per-category breakdown: N/A noted (test.jsonl lacks category metadata)
  - Worst 20 / Best 20 examples ranked by CER
  - Side-by-side: input | expected | generated in both reports
- [x] `--model` flag
- [x] `--force` flag (cleans old output dir)
- [x] `--num-samples` override flag with validation
- [x] Mac-only platform check (clear error on Linux)
- [x] Metric dependency checks upfront (editdistance, nltk)
- [x] Model directory validation (config.json, safetensors)
- [x] Deterministic sampling with `shuffle_seed` from config

**Dependencies:** WU-8 (needs quantized model)  
**Design decisions:**
1. **Mac-only.** Platform check mirrors `08_quantize.py`. Dry-run works on any platform.
2. **CER:** Uses `editdistance` library (already in `requirements-mac.txt`).
3. **BLEU:** Corpus BLEU as headline metric; smoothed sentence BLEU per-sample for diagnostics.
4. **Format accuracy:** NaN for samples with no applicable checks (excluded from aggregate mean). Sub-checks are conditional — e.g., number check only activates when expected output has digits.
5. **Per-category breakdown:** Skipped because `test.jsonl` has no category metadata. Noted as N/A in report.

---

### WU-10: `10_upload.py`
**Spec sections:** `10_upload.py` (lines 610–628), MLX Compatibility (lines 717–725), Inference Settings (lines 727–733)  
**Deliverables:**
- [ ] Create/update HF repo: `{hf_username}/{repo_prefix}-{model_name}-transcriber-4bit`
- [ ] Upload quantized model directory
- [ ] Generate and upload model card (README.md) with:
  - Model description, system prompt, example usage
  - Training details (dataset size, hyperparams, training loss)
  - Eval results
  - Python inference example (`mlx_lm`)
  - Swift usage note (`mlx-swift-lm`)
  - Recommended inference settings
- [ ] Upload system prompt file
- [ ] `--model` flag
- [ ] Respect `upload.enabled` config flag
- [ ] `--dry-run`: show what would be uploaded without uploading

**Dependencies:** WU-9 (wants eval results for the model card, though could run without)  
**Open questions / flags:** None

---

### WU-11: `run_pipeline.py` — Orchestrator
**Spec sections:** `run_pipeline.py` (lines 630–663)  
**Deliverables:**
- [ ] `--all`, `--steps`, `--resume`, `--dry-run`
- [ ] Config overrides: `--platform`, `--model`, `--synthetic-samples`, `--sample-rate`
- [ ] Step dependency validation (e.g., step 6 requires 4+5)
- [ ] State tracking in `.pipeline_state.json`
- [ ] Clear step headers, summaries, elapsed time
- [ ] Error reporting with fix suggestions
- [ ] Steps: 2, 3, 3b, 4, 5, 6, 7, 8, 9, 10 (step 1/setup is manual, not orchestrated)

**Dependencies:** All other WUs (this calls them)  
**Open questions / flags:** None

---

~~WU-12 (README) merged into WU-0 — a concise README is created during scaffolding.~~

---

## Spec Issues & Recommendations

### 1. SPEC QUESTION: System prompt `/no_think` — where exactly?
The spec says two things:
- "In the system prompt, you can also add `/no_think` as a safety measure" (line 471 / Critical Notes)
- "Append `\n/no_think` to the end of the system prompt content" (line 412 / `04_prepare_data.py`)

The `prompts/system_prompt.txt` as written in the spec does NOT include `/no_think`. **Should I:**
- (a) Add `/no_think` to `prompts/system_prompt.txt` itself (means it's always there, including in the model card), or
- (b) Keep `system_prompt.txt` clean and have `04_prepare_data.py` append `/no_think` at data-prep time?

**Decision: (b)** — keep the source prompt file clean, `04_prepare_data.py` appends `/no_think` automatically when building training data. No manual action needed.

### 2. SPEC QUESTION: Project structure mismatch — `models/mlx/` vs spec table
The project structure tree (line 68) shows `models/mlx/` for converted MLX models, and the AGENTS.md file convention table says fused models go to `models/fused/`. The `07_fuse_and_convert.py` spec uses both:
- Linux: fuse to `models/fused/`, then convert to `models/mlx/`
- Mac: fuse directly to `models/fused/` (already MLX format)

Then `08_quantize.py` reads from either. **This is fine and consistent — no issue, just confirming I'll follow this flow.**

### 3. RECOMMENDATION: `load_in_4bit: false` — clarify in config comment
The config has `load_in_4bit: false` with comment "Full precision for best quality fine-tune". This is correct, but worth noting that Unsloth's `FastLanguageModel.from_pretrained` with `load_in_4bit=False` will load in the `dtype` specified (bfloat16). This means ~1.2GB for 0.6B and ~3.4GB for 1.7B — well within your NVIDIA GPU's VRAM. No issue, just confirming.

### 4. RECOMMENDATION: Add `editdistance` and LLM SDK packages to requirements
The requirements files in the spec don't include:
- `editdistance` (for CER computation in `09_evaluate.py`)
- `anthropic` / `openai` (for synthetic data generation and validation — `openai` SDK also covers OpenRouter, Gemini, and any OpenAI-compatible provider via `base_url`)

I'll add these.

### 7. SPEC UPDATE: Config needs `base_url` for OpenAI-compatible providers
The config's `synthetic.provider` and `synthetic.validation.provider` fields currently support `"anthropic"` or `"openai"`. I'll extend this to also support `"openai_compatible"` with an added `base_url` field. Example config for OpenRouter:
```yaml
synthetic:
  provider: "openai_compatible"
  model: "google/gemini-2.5-flash"
  api_key_env: "OPENROUTER_API_KEY"
  base_url: "https://openrouter.ai/api/v1"
```
Generation and validation can use different providers independently (already supported in the spec's config structure).

### 5. RECOMMENDATION: `SFTTrainer` `dataset_text_field` deprecation
In recent TRL versions (≥0.12), `dataset_text_field` is being replaced. The newer approach is to pass a `formatting_func` or use the `messages` field directly with built-in chat template handling. I'll check the current TRL API at implementation time and use the most current approach, but the end result will be functionally identical to what the spec describes.

### 6. RESOLVED: All package management uses `uv` + venv
Spec originally referenced `pip` — updated to `uv`. `01_setup.sh` will create a venv via `uv venv` and install deps via `uv pip install -r`. All scripts assume they run inside the venv.

---

## Execution Order

```
WU-0  (scaffolding)
 ├── WU-1  (pull datasets)
 ├── WU-2  (generate synthetic)  →  WU-3  (validate synthetic)
 │         └──────────────────────────┘
 │                                    ↓
 │                               WU-4  (prepare data)
 ├── WU-5  (download models)
 │                                    ↓
 │                               WU-6  (fine-tune)
 │                                    ↓
 │                               WU-7  (fuse & convert)
 │                                    ↓
 │                               WU-8  (quantize)
 │                                    ↓
 │                               WU-9  (evaluate)
 │                                    ↓
 │                               WU-10 (upload)
 └── WU-11 (orchestrator) — can start after WU-0, but finalized last
```

**Parallelizable:** WU-1, WU-2, and WU-5 can all start immediately after WU-0.

---

## Suggested Build Order

1. **WU-0** — Scaffolding (config, common utils, directory structure, README)
2. **WU-1** — Pull datasets (quick, validates HF integration)
3. **WU-5** — Download models (can run while building data scripts)
4. **WU-2** — Synthetic data generation (includes `llm_client.py`)
5. **WU-3** — Synthetic data validation
6. **WU-4** — Data preparation
7. **WU-6** — Fine-tuning (core script, most complex)
8. **WU-7** — Fuse & convert
9. **WU-8** — Quantize
10. **WU-9** — Evaluate
11. **WU-10** — Upload
12. **WU-11** — Orchestrator
