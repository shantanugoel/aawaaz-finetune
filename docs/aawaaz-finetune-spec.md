# Agent Prompt: Build Complete Qwen3 Transcription Cleanup Fine-Tuning Pipeline

## Context

I am building an app called **aawaaz** (https://github.com/shantanugoel/aawaaz) — a macOS/iOS speech transcription app that uses a two-stage pipeline:

1. **Stage 1 (ASR):** Whisper Large V3 Turbo converts audio → raw text (already working, not part of this task)
2. **Stage 2 (LLM Cleanup):** A fine-tuned Qwen3 model takes the raw messy transcript and outputs clean, formatted text — removing fillers, adding punctuation, fixing grammar, formatting numbers, applying self-corrections, etc.

I need you to build the **complete pipeline** for Stage 2: fine-tuning Qwen3 models for transcript cleanup, producing 4-bit quantized MLX models that I'll run on-device via `mlx-swift-lm`.

## My Hardware

- **Mac:** M1 Max (64GB unified memory) — for MLX-based fine-tuning, testing, and final quantization
- **Linux GPU box:** NVIDIA GPU with CUDA — for faster fine-tuning with Unsloth/HF Transformers
- Both machines have Python 3.11+, uv, and standard dev tools

## Target Models

Build the pipeline for BOTH of these base models:
1. **Qwen3-0.6B** (`Qwen/Qwen3-0.6B`) — fast, small footprint (~400MB at 4-bit)
2. **Qwen3-1.7B** (`Qwen/Qwen3-1.7B`) — better quality, still reasonable size (~1GB at 4-bit)

The final output for each should be a 4-bit quantized MLX model directory uploadable to HuggingFace and loadable by `mlx-swift-lm`.

---

## What to Build

Create a project directory `aawaaz-finetune/` with the following structure and scripts. Everything should be runnable, tested, and well-documented.

### Project Structure

```
aawaaz-finetune/
├── README.md                          # Full setup + usage instructions
├── requirements-linux.txt             # Dependencies for Linux/GPU fine-tuning
├── requirements-mac.txt               # Dependencies for Mac/MLX fine-tuning
├── config.yaml                        # Central config file (all knobs)
├── scripts/
│   ├── 01_setup.sh                    # Install deps for chosen platform
│   ├── 02_pull_datasets.py            # Download existing datasets from HuggingFace
│   ├── 03_generate_synthetic.py       # Generate synthetic training data
│   ├── 03b_validate_synthetic.py      # LLM-as-judge quality validation of synthetic data
│   ├── 04_prepare_data.py             # Combine, format, split into train/valid/test JSONL
│   ├── 05_download_models.py          # Pull base models from HuggingFace
│   ├── 06_finetune.py                 # LoRA fine-tuning (supports both MLX and Unsloth/HF)
│   ├── 07_fuse_and_convert.py         # Fuse LoRA → merge → convert to MLX format
│   ├── 08_quantize.py                 # Quantize to 4-bit MLX
│   ├── 09_evaluate.py                 # Run evaluation suite on the model
│   ├── 10_upload.py                   # Upload to HuggingFace Hub
│   └── run_pipeline.py                # Orchestrator: run any/all steps with one command
├── data/
│   ├── raw/                           # Downloaded raw datasets land here
│   ├── synthetic/                     # Generated synthetic data lands here
│   │   ├── rejected/                  # Categories that failed validation
│   │   ├── validation_report.json     # Full judge results
│   │   └── validation_summary.txt     # Human-readable summary
│   ├── combined/                      # Combined + formatted datasets
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   └── eval/                          # Evaluation test cases
├── models/
│   ├── base/                          # Downloaded base models
│   ├── adapters/                      # LoRA adapter checkpoints
│   ├── fused/                         # Fused (merged) full-precision models
│   ├── mlx/                           # Converted MLX models
│   └── quantized/                     # Final 4-bit quantized MLX models
├── prompts/
│   └── system_prompt.txt              # The system prompt used for training and inference
└── eval_results/                      # Evaluation outputs and reports
```

---

## Detailed Specifications for Each Component

### `config.yaml` — Central Configuration

This is THE knob file. Every script reads from this. Design it so I can change one value and re-run the pipeline.

```yaml
# === Project ===
project_name: "aawaaz-transcriber"
hf_username: "shantanugoel"  # For uploading

# === Models ===
models:
  - name: "qwen3-0.6b"
    base_model: "Qwen/Qwen3-0.6B"
    unsloth_model: "unsloth/Qwen3-0.6B"  # For Unsloth fine-tuning
    enabled: true
  - name: "qwen3-1.7b"
    base_model: "Qwen/Qwen3-1.7B"
    unsloth_model: "unsloth/Qwen3-1.7B"
    enabled: true

# === Platform ===
platform: "linux"  # "linux" (Unsloth/CUDA) or "mac" (MLX)

# === Dataset ===
dataset:
  # Existing datasets to pull
  hf_datasets:
    - repo: "bingbangboom/whisper-transcripts"
      input_col: "Transcript"
      output_col: "Output"
      enabled: true
    - repo: "danielrosehill/Transcription-Cleanup-Trainer"
      enabled: true
  # Synthetic generation
  synthetic:
    enabled: true
    num_samples: 5000           # How many synthetic pairs to generate
    provider: "anthropic"       # "anthropic" or "openai"
    model: "claude-sonnet-4-20250514"
    api_key_env: "ANTHROPIC_API_KEY"  # Env var name holding the API key
    batch_size: 25              # Pairs per API call
    categories:                 # Distribution of synthetic data
      casual_conversation: 0.15
      email_professional: 0.12
      technical_code: 0.12
      medical_clinical: 0.08
      legal_contract: 0.08
      meeting_notes: 0.10
      recipe_cooking: 0.05
      academic_research: 0.08
      creative_writing: 0.05
      financial_business: 0.07
      shopping_lists: 0.03
      self_corrections_heavy: 0.07  # Focus on "wait no, I meant..."
    # LLM-as-judge validation
    validation:
      enabled: true
      sample_rate: 0.12           # Judge 12% of generated pairs (cost-effective sweet spot)
      pass_threshold: 0.90        # If >=90% of sampled pairs pass, accept the whole category
      reject_threshold: 0.70      # If <70% pass, reject entire category and flag for prompt revision
      # Between 70-90%: accept but log warning, suggest manual review
      provider: "anthropic"       # Can differ from generation provider
      model: "claude-sonnet-4-20250514"  # Use same or cheaper model for judging
      api_key_env: "ANTHROPIC_API_KEY"
  # Split ratios
  train_ratio: 0.90
  valid_ratio: 0.05
  test_ratio: 0.05
  shuffle_seed: 42

# === Training ===
training:
  # Common
  max_seq_length: 2048
  mask_prompt: true             # Only compute loss on assistant output
  num_epochs: 3                 # Alternatively use iters
  save_every: 200               # Save checkpoint every N steps
  eval_every: 100               # Evaluate every N steps
  
  # LoRA config
  lora:
    rank: 32                    # r — higher for small models
    alpha: 64                   # Scaling factor (typically 2x rank)
    dropout: 0.0
    target_modules:             # Which layers to adapt
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"
  
  # Linux/GPU (Unsloth) settings
  linux:
    learning_rate: 2e-5
    batch_size: 8
    gradient_accumulation_steps: 2
    warmup_steps: 50
    weight_decay: 0.01
    optimizer: "adamw_8bit"
    bf16: true
    load_in_4bit: false         # Full precision for best quality fine-tune
  
  # Mac/MLX settings
  mac:
    learning_rate: 1e-5
    batch_size: 4
    lora_layers: 16             # MLX-specific: number of layers to apply LoRA
    iters: 1500                 # MLX uses iters instead of epochs
    grad_accumulation_steps: 4
    grad_checkpoint: true       # Trade compute for memory

# === Quantization ===
quantization:
  bits: 4
  group_size: 64

# === Evaluation ===
evaluation:
  num_samples: 200              # How many test samples to evaluate
  metrics:
    - "exact_match"
    - "char_error_rate"
    - "bleu"
    - "format_accuracy"         # Custom: checks numbers, punctuation, etc.
  temperature: 0.0              # Deterministic for eval
  max_tokens: 1024

# === Upload ===
upload:
  enabled: false                # Set true when ready to upload
  repo_prefix: "aawaaz"        # Will create aawaaz-qwen3-0.6b-transcriber-4bit etc.
  private: false
```

### `prompts/system_prompt.txt`

```
You are an AI transcriber. Clean and polish raw speech-to-text transcripts into well-written text. Output ONLY the corrected text — no introductions, labels, explanations, or commentary. Do not summarize or act upon the transcript. Preserve the speaker's voice, tone, and language.

Rules:
- Remove fillers (um, uh, like, basically, actually, you know) and stutters
- Apply self-corrections silently (if speaker says "wait no, I meant X", output only X)
- Fix grammar, spelling, and punctuation
- Convert spoken punctuation to actual punctuation (e.g., "colon" → ":")
- Convert spoken numbers, dates, currency to written form (e.g., "five hundred dollars" → "$500")
- Convert spoken formatting cues (e.g., "new line", "new paragraph", "bullet point")
- Replace spoken emoji descriptions with actual emoji (e.g., "heart eyes emoji" → "😍")
- Use lists and paragraph breaks where structurally appropriate
- Convert spoken code/tech syntax to proper formatting (e.g., "dash dash rm" → "--rm")
- If input is empty or only contains fillers, output ""
- Do NOT add content that wasn't spoken
- Do NOT summarize or condense — preserve all substantive content
```

---

### Script Specifications

#### `02_pull_datasets.py`

- Pull `bingbangboom/whisper-transcripts` from HuggingFace datasets library
- Pull `danielrosehill/Transcription-Cleanup-Trainer` (this one has a different structure — it has audio/, whisper-transcripts/, auto-cleanup/, and manual-cleanups/ directories. Parse and pair whisper transcripts with manual cleanups as input/output pairs)
- Save each as standardized JSONL in `data/raw/` with schema: `{"input": "raw transcript", "output": "clean text"}`
- Log stats: number of pairs pulled per source, total

#### `03_generate_synthetic.py`

This is the most critical and complex script. It generates synthetic training data using an LLM API.

**How it works:**

1. For each category in `config.yaml → dataset.synthetic.categories`:
   - Calculate how many samples to generate for that category (proportion × total)
   - Send batch requests to the LLM API asking it to generate pairs
   
2. The API prompt should ask the LLM to generate BOTH the messy transcript AND the clean version in one call. This is better than "messifying" existing text because:
   - The messy version feels naturally spoken (with organic filler placement)
   - The pairs are guaranteed to be correct
   - You get natural self-correction patterns

3. **The generation prompt sent to the API** (this is critical — get it right):

```
Generate {batch_size} realistic speech-to-text transcript pairs for the category: {category}.

For each pair, create:
1. A "transcript" — what a speech-to-text engine (like Whisper) would output from someone speaking naturally. This should include:
   - Filler words (um, uh, like, basically, actually, you know, so, I mean) placed naturally
   - No punctuation or very minimal/wrong punctuation
   - Numbers, dates, and currency spoken as words (e.g., "two thousand twenty five" not "2025")
   - Occasional self-corrections (e.g., "the meeting is on Tuesday wait no Wednesday")
   - Spoken formatting cues where natural (e.g., "colon", "new line", "bullet point", "dash")
   - Run-on sentences with no clear breaks
   - Some spoken technical terms, code syntax, URLs spelled out
   - Occasional spoken emoji descriptions (e.g., "thumbs up emoji")
   - Realistic speech patterns — not every sentence has fillers, vary the messiness
   
2. An "output" — the clean, properly formatted version that preserves ALL substantive content but:
   - Removes all fillers and stutters
   - Applies self-corrections (only keeps the corrected version)
   - Has proper punctuation, capitalization, and grammar
   - Numbers, dates, currency in written form ($500, January 15, 2025)
   - Proper formatting (bullet lists, paragraphs, code formatting)
   - Emoji characters where described
   - No content added or removed

Category-specific guidance for "{category}":
{category_specific_guidance}

Respond with a JSON array of objects, each with "transcript" and "output" keys. No other text.
Vary the length: some short (1-2 sentences), some medium (paragraph), some long (multiple paragraphs).
```

4. **Category-specific guidance** (include these in the prompt per category):
   - `casual_conversation`: Friend-to-friend messages, voice notes, casual updates. Use contractions, informal language, sometimes trailing off.
   - `email_professional`: Dictating business emails. Include "dear", "regards", salutations. Formal tone but spoken casually.
   - `technical_code`: Dictating code, CLI commands, error messages, technical docs. Include function names, file paths, docker commands, SQL queries, variable names with underscores/camelCase.
   - `medical_clinical`: Patient notes, clinical observations, medication names, dosages, medical abbreviations.
   - `legal_contract`: Contract clauses, legal terminology, article/section references, formal language.
   - `meeting_notes`: Action items, attendee names, deadlines, decisions made.
   - `self_corrections_heavy`: Specifically focus on self-correction patterns: "wait", "no", "scratch that", "actually I meant", "let me rephrase", "correction". Multiple corrections per example. This category is crucial for quality.

5. **Error handling:**
   - Retry failed API calls with exponential backoff (3 retries)
   - Validate each generated pair (input and output should both be non-empty strings, output should be shorter or similar length to input after cleanup)
   - Save progress incrementally — if generation crashes at sample 3000, resuming should skip already-generated samples
   - Log: total generated, failed, retried, per-category counts

6. **Rate limiting:**
   - Respect API rate limits
   - Add configurable delay between batches
   - Show progress bar with ETA

7. Save to `data/synthetic/synthetic_{category}.jsonl` per category, then a combined `data/synthetic/all_synthetic.jsonl`

#### `03b_validate_synthetic.py` — LLM-as-Judge Quality Gate

Rule-based checks in step 04 only catch ~10% of real data quality issues. The dangerous failures are subtle: inputs that aren't messy enough, outputs that silently drop content, outputs that hallucinate new content, or self-corrections applied incorrectly. This script catches those.

**How it works:**

1. For each category file in `data/synthetic/synthetic_{category}.jsonl`:
   - Sample `config.dataset.synthetic.validation.sample_rate` fraction of pairs randomly (e.g., 12%)
   - Send each sampled pair to the LLM judge
   - Score on 4 binary pass/fail criteria
   - Compute per-category pass rate

2. **The judge prompt:**

```
You are evaluating a synthetic training pair for a speech transcript cleanup model.
Your job is to ensure the training data is high quality. Be strict.

INPUT (raw transcript):
{transcript}

OUTPUT (cleaned text):
{output}

Score this pair on 4 criteria. Answer ONLY with a JSON object, no other text.

1. "input_realistic": Is the input realistic raw ASR output? It should have:
   - Little to NO punctuation and NO capitalization (or only accidental/minimal)
   - Numbers, dates, and currency as spoken words (e.g., "five hundred" not "500")
   - Natural filler words placed organically (not "um" mechanically inserted every 5 words)
   - Run-on sentences without clear boundaries
   FAIL if it looks like already-clean text with a few "um"s sprinkled in.
   FAIL if it has proper punctuation, capitalization throughout, or formatted numbers.

2. "content_preserved": Does the output preserve ALL substantive information from the input?
   Compare carefully — no facts, names, numbers, instructions, or meaning should be dropped.
   Minor rewording is fine. Dropping an entire sentence or fact is a FAIL.

3. "no_hallucination": Does the output contain ONLY information present in the input?
   The output should not add facts, opinions, clarifications, context, or conclusions
   that the speaker did not say. Formatting changes (adding punctuation, converting
   numbers) are fine — adding NEW words/ideas is a FAIL.

4. "corrections_applied": Is the cleanup done correctly?
   - Fillers (um, uh, basically, actually, like, you know) removed
   - Self-corrections applied correctly (only the corrected version kept)
   - Numbers/dates/currency properly formatted
   - Punctuation and capitalization added appropriately
   FAIL if obvious cleanup was missed or done incorrectly.

{"input_realistic": true/false, "content_preserved": true/false, "no_hallucination": true/false, "corrections_applied": true/false}
```

3. **Per-category decision logic:**
   - **Pass rate >= 90%** (`pass_threshold`): Accept all data from this category. Log pass rate.
   - **Pass rate 70-90%**: Accept the data, but log a WARNING with the category name, pass rate, and the specific failing criteria breakdown. Suggest manual review or prompt revision for next run.
   - **Pass rate < 70%** (`reject_threshold`): REJECT the entire category. Move the file to `data/synthetic/rejected/`. Log an ERROR with:
     - The category name and pass rate
     - Breakdown of which criteria failed most often (e.g., "input_realistic failed 40% of the time")
     - 5 example failures for each failing criterion
     - Suggest specific prompt improvements based on what failed

4. **Output:**
   - `data/synthetic/validation_report.json`: Full results per category with pass rates, criterion-level breakdown, and example failures
   - `data/synthetic/validation_summary.txt`: Human-readable summary suitable for printing to console
   - For accepted categories: pairs remain in their original files untouched
   - For rejected categories: files moved to `data/synthetic/rejected/` (not deleted)

5. **Individual pair rejection (within accepted categories):**
   - Even in categories that pass overall, individually judged pairs that fail on `content_preserved` or `no_hallucination` should be REMOVED from the data file
   - Pairs failing only `input_realistic` or `corrections_applied` are kept (these are less dangerous — the model can still learn from them, they're just suboptimal)
   - Log how many individual pairs were removed per category

6. **Cost estimation:**
   - Before running, print estimated API cost: (num_samples × sample_rate × avg_tokens_per_judge_call × price_per_token)
   - Ask for confirmation before proceeding (skip confirmation with `--yes` flag)

7. **Flags:**
   - `--sample-rate 0.25` — override the config sample rate (useful for first run: validate more aggressively)
   - `--full` — validate 100% of pairs (expensive, use for final check before training)
   - `--category casual_conversation` — validate only one specific category
   - `--yes` — skip cost confirmation prompt
   - `--verbose` — print every individual pair judgment, not just summary

#### `04_prepare_data.py`

- Load all data from `data/raw/` and `data/synthetic/` (but NOT from `data/synthetic/rejected/` — only load category files that passed the validation gate in step 3b)
- If step 3b was not run (validation disabled in config), load all synthetic data but log a warning
- Convert every pair into **chat-format JSONL** using the system prompt from `prompts/system_prompt.txt`:
  ```json
  {"messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<raw transcript>"},
    {"role": "assistant", "content": "<clean output>"}
  ]}
  ```
- **IMPORTANT for Qwen3:** The assistant response must NOT have any `<think>` blocks. Append `\n/no_think` to the end of the system prompt content to disable thinking mode. Actually, better approach: set `enable_thinking: false` in the chat template application. Research how the Qwen3 tokenizer chat template handles this and implement correctly.
- Deduplicate (by normalizing and comparing input text)
- Shuffle with seed from config
- Split into train/valid/test per config ratios
- Save to `data/combined/{train,valid,test}.jsonl`
- Log stats: total samples, per-source breakdown, split sizes, avg input/output lengths

**CRITICAL NOTE ON QWEN3 CHAT FORMAT:**
- Qwen3 uses a specific chat template. The tokenizer's `apply_chat_template` handles this.
- For training with MLX, the JSONL format with `messages` key should work directly.
- For training with Unsloth/HF, you may need to apply the chat template during data loading.
- Test that the formatted data round-trips correctly through the tokenizer before training.

#### `05_download_models.py`

- Download base models listed in config to `models/base/`
- For Linux: download both the HF model and the Unsloth variant
- For Mac: download the HF model (MLX will handle conversion)
- Verify downloads by loading tokenizer and doing a test generation
- Log: model sizes, parameter counts

#### `06_finetune.py`

This is the core training script. It must support BOTH platforms.

**Linux (Unsloth) path:**

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load model at full precision (NOT 4-bit — we want best quality)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.unsloth_model,
    max_seq_length=config.max_seq_length,
    load_in_4bit=False,
    dtype=torch.bfloat16,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=config.lora.rank,
    target_modules=config.lora.target_modules,
    lora_alpha=config.lora.alpha,
    lora_dropout=config.lora.dropout,
    use_gradient_checkpointing="unsloth",
)

# Load dataset
dataset = load_dataset("json", data_files={"train": "data/combined/train.jsonl", "validation": "data/combined/valid.jsonl"})

# CRITICAL: Use the tokenizer's chat template to format the data
def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  # DISABLE THINKING MODE
    )
    return {"text": text}

dataset = dataset.map(format_chat)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=config.max_seq_length,
    args=TrainingArguments(
        output_dir=f"models/adapters/{model_name}",
        per_device_train_batch_size=config.linux.batch_size,
        gradient_accumulation_steps=config.linux.gradient_accumulation_steps,
        warmup_steps=config.linux.warmup_steps,
        learning_rate=config.linux.learning_rate,
        num_train_epochs=config.num_epochs,
        bf16=config.linux.bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=config.eval_every,
        save_steps=config.save_every,
        save_total_limit=3,
        weight_decay=config.linux.weight_decay,
        optim=config.linux.optimizer,
        seed=42,
        report_to="none",  # or "wandb" if they want
    ),
)

trainer.train()

# Save LoRA adapters
model.save_pretrained(f"models/adapters/{model_name}")
tokenizer.save_pretrained(f"models/adapters/{model_name}")
```

**Mac (MLX) path:**

```bash
mlx_lm.lora \
  --model {base_model_path} \
  --data data/combined \
  --train \
  --batch-size {config.mac.batch_size} \
  --lora-layers {config.mac.lora_layers} \
  --iters {config.mac.iters} \
  --learning-rate {config.mac.learning_rate} \
  --adapter-path models/adapters/{model_name} \
  --steps-per-eval {config.eval_every} \
  --save-every {config.save_every} \
  --mask-prompt \
  --grad-checkpoint \
  --grad-accumulation-steps {config.mac.grad_accumulation_steps}
```

**Things the agent MUST handle:**
- The script should detect platform from config and use the right path
- Log training loss, validation loss, tokens/sec at every eval step
- Save the best checkpoint (lowest validation loss) separately
- After training completes, print a summary: final train loss, final val loss, total time, tokens processed
- If training crashes/is interrupted, support resuming from last checkpoint

#### `07_fuse_and_convert.py`

**Linux path (after Unsloth training):**
```python
# Merge LoRA into base model at 16-bit
model.save_pretrained_merged(
    f"models/fused/{model_name}",
    tokenizer,
    save_method="merged_16bit",
)
```

**Mac path (after MLX training):**
```bash
mlx_lm.fuse \
  --model {base_model_path} \
  --adapter-path models/adapters/{model_name} \
  --save-path models/fused/{model_name} \
  --de-quantize
```

**Cross-platform note:** If you fine-tuned on Linux, the fused model is a standard HF-format model. You then need to convert it to MLX format:
```bash
mlx_lm.convert \
  --hf-path models/fused/{model_name} \
  --mlx-path models/mlx/{model_name}
```

If you fine-tuned on Mac (MLX), the fuse step already produces MLX format.

#### `08_quantize.py`

```bash
mlx_lm.convert \
  --hf-path models/fused/{model_name} \    # or models/mlx/{model_name}
  --mlx-path models/quantized/{model_name}-4bit \
  --quantize \
  --q-bits {config.quantization.bits} \
  --q-group-size {config.quantization.group_size}
```

This step MUST be run on the Mac (MLX is required). If fine-tuning was done on Linux, the fused HF model needs to be transferred to Mac first.

Add a note/warning in the script about this.

#### `09_evaluate.py`

Run the quantized model on the test set and compute metrics:

1. Load the quantized MLX model using `mlx_lm`
2. For each test example:
   - Feed the input (raw transcript) with the system prompt
   - Generate output with temperature=0, max_tokens from config
   - Compare generated output vs expected output
3. Compute metrics:
   - **Exact match rate**: % of outputs that exactly match expected
   - **Character Error Rate (CER)**: Edit distance / reference length
   - **BLEU score**: Standard translation metric
   - **Format accuracy** (custom):
     - Numbers correctly converted (e.g., "five hundred" → "500")
     - Punctuation present and roughly correct
     - Fillers removed (check no "um", "uh", "basically" remain unless they're in quoted speech)
     - Self-corrections applied (no "wait no" / "scratch that" in output)
     - Emoji conversion (if applicable)
   - **Latency**: tokens/second on the target hardware
4. Generate a report in `eval_results/`:
   - Summary statistics
   - Per-category breakdown (if category info is available)
   - Worst 20 examples (highest error) for manual review
   - Best 20 examples for sanity check
   - Side-by-side comparison: input | expected | generated

#### `10_upload.py`

```python
from huggingface_hub import HfApi

api = HfApi()
repo_name = f"{config.hf_username}/{config.upload.repo_prefix}-{model_name}-transcriber-4bit"
api.create_repo(repo_name, private=config.upload.private, exist_ok=True)
api.upload_folder(
    folder_path=f"models/quantized/{model_name}-4bit",
    repo_id=repo_name,
    commit_message="Upload fine-tuned transcriber model"
)
```

Also upload:
- A README.md model card with: model description, system prompt, example usage, training details, eval results
- The system prompt file
- Example inference code for both Python (mlx_lm) and a note about Swift (mlx-swift-lm) usage

#### `run_pipeline.py` — The Orchestrator

This is the main entry point. It should support:

```bash
# Run everything end to end
python scripts/run_pipeline.py --all

# Run specific steps
python scripts/run_pipeline.py --steps 2,3,3b,4       # Just data prep (generate + validate + format)
python scripts/run_pipeline.py --steps 3b              # Just validate existing synthetic data
python scripts/run_pipeline.py --steps 6               # Just fine-tuning
python scripts/run_pipeline.py --steps 7,8,9           # Fuse, quantize, eval
python scripts/run_pipeline.py --steps 6,7,8,9,10      # Train through upload

# Override config values
python scripts/run_pipeline.py --all --platform mac
python scripts/run_pipeline.py --steps 6 --model qwen3-0.6b
python scripts/run_pipeline.py --steps 3 --synthetic-samples 10000
python scripts/run_pipeline.py --steps 3b --sample-rate 0.25  # More aggressive validation

# Dry run (show what would be executed)
python scripts/run_pipeline.py --all --dry-run

# Resume from where it left off (skip completed steps)
python scripts/run_pipeline.py --all --resume
```

Features:
- Tracks which steps have completed successfully (write state to `.pipeline_state.json`)
- Before each step, validate prerequisites (e.g., step 3b requires step 3 to be done, step 4 requires 3b to be done if validation is enabled, step 6 requires step 4 and 5 to be done)
- Print clear step headers and summaries
- Total elapsed time at the end
- If a step fails, print the error clearly and suggest how to fix it

---

## Critical Implementation Notes (READ THESE CAREFULLY)

### Qwen3 Thinking Mode
Qwen3 models have a built-in "thinking" mode that generates `<think>...</think>` blocks before the actual response. This MUST be disabled for our use case — it adds latency and wastes tokens for a deterministic cleanup task.

- When using `tokenizer.apply_chat_template()`, pass `enable_thinking=False`
- In the system prompt, you can also add `/no_think` as a safety measure
- During evaluation, if any output contains `<think>` tags, strip them and log a warning
- Verify this works correctly with the specific Qwen3-0.6B and 1.7B tokenizers — the parameter name or behavior might differ slightly between versions

### Chat Template Handling
- Qwen3 uses `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>` format (ChatML-style)
- MLX's LoRA training expects data in `{"messages": [...]}` JSONL format and applies the chat template internally
- Unsloth/HF expects you to apply the template yourself into a `text` field
- TEST that the templates match between the two paths — a mismatch here will silently ruin training

### mask_prompt / Training Loss
- We ONLY want the model to learn to predict the assistant's response (the cleaned text)
- We do NOT want loss computed on the system prompt or user input
- In MLX: use `--mask-prompt` flag
- In Unsloth/TRL: the SFTTrainer with chat template handling should mask the prompt automatically when using the messages format. Verify this — if it doesn't, implement custom data collator that masks prompt tokens.

### Data Quality Validation
In `04_prepare_data.py`, add validation:
- Reject pairs where output is longer than 2x the input (likely hallucinated content)
- Reject pairs where output is empty but input has substantive content
- Reject pairs with obvious encoding issues (mojibake, null bytes)
- Reject pairs where the output still contains obvious fillers ("um", "uh") — unless inside quoted speech
- Flag (but don't reject) pairs where output adds content not in input
- Log all rejections with reasons

### Synthetic Data Quality
The LLM-as-judge step (`03b_validate_synthetic.py`) is the primary quality gate — see its full spec above. Additional notes:
- The generation prompt MUST emphasize that the "transcript" field should look like REAL Whisper output, not just slightly informal text. Real Whisper output has NO punctuation, numbers are spelled out, there are no paragraph breaks, everything runs together. The `input_realistic` judge criterion specifically catches this — it's the most common failure mode.
- If using Anthropic's API, use claude-sonnet-4-20250514 for both generation and judging — it's great at generating realistic transcript pairs and strict enough as a judge.
- In practice, `technical_code` and `medical_clinical` categories are most prone to hallucination in the output. `casual_conversation` tends to produce inputs that aren't messy enough. Expect to iterate on the generation prompts for these categories based on the judge report.
- On first run, consider using `--sample-rate 0.25` (25%) for the validation step to get a stronger signal on quality before committing to the full pipeline. Once you've tuned your generation prompts and pass rates are consistently >90%, drop back to the default 12%.
- The judge costs roughly 10-15% of the generation API budget — this is worth it. A model trained on subtly bad data will produce subtly bad outputs that are much harder to diagnose.

### Model-Specific Considerations
- **Qwen3-0.6B**: Very small model. Use higher LoRA rank (32+) and more training iterations to compensate. May need more epochs.
- **Qwen3-1.7B**: Better base capabilities. Can use lower rank (16-32). Should converge faster.
- For both: monitor validation loss closely. These small models overfit quickly on small datasets. If val loss starts rising while train loss drops, stop training.

### Quantization
- Always quantize from the full-precision fused model, NOT from a model that was trained in quantized mode
- Use `q-bits=4` and `q-group-size=64` — this is the standard MLX quantization config
- After quantization, run a quick sanity check: generate 5 test outputs and verify they're reasonable
- Compare a few outputs between the full-precision fused model and the 4-bit quantized model — they should be nearly identical. If there's significant degradation, try `q-bits=8` as an alternative.

### MLX Compatibility
- The final model MUST be loadable by `mlx-swift-lm` with code like:
  ```swift
  let model = try await loadModel(id: "shantanugoel/aawaaz-qwen3-0.6b-transcriber-4bit")
  let session = ChatSession(model)
  let cleaned = try await session.respond(to: rawTranscript)
  ```
- Verify the model directory has the expected files: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `*.safetensors`, and the quantization config
- The `config.json` must have the correct architecture identifier that mlx-swift-lm recognizes for Qwen3

### Inference Settings for Production
Document these recommended settings in the model card:
- `temperature: 0.0` (deterministic — this is a formatting task, not creative)
- `top_p: 1.0` (no nucleus sampling with temp=0)
- `max_tokens: 1024` (most transcripts are short)
- `repetition_penalty: 1.1` (slight penalty to avoid degenerate loops)
- System prompt: the one from `prompts/system_prompt.txt`

### Error Handling Throughout
Every script should:
- Use proper argument parsing (argparse) even if primarily driven by config
- Have a `--verbose` flag for debug output
- Catch and report errors clearly (not just stack traces)
- Be idempotent where possible (re-running doesn't duplicate work)
- Support `--dry-run` to show what would happen without doing it

### Dependencies
**Linux (requirements-linux.txt):**
```
torch>=2.1.0
unsloth
transformers>=4.44.0
datasets
trl>=0.12.0
peft
accelerate
bitsandbytes
huggingface_hub
pyyaml
tqdm
evaluate
nltk
rouge-score
```

**Mac (requirements-mac.txt):**
```
mlx>=0.21.0
mlx-lm>=0.21.0
transformers>=4.44.0
datasets
huggingface_hub
pyyaml
tqdm
evaluate
nltk
```

Plus `anthropic` or `openai` SDK for synthetic data generation.

---

## Definition of Done

The pipeline is complete when:
1. I can run `python scripts/run_pipeline.py --all --platform linux --model qwen3-0.6b` on my Linux box and it produces a fine-tuned, quantized 4-bit MLX model
2. I can do the same with `--model qwen3-1.7b`
3. The synthetic data validation step (3b) shows >90% pass rate across categories
4. The eval step shows reasonable metrics (CER < 10%, filler removal > 95%)
5. The quantized models are loadable by `mlx_lm.generate` on my Mac
6. The upload step creates proper HuggingFace repos with model cards
7. Every script works independently and as part of the orchestrated pipeline
8. The README.md has clear setup and usage instructions for both platforms

---

## Stretch Goals (if time permits)

1. **WandB integration**: Add optional Weights & Biases logging for training runs
2. **Curriculum learning**: Train in phases — first on easy examples (short, clean), then hard (long, many corrections)
3. **Multiple quantization outputs**: Produce both 4-bit and 8-bit variants
4. **Comparison script**: Side-by-side compare 0.6B vs 1.7B outputs on the same test set
5. **Data augmentation**: Take existing clean examples and create multiple messy variants of each (varying filler density, different self-correction patterns)
6. **CI/CD**: GitHub Actions workflow that re-trains when data changes
