# Aawaaz Fine-Tuning Guide: Qwen3 for Transcription Cleanup
## Targeting On-Device MLX Inference via mlx-swift-lm

---

## Architecture Overview

The approach you're following mirrors what Wispr Flow does in production (where they use fine-tuned Llama models on Baseten cloud with <700ms p99 latency). The two-stage pipeline is:

```
┌──────────────┐       ┌──────────────────────┐       ┌───────────────┐
│  Audio Input  │──────▶│  ASR Engine           │──────▶│  Raw Transcript│
│  (mic/file)   │       │  (Qwen3-ASR-0.6B,    │       │  (messy text)  │
│               │       │   Whisper, Parakeet)  │       │                │
└──────────────┘       └──────────────────────┘       └───────┬───────┘
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │  LLM Cleanup      │
                                                    │  (Fine-tuned      │
                                                    │   Qwen3 0.6B)     │
                                                    └───────┬──────────┘
                                                            │
                                                            ▼
                                                  ┌──────────────────┐
                                                  │  Clean Output     │
                                                  │  (formatted,      │
                                                  │   punctuated,     │
                                                  │   no fillers)     │
                                                  └──────────────────┘
```

**Critical Clarification:** The `bingbangboom/whisper-transcripts` dataset is a TEXT-to-TEXT dataset (raw messy transcript → clean formatted text). You're fine-tuning the **text LLM** (Qwen3-0.6B) for post-processing, NOT the ASR model (Qwen3-ASR-0.6B). These are completely different model architectures.

**Also note:** The existing model `bingbangboom/Qwen3508B-transcriber` was actually fine-tuned on `unsloth/Qwen3.5-0.8B` (a newer model than Qwen3-0.6B, released later), NOT Qwen3-0.6B. If Qwen3.5-0.8B is available, it may be a better base model. Check compatibility with mlx-swift-lm before committing.

---

## 1. Datasets

### Your Current Dataset
- **bingbangboom/whisper-transcripts**: 1,280 rows. Text-to-text pairs (raw transcript → clean output)
- **Quality**: High quality examples covering diverse domains (medical, legal, code, casual, technical, recipes, emails, etc.)
- **Weakness**: Only 1.28k examples. This is quite small — you'll want to augment significantly.

### Other Recommended Datasets

#### Directly Relevant (Transcript Cleanup)

| Dataset | Size | Description |
|---------|------|-------------|
| `danielrosehill/Transcription-Cleanup-Trainer` | Small | Paired raw Whisper transcripts + human-edited ground truth with real audio. Has audio, auto-cleanup (Gemini), and manual cleanups. |
| `SponSpeech` (arXiv 2409.11241) | Medium | Punctuation restoration from spontaneous/informal speech with stutters and irregularities. Specifically targets real-world ASR messiness. |

#### For Augmentation / Supplementary Training

| Dataset | Use Case |
|---------|----------|
| `oliverguhr/fullstop-punctuation-multilang-large` | Punctuation restoration model — can help generate synthetic training pairs |
| Common Voice transcripts | Use the raw ASR output from Whisper on Common Voice audio, paired with the ground truth clean text, to generate thousands of training pairs |
| LibriSpeech / GigaSpeech | Same approach: run ASR on audio, pair raw output with reference text |

### Synthetic Data Generation (Strongly Recommended)

Your biggest opportunity is **generating synthetic training data**. Here's why: with only 1.28k examples, your model will be undertrained and may not generalize well. The `bingbangboom` examples are well-crafted but cover limited patterns.

**Method 1: Reverse Engineering from Clean Text**
```
1. Take well-written text from diverse sources (Wikipedia, news, emails, code docs, recipes, etc.)
2. Use Claude/GPT-4 to "messify" it — add filler words, remove punctuation,
   spell out numbers as words, add self-corrections, remove capitalization,
   add spoken formatting cues ("colon", "new line", "bullet point")
3. Pair: messy version → original clean version
```

**Method 2: Real ASR Output Pairing**
```
1. Run Qwen3-ASR or Whisper on audio datasets (Common Voice, LibriSpeech, podcasts)
2. The ASR output is your "input" (raw transcript)
3. The ground truth reference text is your "output" (clean version)
4. Filter for pairs where the ASR got the words roughly right but formatting is wrong
```

**Method 3: LLM-Assisted Expansion**
```
1. Use Claude to generate 5,000-10,000 additional transcript/cleanup pairs
2. Cover edge cases: code dictation, math/equations, multi-language mixing,
   medical terminology, legal jargon, emoji descriptions, URLs, email addresses
3. Include examples of: self-corrections ("wait no, I meant..."),
   repeated words, false starts, tangential asides that should be removed
```

**Target dataset size: 5,000-15,000 high-quality pairs minimum** for a 0.6B model to learn this task well. Quality matters more than quantity, but 1.28k is definitively too few.

---

## 2. Fine-Tuning Pipeline: The Optimal Path

### The Key Decision: Fine-Tune First, Quantize Last

**Recommended Path (Best Quality):**
```
Base FP16/BF16 Model → LoRA Fine-tune at Full Precision → Fuse LoRA → Quantize to 4-bit MLX → Deploy
```

**Why NOT QLoRA (quantize-then-fine-tune)?**
- For a 0.6B model, memory is not a constraint. The full BF16 model is only ~1.2GB
- QLoRA introduces quantization noise during training that the LoRA adapters must compensate for
- The QLoRA paper showed near-parity with full fine-tuning for LARGE models (7B+), but for a tiny 0.6B model, every bit of precision matters
- You're going to quantize for deployment anyway — quantizing AFTER training preserves the maximum quality of your fine-tune, and then you take one clean quantization hit

**When QLoRA IS fine:** If you were fine-tuning a 7B+ model and memory-constrained. For 0.6B, it's unnecessary.

### Step-by-Step Pipeline

#### Step 0: Choose Your Base Model

Options (in order of recommendation):
1. **`Qwen/Qwen3-0.6B`** — The model you mentioned. 0.6B params, Qwen3 architecture. Well-supported in mlx-swift-lm.
2. **`Qwen/Qwen3.5-0.8B`** — Newer, slightly larger (0.8B). This is what bingbangboom used. Better base capabilities. Check mlx-swift-lm support.
3. **`unsloth/Qwen3-0.6B`** — Unsloth's optimized version, faster fine-tuning.

Verify your chosen model works with mlx-swift-lm BEFORE fine-tuning:
```bash
# Test with mlx-lm first
pip install mlx-lm
mlx_lm.generate --model mlx-community/Qwen3-0.6B-4bit --prompt "test"
```

#### Step 1: Prepare Your Dataset

Format your data as JSONL with chat template format:

```jsonl
{"messages": [{"role": "system", "content": "You are an AI transcriber. Clean and polish raw speech-to-text transcripts into well-written text. Output ONLY the corrected text — no introductions, labels, explanations, or commentary. Do not summarize or act upon the transcript. Preserve the speaker's voice, tone, and language.\n\nRules: Remove fillers and stutters. Apply self-corrections silently. Fix grammar, spelling, and punctuation. Convert spoken punctuation, numbers, dates, and currency to written form. Replace spoken emoji descriptions with actual emoji. Use lists and paragraph breaks where appropriate. If input is empty, output \"\"."}, {"role": "user", "content": "so uh the argument I want to make in the op-ed is that..."}, {"role": "assistant", "content": "The argument I want to make in the op-ed is that..."}]}
```

Split into train.jsonl (90%), valid.jsonl (5%), test.jsonl (5%).

#### Step 2: Fine-Tune with LoRA (on Mac with MLX)

```bash
pip install mlx-lm

# LoRA fine-tune on full-precision base model
mlx_lm.lora \
  --model Qwen/Qwen3-0.6B \
  --data ./data \
  --train \
  --batch-size 4 \
  --lora-layers 16 \
  --iters 1000 \
  --learning-rate 1e-5 \
  --adapter-path ./adapters \
  --steps-per-eval 100 \
  --mask-prompt
```

**Key flags:**
- `--mask-prompt`: Only compute loss on the assistant's output (the cleanup), not the system prompt or user input. Critical for this task.
- `--lora-layers 16`: For a 0.6B model, 16 layers covers most of the model. Could go higher.
- `--learning-rate 1e-5`: Start conservative. If loss plateaus, try 2e-5 or 5e-5.
- `--batch-size 4`: 0.6B is small enough for batch 4 even on 16GB Mac.

**Alternatively, fine-tune with Unsloth on GPU (faster, recommended if you have access to a GPU):**

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B",
    max_seq_length=2048,
    load_in_4bit=False,  # Full precision for best quality
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,              # Higher rank for small model
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# ... train with SFTTrainer from trl ...
# Save merged model:
model.save_pretrained_merged("./merged_model", tokenizer, save_method="merged_16bit")
```

#### Step 3: Fuse LoRA Adapters into Base Model

If you fine-tuned with MLX:
```bash
mlx_lm.fuse \
  --model Qwen/Qwen3-0.6B \
  --adapter-path ./adapters \
  --save-path ./fused_model \
  --de-quantize   # Ensures full precision output
```

If you used Unsloth, the `save_pretrained_merged` call already does this.

#### Step 4: Convert & Quantize to 4-bit MLX Format

```bash
mlx_lm.convert \
  --hf-path ./fused_model \
  --mlx-path ./qwen3-0.6b-transcriber-4bit \
  --quantize \
  --q-bits 4 \
  --q-group-size 64
```

This produces an MLX-format model directory with 4-bit quantized weights.

#### Step 5: Test Locally

```bash
mlx_lm.generate \
  --model ./qwen3-0.6b-transcriber-4bit \
  --prompt "<system prompt>\n\nso uh basically the the ECS task definition needs to specify the container image memory reservation of five twelve megabytes" \
  --max-tokens 200
```

#### Step 6: Upload to Hugging Face & Use in Swift App

```bash
# Upload
huggingface-cli upload your-username/qwen3-0.6b-transcriber-4bit ./qwen3-0.6b-transcriber-4bit

# In Swift (mlx-swift-lm):
let model = try await loadModel(id: "your-username/qwen3-0.6b-transcriber-4bit")
let session = ChatSession(model)
let cleaned = try await session.respond(to: rawTranscript)
```

---

## 3. Important Things to Know

### Disable Thinking Mode for Qwen3

Qwen3 models have a built-in "thinking mode" that generates `<think>...</think>` blocks before responding. **You absolutely do NOT want this for a transcription cleanup model** — it adds latency and wastes tokens. Disable it:

- In your system prompt, add `/no_think` at the end
- Or when applying the chat template, set `enable_thinking=False`
- In your training data, ensure no `<think>` blocks appear in the assistant responses

### Temperature = 0 for Inference

This is a deterministic task. There's one correct cleanup for a given transcript. Always use temperature=0 (greedy decoding) in production.

### Context Window Considerations

For a dictation app, transcripts are typically short (a few sentences to a few paragraphs). The 0.6B model has a 32K context window, but:
- Keep your system prompt concise (the one from bingbangboom is good)
- For very long dictations (>2 minutes), consider chunking into segments
- Shorter inputs = faster inference = better UX

### Model Size vs. Quality Tradeoff

| Model | Size (4-bit) | Quality | Speed |
|-------|-------------|---------|-------|
| Qwen3-0.6B | ~400MB | Good for structured cleanup | Very fast |
| Qwen3.5-0.8B | ~500MB | Better, newer architecture | Fast |
| Qwen3-1.7B | ~1GB | Significantly better reasoning | Moderate |
| Qwen3-4B | ~2.5GB | Excellent, handles edge cases | Slower |

For a dictation app competing with Wispr Flow, the 0.6B model will handle straightforward cleanup well but may struggle with:
- Complex self-corrections ("wait, no, I meant the opposite")
- Ambiguous sentence boundaries
- Technical jargon it hasn't seen in training
- Multi-language code-switching

Consider starting with 0.6B and having a fallback to 1.7B for complex inputs, or just using 1.7B if your target devices can handle it (any M-series Mac has plenty of RAM).

### Evaluation Strategy

Create a test set with:
1. **Exact match accuracy**: What % of outputs perfectly match expected cleanup
2. **Word Error Rate (WER)**: Standard ASR metric applied to the cleanup task
3. **Edge case coverage**: Self-corrections, empty inputs, code, numbers, emoji
4. **Latency benchmarks**: Measure tokens/second on target hardware
5. **A/B comparison**: Have humans rate your model's output vs. GPT-4's cleanup of the same input

### The Wispr Flow Advantage You Need to Match

What makes Wispr Flow feel magical isn't just cleanup — it's **context-awareness**:
- It knows which app you're typing in (email → formal, Slack → casual)
- It reads surrounding text for context (names, formatting)
- It handles self-corrections intelligently

To match this in aawaaz:
- Include the app context in your system prompt dynamically
- Add "context: replying to email from John about project Alpha" type prefixes
- Train on examples where the output style varies based on stated context
- Train on self-correction patterns extensively

### Data Quality > Data Quantity

The bingbangboom dataset has excellent examples (look at the self-correction patterns, the code dictation, the board meeting email). When generating synthetic data, maintain this quality bar:
- Every pair should have a clear, unambiguous correct output
- Include diverse domains (medical, legal, casual, technical, creative)
- Include tricky patterns (homophones, numbers in context, abbreviations)
- Include "null" cases (empty input → empty output, already-clean input → same output)

---

## Summary: The Complete Path

```
1. DATASET: Expand bingbangboom/whisper-transcripts from 1.28k → 5-15k examples
   using synthetic generation + danielrosehill dataset + ASR output pairing

2. FORMAT: Convert to chat-format JSONL with system prompt

3. FINE-TUNE: LoRA on full-precision Qwen3-0.6B (or Qwen3.5-0.8B)
   - Use MLX LoRA on Mac, or Unsloth on GPU
   - Use --mask-prompt to only train on the cleanup output
   - Monitor validation loss carefully

4. FUSE: Merge LoRA adapters into base model → full-precision merged model

5. QUANTIZE: Convert merged model to 4-bit MLX format

6. DEPLOY: Upload to HuggingFace, load in Swift app via mlx-swift-lm

7. ITERATE: Evaluate, collect real-world failure cases, add to training data, repeat
```
