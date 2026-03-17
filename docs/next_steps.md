# Next Steps: Dataset Preparation & Pipeline Improvements

## Problem Statements

### 1. Script-based synthetic generation produces mediocre quality

Steps 3 and 3b (`03_generate_synthetic.py` and `03b_validate_synthetic.py`) use a
fire-and-forget approach: a single large prompt produces a batch of 25 pairs, basic
regex validation filters obviously broken ones, and the batch moves on. There is no
iteration, no multi-turn refinement, and no ability to course-correct based on earlier
output. The result is that many generated transcripts sound like **written text with
"um" sprinkled in**, rather than realistic speech. Step 3b catches some of these with
an LLM judge, but the fix is to reject and retry — a wasteful loop.

A conversational coding agent (like Claude Code) produces noticeably better data
because it can:
- **Self-correct across turns** — see its own output, spot patterns, and adjust
- **Follow complex instructions more faithfully** — dense single-shot prompts with
  200+ lines of guidance are partially ignored; dialogue enforces compliance
- **Validate as it generates** — the agent is both generator and first-pass reviewer,
  so step 3b becomes redundant for agent-generated data
- **Remember context** — avoids repetitive patterns across batches

### 2. Generated datasets can't be shared or reused

Once generated, synthetic datasets exist only in `data/synthetic/` on the local
machine. There is no mechanism to upload, version, or share them. Other contributors
or machines must regenerate from scratch — wasting time and API costs.

### 3. Pipeline can't use externally prepared datasets

The pipeline assumes data comes from two sources: HF datasets (step 2 → `data/raw/`)
and script-generated synthetic data (step 3 → `data/synthetic/`). There is no way to
feed in agent-generated data, manually curated data, or datasets downloaded from a
shared repository without manually placing files in the right directories and hoping
step 4 picks them up correctly.

### 4. Steps 3 and 3b are extremely slow

Both steps make all API calls **strictly sequentially**:

- **Step 3**: One batch at a time with `time.sleep(0.5)` between batches. For 5000
  samples at batch_size 25 = 200 batches. At ~5-10s per API call + 0.5s delay ≈
  25-90+ minutes.
- **Step 3b**: One pair at a time with `time.sleep(0.2)`. At 12% sample rate of 5000 =
  600 pairs, each taking ~3-5s ≈ 35-50 minutes.
- **Combined**: 1-3+ hours for a full run, with no parallelism.

---

## Solutions

### A. Agent-Based Parallel Data Generation

Replace (or supplement) the script-based pipeline with a **prompt-driven agent
system** that leverages coding agents' native parallelism and multi-turn reasoning.

#### Architecture

```
prompts/agent/
  master.md              Master orchestration — fans out all categories in parallel
  generate.md            Base generation instructions (read by each category agent)
  validate.md            Validation criteria (read by validation sub-agents)
  final_review.md        Cross-category quality check after all generation completes
  categories/
    casual_conversation.md
    email_professional.md
    technical_code.md
    medical_clinical.md
    legal_contract.md
    meeting_notes.md
    recipe_cooking.md
    academic_research.md
    creative_writing.md
    financial_business.md
    shopping_lists.md
    self_corrections_heavy.md
```

#### How It Works

**Two usage modes:**

**Individual category mode** — run one category at a time:
```
User → Agent: "Read prompts/agent/categories/meeting_notes.md and follow its
               instructions. Generate 200 pairs. Use GPT 5.4 for validation."
```
The agent reads the category file, which directs it to read `generate.md` for
base instructions. It generates data, validates each batch using the specified
validation model, loops until done. Output: `data/prepared/meeting_notes.jsonl`.

**Master mode** — fan out all categories in parallel:
```
User → Agent: "Read prompts/agent/master.md and execute.
               Generation model: claude-opus-4-6, validation model: gpt-5.4,
               target per category: 200."
```
The master reads all category files and starts one task per category (in parallel
if the agent system supports it). Each task self-bootstraps by reading its own
file + base instructions. After all complete, a final review checks quality.

#### Task Structure

```
Master Task (user's session)
  ├── reads master.md
  ├── fans out 12 category tasks (parallel where supported)
  │
Category Task (one per category)
  │   ├── reads category .md → reads generate.md
  │   ├── generates batch of 25 pairs (using generation_model)
  │   ├── sends batch to validation_model for evaluation
  │   ├── processes results
  │   ├── fixes/regenerates failures
  │   └── loops until target reached
  │
Final Review Task
      ├── reads final_review.md
      ├── samples across all category output files
      └── reports cross-category quality assessment
```

#### Why This Is Better Than Scripts

| Aspect | Script (step 3) | Agent system |
|--------|-----------------|--------------|
| Quality | Single-shot prompts, basic regex validation | Multi-turn generation, LLM validation per batch |
| Parallelism | None (sequential batches) | 12 categories run simultaneously |
| Self-correction | None (generate → reject → retry) | Agent sees failures and adjusts |
| Validation | Separate step (3b), different model, hours later | Inline per batch, immediate feedback loop |
| Resumability | Counts existing files | Same — reads existing output file |
| Model flexibility | Fixed in config.yaml | Specified per invocation |
| Cost | ~$50-80 for 5000 samples + ~$20 for validation | Comparable or lower (fewer rejects, no step 3b) |

#### Model Selection

The user specifies models at invocation time:
- **Generation model**: The model that creates the pairs (e.g., `claude-opus-4-6`,
  `gpt-5.4`, or any model the agent system supports)
- **Validation model**: The model that evaluates pairs (should be a different model
  family to avoid self-preference bias)

The prompts are agent-system-agnostic — they work with Claude Code, GitHub Copilot
CLI, Cursor, or any agent that can read files, generate text, and write output.

---

### B. Pipeline Integration for Pre-Prepared Data

Add `data/prepared/` as a first-class data source alongside `data/raw/` and
`data/synthetic/`.

#### Config Addition

```yaml
dataset:
  # Pre-prepared datasets (agent-generated, manually curated, or downloaded)
  prepared:
    enabled: true
    paths:
      - "data/prepared/"            # Local directory with JSONL files
```

#### Code Changes

1. **Step 4** (`04_prepare_data.py`): Add `load_prepared_data()` that reads
   `data/prepared/*.jsonl` in the same format as raw data (`{"input": "...",
   "output": "..."}`). Merge alongside raw and synthetic data during the combine
   phase.

2. **`run_pipeline.py`**: When `prepared.enabled` is true and `synthetic.enabled`
   is false, skip steps 3/3b and adjust step 4's dependencies accordingly.

#### Workflow Options

| Workflow | Steps Run | Data Sources |
|----------|-----------|--------------|
| Full pipeline | 2 → 3 → 3b → 4 → 5 → 6+ | raw + synthetic |
| Agent + raw | 2 → 4 → 5 → 6+ | raw + prepared |
| Prepared only | 4 → 5 → 6+ | prepared only |
| Mixed | 2 → 3 → 3b → 4 → 5 → 6+ | raw + synthetic + prepared |

---

### C. Dataset Sharing via Hugging Face

Upload prepared datasets to Hugging Face for versioning, sharing, and easy
consumption by the existing pipeline.

#### Upload Process

```bash
pip install huggingface_hub
huggingface-cli login

python scripts/upload_dataset.py \
  --source data/prepared/ \
  --repo shantanugoel/aawaaz-transcript-cleanup \
  --description "Agent-generated transcript-cleanup training pairs"
```

Or extend step 10 to also handle dataset uploads.

#### Consumption

Once uploaded, add to `config.yaml` as an HF dataset source — step 2 downloads it
automatically:

```yaml
dataset:
  hf_datasets:
    - repo: "shantanugoel/aawaaz-transcript-cleanup"
      input_col: "input"
      output_col: "output"
      enabled: true
```

---

### D. Script Parallelization (if keeping scripts)

If the script path (steps 3/3b) is still used for bulk generation, parallelize
API calls using `concurrent.futures.ThreadPoolExecutor`.

#### Changes

1. Add `max_concurrency` to config:
   ```yaml
   synthetic:
     max_concurrency: 10    # Parallel API calls
   ```

2. In `03_generate_synthetic.py`: Replace the sequential batch loop with a thread
   pool. Each worker handles one batch (generate → validate → append).

3. In `03b_validate_synthetic.py`: Replace the sequential pair-by-pair evaluation
   with parallel judge calls.

4. File writes protected by a threading lock.

#### Expected Speedup

- Step 3: 200 sequential batches → ~20 rounds of 10 parallel = **8-10x faster**
- Step 3b: 600 sequential calls → ~60 rounds of 10 parallel = **8-10x faster**
- Combined: 1-3 hours → **10-20 minutes**

---

## Implementation Tasks

### Phase 1: Agent prompt system (enables agent-based generation now)
- [x] Create `prompts/agent/generate.md` — base generation instructions
- [x] Create `prompts/agent/validate.md` — validation criteria for sub-agents
- [x] Create `prompts/agent/final_review.md` — cross-category review
- [x] Create `prompts/agent/master.md` — orchestration prompt
- [x] Create 12 category files in `prompts/agent/categories/`
- [ ] Test with a single category end-to-end
- [ ] Test master mode with all categories in parallel

### Phase 2: Pipeline integration (enables prepared data in pipeline)
- [ ] Add `dataset.prepared` section to `config.yaml`
- [ ] Add `load_prepared_data()` to `04_prepare_data.py`
- [ ] Update `run_pipeline.py` to handle skip-synthetic workflow
- [ ] Test: agent-generated data → step 4 → step 6

### Phase 3: Dataset sharing (enables reuse across machines)
- [ ] Create `scripts/upload_dataset.py` (or extend step 10)
- [ ] Upload agent-generated dataset to HF
- [ ] Test: HF download via step 2 → step 4

### Phase 4: Script parallelization (optional, improves script path)
- [ ] Add `max_concurrency` to config
- [ ] Parallelize `03_generate_synthetic.py` with ThreadPoolExecutor
- [ ] Parallelize `03b_validate_synthetic.py`
- [ ] Test with rate-limit-sensitive providers (add backoff on 429s)
