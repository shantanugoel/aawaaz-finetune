# Prompts — Aawaaz Data Generation

This directory contains the prompt system for generating synthetic training data
using coding agents. The prompts are agent-system-agnostic — they work with any
agent that can read files, generate text, and write output (Claude Code, GitHub
Copilot CLI, Cursor, Aider, etc.).

## Directory Structure

```
prompts/
  system_prompt.txt              The inference-time system prompt for Aawaaz
  README.md                      This file
  agent/
    master.md                    Orchestration — fan out all categories in parallel
    generate.md                  Base generation instructions (voice preservation,
                                 realism rules, output format, workflow)
    validate.md                  Validation criteria (4 criteria including voice
                                 preservation check)
    final_review.md              Cross-category quality check
    categories/
      casual_conversation.md     Core — casual voice messages, daily life updates
      self_corrections_heavy.md  Core — heavy self-correction transcripts
      technical_code.md          Core — dictated code, CLI commands, tech docs
      financial_business.md      Core — financial reports, earnings, budgets
      academic_research.md       Core — research dictation, stats, citations
      legal_contract.md          Core — legal dictation, contracts, court notes
      meeting_notes.md           Domain Specific — structured meeting notes
      email_professional.md      Domain Specific — dictated emails
      recipe_cooking.md          Domain Specific — recipes with ingredients + steps
      shopping_lists.md          Domain Specific — bulleted lists
      medical_clinical.md        Domain Specific — clinical notes with vitals
      creative_writing.md        Domain Specific — literary prose, poetry
```

### Category Types

Each category file is tagged with a type (`> **Type**: Core` or
`> **Type**: Domain Specific`). This controls which categories are generated
when filtering by type — it does not affect training behavior.

- **Core**: Conservative cleanup — removes fillers, fixes grammar, preserves the
  speaker's structure and voice
- **Domain Specific**: Reformats output into a domain-appropriate structure (bullets,
  headers, etc.) while preserving all content

## Quick Start

### Parameters

All parameters have sensible defaults. Specify only what you want to override.

| Parameter | Default | Description |
|-----------|---------|-------------|
| generation_model | (current model) | Model for generating data |
| validation_model | (a different model) | Model for validating — should differ from generation model |
| target_per_category | 200 | Number of pairs to generate per category |
| batch_size | 50 | Pairs per generation-validation cycle |
| category_type | all | Which types to run: `core`, `domain_specific`, or `all` |
| categories | (from type) | Comma-separated list — overrides category_type |

### Generate a single category

```
Read prompts/agent/categories/meeting_notes.md and follow its instructions.
Target: 2000 pairs. Batch size: 50. Validation model: gpt-5.4.
Output to data/prepared/meeting_notes.jsonl.
```

The agent reads the category file, which directs it to read `generate.md` for base
instructions. It generates data in batches, validates each batch using the specified
validation model, and loops until the target is reached.

### Generate all categories (recommended for training)

```
Read prompts/agent/master.md and execute.
Generation model: claude-opus-4-6. Validation model: gpt-5.4.
Target per category: 2000. Batch size: 50.
```

The master prompt fans out one task per category (in parallel if the agent supports
it), waits for all to complete, then runs a final cross-category quality review.

### Generate only Core or Domain Specific categories

```
Read prompts/agent/master.md and execute.
category_type: core
Target per category: 2000.
```

or

```
Read prompts/agent/master.md and execute.
category_type: domain_specific
Target per category: 2000.
```

### Generate specific categories only

```
Read prompts/agent/master.md and execute.
categories: meeting_notes, self_corrections_heavy, financial_business
Target per category: 2000.
```

The `categories` parameter overrides `category_type` when both are specified.

## Context Freshness

**Important for large generation runs:** Each category should be generated in its
own agent session (a fresh context window). This avoids context overload and
compaction issues that degrade quality in later batches.

When using the master prompt with agents that support parallel execution:
- Each category task is launched as a separate sub-agent (fresh context)
- The master agent only orchestrates — it doesn't generate data itself

When running categories individually:
- Start a new agent session for each category
- Don't generate multiple categories in the same session

Within a single category, the batch loop (generate → validate → fix → repeat) runs
in one session. This is fine — individual batches are small (50 pairs) and the
context stays manageable. The resume mechanism means you can also split a large
category across sessions if needed.

## Model Selection

You specify two models when invoking:

**Generation model** — creates the training pairs. This is the model doing the
creative work, so quality matters. Use the best model you have access to.

**Validation model** — evaluates generated pairs for quality. Should ideally be a
**different model family** from the generation model to avoid self-preference bias
(the generator shouldn't judge its own work).

### Recommended combinations

| Use case | Generation | Validation | Notes |
|----------|-----------|------------|-------|
| Best quality | Claude Opus 4.6 | GPT 5.4 | Different families, both top-tier |
| Good quality | Claude Sonnet 4.6 | GPT 5.4 | Cheaper generation, strong validation |
| Budget | Claude Sonnet 4.6 | Claude Haiku 4.5 | All Claude, lowest cost |
| Reversed | GPT 5.4 | Claude Opus 4.6 | Generation with GPT, Claude validates |

Any model your agent system supports can be used for either role. The prompts don't
assume any specific provider.

## How Validation Works

After each batch of generated pairs, the workflow sends them to the validation model
for evaluation on 4 criteria:

1. **input_realistic** — Does the transcript sound like real speech?
2. **content_preserved** — Is all substantive content kept in the output?
3. **no_hallucination** — Does the output add anything not in the input?
4. **corrections_applied** — Are all cleanup rules applied correctly? This includes
   a **voice preservation check** — outputs that formalize or reword the speaker's
   personality (e.g., "it's been studied to death" → "has been extensively studied")
   are flagged as failures.

The validation model returns a JSON evaluation. Passing pairs are appended to the
output file; failing pairs are regenerated with the failure reasons as guidance.

The criteria and expected JSON schema are defined in `prompts/agent/validate.md`.

### Standalone validation script

For batch validation outside an agent context (e.g., validating existing files), a
standalone script is also available:

```bash
# Uses config.yaml defaults (dataset.synthetic.validation section)
python scripts/validate_batch.py --input data/prepared/meeting_notes.jsonl

# Override model on command line
python scripts/validate_batch.py --input batch.jsonl \
    --provider openai_compatible \
    --model openai/gpt-5.4 \
    --api-key-env OPENROUTER_API_KEY \
    --base-url https://openrouter.ai/api/v1
```

This script supports any model via the pipeline's `llm_client.py` infrastructure.

## Output Format

All generated data is written to `data/prepared/` as JSONL:

```
data/prepared/
  casual_conversation.jsonl
  email_professional.jsonl
  technical_code.jsonl
  ...
```

Each line:
```json
{"input": "messy transcript...", "output": "cleaned text..."}
```

## Feeding Output into the Pipeline

The generated data in `data/prepared/` is designed to be consumed by step 4
(`04_prepare_data.py`), which combines, validates, deduplicates, formats as chat
messages, and splits into train/valid/test sets.

**Current workaround** (until `data/prepared/` is natively supported by step 4):
Copy the files into `data/synthetic/` — the format is identical and step 4 already
loads all `data/synthetic/*.jsonl`. Rename files to match the expected prefix:

```bash
for f in data/prepared/*.jsonl; do
  cp "$f" "data/synthetic/synthetic_$(basename "$f")"
done
```

## Resume Behavior

Each category task checks its output file before starting. If the file exists with N
pairs already generated, it only generates the remaining (target - N) pairs. Re-running
after a partial failure or interruption automatically resumes.

## Tips

- **Start small**: Test with one category and 20-50 pairs before running the full
  master orchestration.
- **Check quality early**: Read the first batch of output manually. If the transcripts
  don't sound like real speech, adjust your instructions.
- **Fresh context per category**: For large runs (1000+ per category), start a new
  agent session for each category to avoid context degradation.
- **Different models per category**: Run categories individually with different models.
  Use your best model for hard categories (creative_writing, self_corrections_heavy)
  and a cheaper one for easier ones.
- **Validation is optional for iteration**: For quick drafts, skip validation and just
  rely on step 4's built-in quality checks. Use validation for production datasets.
