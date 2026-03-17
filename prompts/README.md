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
    generate.md                  Base generation instructions for category agents
    validate.md                  Validation criteria for evaluating generated pairs
    final_review.md              Cross-category quality check
    categories/
      casual_conversation.md     12 category-specific prompt files,
      email_professional.md      each with description, speech patterns,
      technical_code.md          example pairs, and pitfalls
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

## Quick Start

### Generate a single category

Tell your coding agent:

```
Read prompts/agent/categories/meeting_notes.md and follow its instructions.
Generate 200 pairs using GPT 5.4 for validation.
Output to data/prepared/meeting_notes.jsonl.
```

The agent reads the category file, which directs it to read `generate.md` for base
instructions. It generates data in batches, validates each batch using the specified
validation model, and loops until the target is reached.

### Generate all categories in parallel

```
Read prompts/agent/master.md and execute.
Generation model: claude-opus-4-6. Validation model: gpt-5.4.
Target per category: 200.
```

The master prompt fans out one task per category (in parallel if the agent supports
it), waits for all to complete, then runs a final cross-category quality review.

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
4. **corrections_applied** — Are all cleanup rules applied correctly?

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
Copy the files into `data/raw/` — the format is identical and step 4 already loads
all `data/raw/*.jsonl`.

```bash
cp data/prepared/*.jsonl data/raw/
```

Native `data/prepared/` support is tracked in `docs/next_steps.md` (Phase 2).

## Resume Behavior

Each category task checks its output file before starting. If the file exists with N
pairs already generated, it only generates the remaining (target - N) pairs. Re-running
after a partial failure or interruption automatically resumes.

## Tips

- **Start small**: Test with one category and 20-50 pairs before running the full
  master orchestration.
- **Check quality early**: Read the first batch of output manually. If the transcripts
  don't sound like real speech, adjust your instructions.
- **Different models per category**: Run categories individually with different models.
  Use your best model for hard categories (creative_writing, self_corrections_heavy)
  and a cheaper one for easier ones.
- **Validation is optional for iteration**: For quick drafts, skip validation and just
  rely on step 4's built-in quality checks. Use validation for production datasets.
