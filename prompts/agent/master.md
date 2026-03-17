# Master Orchestration — Agent-Based Synthetic Data Generation

## Overview

This prompt orchestrates parallel generation of synthetic transcript-cleanup training
data across all categories. It fans out one task per category (running in parallel
where possible), then runs a final cross-category review.

## Parameters

The user should specify these when invoking. Use defaults if not provided.

| Parameter | Default | Description |
|-----------|---------|-------------|
| generation_model | (current model) | Model for generating data (e.g., `claude-opus-4-6`, `gpt-5.4`) |
| validation_model | (a different model) | Model for validating data — should differ from generation_model to avoid self-preference bias |
| target_per_category | 200 | Number of pairs to generate per category |
| batch_size | 50 | Pairs per generation-validation cycle |
| category_type | all | Which category types to run: `core`, `domain_specific`, or `all` |
| categories | (from type) | Comma-separated list to override type-based selection with specific categories |

**Choosing models**: Using different model families for generation vs. validation
produces the best results (e.g., Claude for generation, GPT for validation — or vice
versa). This avoids the generator judging its own work favorably.

---

## Execution Steps

### Step 1: Discover Categories

List all `.md` files in `prompts/agent/categories/`. Each file defines one category.
The category name is the filename without extension (e.g., `meeting_notes.md` →
`meeting_notes`).

Each category file contains a **Type** line (e.g., `> **Type**: Core` or
`> **Type**: Domain Specific`). Filter categories based on the `category_type`
parameter:

- `core` → only categories marked `Core` (conservative cleanup, no restructuring)
- `domain_specific` → only categories marked `Domain Specific` (restructured output)
- `all` → all categories regardless of type

If the user specified specific categories via the `categories` parameter, use those
instead (overrides `category_type`).

**Current category types:**

| Type | Categories |
|------|-----------|
| Core | casual_conversation, self_corrections_heavy, technical_code, financial_business, academic_research, legal_contract |
| Domain Specific | meeting_notes, email_professional, recipe_cooking, shopping_lists, medical_clinical, creative_writing |

Report: "Found N categories (type: [type]): [list]"

### Step 2: Ensure Output Directory

Create `data/prepared/` if it doesn't exist.

### Step 3: Fan Out — One Task Per Category

For EACH category, start a generation task using the **generation_model**. If your
agent system supports parallel execution (background agents, concurrent tasks, etc.),
launch all categories simultaneously. Otherwise, process them sequentially.

Each category task should receive:

```
Read prompts/agent/categories/[category_name].md and follow its instructions.

Parameters for this run:
- target_count: [target_per_category]
- batch_size: [batch_size]
- validation_model: [validation_model]
- output_file: data/prepared/[category_name].jsonl
```

The category prompt will direct the agent to also read `prompts/agent/generate.md`
for base instructions and `prompts/agent/validate.md` for validation criteria.

### Step 4: Wait for Completion

Wait for all category tasks to finish. As each completes, note:
- Category name
- Success/failure
- Number of pairs generated (from the task's report)

### Step 5: Final Review

Once all category tasks have completed:

1. Read `prompts/agent/final_review.md`
2. Start a review task (using either model) that follows those instructions to do a
   cross-category quality check across all files in `data/prepared/`
3. Wait for the review to complete

### Step 6: Report to User

Summarize the full run:

```
## Generation Complete

### Per-Category Results
| Category | Target | Generated | Status |
|----------|--------|-----------|--------|
| casual_conversation | 200 | 200 | Done |
| email_professional | 200 | 198 | Done |
| ... | ... | ... | ... |

### Total: XXXX pairs across N categories
### Output: data/prepared/*.jsonl

### Final Review: [PASS/NEEDS_ATTENTION/FAIL]
[Summary from final review agent]
```

If any categories failed or the final review flagged issues, highlight them clearly
and suggest re-running just those categories individually.

---

## Error Handling

- If a category task fails: note it, continue with others, report at the end
- If most categories fail: stop and report the common error (likely API/model issue)
- If the final review flags serious issues in specific categories: suggest re-running
  just those categories

## Resume Behavior

Each category task independently checks its output file for existing pairs and only
generates the remaining count. So re-running the master prompt after a partial failure
will skip already-completed categories and resume incomplete ones.
