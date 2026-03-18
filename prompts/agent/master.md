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
| orchestrator_model | claude-sonnet-4.6 | Model for category orchestrator agents — these are loop controllers that don't generate data, so a capable but cheaper model works well |
| target_per_category | 200 | Number of pairs to generate per category |
| batch_size | 50 | Pairs per generation-validation cycle |
| max_concurrent | 3 | Maximum number of category orchestrators running simultaneously — controls rate limit pressure |
| category_type | all | Which category types to run: `core`, `domain_specific`, or `all` |
| categories | (from type) | Comma-separated list to override type-based selection with specific categories |

**Choosing models**: Using different model families for generation vs. validation
produces the best results (e.g., Claude for generation, GPT for validation — or vice
versa). This avoids the generator judging its own work favorably. The orchestrator
model can be a cheaper model (like Sonnet 4.6) since it only manages the batch loop —
it never generates or evaluates data itself.

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

### Step 3: Pre-Check — Skip Completed Categories

Before launching any orchestrators, check each category's output file to avoid
unnecessary work:

For each category in the list:
1. Check if `data/prepared/[category_name].jsonl` exists
2. If yes, count the lines (each line = one pair)
3. If line count >= `target_per_category`, mark this category as **already complete**

Partition the category list into:
- **complete**: already at or above target — these will be skipped entirely
- **pending**: need work (file missing, partially done, or empty)

Report:
```
Pre-check results:
- Complete (skipped): [list or "none"]
- Pending (will generate): [list]
```

If all categories are complete, skip to Step 7 (report).

### Step 4: Queue and Launch — Concurrent Category Orchestrators

Use a queue to control how many category orchestrators run simultaneously, respecting
the `max_concurrent` parameter. This prevents hitting API rate limits when generation
and validation sub-agents are making model calls.

**Queue logic:**

1. Place all **pending** categories into a queue (ordered as discovered)
2. Launch up to `max_concurrent` category orchestrators from the front of the queue
3. As each orchestrator **completes** (success or failure):
   - Note its result (category name, success/failure, pairs generated)
   - If the queue is not empty, launch the next category from the queue
4. Repeat until the queue is empty and all running orchestrators have finished

Each category orchestrator task should receive:

```
Read prompts/agent/generate.md and prompts/agent/categories/[category_name].md.
Follow the orchestrator workflow defined in generate.md.

Parameters for this run:
- target_count: [target_per_category]
- batch_size: [batch_size]
- generation_model: [generation_model]
- validation_model: [validation_model]
- output_file: data/prepared/[category_name].jsonl
```

**Important**: The category orchestrator does NOT generate data itself. It runs the
batch loop and spawns fresh sub-agents (using the generation_model and
validation_model) for each batch. This keeps every batch in a clean context window.

The category orchestrator reads `generate.md` for its workflow and spawns sub-agents
that also read `generate.md` (for quality guidance) and `validate.md` (for evaluation
criteria).

**Progress reporting**: As orchestrators complete and new ones launch, report:
```
[category_name] complete (N pairs). Queue: M remaining, K running.
Launching: [next_category_name]
```

### Step 5: Wait for All Orchestrators

Wait for all category orchestrators to finish. Track the final results for each:
- Category name
- Success/failure
- Number of pairs generated (from the task's report)

### Step 6: Final Review

Once all category tasks have completed:

1. Read `prompts/agent/final_review.md`
2. Start a review task (using either model) that follows those instructions to do a
   cross-category quality check across all files in `data/prepared/`
3. Wait for the review to complete

### Step 7: Report to User

Summarize the full run:

```
## Generation Complete

### Per-Category Results
| Category | Target | Generated | Status |
|----------|--------|-----------|--------|
| casual_conversation | 200 | 200 | Done |
| email_professional | 200 | 198 | Done |
| shopping_lists | 200 | 200 | Skipped (already complete) |
| ... | ... | ... | ... |

### Total: XXXX pairs across N categories (M skipped, K generated this run)
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

The master prompt handles resume at two levels:

1. **Category-level skip** (Step 3): Before launching any orchestrators, the master
   checks each output file. Categories already at target are skipped entirely — no
   orchestrator agent is spawned for them.

2. **Batch-level resume** (in generate.md Step 0): Each category orchestrator checks
   its output file for existing pairs and only generates the remaining count.

So re-running after a partial failure will skip completed categories and resume
incomplete ones from where they left off.
