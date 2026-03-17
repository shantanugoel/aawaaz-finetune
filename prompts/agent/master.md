# Master Orchestration — Agent-Based Synthetic Data Generation

## Overview

This prompt orchestrates parallel generation of synthetic transcript-cleanup training
data across all categories. It fans out one agent per category (all running
simultaneously), then runs a final cross-category review.

## Parameters

The user should specify these when invoking. Use defaults if not provided.

| Parameter | Default | Description |
|-----------|---------|-------------|
| generation_model | (inherit) | Claude model for category agents: `opus`, `sonnet`, or `haiku` |
| validation_model | haiku | Claude model for validation sub-agents |
| target_per_category | 200 | Number of pairs to generate per category |
| batch_size | 25 | Pairs per generation-validation cycle |
| categories | all | Comma-separated list, or "all" for every category file |

---

## Execution Steps

### Step 1: Discover Categories

List all `.md` files in `prompts/agent/categories/`. Each file defines one category.
The category name is the filename without extension (e.g., `meeting_notes.md` →
`meeting_notes`).

If the user specified specific categories, filter to only those. Otherwise use all.

Report: "Found N categories: [list]"

### Step 2: Ensure Output Directory

```bash
mkdir -p data/prepared
```

### Step 3: Fan Out — Launch One Agent Per Category

For EACH category, launch a **background** agent. All agents should be launched in a
**single message** (parallel background agents).

For each category agent:
- **model**: the `generation_model` parameter
- **run_in_background**: true
- **description**: "Generate [category_name] data"
- **prompt**: Compose as follows:

```
Read prompts/agent/categories/[category_name].md and follow its instructions.

Parameters for this run:
- target_count: [target_per_category]
- batch_size: [batch_size]
- validation_model: [validation_model]
- output_file: data/prepared/[category_name].jsonl
```

Launch ALL category agents in a single message to maximize parallelism.

### Step 4: Wait for Completion

You will be automatically notified as each background agent completes. Do NOT poll
or sleep. Continue to wait until ALL agents have reported back.

As each completes, note:
- Category name
- Success/failure
- Number of pairs generated (from the agent's report)

### Step 5: Final Review

Once all category agents have completed:

1. Read `prompts/agent/final_review.md`
2. Launch a **foreground** agent with:
   - **model**: the `validation_model` parameter
   - **description**: "Final review of all categories"
   - **prompt**: The content of `final_review.md`

Wait for the review to complete.

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
and suggest next steps (e.g., "Re-run casual_conversation with: Read
prompts/agent/categories/casual_conversation.md ...").

---

## Error Handling

- If a category agent fails: note it, continue with others, report at the end
- If most categories fail: stop and report the common error (likely API/model issue)
- If the final review flags serious issues in specific categories: suggest re-running
  just those categories individually

## Resume Behavior

Each category agent independently checks its output file for existing pairs and only
generates the remaining count. So re-running the master prompt after a partial failure
will skip already-completed categories and resume incomplete ones.
