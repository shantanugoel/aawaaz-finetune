# Final Review — Cross-Category Quality Check

## Your Role

You are performing a final quality review across ALL categories of generated synthetic
data. Your job is to catch systematic issues that per-category validation might miss:
format inconsistencies, repetitive patterns, distribution imbalances, or quality
problems that only become visible when looking at the full dataset.

---

## Steps

### 1. Inventory

List all `.jsonl` files in `data/prepared/`. For each file:
- Count total pairs (lines)
- Note the file size
- Flag any missing categories or suspiciously small files

### 2. Sample and Evaluate

For each file, read a random sample of **5 pairs** (or all if fewer than 5). For each
sampled pair, check:

**Format:**
- Valid JSON with `input` and `output` string fields
- No extra fields, no malformed JSON
- Proper UTF-8, no null bytes or mojibake

**Input realism (quick check):**
- Does it sound like speech, not writing?
- Are fillers placed naturally?
- Are numbers in spoken form?

**Output quality (quick check):**
- Are all fillers removed?
- Are all facts from the input preserved?
- Is formatting appropriate for the category?
- No hallucinated content?

### 3. Cross-Category Checks

Look across all categories for:

**Repetitive patterns:**
- Same names, places, or numbers appearing across categories
- Similar sentence structures reused ("so I was talking to [name] and...")
- Templated feel — all pairs following the same pattern within a category

**Distribution:**
- Are file sizes roughly proportional to expectations?
- Any category dramatically larger or smaller than others?

**Consistency:**
- Same cleanup rules applied across categories (fillers removed, numbers formatted)
- No category applying different formatting standards than others

### 4. Spot Check — Deep Evaluation

Pick **3 random pairs from 3 different categories** (9 total). For each, do a thorough
evaluation:
- Read the input aloud (mentally). Does it genuinely sound like speech?
- Compare input to output word by word. Is every fact preserved?
- Check for any hallucinated content in the output.
- Verify all cleanup rules are applied.

### 5. Report

Return a structured summary:

```
## Final Review Report

### Per-Category Summary
| Category | Pairs | Status | Notes |
|----------|-------|--------|-------|
| casual_conversation | 200 | OK | |
| email_professional | 200 | OK | |
| ... | ... | ... | ... |

### Overall Assessment
[PASS / NEEDS_ATTENTION / FAIL]

### Issues Found
- [List any specific issues with examples]

### Recommendations
- [Any suggestions for improvement]

### Cross-Category Observations
- [Notes on diversity, consistency, patterns]
```

If the assessment is **NEEDS_ATTENTION** or **FAIL**, be specific about which
categories have problems and what the problems are, so the user can re-run those
categories.
