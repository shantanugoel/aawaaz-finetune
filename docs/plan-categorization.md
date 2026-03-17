# Category Type System & Training Plan

## The Problem

The 12 data generation categories produce fundamentally different output styles:

- **Conservative cleanup** (e.g., casual_conversation): Removes fillers, fixes grammar,
  preserves the speaker's structure and words. Output looks like a cleaned version of
  the input.
- **Structured reformatting** (e.g., meeting_notes): Transforms a rambling transcript
  into a formatted document with headers, bullet points, and structured sections.
  Output preserves all content but reorganizes the structure.

Training a single model on both styles risks the model confusing when to restructure
vs. when to just clean. However, with strong voice-preservation rules in the training
data and clear content signals in the inputs, the model can learn to distinguish these
modes from context alone — without explicit conditioning tags.

---

## The Type System

Each category file in `prompts/agent/categories/` is tagged with a type. This is used
for **orchestration** (selectively generating categories) — not as a training signal.

### Core (conservative cleanup)

These categories all share the same behavior: remove fillers, fix grammar, format
numbers — but preserve the speaker's structure and voice. The output reads like a
polished version of what the speaker said, not a reformatted document.

| Category | Why it's Core |
|----------|--------------|
| casual_conversation | Casual tone preserved, just cleaned |
| self_corrections_heavy | Corrections collapsed, prose preserved |
| technical_code | Code terms formatted, explanations preserved |
| financial_business | Numbers formatted, speaker's phrasing preserved |
| academic_research | Stats formatted, speaker's voice preserved (not formalized) |
| legal_contract | Legal terms capitalized, structure preserved |

### Domain Specific (structured output)

These categories transform the input's structure into a domain-appropriate format.
The output is recognizably different in structure from the input, but preserves all
substantive content — this is reformatting, not summarization.

| Category | Output format |
|----------|--------------|
| meeting_notes | Headers, bold sections, bullet-point action items |
| email_professional | Greeting / body / closing email structure |
| recipe_cooking | Title, ingredients list, numbered instructions |
| shopping_lists | Bulleted item lists |
| medical_clinical | Structured clinical note with vitals, assessment |
| creative_writing | Poetry line breaks, literary prose formatting |

---

## Training Approach

### Current Plan: Train on All Categories

**Goal:** Train a single model on all 12 categories (Core + Domain Specific) without
explicit context tags. The model learns when to restructure vs. when to just clean
from the content of the input itself.

**Why this works:**
- Strong voice-preservation rules in `generate.md` and `validate.md` ensure all
  categories teach "clean the fillers, not the personality"
- Domain Specific inputs naturally contain content signals ("okay meeting notes for
  the product sync...", "dear mr thompson comma...", "grocery list...") that the model
  can learn to detect
- More diverse training data (24,000 samples across 12 categories) produces a more
  robust model than restricting to 6 categories

**Generation:**
```
category_type=all, target_per_category=2000
```
This generates data for all 12 categories (24,000 total pairs).

**Training data format** (unchanged):
```json
{"input": "so um I was talking to sarah and she said...", "output": "I was talking to Sarah and she said..."}
```

No context tags — the model sees raw transcript → cleaned output.

**Inference in Aawaaz** (unchanged): The model receives raw text and cleans it up.
The system prompt covers all cleanup rules. The model infers appropriate formatting
from the content itself.

---

## Key Principle: Voice Preservation

Across ALL categories (Core and Domain Specific), one rule is absolute: **preserve
the speaker's voice**. Cleanup means removing fillers and fixing grammar, not
rewriting the speaker's word choices.

- "it's been studied to death" stays as "it's been studied to death" (not
  "has been extensively studied")
- "the pad thai was insane" stays as "the pad thai was insane" (not "the pad thai
  was excellent")
- "we still get it wrong" stays (not "errors persist")

This rule is enforced in:
- `prompts/agent/generate.md` — concrete anti-patterns with examples
- `prompts/agent/validate.md` — explicit failure criterion for voice formalization

The only categories that change structure (not voice) are Domain Specific ones, and
only when the content naturally calls for it.

---

## Orchestrator Usage

The `master.md` orchestrator accepts a `category_type` parameter for convenience:

```
# Generate all data (recommended)
category_type=all, target_per_category=2000

# Generate only Core data (if you want to test conservative cleanup alone)
category_type=core, target_per_category=2000

# Generate only Domain Specific data (if you want to supplement structured examples)
category_type=domain_specific, target_per_category=2000
```

Categories self-identify their type via a `> **Type**: Core` or
`> **Type**: Domain Specific` tag in their `.md` file. The orchestrator reads this
to filter which categories to run.

---

## Fallback: Context Tags (If Needed)

If evaluation shows the model confuses cleanup modes (e.g., restructures casual
conversation into bullet points, or fails to structure meeting notes), a context-tag
system can be added as a follow-up:

**Training data format with tags:**
```json
{"input": "<context>core</context>\nso um I was talking to sarah...", "output": "I was talking to Sarah..."}
{"input": "<context>meeting_notes</context>\nokay meeting notes for...", "output": "# Meeting Notes\n\n..."}
```

**Inference:** Aawaaz prepends `<context>core</context>` by default, with
app-specific overrides (e.g., Mail.app → `email_professional`).

**Implementation:** Requires changes to `04_prepare_data.py` to prepend tags based
on category type, and to Aawaaz's `LocalLLMProcessor.swift` to prepend the tag at
inference time.

This is documented here for reference but should only be implemented if the
all-categories-no-tags approach shows clear mode confusion in evaluation.
