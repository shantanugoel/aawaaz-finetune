# Category: Legal / Contract

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions.

> **Type**: Core

## Description

Legal professionals dictating contract clauses, legal correspondence, court notes,
or case summaries. Formal legal language spoken conversationally — they know the terms
but are composing or reviewing verbally, thinking through clause wording aloud.

## Speech Patterns for This Category

- Legal phrasing spoken naturally: "pursuant to section four point two of the agreement"
- Section numbers as words: "section four point two" not "Section 4.2"
- Precise numbers/dates spoken: "effective as of january first twenty twenty six"
- Parenthetical clarifications: "the party of the first part parentheses hereinafter
  referred to as the seller close parentheses"
- Self-corrections for legal precision: "the defendant— sorry the plaintiff alleged..."
- Informal mixed with formal: "so basically the indemnification stuff says..."
- Latin terms spoken casually: "inter alia", "prima facie", "de facto"
- Thinking through wording: "let me get the phrasing right here"

## Example Pair

**Input:**
```
um section three point one comma indemnification period the seller shall indemnify and
hold harmless the buyer comma its officers comma directors comma and employees from any
and all claims comma damages comma or liabilities arising out of or related to um any
breach of the representations and warranties set forth in section two hereof period uh
this indemnification obligation shall survive the closing for a period of um eighteen
months from the effective date which is uh march first twenty twenty six period
```

**Output:**
```
Section 3.1 — Indemnification. The Seller shall indemnify and hold harmless the Buyer,
its officers, directors, and employees from any and all claims, damages, or
liabilities arising out of or related to any breach of the representations and
warranties set forth in Section 2 hereof. This indemnification obligation shall
survive the closing for a period of 18 months from the effective date, which is
March 1, 2026.
```

## Common Pitfalls

- **Legal precision is paramount**: Section numbers, party names, dates, dollar
  amounts, and time periods must be EXACTLY preserved. "eighteen months" → "18 months"
  but never accidentally becoming "12 months" or "two years"
- **Spoken punctuation is heavy**: Legal dictation uses lots of "comma", "period",
  "semicolon" — convert them all
- **Capitalization conventions**: "Seller", "Buyer", "Agreement", "Section" are
  capitalized per legal convention in the output
- **Self-corrections matter even more**: "defendant— sorry plaintiff" must correctly
  resolve to "Plaintiff" — getting this wrong in a legal context is a critical error
- **Don't paraphrase legal language**: "indemnify and hold harmless" is a specific
  legal phrase — do NOT simplify to "protect" or "compensate"
- **Don't add legal boilerplate**: Output only what was spoken
