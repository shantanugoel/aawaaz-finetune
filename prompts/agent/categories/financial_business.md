# Category: Financial / Business

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions.

> **Type**: Core

## Description

Business professionals dictating financial reports, budget summaries, investment notes,
earnings recaps, or business communications. Numbers-heavy content with financial
terminology — someone reviewing or presenting figures verbally.

## Speech Patterns for This Category

- Money amounts as words: "three point five million dollars", "revenue of twelve point
  eight million", "one thousand two hundred dollars"
- Percentages spoken: "up seventeen percent year over year", "sixty two point three
  percent"
- Financial terms spoken casually: "ebitda", "gross margin", "quarter over quarter",
  "the bottom line"
- Comparisons: "compared to last quarter where we saw...", "that's up from..."
- Decimal numbers: "four point seven percent", "zero point oh three"
- Quarters spoken: "q one twenty twenty six", "first quarter"
- Abbreviations spoken: "return on investment" or "r o i" not "ROI"

## Example Pair

**Input:**
```
uh so the q one twenty twenty six financial summary um total revenue came in at twelve
point eight million which is up uh seventeen percent year over year period gross margin
improved to um sixty two point three percent from fifty eight point one percent in q
one twenty twenty five period uh operating expenses were seven point two million um
mainly driven by the new hires in engineering period uh ebitda was um three point five
million representing a twenty seven point three percent ebitda margin period uh net
income after tax was two point one million or uh one dollar and forty three cents per
diluted share period um we're raising our full year guidance to uh fifty two to fifty
four million in revenue
```

**Output:**
```
Q1 2026 Financial Summary: Total revenue came in at $12.8 million, up 17% year over
year. Gross margin improved to 62.3% from 58.1% in Q1 2025.

Operating expenses were $7.2 million, mainly driven by the new hires in engineering.
EBITDA was $3.5 million, representing a 27.3% EBITDA margin.

Net income after tax was $2.1 million, or $1.43 per diluted share.

We're raising our full-year guidance to $52-54 million in revenue.
```

## Common Pitfalls

- **Number formatting is the hardest part**: "twelve point eight million" → "$12.8
  million", "sixty two point three percent" → "62.3%", "one dollar and forty three
  cents" → "$1.43"
- **Financial abbreviations in output**: "ebitda" → "EBITDA", "q one" → "Q1",
  "r o i" → "ROI", "return on investment" → "ROI" (when clearly used as abbreviation)
- **EVERY number must be exactly correct**: Financial data precision is critical.
  Don't round, don't approximate, don't change values.
- **Ranges**: "fifty two to fifty four million" → "$52-54 million"
- **Per-share data**: "one dollar and forty three cents per diluted share" → "$1.43
  per diluted share"
- **THE #1 FAILURE**: Pre-formatted numbers in the input. "The acquisition cost is
  $8.5 million" is written text. "the acquisition cost was eight point five million
  dollars" is speech.
