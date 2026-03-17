# Category: Email Professional

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions.

> **Type**: Domain Specific

## Description

Someone DICTATING a business email out loud — thinking through what they want to write
as they speak. NOT reading a pre-written email. They compose on-the-fly, hesitate on
wording, change their mind about phrasing, and speak punctuation aloud.

## Speech Patterns for This Category

- Spoken punctuation is essential: "comma", "period", "new paragraph", "colon",
  "exclamation mark"
- Meta-commentary: "okay so the email should say...", "actually scratch that let me
  rephrase", "how should I put this"
- Numbers as words: "twelve thousand five hundred" not "$12,500"
- Thinking aloud: "hmm what's the right word here", "no that sounds too harsh"
- Dictation cues: "capital", "all caps", "new line"
- Composing = messy: they don't speak in finished sentences, they build them

## Example Pair

**Input:**
```
okay so uh dear mister thompson comma new paragraph I wanted to follow up on our
conversation from last week regarding the um the quarterly budget proposal period um
the revised numbers are looking at twelve thousand five hundred for marketing and uh
eight thousand three hundred for operations period new paragraph I've attached the the
spreadsheet with the full breakdown and um I think we should schedule a meeting to
discuss maybe uh thursday at two pm if that works for you question mark new paragraph
uh kind regards comma new line patricia chen
```

**Output:**
```
Dear Mr. Thompson,

I wanted to follow up on our conversation from last week regarding the quarterly
budget proposal. The revised numbers are looking at $12,500 for marketing and $8,300
for operations.

I've attached the spreadsheet with the full breakdown, and I think we should schedule
a meeting to discuss — maybe Thursday at 2 PM if that works for you?

Kind regards,
Patricia Chen
```

## Common Pitfalls

- **THE #1 FAILURE**: Making the input sound like a typed email with "um" added. Real
  dictation has someone COMPOSING aloud — they speak punctuation, think about phrasing,
  change their mind mid-sentence.
- **BAD input example**: "Dear Mr. Smith, um, I am writing to inform you that the
  budget has been approved." — This is written text with one filler. No one dictates
  this cleanly.
- **Numbers must be spoken**: "twelve thousand five hundred" in input, "$12,500" in
  output. Never pre-formatted numbers in the input.
- **Output must be a proper email**: Salutation, paragraphs, sign-off, formatted
  numbers, proper punctuation.
- **Spoken punctuation must be converted**: "comma" → , / "period" → . /
  "new paragraph" → paragraph break / "question mark" → ?
