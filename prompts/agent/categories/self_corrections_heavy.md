# Category: Self-Corrections Heavy

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions.

## Description

Transcripts where the speaker heavily self-corrects: changing numbers, names, dates,
word choices, and even whole sentences. This is the **hardest category** because the
model must correctly identify and resolve each correction, keeping ONLY the final
version. Every transcript should have at least 2-3 distinct corrections.

## Speech Patterns for This Category

- **Number corrections**: "it was three hundred... wait no sorry three thousand
  dollars"
- **Name corrections**: "send it to michael uh I mean matthew"
- **Date/time corrections**: "the deadline is friday... actually no it got pushed to
  monday"
- **Word choice corrections**: "the project is delayed... well not delayed exactly,
  more like reprioritized"
- **Instruction corrections**: "put it in the blue folder wait the red one"
- **Whole sentence rewrites**: "actually scratch that let me start over"
- **Cascading corrections**: "on Tuesday no Wednesday actually make that Thursday" —
  multiple revisions of the same fact

Include AT LEAST 2-3 corrections per transcript. Mix different correction types
within a single transcript.

## Example Pair

**Input:**
```
so the meeting with um the client is on tuesday no wait wednesday march the eighteenth
uh at two pm no sorry two thirty pm and it's at their office on elm street uh actually
no we changed it to a zoom call so yeah it's a zoom meeting at two thirty on wednesday
and the attendees are me sarah david and uh no not david um james will be joining
instead and we need to prepare the q four numbers wait q one I mean the q one twenty
twenty six numbers showing revenue of uh eight point five no sorry eight point seven
million and make sure to mention that the timeline got moved from june to august no
sorry july we moved it to july
```

**Output:**
```
The meeting with the client is on Wednesday, March 18th at 2:30 PM. It's a Zoom
meeting. The attendees are me, Sarah, and James.

We need to prepare the Q1 2026 numbers showing revenue of $8.7 million, and make sure
to mention that the timeline got moved from June to July.
```

## Common Pitfalls

- **Every correction must resolve to the FINAL version**: Tuesday→Wednesday =
  Wednesday. David→James = James. 8.5→8.7 = 8.7. June→August→July = July. Q4→Q1 = Q1.
- **Intermediate values must DISAPPEAR**: The output should NOT contain "Tuesday",
  "David", "8.5", "August", "Q4", or "two pm" — these are all superseded.
- **Correction phrases removed**: "no wait", "actually", "sorry", "I mean", "scratch
  that", "well not X exactly" — all removed
- **Cascading corrections**: If they correct A to B then B to C, output should have
  ONLY C. This is the hardest pattern — test these specifically.
- **Don't lose content around corrections**: When resolving a correction, make sure
  the surrounding sentence still makes sense and retains all non-corrected information.
- **"Not X exactly, more like Y" = choose Y**: "delayed... well not delayed exactly,
  more like reprioritized" → use "reprioritized"
- **This category's quality depends entirely on correction resolution accuracy**.
  Double-check every correction before finalizing each pair.
