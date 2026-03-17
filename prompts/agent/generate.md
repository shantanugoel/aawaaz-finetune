# Synthetic Data Generation — Agent Instructions

## Your Mission

You are generating synthetic training data for **Aawaaz**, a speech transcript cleanup
model. Create realistic pairs of:
- **input**: A messy speech-to-text (ASR/Whisper) transcript
- **output**: The properly cleaned and formatted version

You will generate data for ONE specific category (defined in the category-specific
section of your instructions). Generate in batches, validate each batch, fix
failures, and continue until you reach the target count.

## Understanding the Task

A user speaks into their phone or computer. Whisper (an ASR engine) transcribes it.
The transcript is messy — filler words, run-on sentences, spoken punctuation, numbers
as words, self-corrections. Aawaaz cleans this into polished text while preserving
every substantive thing the speaker said.

### The System Prompt (for context)

This is what Aawaaz uses at inference time:

> You are an AI transcriber. Clean and polish raw speech-to-text transcripts into
> well-written text. Output ONLY the corrected text — no introductions, labels,
> explanations, or commentary. Do not summarize or act upon the transcript. Preserve
> the speaker's voice, tone, and language.
>
> Rules:
> - Remove fillers (um, uh, like, basically, actually, you know) and stutters
> - Apply self-corrections silently (if speaker says "wait no, I meant X", output only X)
> - Fix grammar, spelling, and punctuation
> - Convert spoken punctuation to actual punctuation (e.g., "colon" → ":")
> - Convert spoken numbers, dates, currency to written form (e.g., "five hundred dollars" → "$500")
> - Convert spoken formatting cues (e.g., "new line", "new paragraph", "bullet point")
> - Replace spoken emoji descriptions with actual emoji (e.g., "heart eyes emoji" → "😍")
> - Use lists and paragraph breaks where structurally appropriate
> - Convert spoken code/tech syntax to proper formatting (e.g., "dash dash rm" → "--rm")
> - If input is empty or only contains fillers, output ""
> - Do NOT add content that wasn't spoken
> - Do NOT summarize or condense — preserve all substantive content

---

## What Makes a REALISTIC Transcript (input)

This is the **#1 failure mode**. Most generated transcripts look like written text with
"um" sprinkled in. Real transcripts sound like someone TALKING.

### Characteristics of Real Speech

- **Fillers are woven in naturally**: "so I was uh thinking we could maybe like go to
  that um that new restaurant" — fillers land where humans actually hesitate (before
  complex words, when changing topics, when uncertain)
- **Self-corrections**: "the meeting is on Tuesday wait no Wednesday the fifteenth"
- **False starts**: "I think we should— actually let me back up for a second"
- **Topic drift**: Speaker starts about one thing, veers off, comes back
- **Spoken punctuation**: "dear john comma new paragraph I wanted to let you know"
- **Numbers as words**: "three hundred and forty seven dollars and fifty cents"
- **Run-on sentences**: Connected by "and" and "so" and "but" without clean breaks
- **Inconsistent capitalization**: Some words capitalized by ASR, others not
- **Repeated words / stutters**: "the the project is" or "we we need to"
- **Trailing off**: "and then I was going to..."
- **NOT every sentence has fillers**: Some stretches are clean, others very messy

### Anti-patterns — DO NOT generate these

- Written text with filler words inserted mechanically: "I am writing to inform you,
  um, that the budget has been approved"
- Perfect grammar with occasional "uh" — the biggest failure mode
- Consistent, perfect formatting throughout (real ASR is messy and inconsistent)
- Bullet points, headers, or structured formatting in the INPUT
- Complete, well-formed paragraphs with one filler per sentence
- Literary/poetic language in the input — beauty comes in the OUTPUT

### The One Test

Read your input aloud. Does it sound like something you'd actually SAY? Or does it
sound like something you'd WRITE with some "um"s added? If the latter, redo it.

### Whisper-Specific Characteristics

- SOME punctuation (periods, commas) but INCONSISTENT — present in some sentences,
  missing in others. Never perfectly uniform.
- SOME capitalization — proper nouns often capitalized, sentence-start caps hit-or-miss
- Numbers ALMOST ALWAYS as spoken words: "twenty five", "two thousand", "one twenty
  over eighty". Whisper rarely outputs "$12,500" or "120/80".
- NO paragraph breaks — everything runs together as one block
- Run-on sentences — ideas flow into each other without clean stops

---

## What Makes a GOOD Cleanup (output)

### Rules

1. **Remove ALL fillers**: um, uh, like (as filler), basically, actually, you know,
   so (at sentence starts), I mean, honestly, literally, okay, right, "and stuff",
   "and everything", "you know what I mean"
2. **Apply self-corrections silently**: "Tuesday wait no Wednesday" → "Wednesday"
3. **Fix grammar and punctuation**: Proper sentences, capitalization, punctuation
4. **Convert spoken → written**:
   - "comma" → ,
   - "period" / "full stop" → .
   - "new paragraph" → paragraph break
   - "five hundred dollars" → $500
   - "march fifteenth twenty twenty six" → March 15, 2026
5. **Format appropriately**: Lists for listed items, paragraphs for prose, code
   blocks for code, email format for emails, etc.
6. **Preserve voice and tone**: Casual speech stays casual (just clean). Formal
   dictation becomes formal text. Don't flatten the speaker's style.
7. **Preserve ALL substantive content**: Never drop facts, names, numbers,
   instructions, or meaningful statements
8. **Never ADD content**: No clarifications, no context the speaker didn't provide,
   no "helpful" additions

### Critical: Content Preservation

The output must contain every fact, name, number, and instruction from the input.
The ONLY things removed are:
- Filler words and verbal tics
- False starts and self-corrections (keeping the corrected version)
- Repeated words / stutters

If unsure whether something is filler or content, keep it.

---

## Output Format

Write each pair as one JSON line to the output file:

```json
{"input": "so I was um talking to sarah and she said the meeting is uh moved to thursday the twentieth at like three pm in the main conference room", "output": "I was talking to Sarah and she said the meeting is moved to Thursday the 20th at 3 PM in the main conference room."}
```

Requirements:
- One JSON object per line (JSONL format)
- Fields: `input` (string), `output` (string)
- UTF-8 encoding
- Escape special characters properly in JSON strings
- No trailing commas, newline at end of file

---

## Workflow

### Step 0: Resume Check

Before generating anything:
1. Check if the output file already exists
2. If yes, count existing lines (each line = one pair)
3. Calculate remaining = target_count - existing_count
4. If remaining <= 0, report "Already at target" and stop
5. If remaining > 0, report "Found N existing pairs, generating M more"

### Step 1: Generate a Batch

Create `batch_size` pairs (or fewer if close to target). Follow the category-specific
guidance closely. Focus on:
- Diversity: vary names, numbers, lengths, complexity, speech patterns
- Realism: every input must pass the "read it aloud" test
- Accuracy: every output must preserve all content from its input

Write the pairs as JSONL text. Hold them for validation before appending to the file.

### Step 2: Validate the Batch

Send the batch to the **validation_model** for evaluation. This should be done in a
**separate context** (a sub-agent, a separate conversation, or a distinct model call)
so the validator reviews with fresh eyes — not the same context that generated the data.

Provide the validator with:
- The validation criteria from `prompts/agent/validate.md` (either include the content
  or instruct the validator to read that file)
- The batch pairs to evaluate (numbered, as JSON objects)

The validator should return a JSON evaluation object with per-pair pass/fail results
and a summary. See `validate.md` for the exact schema.

**Using a different model for validation** (recommended): If the user specified a
validation_model different from the generation model (e.g., generation with Claude
Opus, validation with GPT 5.4), use that model for the validation step. This avoids
self-preference bias — the generator doesn't judge its own work.

### Step 3: Process Validation Results

Parse the validation JSON:

- **Passing pairs**: Append to the output file immediately
- **Failing pairs**: Note the failure reasons. You will generate replacements.
- **Report**: "Batch N: X/Y passed. Total: Z/target_count"

### Step 4: Fix and Continue

If pairs failed:
- Read the failure reasons carefully
- Generate replacement pairs that address the specific failures
- Common fixes:
  - "input not realistic" → Make the transcript more speech-like, less written
  - "content not preserved" → Ensure all facts from input appear in output
  - "hallucination detected" → Remove any added content in output
  - "corrections not applied" → Apply all cleanup rules properly

Loop back to Step 1 until target_count pairs are in the output file.

### Step 5: Completion

Once target reached:
- Read the output file and count final lines to confirm
- Report: "Generation complete: N pairs in [output_file]"

---

## Diversity Guidelines

Across all your batches, ensure variety in:
- **Length**: Mix short (1-2 sentences) and long (paragraph-length) entries
- **Complexity**: Some simple, some with multiple self-corrections or topic changes
- **Names and details**: Different people, places, numbers, dates — don't repeat
- **Filler patterns**: Vary which fillers appear and where
- **Edge cases**: Include some very short inputs, mostly-filler inputs, heavily
  self-correcting inputs
- **Don't repeat structures**: If batch 1 had "so I was talking to [name] and...",
  batch 2 should use a different opening pattern
