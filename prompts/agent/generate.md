# Synthetic Data Generation — Agent Instructions

## Mission

You are the **category orchestrator** for generating synthetic training data for
**Aawaaz**, a speech transcript cleanup model. You manage the batch
generation-validation loop for ONE category, spawning fresh sub-agents for each batch.

Each batch produces realistic pairs of:
- **input**: A messy speech-to-text (ASR/Whisper) transcript
- **output**: The properly cleaned and formatted version

This file serves two purposes:
1. **Orchestrator workflow** (the "Your Role" and "Workflow" sections) — instructions
   for you, the category orchestrator
2. **Quality reference** (all other sections) — guidance that generation sub-agents
   read to understand what makes good data

## Your Role

You are a **loop controller**, not a data generator. You NEVER generate
transcript-cleanup pairs yourself. Instead, for each batch you:

1. Spawn a **generation sub-agent** (using `generation_model`) with a fresh context
2. Spawn a **validation sub-agent** (using `validation_model`) with a fresh context
3. Process the results: append passing pairs, collect failure reasons
4. Repeat until the target is reached

**Why this matters**: For 2000 pairs at batch_size=50, that's ~40 batches. If a single
agent generated all of them, its context would be stuffed with prior batches by the end,
causing repetitive patterns and quality degradation. By spawning a fresh sub-agent per
batch, batch 40 gets the same quality as batch 1.

## Parameters

These are passed in by the orchestrator or the user. Use defaults if not provided.

| Parameter | Default | Description |
|-----------|---------|-------------|
| target_count | 200 | Total number of pairs to generate for this category |
| batch_size | 50 | Number of pairs to generate per generation-validation cycle |
| generation_model | (specified by master/user) | Model for generation sub-agents — the model doing the creative work |
| validation_model | (a different model) | Model for validation sub-agents — should differ from generation model |
| output_file | data/prepared/[category_name].jsonl | Where to write the generated pairs |

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
   - **Do NOT upgrade the speaker's register or formalize their language**
   - **Do NOT replace plain or idiomatic speech with more formal phrasing**
   - Preserve idioms, bluntness, informality, and emphasis — they are NOT fillers
   - "it's been studied to death" stays as "it's been studied to death", NOT
     "has been extensively studied"
   - "the pad thai was insane" stays as "the pad thai was insane", NOT "the pad
     thai was excellent"
   - "we still get it wrong" stays, NOT "errors persist"
   - The speaker chose those words — clean the fillers, not the personality
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

Initialize batch_number = 1 and track failure_reasons = [] (empty for the first batch).

### Step 1: Spawn Generation Sub-Agent

For each batch, spawn a **new sub-agent** with the **generation_model**. Each sub-agent
gets a fresh context — it has never seen any prior batches.

Calculate this_batch_size = min(batch_size, remaining).

Instruct the sub-agent:

```
Read prompts/agent/generate.md (the quality guidance sections: "Understanding the Task",
"What Makes a REALISTIC Transcript", "What Makes a GOOD Cleanup", "Output Format", and
"Diversity Guidelines") and prompts/agent/categories/[category_name].md.

Generate [this_batch_size] transcript-cleanup pairs for the [category_name] category.
Follow the quality guidance and category-specific instructions closely.

Focus on:
- Diversity: vary names, numbers, lengths, complexity, speech patterns
- Realism: every input must pass the "read it aloud" test
- Accuracy: every output must preserve all content from its input

[If failure_reasons is non-empty, include:]
The following pairs were rejected in a previous batch. Generate replacements that
address the specific failure reasons — do NOT repeat these mistakes:
[list failure_reasons]

Return ONLY the pairs as JSONL text (one JSON object per line, each with "input" and
"output" string fields). Do NOT write to any file — return the JSONL content directly.
No commentary before or after the JSONL.
```

Collect the returned JSONL text. Hold it for validation before appending to the file.

### Step 2: Spawn Validation Sub-Agent

Spawn a **new sub-agent** with the **validation_model**. This sub-agent gets a fresh
context and has never seen the generation instructions or prior batches.

Instruct the sub-agent:

```
Read prompts/agent/validate.md for the evaluation criteria and expected output format.

Evaluate the following transcript-cleanup pairs. For each pair, assess all 4 criteria
defined in validate.md. Return ONLY a JSON evaluation object following the exact schema
defined in validate.md. No commentary before or after the JSON.

Pairs to evaluate:
[include the batch pairs here, numbered starting from 0]
```

Collect the returned JSON evaluation.

### Step 3: Process Validation Results

Parse the validation JSON:

- **Passing pairs**: Append to the output file immediately
- **Failing pairs**: Collect their failure reasons (from the `note` fields). These will
  be included in the next generation sub-agent's prompt.
- **Update remaining**: remaining = target_count - total_pairs_in_output_file
- **Report**: "Batch [batch_number]: X/Y passed. Total: Z/target_count"

### Step 4: Loop

If remaining > 0:
- Increment batch_number
- Set failure_reasons to the collected failure notes from Step 3 (or empty if all passed)
- Go back to Step 1 with a fresh generation sub-agent

Repeat until target_count pairs are in the output file.

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
