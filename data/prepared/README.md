---
language:
  - en
license: mit
task_categories:
  - text-generation
tags:
  - speech-transcript-cleanup
  - asr-post-processing
  - text-correction
  - voice-dictation
pretty_name: Aawaaz Transcript Cleanup
size_categories:
  - 10K<n<50K
---

# Aawaaz Transcript Cleanup Dataset

Training pairs for cleaning messy speech transcripts (ASR output, voice dictation) into well-formatted text while preserving the speaker's voice and meaning.

## Dataset Description

Each example is a pair of:
- **input**: A realistic messy transcript with filler words, false starts, self-corrections, grammar errors, and missing punctuation
- **output**: The cleaned version with fillers removed, grammar fixed, punctuation added, and domain-appropriate formatting applied

The cleanup preserves the speaker's personality, vocabulary, and sentence structure. It does not rewrite or formalize — "it's been studied to death" stays as-is rather than becoming "has been extensively studied".

## Format

JSONL files, one per category. Each line:

```json
{"input": "messy transcript...", "output": "cleaned text..."}
```

## Categories

| Category | Pairs | Type | Description |
|----------|------:|------|-------------|
| `casual_conversation` | 2,000 | Core | Voice messages, daily life updates, informal chatter |
| `self_corrections_heavy` | 2,000 | Core | Transcripts with heavy "wait no, I meant..." patterns |
| `technical_code` | 2,002 | Core | Dictated code, CLI commands, technical documentation |
| `financial_business` | 2,000 | Core | Financial reports, earnings calls, budget discussions |
| `academic_research` | 2,000 | Core | Research dictation, statistics, citations |
| `legal_contract` | 2,008 | Core | Legal dictation, contract terms, court notes |
| `meeting_notes` | 2,000 | Domain Specific | Structured meeting notes with attendees, action items |
| `email_professional` | 1,914 | Domain Specific | Dictated professional emails |
| `medical_clinical` | 2,007 | Domain Specific | Clinical notes with vitals and diagnoses |
| `creative_writing` | 1,941 | Domain Specific | Literary prose, poetry, creative dictation |
| `shopping_lists` | 1,954 | Domain Specific | Bulleted shopping and to-do lists |
| `recipe_cooking` | 959 | Domain Specific | Recipes with ingredients and steps |
| **Total** | **22,785** | | |

**Core** categories apply conservative cleanup — remove fillers, fix grammar, preserve the speaker's structure. **Domain Specific** categories reformat into domain-appropriate structures (bullets, headers, sections) while preserving all content.

## Examples

### Casual Conversation

**Input:**
> oh my god dude you will not believe what happened to me today so I was like walking to the the grocery store right and I ran into uh marcus from high school like I haven't seen him in like ten years and he was like oh hey and I was like oh hey and it was so awkward because you know we we didn't really end on good terms

**Output:**
> You will not believe what happened to me today. I was walking to the grocery store and I ran into Marcus from high school. I haven't seen him in like ten years. He said hey and I said hey, and it was so awkward because we didn't really end on good terms.

### Meeting Notes

**Input:**
> okay so uh meeting notes from the standup this morning um it was me and priya and uh james and also wei joined late I think like five minutes in um so priya said the the uh login flow redesign is about seventy percent done she's uh she said she needs the new api endpoints from james before she can finish the uh the integration testing part

**Output:**
> **Standup Meeting Notes**
>
> Attendees: Priya, James, Wei (joined 5 minutes late), and myself.
>
> **Login Flow Redesign**
> Priya reported the login flow redesign is about 70% done. She needs the new API endpoints from James before she can finish integration testing.
>
> - Action item: James to have API endpoints ready by Thursday.

### Technical Code

**Input:**
> so I'm trying to install um requests, like the requests library, and I did pip install requests but it says um permission denied or something? I think I need to use like sudo or maybe pip three instead of just pip, I'm not really sure which one to use

**Output:**
> So I'm trying to install the requests library, and I did `pip install requests` but it says permission denied or something. I think I need to use `sudo` or maybe `pip3` instead of just `pip`. I'm not really sure which one to use.

## How It Was Generated

Data was generated using an agent-based system ([prompts/agent/](https://github.com/shantanugoel/aawaaz-finetune/tree/main/prompts/agent)) with a multi-stage pipeline:

1. **Generation**: A coding agent reads category-specific prompts and generates transcript pairs in batches of 50
2. **Validation**: A different model evaluates each batch on 4 criteria — input realism, content preservation, no hallucination, and voice preservation
3. **Iteration**: Failed pairs are regenerated with failure reasons as guidance

Each batch gets a fresh agent context to maintain consistent quality across thousands of pairs.

## Intended Use

Fine-tuning small language models (0.6B–1.7B parameters) for on-device speech transcript cleanup in the [Aawaaz](https://github.com/shantanugoel/aawaaz) macOS app.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("shantanugoel/aawaaz-transcript-cleanup")
```

Or load individual category files:

```python
import json

with open("casual_conversation.jsonl") as f:
    pairs = [json.loads(line) for line in f]
```
