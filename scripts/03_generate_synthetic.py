#!/usr/bin/env python3
"""Generate synthetic transcript-cleanup training pairs via LLM API.

Reads category proportions from ``config.yaml``, calls an LLM to produce
realistic Whisper-style transcript / clean-output pairs, validates each
pair, and saves per-category JSONL plus a combined output file.

Supports resume: on restart, existing records are counted and already-done
batches are skipped.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    DATA_SYNTHETIC,
    SyntheticConfig,
    base_arg_parser,
    ensure_dirs,
    load_config,
    setup_logging,
)
from llm_client import create_client_from_config, LLMClient

logger = logging.getLogger("aawaaz.generate_synthetic")

# ── Category-specific generation guidance ──────────────────────────────────

CATEGORY_GUIDANCE: dict[str, str] = {
    "casual_conversation": (
        "Friend-to-friend voice messages, voice notes about daily life, casual updates about "
        "weekend plans, stories about what happened, recommendations. "
        "Use contractions heavily, informal language, sometimes trailing off mid-thought. "
        "The transcript should sound like someone rambling to a friend — lots of 'like', "
        "'you know', 'so basically', changes of topic.\n\n"
        "OUTPUT RULES (CRITICAL — this category fails on cleanup quality):\n"
        "The output must aggressively remove ALL filler words with zero exceptions:\n"
        "- Remove: um, uh, like (when used as filler, NOT as 'such as' or 'similar to'), "
        "basically, actually, you know, so (at sentence starts), I mean, honestly, literally, "
        "right, okay so, oh my god, seriously, oh wait\n"
        "- Remove: 'and like', 'but like', 'so like', 'like seriously', 'like honestly'\n"
        "- Remove: trailing 'and stuff', 'and everything', 'and all that'\n"
        "- Casual content does NOT mean casual formatting — output must have clean punctuation "
        "and proper sentences even though the topic is informal\n"
        "- Do NOT drop substantive emotional content ('I'm so excited' should stay, but "
        "'oh my god I'm like so excited' becomes 'I'm so excited')"
    ),
    "email_professional": (
        "Someone DICTATING a business email out loud — talking through what they want to write, "
        "NOT reading a pre-written email. They think out loud, hesitate on wording, change "
        "their mind about phrasing.\n\n"
        "TRANSCRIPT MUST sound like DICTATION, not a formatted email with ums:\n"
        "- BAD: 'Dear Mr. Smith, um, I am writing to inform you that the budget has been approved.'\n"
        "  (This is a typed email with one filler — NO real person dictates this cleanly)\n"
        "- GOOD: 'okay so uh dear mister smith comma new paragraph I wanted to let you know that "
        "uh the budget got approved so the the number was um twelve thousand five hundred and uh "
        "we should be good to go from here new paragraph kind regards'\n"
        "  (This sounds like someone talking through their email, thinking as they go)\n\n"
        "Key patterns: spoken punctuation ('comma', 'period', 'new paragraph'), numbers as words "
        "('twelve thousand five hundred' not '$12,500'), no consistent capitalization, "
        "run-on structure where they're composing on the fly.\n"
        "The OUTPUT should be a properly formatted email with correct salutations, paragraph "
        "breaks, and formatted numbers ($12,500)."
    ),
    "technical_code": (
        "Someone dictating code, CLI commands, error messages, or technical docs OUT LOUD. "
        "They're speaking to a transcription system, not typing.\n\n"
        "TRANSCRIPT REALISM (this category fails because inputs look too structured):\n"
        "- BAD: 'The function processData() takes two arguments: input_file and output_dir.'\n"
        "  (This is written documentation with no speech characteristics)\n"
        "- GOOD: 'so the function is um process underscore data and it takes two arguments "
        "uh the first one is input underscore file and then output underscore dir'\n\n"
        "Key patterns:\n"
        "- Spell out symbols: 'open paren', 'close bracket', 'equals equals', 'hash', "
        "'forward slash', 'underscore', 'dash dash'\n"
        "- File paths spoken out: 'slash user slash local slash bin slash python three'\n"
        "- Variable/function names spoken with underscores: 'get underscore user underscore by "
        "underscore id'\n"
        "- Code is described haltingly with pauses and restarts, NOT rattled off fluently\n"
        "- Numbers as words: 'port eight zero eight zero' not 'port 8080'\n\n"
        "The OUTPUT should have properly formatted code with actual symbols, file paths, etc. "
        "CRITICAL: Do NOT hallucinate code details — if the speaker said 'import react', "
        "do NOT expand it to 'import React from \"react\"' unless they specifically said that."
    ),
    "medical_clinical": (
        "A clinician DICTATING patient notes out loud, thinking through their observations. "
        "NOT reading from a chart — speaking from memory/observation.\n\n"
        "TRANSCRIPT REALISM (this category fails because inputs are too structured/clinical):\n"
        "- BAD: 'Patient presents with shortness of breath. BP: 120/80. Prescribed metformin 500mg.'\n"
        "  (This is a written chart note, not speech)\n"
        "- GOOD: 'okay so the patient uh came in with shortness of breath and um the b p was "
        "one twenty over eighty and I'm gonna put them on uh metformin five hundred milligrams "
        "twice daily'\n\n"
        "Key patterns:\n"
        "- Abbreviations spoken out fully: 'b p' not 'BP', 'c t scan' not 'CT scan'\n"
        "- Numbers as spoken words: 'one twenty over eighty' not '120/80'\n"
        "- Dosages spoken: 'five hundred milligrams' not '500mg'\n"
        "- Casual clinical speech: 'gonna put them on', 'looks like', 'so basically the labs came back'\n"
        "- Thinking out loud: 'let me think what else... oh yeah the hemoglobin was'\n\n"
        "The OUTPUT should format these correctly (BP: 120/80, metformin 500 mg BID) while "
        "preserving EVERY clinical detail. Missing a dosage or lab value is a critical failure."
    ),
    "legal_contract": (
        "A lawyer DICTATING contract language, working through clause wording out loud. "
        "They know the legal terms but are composing/reviewing verbally.\n\n"
        "TRANSCRIPT REALISM (this category fails because inputs read like written contracts):\n"
        "- BAD: 'Whereas the Party of the First Part agrees to the terms set forth in Section 4.2...'\n"
        "  (This is a typed contract with perfect formatting)\n"
        "- GOOD: 'okay so um whereas the party of the first part uh agrees to the terms set "
        "forth in section four point two and uh this includes all the indemnification stuff "
        "from the previous um the previous version'\n\n"
        "Key patterns:\n"
        "- Section numbers spoken: 'section four point two' not 'Section 4.2'\n"
        "- Asides and thinking: 'let me get the wording right here', 'so the clause should say'\n"
        "- Informal speech MIXED with formal legal terms: 'so the indemnification stuff basically says'\n"
        "- No consistent capitalization of legal terms in transcript\n"
        "- Numbers as words: 'thirty days' not '30 days'\n\n"
        "The OUTPUT must preserve EVERY legal term, section number, party name, and clause "
        "exactly. Do NOT paraphrase or simplify legal language — keep it verbatim minus fillers. "
        "Do NOT add 'for clarity' explanations or restructure clauses."
    ),
    "meeting_notes": (
        "Someone dictating meeting notes AFTER a meeting, recalling what happened, or "
        "speaking during the meeting to capture action items. They're recalling from memory, "
        "jumping between topics.\n\n"
        "TRANSCRIPT REALISM:\n"
        "- BAD: 'Meeting attendees: John, Sarah, Mike. Agenda item 1: Q2 budget review.'\n"
        "  (This is written minutes, not speech)\n"
        "- GOOD: 'so the meeting was with uh john and sarah and mike was there too and um "
        "first we talked about the q two budget and john said uh we need to cut like fifteen "
        "percent from the marketing spend'\n\n"
        "Key patterns:\n"
        "- Stream of consciousness recall: 'oh and I forgot to mention', 'what else... right'\n"
        "- Names without capitalization in transcript\n"
        "- Dates/deadlines spoken: 'by next friday', 'end of q two', 'march fifteenth'\n"
        "- Action items emerge naturally: 'so sarah's gonna handle that' not 'Action: Sarah'\n\n"
        "The OUTPUT should be well-structured meeting notes with bullet points or numbered "
        "items. Preserve EVERY name, date, action item, and decision."
    ),
    "recipe_cooking": (
        "Someone dictating a recipe from memory or while cooking — casual, instructional, "
        "sometimes distracted.\n\n"
        "TRANSCRIPT REALISM:\n"
        "- BAD: 'Ingredients: 2 cups flour, 1 tsp salt. Step 1: Preheat oven to 350°F.'\n"
        "  (This is a written recipe card)\n"
        "- GOOD: 'okay so you're gonna need um two cups of flour and uh a teaspoon of salt "
        "and then preheat your oven to like three fifty and uh while that's heating up you "
        "wanna mix the dry ingredients'\n\n"
        "Key patterns:\n"
        "- Quantities as spoken words: 'two cups', 'a teaspoon', 'three fifty'\n"
        "- Conversational instruction: 'you wanna', 'go ahead and', 'what I usually do is'\n"
        "- Tangents: 'oh and make sure your butter is room temp that's important'\n"
        "- Time as words: 'twenty minutes', 'about an hour'\n\n"
        "The OUTPUT should have a properly formatted ingredients list and numbered steps "
        "with all quantities converted to written form (2 cups, 350°F, 20 minutes)."
    ),
    "academic_research": (
        "A researcher TALKING through their notes, literature review, or findings — dictating "
        "to capture ideas, NOT reading from a polished paper.\n\n"
        "TRANSCRIPT REALISM (this category fails badly because inputs look like written papers):\n"
        "- BAD: 'The study by Chen and Rodriguez (2023) examined machine learning applications...'\n"
        "  (This is a written citation — formatted year, proper punctuation, polished prose)\n"
        "- GOOD: 'so um that paper by chen and rodriguez from uh twenty twenty three looked at "
        "machine learning applications in healthcare and they found uh the sample size was like "
        "n equals forty two and the p value was less than point oh five'\n\n"
        "Key patterns:\n"
        "- Years spoken as words: 'twenty twenty three' not '2023'\n"
        "- Statistics spoken out: 'n equals forty two', 'p less than point oh five', "
        "'r squared was like point eight three'\n"
        "- Informal academic talk: 'they basically found that', 'the methodology was um'\n"
        "- No parenthetical citations — just spoken: 'that paper by chen and rodriguez'\n"
        "- Thinking through: 'wait what was the sample size... I think it was forty two'\n\n"
        "The OUTPUT should format citations (Chen & Rodriguez, 2023), statistics (p < 0.05, "
        "n = 42), and technical terms properly. Preserve every author name, statistic, and "
        "finding exactly."
    ),
    "creative_writing": (
        "Someone DICTATING a story, poem, blog post, or journal entry — speaking their creative "
        "ideas aloud as they compose. They are THINKING and CREATING verbally, not reading "
        "finished prose.\n\n"
        "TRANSCRIPT REALISM (this is the WORST category — 76% of inputs fail because they sound "
        "like written prose, not dictation):\n"
        "- BAD: 'She walked into the room and the light was like gold spilling across the floor.'\n"
        "  (This is finished prose with one filler word — it reads like a book, not speech)\n"
        "- BAD: 'The forest was ancient and the trees whispered secrets if you listened carefully.'\n"
        "  (Beautiful writing, but no one speaks this fluently while composing)\n"
        "- GOOD: 'okay so she walks into the room and um the light is I want to say like golden "
        "you know like spilling across the the wooden floor and he's just sitting there waiting "
        "for her and uh he looks up and smiles or no he looks up and like their eyes meet'\n"
        "- GOOD: 'so for this next part um I'm thinking the forest is really old like ancient and "
        "the trees kind of uh what's the word whisper I guess the trees whisper secrets to you if "
        "you actually listen like stories about the people who used to live here'\n\n"
        "Key patterns that make creative dictation REALISTIC:\n"
        "- Writer thinking about word choices: 'I want to say', 'what's the word', 'how do I put this'\n"
        "- Composing in real-time: tense shifts (present then past), trying different phrasings\n"
        "- Direction changes: 'actually no let me start this part differently'\n"
        "- Spoken formatting: 'new paragraph', 'open quote', 'end quote'\n"
        "- NOT literary/poetic in the transcript — the beauty comes in the OUTPUT after cleanup\n"
        "- Messy sentence boundaries — ideas flow into each other without clean stops\n\n"
        "The OUTPUT should format dialogue with quotation marks, add proper paragraph "
        "breaks, and clean up prose while preserving the writer's voice and every detail."
    ),
    "financial_business": (
        "Someone TALKING through financial figures — reviewing a report verbally, dictating "
        "an analysis, or discussing numbers in a call.\n\n"
        "TRANSCRIPT REALISM (this category fails because inputs have pre-formatted numbers):\n"
        "- BAD: 'The acquisition cost for the new property is $8.5 million with a 15% ROI.'\n"
        "  (This is a written report — formatted currency, percentages, abbreviations)\n"
        "- GOOD: 'so the uh acquisition cost for the new property was eight point five million "
        "dollars and the return on investment is looking like about fifteen percent which is "
        "um actually pretty good for this market'\n\n"
        "Key patterns:\n"
        "- Dollar amounts spoken: 'twelve hundred dollars', 'eight point five million'\n"
        "- Percentages spoken: 'fifteen percent' not '15%'\n"
        "- Quarters spoken: 'q three twenty twenty five', 'third quarter'\n"
        "- Abbreviations spoken: 'return on investment' or 'r o i' not 'ROI'\n"
        "- Casual financial talk: 'the numbers look pretty good', 'we're a bit over budget'\n\n"
        "The OUTPUT must format all numbers correctly ($1,200, 15%, Q3 2025) and preserve "
        "every financial figure exactly."
    ),
    "shopping_lists": (
        "Dictating a shopping or to-do list quickly — rapid-fire items, sometimes with "
        "quantities, brands, or notes. Speaker is usually walking around or checking the "
        "fridge/pantry.\n\n"
        "Key patterns:\n"
        "- Items rattled off: 'milk eggs bread and oh we need butter too'\n"
        "- Quantities as words: 'like three avocados', 'a dozen eggs'\n"
        "- Afterthoughts: 'oh and get the organic kind', 'wait do we have rice... no get rice'\n"
        "- Brands/specifics: 'the kirkland one', 'get the unsalted butter'\n\n"
        "The OUTPUT should be a clean bulleted or numbered list with formatted quantities."
    ),
    "self_corrections_heavy": (
        "Focus heavily on self-correction patterns. The speaker frequently changes their mind "
        "mid-sentence.\n\n"
        "Include MULTIPLE corrections per example — at least 2-3 per transcript. Mix different "
        "correction patterns:\n"
        "- Number corrections: 'the meeting is at two pm wait no three pm'\n"
        "- Name corrections: 'send it to john no actually send it to sarah'\n"
        "- Fact corrections: 'it was on tuesday or wait no it was wednesday'\n"
        "- Phrasing corrections: 'we need to cancel the uh no not cancel just postpone'\n"
        "- Instruction corrections: 'put it in the blue folder wait the red one'\n\n"
        "CRITICAL for OUTPUT:\n"
        "- Keep ONLY the final corrected version of each correction\n"
        "- If speaker said 'tuesday no wednesday', output only Wednesday\n"
        "- Do NOT include both versions or any correction markers\n"
        "- Do NOT add notes like 'corrected from'\n"
        "- Preserve ALL non-corrected content exactly — do NOT accidentally drop surrounding "
        "facts when applying corrections"
    ),
}

# ── Generation prompt template ─────────────────────────────────────────────

GENERATION_PROMPT = """\
You are generating training data for a speech transcript cleanup model. \
Generate {batch_size} realistic input/output pairs for the category: {category}.

## RULES FOR THE "transcript" FIELD (the messy input)

The transcript must look like REAL output from Whisper (a modern speech-to-text engine) \
transcribing someone who is SPEAKING NATURALLY.

### What makes a transcript REALISTIC vs FAKE

FAKE (will be rejected — sounds like written text with fillers inserted):
- "The patient presents with shortness of breath. Um, blood pressure is 120/80."
  → Why fake: Perfect sentence structure, formatted numbers, consistent punctuation, \
one mechanical filler
- "She walked into the room and the light was like gold spilling across the floor."
  → Why fake: This is polished prose, not how someone speaks while composing
- "Dear Mr. Smith, I am writing to inform you that the Q3 budget has been approved."
  → Why fake: This is a typed email, not someone dictating

REALISTIC (will pass — sounds like someone actually talking):
- "so the patient uh came in with shortness of breath and the b p was one twenty \
over eighty and I'm gonna I'm thinking we put them on metformin"
  → Why real: Spoken numbers, casual phrasing, false start, inconsistent flow
- "okay so she walks into the room and um the light is like I want to say golden \
you know spilling across the the wooden floor and he's just sitting there"
  → Why real: Thinking out loud, tense shifts, stutter, composing in real-time
- "okay uh dear mister smith comma new paragraph so I wanted to let you know the \
q three budget got approved the number was uh twelve thousand five hundred"
  → Why real: Spoken punctuation, numbers as words, thinking as they compose

### Whisper transcription characteristics:
- SOME punctuation (periods, commas) but INCONSISTENT — present in some sentences, \
missing in others. Never perfectly uniform.
- SOME capitalization — proper nouns often capitalized, but sentence-start caps are \
hit-or-miss. Never consistently perfect.
- Numbers ALMOST ALWAYS as spoken words: "twenty five", "two thousand", "one twenty \
over eighty". Whisper rarely outputs formatted numbers like "$12,500" or "120/80".
- NO paragraph breaks — everything runs together as one block
- Filler words placed where humans ACTUALLY hesitate: before complex words, when \
changing topics, when uncertain about next words. NOT at regular intervals.
- Run-on sentences — ideas flow into each other without clean stops
- NOT every sentence has fillers — some stretches are clean, others are messy

### The ONE test your transcript must pass:
Read it aloud. Does it sound like a real person talking? If it sounds like someone \
reading from a well-written document, it is FAKE. Rewrite it.

## RULES FOR THE "output" FIELD (the clean version)

The output is what the cleanup model should produce. It must:
- Remove ALL fillers and stutters with ZERO exceptions (um, uh, like [as filler], \
basically, actually, you know, so [at sentence starts], I mean, honestly, literally, \
okay so, right, and stuff, and everything)
- Apply self-corrections (keep ONLY the corrected version, drop the mistake entirely)
- Add proper, consistent punctuation and capitalization throughout
- Convert spoken numbers/dates/currency to written form ($500, January 15, 2025)
- Apply proper formatting (bullet lists, paragraphs, code blocks) where the speaker indicated them
- Convert spoken emoji descriptions to actual emoji characters
- Convert spoken punctuation cues to actual punctuation ("comma" → ,  "new paragraph" → ¶)

CRITICAL:
- Preserve EVERY substantive fact, name, number, and instruction from the input \
— do NOT drop or summarize anything
- Only REMOVE noise (fillers, stutters, corrections) and ADD formatting (punctuation, structure)
- Do NOT add information, context, conclusions, or clarifications the speaker did not say
- Do NOT rephrase in your own words — preserve the speaker's wording minus the noise
- Be THOROUGH with cleanup — every filler removed, every number formatted, every \
self-correction resolved. Incomplete cleanup will be rejected.

## Category-specific guidance for "{category}":
{category_guidance}

## Response format
Respond with a JSON array of objects, each with "transcript" and "output" keys. No other text.
Vary the length: some short (1-2 sentences, ~15-30 words), some medium (paragraph, ~50-100 words), \
some long (multiple paragraphs, ~150-300 words)."""

# ── Messify prompt for clean→messy mode ────────────────────────────────────

MESSIFY_PROMPT = """\
Convert these {batch_size} clean, well-formatted texts into realistic speech-to-text (ASR) \
transcripts, as if someone SPOKE this content aloud and it was transcribed by Whisper.

### What REALISTIC ASR transcription sounds like:
The output should sound like someone TALKING, not someone reading a document aloud. \
People who dictate content think as they speak — they hesitate, restart, use filler words, \
and their formatting is messy.

FAKE (do NOT produce this):
"The quarterly revenue was $2.5 million, um, representing a 15% increase over Q2."
→ Why: formatted numbers, perfect punctuation, one mechanical filler

REALISTIC (produce this):
"so the quarterly revenue was uh two point five million dollars which is like a fifteen \
percent increase over q two and that's um actually pretty solid"
→ Why: spoken numbers, natural fillers, casual phrasing, run-on structure

### Key transformation rules:
- Convert ALL formatted numbers to spoken words: "$2,500" → "twenty five hundred dollars", \
"15%" → "fifteen percent", "2023" → "twenty twenty three", "120/80" → "one twenty over eighty"
- REMOVE all consistent punctuation patterns — add some back inconsistently (a period here, \
a comma there, but NOT on every sentence)
- REMOVE paragraph breaks — everything becomes one continuous block
- ADD filler words where people naturally hesitate (NOT at regular intervals):
  - Before complex terms: "the uh indemnification clause"
  - When changing topics: "so um anyway the next thing"
  - When thinking: "I think it was like forty two or wait yeah forty two"
- ADD natural speech patterns: false starts, self-corrections, run-on sentences, topic restarts
- VARY the messiness — some sentences cleaner, others much messier
- Capitalization should be inconsistent — some proper nouns capped, some not

CRITICAL: Include ALL content from the original — do not drop facts, names, or details.

Category context: "{category}" — {category_guidance}

CLEAN TEXTS:
{numbered_texts}

Respond with a JSON array of {batch_size} objects, each with a "transcript" key \
containing ONLY the messy version. Same order as input. No other text."""

# Categories that work well with clean→messy (have natural source text in datasets)
CLEAN_TO_MESSY_CATEGORIES = {
    "medical_clinical",
    "legal_contract",
    "technical_code",
    "academic_research",
    "financial_business",
    "recipe_cooking",
    "meeting_notes",
    "email_professional",
    "creative_writing",
}

# Keywords for categorizing text from generic datasets
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "medical_clinical": [
        "patient", "diagnosis", "clinical", "medication", "dosage", "symptom",
        "treatment", "hospital", "surgery", "prescription", "disease", "therapy",
    ],
    "legal_contract": [
        "contract", "agreement", "clause", "hereby", "whereas", "provision",
        "indemnif", "liability", "jurisdiction", "arbitration", "statute",
    ],
    "technical_code": [
        "function", "variable", "algorithm", "database", "server", "compile",
        "debug", "software", "programming", "API", "framework", "deploy",
    ],
    "academic_research": [
        "research", "study", "hypothesis", "methodology", "findings",
        "journal", "experiment", "analysis", "university", "published",
    ],
    "financial_business": [
        "revenue", "budget", "fiscal", "investment", "quarterly", "profit",
        "dividend", "financial", "accounting", "market", "stock",
    ],
    "recipe_cooking": [
        "recipe", "ingredient", "tablespoon", "teaspoon", "preheat", "oven",
        "simmer", "bake", "cook", "stir", "cuisine", "dish",
    ],
    "meeting_notes": [
        "meeting", "agenda", "minutes", "action item", "attendee", "discuss",
        "decided", "follow-up", "deadline", "committee", "board",
    ],
    "email_professional": [
        "dear", "regards", "sincerely", "attached", "forwarding", "subject",
        "memo", "correspondence", "notify",
    ],
    "creative_writing": [
        "story", "character", "narrative", "novel", "poem", "fiction",
        "protagonist", "literary", "author", "chapter",
    ],
}


def categorize_text(text: str) -> str | None:
    """Assign a category to clean text based on keyword matching.

    Returns the best-matching category or None if no keywords match.
    """
    text_lower = text.lower()
    best_category: str | None = None
    best_score = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > best_score:
            best_score = score
            best_category = category

    # Require at least 2 keyword matches to avoid weak categorization
    return best_category if best_score >= 2 else None


def load_clean_texts(
    source: Any,
    max_samples: int,
    verbose: bool = False,
) -> list[dict[str, str]]:
    """Load and categorize clean texts from a HuggingFace dataset.

    Returns a list of dicts with 'text' and 'category' keys.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' package is required for clean_to_messy mode. "
            "Install it with: pip install datasets"
        )
        return []

    logger.info(
        "Loading clean text dataset: %s (subset: %s)",
        source.dataset,
        source.subset,
    )

    try:
        ds = load_dataset(
            source.dataset,
            source.subset,
            split="train",
            streaming=True,
        )
    except Exception as exc:
        logger.error("Failed to load dataset '%s': %s", source.dataset, exc)
        return []

    categorized: list[dict[str, str]] = []
    seen = 0
    max_scan = max_samples * 20  # Scan more texts than needed to find categorizable ones

    for item in ds:
        if seen >= max_scan or len(categorized) >= max_samples:
            break
        seen += 1

        text = item.get(source.text_column, "")
        if not isinstance(text, str):
            continue

        # For Wikipedia, extract a single paragraph (not whole article)
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if source.min_text_length <= len(p.strip()) <= source.max_text_length
        ]
        if not paragraphs:
            continue

        # Use first suitable paragraph
        para = paragraphs[0]
        category = categorize_text(para)
        if category and category in CLEAN_TO_MESSY_CATEGORIES:
            categorized.append({"text": para, "category": category})

        if verbose and seen % 1000 == 0:
            logger.debug(
                "Scanned %d texts, categorized %d so far", seen, len(categorized)
            )

    logger.info(
        "Loaded %d categorized texts from %d scanned", len(categorized), seen
    )

    # Log distribution
    from collections import Counter
    dist = Counter(item["category"] for item in categorized)
    for cat, count in sorted(dist.items()):
        logger.info("  %s: %d texts", cat, count)

    return categorized


def messify_batch(
    client: LLMClient,
    texts: list[str],
    category: str,
) -> list[str]:
    """Convert a batch of clean texts to messy transcripts via LLM.

    Returns list of messy transcript strings.
    """
    guidance = CATEGORY_GUIDANCE.get(category, "General content.")
    numbered = "\n".join(
        f"{i+1}. {text}" for i, text in enumerate(texts)
    )

    prompt = MESSIFY_PROMPT.format(
        batch_size=len(texts),
        category=category,
        category_guidance=guidance,
        numbered_texts=numbered,
    )

    messages = [{"role": "user", "content": prompt}]
    raw_response = client.generate(messages, max_tokens=8192, temperature=1.0)

    parsed = parse_llm_response(raw_response)
    transcripts: list[str] = []
    for item in parsed:
        if isinstance(item, dict) and "transcript" in item:
            transcripts.append(item["transcript"])

    return transcripts


def generate_clean_to_messy(
    client: LLMClient,
    synth: Any,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, dict[str, int]]:
    """Run the clean→messy generation mode.

    Loads clean text from the configured dataset, categorizes it, and
    generates messy transcript versions via LLM.

    Returns per-category stats.
    """
    source = synth.clean_text_source
    stats: dict[str, dict[str, int]] = {}

    clean_texts = load_clean_texts(source, source.max_samples, verbose)
    if not clean_texts:
        logger.warning("No clean texts loaded — skipping clean_to_messy mode")
        return stats

    # Group by category
    by_category: dict[str, list[str]] = {}
    for item in clean_texts:
        by_category.setdefault(item["category"], []).append(item["text"])

    if dry_run:
        logger.info("")
        logger.info("=== DRY RUN — Clean→Messy Plan ===")
        logger.info("Dataset: %s (%s)", source.dataset, source.subset)
        logger.info("Total categorized texts: %d", len(clean_texts))
        logger.info("")
        logger.info("%-25s %8s %8s", "Category", "Texts", "Batches")
        logger.info("-" * 45)
        for cat in sorted(by_category):
            n = len(by_category[cat])
            batches = math.ceil(n / synth.batch_size)
            logger.info("%-25s %8d %8d", cat, n, batches)
        logger.info("-" * 45)
        return stats

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    for category, texts in by_category.items():
        cat_stats = {
            "generated": 0,
            "failed_batches": 0,
            "rejected_pairs": 0,
            "total_parsed": 0,
        }
        out_path = category_output_path(category)

        logger.info(
            "Clean→messy '%s': messifying %d texts", category, len(texts)
        )

        pbar = None
        if tqdm is not None:
            pbar = tqdm(
                total=len(texts),
                desc=f"  {category} (c2m)",
                unit="pair",
                leave=True,
            )

        for batch_start in range(0, len(texts), synth.batch_size):
            batch_texts = texts[batch_start : batch_start + synth.batch_size]
            try:
                messy_transcripts = messify_batch(client, batch_texts, category)
                cat_stats["total_parsed"] += len(messy_transcripts)

                # Pair up: transcript=messy, output=original clean text
                pairs: list[dict[str, str]] = []
                for clean_text, messy in zip(batch_texts, messy_transcripts):
                    pair = {"transcript": messy, "output": clean_text}
                    reason = validate_pair(pair, category)
                    if reason:
                        logger.debug(
                            "Rejected c2m pair in '%s': %s", category, reason
                        )
                        cat_stats["rejected_pairs"] += 1
                    else:
                        pairs.append(pair)

                if pairs:
                    append_pairs(out_path, pairs, category)
                    cat_stats["generated"] += len(pairs)
                    if pbar:
                        pbar.update(len(pairs))

            except (RuntimeError, ValueError) as exc:
                logger.error("C2M batch for '%s' failed: %s", category, exc)
                cat_stats["failed_batches"] += 1

            # Rate limit
            time.sleep(0.5)

        if pbar:
            pbar.close()

        logger.info(
            "Clean→messy '%s' done: generated=%d, rejected=%d, failed=%d",
            category,
            cat_stats["generated"],
            cat_stats["rejected_pairs"],
            cat_stats["failed_batches"],
        )
        stats[category] = cat_stats

    return stats


# ── Pair validation ────────────────────────────────────────────────────────


def validate_pair(pair: Any, category: str) -> str | None:
    """Validate a single transcript/output pair.

    Returns
    -------
    str | None
        ``None`` if the pair is valid; otherwise a human-readable reason
        for rejection.
    """
    if not isinstance(pair, dict):
        return f"expected dict, got {type(pair).__name__}"

    transcript = pair.get("transcript", "")
    output = pair.get("output", "")

    if not isinstance(transcript, str) or not transcript.strip():
        return "empty or missing 'transcript' field"
    if not isinstance(output, str) or not output.strip():
        return "empty or missing 'output' field"

    # Reject encoding issues: null bytes
    if "\x00" in transcript or "\x00" in output:
        return "contains null bytes"

    # Output should not be excessively longer than input (≤ ~2× input length)
    transcript_len = len(transcript.strip())
    output_len = len(output.strip())
    if transcript_len > 0 and output_len > 2.0 * transcript_len:
        return (
            f"output too long relative to input "
            f"(output={output_len}, input={transcript_len}, "
            f"ratio={output_len / transcript_len:.1f}x)"
        )

    # Reject output that still contains obvious filler words
    # (only standalone fillers, not substrings of real words)
    _filler_pattern = r"\b(?:um|uh|uh huh|you know|I mean|like basically)\b"
    if re.search(_filler_pattern, output, re.IGNORECASE):
        return "output still contains obvious filler words"

    return None


# ── JSON response parsing ──────────────────────────────────────────────────


def parse_llm_response(raw: str) -> list[dict[str, Any]]:
    """Parse the LLM response into a list of pair dicts.

    Handles cases where the LLM wraps the JSON in markdown code fences.

    Raises
    ------
    ValueError
        If the response cannot be parsed as a JSON array.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # Try to find the JSON array within the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not parse LLM response as JSON: {exc}\n"
                    f"Response (first 500 chars): {raw[:500]}"
                ) from exc
        else:
            raise ValueError(
                f"Could not parse LLM response as JSON array: {exc}\n"
                f"Response (first 500 chars): {raw[:500]}"
            ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected JSON array, got {type(parsed).__name__}: "
            f"{raw[:200]}"
        )

    return parsed


# ── File I/O ───────────────────────────────────────────────────────────────


def category_output_path(category: str) -> Path:
    """Return the JSONL path for a given category."""
    return DATA_SYNTHETIC / f"synthetic_{category}.jsonl"


def combined_output_path() -> Path:
    """Return the combined JSONL output path."""
    return DATA_SYNTHETIC / "all_synthetic.jsonl"


def count_existing_records(path: Path) -> int:
    """Count valid JSON records in an existing JSONL file.

    Only lines that parse as valid JSON objects with 'input' and 'output'
    keys are counted, protecting against partial/corrupt lines from crashes.
    """
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if isinstance(record, dict) and "input" in record and "output" in record:
                    count += 1
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSONL line in %s", path)
    return count


def append_pairs(
    path: Path,
    pairs: list[dict[str, str]],
    category: str,
) -> None:
    """Append validated pairs to a JSONL file."""
    with open(path, "a", encoding="utf-8") as fh:
        for pair in pairs:
            record = {
                "input": pair["transcript"],
                "output": pair["output"],
                "category": category,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Cost estimation ────────────────────────────────────────────────────────


def estimate_cost(
    synth: SyntheticConfig,
    total_samples: int,
) -> dict[str, Any]:
    """Estimate API cost for generation.

    Uses rough token estimates: ~1500 input tokens + ~2000 output tokens
    per batch call for Anthropic Claude Sonnet pricing.
    """
    total_batches = math.ceil(total_samples / synth.batch_size)

    input_tokens_per_batch = 1500
    output_tokens_per_batch = 2000

    total_input_tokens = total_batches * input_tokens_per_batch
    total_output_tokens = total_batches * output_tokens_per_batch

    # Pricing varies by provider/model; use Anthropic Sonnet as baseline
    # claude-sonnet-4-20250514: $3/1M input, $15/1M output
    cost_per_m_input = 3.0
    cost_per_m_output = 15.0

    input_cost = (total_input_tokens / 1_000_000) * cost_per_m_input
    output_cost = (total_output_tokens / 1_000_000) * cost_per_m_output
    total_cost = input_cost + output_cost

    return {
        "total_samples": total_samples,
        "batch_size": synth.batch_size,
        "total_batches": total_batches,
        "est_input_tokens": total_input_tokens,
        "est_output_tokens": total_output_tokens,
        "est_cost_usd": round(total_cost, 2),
        "pricing_note": (
            f"Estimate based on ~{input_tokens_per_batch} input + "
            f"~{output_tokens_per_batch} output tokens/batch at "
            f"${cost_per_m_input}/M input, ${cost_per_m_output}/M output "
            f"(Anthropic Sonnet baseline)"
        ),
    }


# ── Category plan ──────────────────────────────────────────────────────────


def build_category_plan(
    synth: SyntheticConfig,
    total_samples: int,
) -> list[dict[str, Any]]:
    """Build a list of per-category generation targets.

    Each entry has: category, target_count, num_batches, existing_count.
    """
    plan: list[dict[str, Any]] = []
    for category, proportion in synth.categories.items():
        target = round(total_samples * proportion)
        if target == 0:
            continue
        existing = count_existing_records(category_output_path(category))
        remaining = max(0, target - existing)
        num_batches = math.ceil(remaining / synth.batch_size) if remaining > 0 else 0
        plan.append(
            {
                "category": category,
                "proportion": proportion,
                "target": target,
                "existing": existing,
                "remaining": remaining,
                "num_batches": num_batches,
            }
        )
    return plan


# ── Batch generation ───────────────────────────────────────────────────────


def generate_batch(
    client: LLMClient,
    category: str,
    batch_size: int,
) -> tuple[list[dict[str, str]], int, int]:
    """Generate one batch of pairs from the LLM.

    Returns
    -------
    tuple
        (valid_pairs, total_parsed, rejected_count)
    """
    guidance = CATEGORY_GUIDANCE.get(category, "General content for this category.")

    prompt = GENERATION_PROMPT.format(
        batch_size=batch_size,
        category=category,
        category_guidance=guidance,
    )

    messages = [{"role": "user", "content": prompt}]

    raw_response = client.generate(
        messages,
        max_tokens=8192,
        temperature=1.0,
    )

    parsed = parse_llm_response(raw_response)

    valid: list[dict[str, str]] = []
    rejected = 0
    for pair in parsed:
        reason = validate_pair(pair, category)
        if reason:
            logger.debug(
                "Rejected pair in '%s': %s (transcript: %.60s...)",
                category,
                reason,
                pair.get("transcript", "")[:60],
            )
            rejected += 1
        else:
            valid.append(pair)

    return valid, len(parsed), rejected


def generate_category(
    client: LLMClient,
    category: str,
    plan_entry: dict[str, Any],
    synth: SyntheticConfig,
    dry_run: bool = False,
) -> dict[str, int]:
    """Generate all pairs for a single category.

    Uses a target-based while loop so rejections/failures don't cause
    silent under-generation.  Caps accepted pairs to ``still_needed``
    so the category doesn't overshoot.

    Returns per-category stats dict.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    target = plan_entry["target"]
    existing = plan_entry["existing"]

    out_path = category_output_path(category)

    stats = {
        "generated": 0,
        "failed_batches": 0,
        "rejected_pairs": 0,
        "total_parsed": 0,
    }

    current_count = existing
    if current_count >= target:
        logger.info(
            "Category '%s': target=%d, existing=%d — already complete, skipping",
            category,
            target,
            existing,
        )
        return stats

    if dry_run:
        remaining = target - current_count
        num_batches = math.ceil(remaining / synth.batch_size)
        logger.info(
            "Category '%s': would generate %d samples in ~%d batches (target=%d, existing=%d)",
            category,
            remaining,
            num_batches,
            target,
            existing,
        )
        return stats

    logger.info(
        "Category '%s': generating up to %d samples (target=%d, existing=%d)",
        category,
        target - current_count,
        target,
        existing,
    )

    # Use tqdm for progress if available
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=target - existing,
            desc=f"  {category}",
            unit="pair",
            leave=True,
        )

    max_consecutive_failures = 5
    consecutive_failures = 0

    while current_count < target:
        still_needed = target - current_count
        batch_size = min(synth.batch_size, still_needed)

        try:
            valid_pairs, total_parsed, rejected = generate_batch(
                client, category, batch_size
            )
            stats["total_parsed"] += total_parsed
            stats["rejected_pairs"] += rejected
            consecutive_failures = 0

            if valid_pairs:
                # Cap to avoid overshooting
                accepted = valid_pairs[:still_needed]
                append_pairs(out_path, accepted, category)
                current_count += len(accepted)
                stats["generated"] += len(accepted)
                if pbar:
                    pbar.update(len(accepted))

        except (RuntimeError, ValueError) as exc:
            logger.error(
                "Batch for '%s' failed: %s",
                category,
                exc,
            )
            stats["failed_batches"] += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    "Category '%s': %d consecutive failures, stopping",
                    category,
                    max_consecutive_failures,
                )
                break
            continue

        # Small delay between batches to respect rate limits
        if current_count < target:
            time.sleep(0.5)

    if pbar:
        pbar.close()

    if current_count < target:
        logger.warning(
            "Category '%s' under target: generated=%d, target=%d (shortfall=%d)",
            category,
            current_count,
            target,
            target - current_count,
        )

    logger.info(
        "Category '%s' done: generated=%d, rejected=%d, failed_batches=%d, total=%d/%d",
        category,
        stats["generated"],
        stats["rejected_pairs"],
        stats["failed_batches"],
        current_count,
        target,
    )

    return stats


# ── Combined output ────────────────────────────────────────────────────────


def build_combined_output(categories: list[str]) -> int:
    """Merge all per-category JSONL files into a single combined file.

    Returns total record count.
    """
    combined_path = combined_output_path()
    total = 0

    with open(combined_path, "w", encoding="utf-8") as out_fh:
        for category in categories:
            cat_path = category_output_path(category)
            if not cat_path.exists():
                continue
            with open(cat_path, encoding="utf-8") as in_fh:
                for line in in_fh:
                    stripped = line.strip()
                    if stripped:
                        out_fh.write(stripped + "\n")
                        total += 1

    logger.info("Combined output: %d records → %s", total, combined_path)
    return total


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> Any:
    """Parse CLI arguments."""
    parser = base_arg_parser(
        "Generate synthetic transcript-cleanup pairs via LLM API."
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=None,
        help="Override num_samples from config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing category files and regenerate from scratch.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success, 1 on failure."""
    args = parse_args()
    log = setup_logging(args.verbose)

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Configuration error: %s", exc)
        return 1

    synth = cfg.dataset.synthetic

    if not synth.enabled:
        log.info("Synthetic data generation is disabled in config. Exiting.")
        return 0

    total_samples = args.synthetic_samples or synth.num_samples
    log.info("Target: %d total synthetic samples", total_samples)

    # Build plan (before ensure_dirs so dry-run has no side effects)
    plan = build_category_plan(synth, total_samples)

    # ── Dry-run output ────────────────────────────────────────────────────
    if args.dry_run:
        modes = synth.generation_modes
        log.info("=== DRY RUN — Generation Plan ===")
        log.info("Provider: %s | Model: %s", synth.provider, synth.model)
        log.info(
            "Modes: generate_both=%s, clean_to_messy=%s",
            modes.generate_both,
            modes.clean_to_messy,
        )

        if not modes.generate_both and not modes.clean_to_messy:
            log.info(
                "Both generation modes disabled — "
                "pipeline will use only pre-pulled datasets from step 02."
            )
            return 0

        if modes.generate_both:
            cost = estimate_cost(synth, total_samples)
            log.info("")
            log.info("--- Generate-Both Mode ---")
            log.info("Total samples: %d | Batch size: %d", total_samples, synth.batch_size)
            log.info("")
            log.info("%-25s %8s %8s %8s %8s", "Category", "Target", "Existing", "Remaining", "Batches")
            log.info("-" * 65)
            total_remaining = 0
            total_batches = 0
            for entry in plan:
                log.info(
                    "%-25s %8d %8d %8d %8d",
                    entry["category"],
                    entry["target"],
                    entry["existing"],
                    entry["remaining"],
                    entry["num_batches"],
                )
                total_remaining += entry["remaining"]
                total_batches += entry["num_batches"]
            log.info("-" * 65)
            log.info("%-25s %8s %8s %8d %8d", "TOTAL", "", "", total_remaining, total_batches)
            log.info("")
            log.info("Estimated cost: $%.2f", cost["est_cost_usd"])
            log.info("  %s", cost["pricing_note"])

        if modes.clean_to_messy:
            log.info("")
            log.info("--- Clean→Messy Mode ---")
            src = synth.clean_text_source
            log.info("Dataset: %s (%s)", src.dataset, src.subset)
            log.info("Max samples: %d", src.max_samples)
            log.info(
                "Text length filter: %d-%d chars",
                src.min_text_length,
                src.max_text_length,
            )
            log.info(
                "Eligible categories: %s",
                ", ".join(sorted(CLEAN_TO_MESSY_CATEGORIES)),
            )
            log.info("(Dataset will be loaded and categorized during real run)")

        return 0

    # ── Real run ──────────────────────────────────────────────────────────
    ensure_dirs()
    modes = synth.generation_modes

    if not modes.generate_both and not modes.clean_to_messy:
        log.info(
            "Both generation modes disabled — "
            "pipeline will use only pre-pulled datasets from step 02."
        )
        return 0

    # Handle --force: remove existing per-category files
    if args.force:
        for category in synth.categories:
            cat_path = category_output_path(category)
            if cat_path.exists():
                log.info("Removing existing file (--force): %s", cat_path)
                cat_path.unlink()
        combined = combined_output_path()
        if combined.exists():
            combined.unlink()
        # Rebuild plan after deleting files
        plan = build_category_plan(synth, total_samples)

    # ── Create LLM client ─────────────────────────────────────────────────
    try:
        client = create_client_from_config(
            provider=synth.provider,
            model=synth.model,
            api_key_env=synth.api_key_env,
            base_url=synth.base_url,
        )
    except (EnvironmentError, ImportError, ValueError) as exc:
        log.error("Failed to create LLM client: %s", exc)
        return 1

    # ── Generate-both mode ────────────────────────────────────────────────
    start_time = time.monotonic()
    all_stats: dict[str, dict[str, int]] = {}

    if modes.generate_both:
        log.info("")
        log.info("=== Running Generate-Both Mode ===")
        for entry in plan:
            category = entry["category"]
            cat_stats = generate_category(
                client=client,
                category=category,
                plan_entry=entry,
                synth=synth,
                dry_run=False,
            )
            all_stats[category] = cat_stats
    else:
        log.info("Generate-both mode disabled, skipping.")

    # ── Clean→messy mode ──────────────────────────────────────────────────
    c2m_stats: dict[str, dict[str, int]] = {}

    if modes.clean_to_messy:
        log.info("")
        log.info("=== Running Clean→Messy Mode ===")
        c2m_stats = generate_clean_to_messy(
            client=client,
            synth=synth,
            dry_run=False,
            verbose=args.verbose,
        )
    else:
        log.info("Clean-to-messy mode disabled, skipping.")

    # ── Build combined output ─────────────────────────────────────────────
    all_categories = list(synth.categories.keys())
    combined_count = build_combined_output(all_categories)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    total_generated = sum(s["generated"] for s in all_stats.values())
    total_rejected = sum(s["rejected_pairs"] for s in all_stats.values())
    total_failed = sum(s["failed_batches"] for s in all_stats.values())
    c2m_generated = sum(s["generated"] for s in c2m_stats.values())
    c2m_rejected = sum(s["rejected_pairs"] for s in c2m_stats.values())
    c2m_failed = sum(s["failed_batches"] for s in c2m_stats.values())

    log.info("")
    log.info("=== Generation Summary ===")
    log.info("Elapsed: %.1fs", elapsed)

    if modes.generate_both:
        log.info("")
        log.info("Generate-Both: %d generated, %d rejected, %d failed batches",
                 total_generated, total_rejected, total_failed)
    if modes.clean_to_messy:
        log.info("Clean→Messy:   %d generated, %d rejected, %d failed batches",
                 c2m_generated, c2m_rejected, c2m_failed)

    log.info("Grand total:   %d pairs", total_generated + c2m_generated)
    log.info("Combined file: %d records", combined_count)
    log.info("")
    log.info("%-25s %10s %10s %10s", "Category", "Generated", "Rejected", "Failed")
    log.info("-" * 60)
    for category in all_categories:
        g = all_stats.get(category, {"generated": 0, "rejected_pairs": 0, "failed_batches": 0})
        c = c2m_stats.get(category, {"generated": 0, "rejected_pairs": 0, "failed_batches": 0})
        log.info(
            "%-25s %10d %10d %10d",
            category,
            g["generated"] + c["generated"],
            g["rejected_pairs"] + c["rejected_pairs"],
            g["failed_batches"] + c["failed_batches"],
        )
    log.info("-" * 60)

    if total_failed > 0:
        log.warning("Some batches failed — generated data may be below target.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
