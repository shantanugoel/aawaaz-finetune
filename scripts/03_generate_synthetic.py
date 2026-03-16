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
        "'you know', 'so basically', changes of topic. "
        "The OUTPUT must STILL properly format this: add punctuation, fix grammar, remove "
        "ALL fillers. Casual content does NOT mean casual formatting. The output should be "
        "clean, readable text even if the content is informal."
    ),
    "email_professional": (
        "Someone DICTATING a business email out loud. They say things like 'dear mister smith' "
        "or 'hi team comma', 'new paragraph', 'kind regards'. The transcript should sound like "
        "someone talking their email out, not like a typed email with ums added. "
        "Include salutations, sign-offs, and professional language but spoken naturally. "
        "Numbers should be spoken: 'the budget is twelve thousand five hundred dollars'. "
        "The OUTPUT should be a properly formatted email with correct salutations, paragraph "
        "breaks, and formatted numbers ($12,500)."
    ),
    "technical_code": (
        "Dictating code, CLI commands, error messages, or technical documentation. "
        "Include function names ('def process underscore data'), file paths "
        "('slash user slash bin slash app'), docker commands, SQL queries, variable names. "
        "The speaker might spell out symbols: 'open paren', 'close bracket', 'equals equals'. "
        "The OUTPUT should have properly formatted code with actual symbols, file paths, etc. "
        "CRITICAL: Do NOT hallucinate code details — if the speaker said 'import react', "
        "do NOT expand it to 'import React from \"react\"' unless they specifically said that."
    ),
    "medical_clinical": (
        "Dictating patient notes, clinical observations, medication names and dosages, "
        "lab results, diagnoses. Include medical abbreviations spoken out "
        "('b p one twenty over eighty', 'patient presented with shortness of breath'). "
        "The OUTPUT should format these correctly (BP: 120/80, SOB) while preserving EVERY "
        "clinical detail. Missing a dosage or lab value is a critical failure."
    ),
    "legal_contract": (
        "Dictating contract clauses, legal terminology, section references. "
        "Include formal legal phrasing spoken naturally: 'whereas the party of the first part', "
        "'section four point two', 'hereinafter referred to as'. "
        "The OUTPUT must preserve EVERY legal term, section number, party name, and clause "
        "exactly. Do NOT paraphrase or simplify legal language — keep it verbatim minus fillers. "
        "Do NOT add 'for clarity' explanations or restructure clauses."
    ),
    "meeting_notes": (
        "Someone dictating meeting minutes or action items. Include attendee names, "
        "deadlines ('by next friday', 'end of q two'), decisions made, who is responsible. "
        "The OUTPUT should be well-structured meeting notes with bullet points or numbered "
        "items. Preserve EVERY name, date, action item, and decision."
    ),
    "recipe_cooking": (
        "Dictating recipes or cooking instructions. Ingredient quantities spoken as words "
        "('two cups of flour', 'three hundred fifty degrees'). Step-by-step with natural "
        "speech: 'then you wanna let it sit for like twenty minutes'. "
        "The OUTPUT should have a properly formatted ingredients list and numbered steps "
        "with all quantities converted to written form (2 cups, 350°F, 20 minutes)."
    ),
    "academic_research": (
        "Dictating research notes, paper summaries, citations, methodology descriptions. "
        "Include author names, journal titles, years, statistical values spoken as words "
        "('p less than point oh five', 'n equals forty two'). "
        "The OUTPUT should format citations, statistics (p < 0.05, n = 42), and technical "
        "terms properly. Preserve every author name, statistic, and finding exactly."
    ),
    "creative_writing": (
        "Dictating stories, poems, blog posts, personal essays, or journal entries. "
        "Include descriptive language, dialogue (said with 'quote' / 'end quote' or "
        "'open quote'), and natural pauses or direction changes. "
        "The OUTPUT should format dialogue with quotation marks, add proper paragraph "
        "breaks, and clean up prose while preserving the writer's voice and every detail. "
        "Remove ALL fillers even if they feel like part of the writing style — they are "
        "from the dictation process, not the content."
    ),
    "financial_business": (
        "Dictating financial reports, budget summaries, invoice details, or business plans. "
        "Include dollar amounts ('twelve hundred dollars'), percentages ('fifteen percent'), "
        "dates ('q three twenty twenty five'), company names. "
        "The OUTPUT must format all numbers correctly ($1,200, 15%, Q3 2025) and preserve "
        "every financial figure exactly."
    ),
    "shopping_lists": (
        "Dictating a shopping or to-do list quickly. Items in rapid succession, sometimes "
        "with quantities ('like three avocados', 'a dozen eggs'), brands, or notes. Often "
        "short and list-oriented. "
        "The OUTPUT should be a clean bulleted or numbered list with formatted quantities."
    ),
    "self_corrections_heavy": (
        "Focus heavily on self-correction patterns. The speaker frequently changes their mind: "
        "'the meeting is at two pm wait no three pm', "
        "'send it to john at gmail no actually his work email', "
        "'we need five hundred units scratch that make it six hundred'. "
        "Include multiple corrections per example. Mix corrections of: names, numbers, dates, "
        "instructions, and facts. "
        "The OUTPUT must apply ALL corrections correctly — keep ONLY the final corrected "
        "version. If the speaker said 'tuesday no wednesday', output Wednesday. "
        "Do NOT include both wrong and right versions. Do NOT add notes like 'corrected from'. "
        "Preserve all non-corrected content exactly."
    ),
}

# ── Generation prompt template ─────────────────────────────────────────────

GENERATION_PROMPT = """\
You are generating training data for a speech transcript cleanup model. \
Generate {batch_size} realistic input/output pairs for the category: {category}.

## RULES FOR THE "transcript" FIELD (the messy input)

The transcript must look like real output from a modern speech-to-text engine (like Whisper). \
Whisper output characteristics:
- MAY have SOME punctuation (periods, commas) but it is often inconsistent — sometimes present, sometimes missing within the same transcript
- MAY have SOME capitalization — proper nouns are often capitalized, sentence starts may or may not be
- Numbers are USUALLY spelled out as words ("twenty five", "two thousand") but Whisper sometimes outputs digits for common numbers
- No paragraph breaks — everything runs together as one or a few long blocks of text
- Filler words (um, uh, like, basically, actually, you know, so, I mean) placed where real people hesitate
- Self-corrections ("the meeting is on Tuesday wait no Wednesday")
- Spoken formatting cues where natural ("new line", "bullet point", "colon", "dash")
- Run-on sentences where the speaker's natural pauses are not reflected as punctuation
- NOT every sentence has fillers — vary the messiness naturally, some stretches are clean

CRITICAL — DO NOT:
- Generate perfectly clean text with a few "um"s inserted — that is fake and obvious
- Use perfectly consistent formatting throughout (real speech transcription varies in quality)
- Add fillers mechanically at regular intervals

DO:
- Vary quality throughout — some parts more coherent, others messier
- Place fillers where people actually hesitate (before complex words, when changing topics, when uncertain)
- Include natural speech patterns: false starts, topic changes, backtracking
- Make it sound like someone SPEAKING, not someone TYPING

## RULES FOR THE "output" FIELD (the clean version)

The output is what the cleanup model should produce. It must:
- Remove ALL fillers and stutters completely
- Apply self-corrections (keep ONLY the corrected version, drop the mistake entirely)
- Add proper, consistent punctuation and capitalization throughout
- Convert spoken numbers/dates/currency to written form ($500, January 15, 2025)
- Apply proper formatting (bullet lists, paragraphs, code blocks) where the speaker indicated them
- Convert spoken emoji descriptions to actual emoji characters

CRITICAL:
- Preserve EVERY substantive fact, name, number, and instruction from the input — do NOT drop or summarize anything
- Only REMOVE noise (fillers, stutters, corrections) and ADD formatting (punctuation, structure)
- Do NOT add information, context, conclusions, or clarifications the speaker did not say
- Do NOT rephrase in your own words — preserve the speaker's wording minus the noise

## Category-specific guidance for "{category}":
{category_guidance}

## Response format
Respond with a JSON array of objects, each with "transcript" and "output" keys. No other text.
Vary the length: some short (1-2 sentences, ~15-30 words), some medium (paragraph, ~50-100 words), some long (multiple paragraphs, ~150-300 words)."""

# ── Messify prompt for clean→messy mode ────────────────────────────────────

MESSIFY_PROMPT = """\
Convert these {batch_size} clean, well-formatted texts into realistic speech-to-text (ASR) \
transcripts, as if someone SPOKE this content aloud and it was transcribed by Whisper.

For each text, produce a realistic transcript that:
- Sounds like natural speech with fillers (um, uh, like, you know, basically) placed where \
people actually hesitate
- Has inconsistent punctuation — some present, some missing
- Has some capitalization but not perfectly consistent
- Numbers mostly as spoken words ("twenty five" not "25")
- Run-on sentences, natural topic transitions
- Self-corrections where natural ("wait no I meant", "actually scratch that")
- Spoken formatting cues where the original has structure ("bullet point", "new line")
- Varies the messiness — not every sentence needs fillers

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
