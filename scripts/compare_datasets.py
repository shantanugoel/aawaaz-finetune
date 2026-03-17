#!/usr/bin/env python3
"""Compare two synthetic transcript-cleanup datasets: script-generated vs prompt-generated."""

import json
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEED = 42
SAMPLE_SIZE = 10
VOCAB_SAMPLE = 100

CATEGORIES = [
    "academic_research", "casual_conversation", "creative_writing",
    "email_professional", "financial_business", "legal_contract",
    "medical_clinical", "meeting_notes", "recipe_cooking",
    "self_corrections_heavy", "shopping_lists", "technical_code",
]

SCRIPT_DIR = Path("/Users/shantanugoel/dev/aawaaz-finetune/data/synthetic")
PROMPT_DIR = Path("/Users/shantanugoel/dev/aawaaz-finetune/data/prepared")

FILLER_PATTERNS = [
    r'\bum\b', r'\buh\b', r'\blike\b', r'\byou know\b', r'\bbasically\b',
    r'\bactually\b', r'\bI mean\b', r'\bliterally\b', r'\bhonestly\b',
    r'\bso\b(?=\s+(?:um|uh|like|the|we|I|it|he|she|they))',
]

# Standalone filler detection (more precise for output checking)
OUTPUT_FILLERS = [
    r'\bum\b', r'\buh\b', r'\byou know\b', r'\bI mean\b',
    r'\bbasically\b', r'\bliterally\b',
]

SELF_CORRECTION_MARKERS = [
    r'\bwait no\b', r'\bactually\b', r'\bI mean\b', r'\bscratch that\b',
    r'\blet me rephrase\b', r'\bno wait\b', r'\bsorry\b',
    r'\bno\s+(?:it\'?s|that\'?s|I meant)\b', r'\bwait\b.*\bno\b',
]

FALSE_START_PATTERNS = [
    r'\b\w+\s*--\s*', r'\.\.\.',  r'—',
    r'\bso\s+(?:the|um|uh)\b',
]

SPOKEN_NUMBER_PATTERNS = [
    r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b',
    r'\b(?:eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b',
    r'\b(?:thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\bpoint\s+\w+\b',
]


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, one JSON object per line."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_all_data() -> tuple[dict[str, list], dict[str, list]]:
    """Load both datasets organized by category."""
    script_data: dict[str, list] = {}
    prompt_data: dict[str, list] = {}
    for cat in CATEGORIES:
        script_file = SCRIPT_DIR / f"synthetic_{cat}.jsonl"
        prompt_file = PROMPT_DIR / f"{cat}.jsonl"
        script_data[cat] = load_jsonl(script_file) if script_file.exists() else []
        prompt_data[cat] = load_jsonl(prompt_file) if prompt_file.exists() else []
    return script_data, prompt_data


def count_fillers(text: str) -> int:
    """Count filler words in text."""
    total = 0
    text_lower = text.lower()
    for pat in FILLER_PATTERNS:
        total += len(re.findall(pat, text_lower))
    return total


def count_output_fillers(text: str) -> list[str]:
    """Find filler words remaining in output."""
    found = []
    text_lower = text.lower()
    for pat in OUTPUT_FILLERS:
        matches = re.findall(pat, text_lower)
        found.extend(matches)
    return found


def filler_position_analysis(text: str) -> dict[str, int]:
    """Check if fillers are at sentence boundaries (mechanical) or mid-phrase (natural)."""
    sentences = re.split(r'[.!?]+', text)
    boundary = 0
    mid_phrase = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = sent.lower().split()
        if len(words) < 2:
            continue
        for i, w in enumerate(words):
            if w in ('um', 'uh', 'like', 'basically', 'honestly'):
                if i <= 1 or i >= len(words) - 1:
                    boundary += 1
                else:
                    mid_phrase += 1
    return {"boundary": boundary, "mid_phrase": mid_phrase}


def count_self_corrections(text: str) -> int:
    """Count self-correction markers."""
    total = 0
    text_lower = text.lower()
    for pat in SELF_CORRECTION_MARKERS:
        total += len(re.findall(pat, text_lower))
    return total


def check_multi_step_corrections(text: str) -> bool:
    """Check for multi-step corrections like 'Tuesday, wait no Wednesday, actually Thursday'."""
    pattern = r'(?:wait\s+no|no\s+wait|actually|sorry).*(?:wait\s+no|no\s+wait|actually|sorry)'
    return bool(re.search(pattern, text.lower()))


def written_text_smell(text: str) -> dict[str, Any]:
    """Check for signs the input reads like written text rather than speech."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    caps_start = sum(1 for s in sentences if s and s[0].isupper()) if sentences else 0
    caps_pct = caps_start / len(sentences) * 100 if sentences else 0

    has_bullets = bool(re.search(r'^\s*[-•*]\s', text, re.MULTILINE))
    has_numbered = bool(re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE))
    has_headers = bool(re.search(r'^#+\s', text, re.MULTILINE))

    period_endings = sum(1 for s in re.split(r'\n', text) if s.strip().endswith('.'))
    total_lines = sum(1 for s in re.split(r'\n', text) if s.strip())
    period_pct = period_endings / total_lines * 100 if total_lines else 0

    return {
        "caps_start_pct": round(caps_pct, 1),
        "has_bullets": has_bullets,
        "has_numbered_lists": has_numbered,
        "has_headers": has_headers,
        "period_ending_pct": round(period_pct, 1),
    }


def check_content_preservation(inp: str, out: str) -> float:
    """Check what fraction of key nouns/names from input appear in output."""
    # Extract capitalized words (likely proper nouns) and long words (likely key terms)
    inp_lower = inp.lower()
    out_lower = out.lower()

    # Get significant words (4+ chars, not fillers)
    filler_set = {'like', 'just', 'really', 'actually', 'basically', 'literally',
                  'honestly', 'that', 'this', 'with', 'from', 'they', 'them',
                  'have', 'been', 'were', 'would', 'could', 'should', 'about',
                  'which', 'their', 'there', 'know', 'mean', 'think', 'going',
                  'some', 'what', 'when', 'than', 'also', 'very', 'much',
                  'well', 'yeah', 'okay'}
    inp_words = set(re.findall(r'\b[a-z]{4,}\b', inp_lower)) - filler_set
    out_words = set(re.findall(r'\b[a-z]{4,}\b', out_lower))

    if not inp_words:
        return 1.0
    preserved = inp_words & out_words
    return len(preserved) / len(inp_words)


def check_hallucination(inp: str, out: str) -> list[str]:
    """Find significant words in output that are absent from input (potential hallucinations)."""
    inp_lower = inp.lower()
    out_lower = out.lower()

    common = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
              'had', 'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how',
              'its', 'may', 'new', 'now', 'old', 'see', 'way', 'who', 'did',
              'get', 'let', 'say', 'she', 'too', 'use', 'will', 'with', 'that',
              'this', 'from', 'they', 'been', 'have', 'each', 'make', 'like',
              'than', 'them', 'then', 'into', 'just', 'over', 'such', 'take',
              'year', 'also', 'back', 'come', 'made', 'find', 'here', 'more',
              'only', 'well', 'what', 'when', 'your', 'which', 'their', 'there',
              'these', 'about', 'would', 'could', 'should', 'approximately',
              'regarding', 'specifically', 'mentioned', 'discussed', 'noted',
              'included', 'regarding', 'however', 'therefore', 'initial',
              'effective', 'following', 'confirmed', 'estimated', 'current'}

    out_words = set(re.findall(r'\b[a-z]{5,}\b', out_lower)) - common
    inp_words = set(re.findall(r'\b[a-z]{3,}\b', inp_lower))

    hallucinated = []
    for w in out_words:
        # Check if word or close variant exists in input
        if w not in inp_words and w + 's' not in inp_words and w.rstrip('s') not in inp_words:
            if w + 'ed' not in inp_words and w + 'ing' not in inp_words:
                if w.rstrip('ed') not in inp_words and w.rstrip('ing') not in inp_words:
                    hallucinated.append(w)
    return hallucinated


def check_number_conversion(inp: str, out: str) -> dict[str, Any]:
    """Check if spoken numbers are converted to formatted form in output."""
    spoken_nums = []
    for pat in SPOKEN_NUMBER_PATTERNS:
        spoken_nums.extend(re.findall(pat, inp.lower()))

    has_formatted = bool(re.search(r'\$[\d,.]+|\d+%|\d{1,3}(?:,\d{3})+|\d+\.\d+', out))
    has_spoken_in_output = []
    for pat in SPOKEN_NUMBER_PATTERNS:
        found = re.findall(pat, out.lower())
        has_spoken_in_output.extend(found)

    return {
        "spoken_in_input": len(spoken_nums),
        "formatted_in_output": has_formatted,
        "spoken_remaining_in_output": len(has_spoken_in_output),
    }


def output_formatting_check(text: str) -> dict[str, Any]:
    """Check output formatting quality."""
    has_caps_start = text and text[0].isupper()
    has_period_end = text.rstrip().endswith('.') or text.rstrip().endswith('!') or text.rstrip().endswith('?')
    has_bullets = bool(re.search(r'^\s*[-•*]\s', text, re.MULTILINE))
    has_paragraphs = '\n\n' in text
    has_headers = bool(re.search(r'^#+\s', text, re.MULTILINE))
    has_code_formatting = '`' in text

    return {
        "starts_capitalized": has_caps_start,
        "ends_punctuation": has_period_end,
        "has_bullets": has_bullets,
        "has_paragraphs": has_paragraphs,
        "has_headers": has_headers,
        "has_code_formatting": has_code_formatting,
    }


def vocab_diversity(samples: list[dict], n: int = VOCAB_SAMPLE) -> float:
    """Compute unique_words / total_words for a sample of texts."""
    rng = random.Random(SEED + 1)
    chosen = rng.sample(samples, min(n, len(samples)))
    all_words = []
    for s in chosen:
        words = re.findall(r'\b\w+\b', s.get("input", "").lower())
        all_words.extend(words)
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def compute_stats(samples: list[dict]) -> dict[str, Any]:
    """Compute statistical measures across all samples."""
    if not samples:
        return {}

    inp_chars = [len(s["input"]) for s in samples]
    inp_words = [len(s["input"].split()) for s in samples]
    out_chars = [len(s["output"]) for s in samples]
    out_words = [len(s["output"].split()) for s in samples]

    compression = [
        len(s["output"]) / len(s["input"]) if len(s["input"]) > 0 else 0
        for s in samples
    ]

    filler_densities = []
    correction_counts = []
    for s in samples:
        words = s["input"].split()
        fc = count_fillers(s["input"])
        density = fc / len(words) * 100 if words else 0
        filler_densities.append(density)
        correction_counts.append(count_self_corrections(s["input"]))

    # Length distribution
    import statistics
    inp_std = statistics.stdev(inp_words) if len(inp_words) > 1 else 0
    inp_cv = inp_std / (sum(inp_words) / len(inp_words)) if sum(inp_words) > 0 else 0

    return {
        "count": len(samples),
        "avg_input_chars": round(sum(inp_chars) / len(inp_chars), 1),
        "avg_input_words": round(sum(inp_words) / len(inp_words), 1),
        "avg_output_chars": round(sum(out_chars) / len(out_chars), 1),
        "avg_output_words": round(sum(out_words) / len(out_words), 1),
        "avg_compression": round(sum(compression) / len(compression), 3),
        "avg_filler_density": round(sum(filler_densities) / len(filler_densities), 2),
        "avg_corrections": round(sum(correction_counts) / len(correction_counts), 2),
        "input_length_cv": round(inp_cv, 3),
        "input_word_min": min(inp_words),
        "input_word_max": max(inp_words),
        "input_word_std": round(inp_std, 1),
        "vocab_diversity": round(vocab_diversity(samples), 4),
    }


def truncate(text: str, max_len: int = 120) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def indent(text: str, prefix: str = "    ") -> str:
    """Indent text block."""
    return textwrap.indent(text, prefix)


def print_divider(char: str = "=", width: int = 100) -> None:
    print(char * width)


def print_header(title: str) -> None:
    print()
    print_divider("=")
    print(f"  {title}")
    print_divider("=")
    print()


def print_subheader(title: str) -> None:
    print()
    print_divider("-", 80)
    print(f"  {title}")
    print_divider("-", 80)


def analyze_sample(sample: dict, label: str) -> dict[str, Any]:
    """Analyze a single sample comprehensively."""
    inp = sample["input"]
    out = sample["output"]

    fillers = count_fillers(inp)
    filler_pos = filler_position_analysis(inp)
    corrections = count_self_corrections(inp)
    smell = written_text_smell(inp)
    remaining_fillers = count_output_fillers(out)
    preservation = check_content_preservation(inp, out)
    hallucinated = check_hallucination(inp, out)
    num_conv = check_number_conversion(inp, out)
    fmt = output_formatting_check(out)

    return {
        "label": label,
        "input_len": len(inp.split()),
        "output_len": len(out.split()),
        "filler_count": fillers,
        "filler_boundary": filler_pos["boundary"],
        "filler_mid_phrase": filler_pos["mid_phrase"],
        "corrections": corrections,
        "has_multi_step_correction": check_multi_step_corrections(inp),
        "written_smell": smell,
        "remaining_fillers": remaining_fillers,
        "content_preservation": round(preservation, 3),
        "hallucinated_words": hallucinated[:5],
        "number_conversion": num_conv,
        "formatting": fmt,
    }


def category_specific_check(cat: str, script_samples: list[dict],
                             prompt_samples: list[dict]) -> str:
    """Run category-specific deeper checks and return report text."""
    lines = []

    if cat == "self_corrections_heavy":
        lines.append("  DEEP CHECK: Self-Correction Complexity")
        for label, samples in [("SCRIPT", script_samples), ("PROMPT", prompt_samples)]:
            multi = sum(1 for s in samples if check_multi_step_corrections(s["input"]))
            avg_corr = sum(count_self_corrections(s["input"]) for s in samples) / max(len(samples), 1)
            lines.append(f"    {label}: {multi}/{len(samples)} have multi-step corrections, "
                         f"avg {avg_corr:.1f} correction markers per sample")
            # Show example
            for s in samples[:3]:
                if check_multi_step_corrections(s["input"]):
                    lines.append(f"    Example multi-step: \"{truncate(s['input'], 150)}\"")
                    break
            else:
                lines.append(f"    (No multi-step correction found in first 3 samples)")

    elif cat == "shopping_lists":
        lines.append("  DEEP CHECK: Output List Structure")
        for label, samples in [("SCRIPT", script_samples), ("PROMPT", prompt_samples)]:
            bulleted = sum(1 for s in samples if re.search(r'^\s*[-•*]\s', s["output"], re.MULTILINE))
            numbered = sum(1 for s in samples if re.search(r'^\s*\d+[.)]\s', s["output"], re.MULTILINE))
            flat = len(samples) - bulleted - numbered
            lines.append(f"    {label}: {bulleted} bulleted, {numbered} numbered, {flat} flat paragraph")
            if samples:
                lines.append(f"    Example output: \"{truncate(samples[0]['output'], 150)}\"")

    elif cat == "technical_code":
        lines.append("  DEEP CHECK: Code Formatting")
        for label, samples in [("SCRIPT", script_samples), ("PROMPT", prompt_samples)]:
            backtick = sum(1 for s in samples if '`' in s["output"])
            underscore_spoken = sum(1 for s in samples if 'underscore' in s["input"].lower())
            underscore_converted = sum(
                1 for s in samples
                if 'underscore' in s["input"].lower() and '_' in s["output"]
            )
            code_block = sum(1 for s in samples if '```' in s["output"])
            lines.append(f"    {label}: {backtick}/{len(samples)} use backticks, "
                         f"{underscore_converted}/{underscore_spoken} convert spoken underscores, "
                         f"{code_block} have code blocks")
            for s in samples:
                if '`' in s["output"]:
                    lines.append(f"    Example: IN=\"{truncate(s['input'], 100)}\"")
                    lines.append(f"             OUT=\"{truncate(s['output'], 100)}\"")
                    break

    elif cat == "financial_business":
        lines.append("  DEEP CHECK: Number Accuracy")
        for label, samples in [("SCRIPT", script_samples), ("PROMPT", prompt_samples)]:
            has_spoken = sum(1 for s in samples
                            if any(re.search(p, s["input"].lower()) for p in SPOKEN_NUMBER_PATTERNS))
            has_formatted = sum(1 for s in samples
                                if re.search(r'\$[\d,.]+|\d+%', s["output"]))
            lines.append(f"    {label}: {has_spoken}/{len(samples)} inputs have spoken numbers, "
                         f"{has_formatted}/{len(samples)} outputs have formatted numbers")
            # Check for magnitude issues
            for s in samples:
                # Look for dollar amounts in output
                amounts = re.findall(r'\$[\d,.]+\s*(?:million|billion)?', s["output"])
                if amounts:
                    lines.append(f"    Sample numbers: IN=\"{truncate(s['input'], 100)}\"")
                    lines.append(f"                   OUT amounts: {amounts[:3]}")
                    break

    elif cat == "meeting_notes":
        lines.append("  DEEP CHECK: Output Structure")
        for label, samples in [("SCRIPT", script_samples), ("PROMPT", prompt_samples)]:
            has_headers = sum(1 for s in samples if re.search(r'^#+\s', s["output"], re.MULTILINE))
            has_bullets = sum(1 for s in samples if re.search(r'^\s*[-•*]\s', s["output"], re.MULTILINE))
            has_action = sum(1 for s in samples
                            if re.search(r'action\s*item|decision|attendee|agenda', s["output"].lower()))
            prose_only = sum(1 for s in samples
                            if not re.search(r'[-•*#]', s["output"]))
            lines.append(f"    {label}: {has_headers} w/headers, {has_bullets} w/bullets, "
                         f"{has_action} w/structured fields, {prose_only} prose-only")
            for s in samples:
                if re.search(r'^#+\s', s["output"], re.MULTILINE):
                    out_preview = s["output"][:200].replace('\n', ' | ')
                    lines.append(f"    Structured example: \"{truncate(out_preview, 150)}\"")
                    break

    return "\n".join(lines) if lines else ""


def render_verdict(cat: str, script_analysis: list[dict], prompt_analysis: list[dict],
                   script_stats: dict, prompt_stats: dict,
                   script_samples: list[dict], prompt_samples: list[dict]) -> tuple[str, str]:
    """Determine head-to-head verdict for a category."""
    scores = {"script": 0, "prompt": 0}
    reasons = []

    # 1. Input realism: filler naturalness
    s_mid = sum(a["filler_mid_phrase"] for a in script_analysis)
    s_bound = sum(a["filler_boundary"] for a in script_analysis)
    p_mid = sum(a["filler_mid_phrase"] for a in prompt_analysis)
    p_bound = sum(a["filler_boundary"] for a in prompt_analysis)

    s_natural_ratio = s_mid / (s_mid + s_bound) if (s_mid + s_bound) > 0 else 0
    p_natural_ratio = p_mid / (p_mid + p_bound) if (p_mid + p_bound) > 0 else 0

    if s_natural_ratio > p_natural_ratio + 0.1:
        scores["script"] += 1
        reasons.append(f"script has more natural filler placement ({s_natural_ratio:.0%} mid-phrase vs {p_natural_ratio:.0%})")
    elif p_natural_ratio > s_natural_ratio + 0.1:
        scores["prompt"] += 1
        reasons.append(f"prompt has more natural filler placement ({p_natural_ratio:.0%} mid-phrase vs {s_natural_ratio:.0%})")

    # 2. Self-corrections
    s_corr = script_stats.get("avg_corrections", 0)
    p_corr = prompt_stats.get("avg_corrections", 0)
    if p_corr > s_corr + 0.3:
        scores["prompt"] += 1
        reasons.append(f"prompt has more self-corrections ({p_corr:.1f} vs {s_corr:.1f} avg)")
    elif s_corr > p_corr + 0.3:
        scores["script"] += 1
        reasons.append(f"script has more self-corrections ({s_corr:.1f} vs {p_corr:.1f} avg)")

    # 3. Written-text smell (lower = better for inputs)
    s_caps = sum(a["written_smell"]["caps_start_pct"] for a in script_analysis) / max(len(script_analysis), 1)
    p_caps = sum(a["written_smell"]["caps_start_pct"] for a in prompt_analysis) / max(len(prompt_analysis), 1)
    if s_caps < p_caps - 10:
        scores["script"] += 1
        reasons.append(f"script inputs feel less 'written' (caps start: {s_caps:.0f}% vs {p_caps:.0f}%)")
    elif p_caps < s_caps - 10:
        scores["prompt"] += 1
        reasons.append(f"prompt inputs feel less 'written' (caps start: {p_caps:.0f}% vs {s_caps:.0f}%)")

    # 4. Output quality: remaining fillers
    s_remaining = sum(len(a["remaining_fillers"]) for a in script_analysis)
    p_remaining = sum(len(a["remaining_fillers"]) for a in prompt_analysis)
    if s_remaining < p_remaining:
        scores["script"] += 1
        reasons.append(f"script outputs have fewer remaining fillers ({s_remaining} vs {p_remaining})")
    elif p_remaining < s_remaining:
        scores["prompt"] += 1
        reasons.append(f"prompt outputs have fewer remaining fillers ({p_remaining} vs {s_remaining})")

    # 5. Content preservation
    s_pres = sum(a["content_preservation"] for a in script_analysis) / max(len(script_analysis), 1)
    p_pres = sum(a["content_preservation"] for a in prompt_analysis) / max(len(prompt_analysis), 1)
    if s_pres > p_pres + 0.05:
        scores["script"] += 1
        reasons.append(f"script preserves more content ({s_pres:.0%} vs {p_pres:.0%})")
    elif p_pres > s_pres + 0.05:
        scores["prompt"] += 1
        reasons.append(f"prompt preserves more content ({p_pres:.0%} vs {p_pres:.0%})")

    # 6. Length diversity (higher CV = more diverse)
    s_cv = script_stats.get("input_length_cv", 0)
    p_cv = prompt_stats.get("input_length_cv", 0)
    if s_cv > p_cv + 0.1:
        scores["script"] += 1
        reasons.append(f"script has more diverse lengths (CV: {s_cv:.2f} vs {p_cv:.2f})")
    elif p_cv > s_cv + 0.1:
        scores["prompt"] += 1
        reasons.append(f"prompt has more diverse lengths (CV: {p_cv:.2f} vs {s_cv:.2f})")

    # 7. Vocabulary diversity
    s_vd = script_stats.get("vocab_diversity", 0)
    p_vd = prompt_stats.get("vocab_diversity", 0)
    if s_vd > p_vd + 0.02:
        scores["script"] += 1
        reasons.append(f"script has higher vocab diversity ({s_vd:.3f} vs {p_vd:.3f})")
    elif p_vd > s_vd + 0.02:
        scores["prompt"] += 1
        reasons.append(f"prompt has higher vocab diversity ({p_vd:.3f} vs {p_vd:.3f})")

    # 8. Filler density (moderate is best: 3-8%)
    s_fd = script_stats.get("avg_filler_density", 0)
    p_fd = prompt_stats.get("avg_filler_density", 0)
    s_fd_score = abs(s_fd - 5.5)  # distance from ideal
    p_fd_score = abs(p_fd - 5.5)
    if s_fd_score < p_fd_score - 1:
        scores["script"] += 1
        reasons.append(f"script has more realistic filler density ({s_fd:.1f}% vs {p_fd:.1f}%)")
    elif p_fd_score < s_fd_score - 1:
        scores["prompt"] += 1
        reasons.append(f"prompt has more realistic filler density ({p_fd:.1f}% vs {s_fd:.1f}%)")

    # 9. Sample size advantage
    s_count = script_stats.get("count", 0)
    p_count = prompt_stats.get("count", 0)
    if s_count > p_count * 2:
        scores["script"] += 1
        reasons.append(f"script has {s_count} samples vs prompt's {p_count}")
    elif p_count > s_count * 2:
        scores["prompt"] += 1
        reasons.append(f"prompt has {p_count} samples vs script's {s_count}")

    # 10. Average input complexity (longer = generally more realistic)
    s_avg_w = script_stats.get("avg_input_words", 0)
    p_avg_w = prompt_stats.get("avg_input_words", 0)
    if p_avg_w > s_avg_w * 1.5:
        scores["prompt"] += 1
        reasons.append(f"prompt inputs are more complex ({p_avg_w:.0f} vs {s_avg_w:.0f} avg words)")
    elif s_avg_w > p_avg_w * 1.5:
        scores["script"] += 1
        reasons.append(f"script inputs are more complex ({s_avg_w:.0f} vs {p_avg_w:.0f} avg words)")

    # Verdict
    if scores["script"] > scores["prompt"] + 1:
        verdict = "SCRIPT_BETTER"
    elif scores["prompt"] > scores["script"] + 1:
        verdict = "PROMPT_BETTER"
    else:
        verdict = "TIE"

    reason_text = f"Score: script={scores['script']} prompt={scores['prompt']}. " + "; ".join(reasons[:5])
    return verdict, reason_text


def main() -> None:
    random.seed(SEED)

    print_header("DATASET QUALITY COMPARISON: SCRIPT vs PROMPT")
    print("Script dataset: data/synthetic/synthetic_*.jsonl")
    print("Prompt dataset: data/prepared/*.jsonl")
    print()

    script_data, prompt_data = load_all_data()

    total_script = sum(len(v) for v in script_data.values())
    total_prompt = sum(len(v) for v in prompt_data.values())
    print(f"Total samples: Script={total_script}, Prompt={total_prompt}")

    verdicts = {}
    all_script_stats = {}
    all_prompt_stats = {}

    for cat in CATEGORIES:
        print_header(f"CATEGORY: {cat.upper().replace('_', ' ')}")

        s_all = script_data[cat]
        p_all = prompt_data[cat]

        # Compute full stats
        s_stats = compute_stats(s_all)
        p_stats = compute_stats(p_all)
        all_script_stats[cat] = s_stats
        all_prompt_stats[cat] = p_stats

        # --- Section 3: Statistical Comparison ---
        print_subheader("STATISTICAL COMPARISON (all samples)")
        stat_keys = [
            ("count", "Sample count"),
            ("avg_input_chars", "Avg input (chars)"),
            ("avg_input_words", "Avg input (words)"),
            ("avg_output_chars", "Avg output (chars)"),
            ("avg_output_words", "Avg output (words)"),
            ("avg_compression", "Compression ratio"),
            ("avg_filler_density", "Filler density (/100w)"),
            ("avg_corrections", "Self-correction freq"),
            ("input_length_cv", "Input length CV"),
            ("input_word_min", "Min input words"),
            ("input_word_max", "Max input words"),
            ("input_word_std", "Input word StdDev"),
            ("vocab_diversity", "Vocab diversity"),
        ]
        print(f"  {'Metric':<25} {'SCRIPT':>12} {'PROMPT':>12} {'Winner':>10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        for key, label in stat_keys:
            sv = s_stats.get(key, "N/A")
            pv = p_stats.get(key, "N/A")
            if isinstance(sv, (int, float)) and isinstance(pv, (int, float)):
                if key in ("avg_filler_density",):
                    # Moderate is best
                    s_dist = abs(sv - 5.5)
                    p_dist = abs(pv - 5.5)
                    winner = "script" if s_dist < p_dist - 0.5 else ("prompt" if p_dist < s_dist - 0.5 else "~")
                elif key in ("avg_compression",):
                    winner = "~"  # hard to judge
                elif key in ("count", "input_length_cv", "avg_corrections", "vocab_diversity",
                              "input_word_max", "input_word_std"):
                    winner = "script" if sv > pv else ("prompt" if pv > sv else "~")
                else:
                    winner = "~"
            else:
                winner = "~"
            print(f"  {label:<25} {sv!s:>12} {pv!s:>12} {winner:>10}")

        # --- Sample 10 from each ---
        # Use a deterministic per-category seed (hash() is not stable across sessions)
        cat_seed = SEED + sum(ord(c) for c in cat)
        rng = random.Random(cat_seed)
        s_sampled = rng.sample(s_all, min(SAMPLE_SIZE, len(s_all)))
        p_sampled = rng.sample(p_all, min(SAMPLE_SIZE, len(p_all)))

        # --- Section 1: Input Realism ---
        print_subheader("INPUT REALISM (sampled 10)")

        s_analyses = [analyze_sample(s, "script") for s in s_sampled]
        p_analyses = [analyze_sample(s, "prompt") for s in p_sampled]

        # Filler analysis
        s_total_fillers = sum(a["filler_count"] for a in s_analyses)
        p_total_fillers = sum(a["filler_count"] for a in p_analyses)
        s_mid = sum(a["filler_mid_phrase"] for a in s_analyses)
        s_bound = sum(a["filler_boundary"] for a in s_analyses)
        p_mid = sum(a["filler_mid_phrase"] for a in p_analyses)
        p_bound = sum(a["filler_boundary"] for a in p_analyses)

        print("  FILLER NATURALNESS:")
        print(f"    Script: {s_total_fillers} fillers in 10 samples "
              f"(mid-phrase: {s_mid}, boundary: {s_bound})")
        print(f"    Prompt: {p_total_fillers} fillers in 10 samples "
              f"(mid-phrase: {p_mid}, boundary: {p_bound})")

        # Show contrasting examples
        if s_sampled:
            print(f"    Script example: \"{truncate(s_sampled[0]['input'], 130)}\"")
        if p_sampled:
            print(f"    Prompt example: \"{truncate(p_sampled[0]['input'], 130)}\"")

        # Self-corrections
        s_corr_total = sum(a["corrections"] for a in s_analyses)
        p_corr_total = sum(a["corrections"] for a in p_analyses)
        s_false_starts = sum(
            1 for s in s_sampled
            if any(re.search(p, s["input"]) for p in [r'\.\.\.', r'—', r'\b\w+\s*--\s*'])
        )
        p_false_starts = sum(
            1 for s in p_sampled
            if any(re.search(p, s["input"]) for p in [r'\.\.\.', r'—', r'\b\w+\s*--\s*'])
        )

        print(f"\n  SPEECH PATTERNS:")
        print(f"    Self-corrections: Script={s_corr_total}, Prompt={p_corr_total}")
        print(f"    False starts/abandoned thoughts: Script={s_false_starts}/10, Prompt={p_false_starts}/10")

        # Written-text smell
        s_avg_caps = sum(a["written_smell"]["caps_start_pct"] for a in s_analyses) / len(s_analyses)
        p_avg_caps = sum(a["written_smell"]["caps_start_pct"] for a in p_analyses) / len(p_analyses)
        s_bullets_in = sum(1 for a in s_analyses if a["written_smell"]["has_bullets"])
        p_bullets_in = sum(1 for a in p_analyses if a["written_smell"]["has_bullets"])

        print(f"\n  WRITTEN-TEXT SMELL (in inputs):")
        print(f"    Avg caps-start sentences: Script={s_avg_caps:.1f}%, Prompt={p_avg_caps:.1f}%")
        print(f"    Inputs with bullets/lists: Script={s_bullets_in}/10, Prompt={p_bullets_in}/10")

        # --- Section 2: Output Quality ---
        print_subheader("OUTPUT QUALITY (sampled 10)")

        # Remaining fillers
        s_rem = sum(len(a["remaining_fillers"]) for a in s_analyses)
        p_rem = sum(len(a["remaining_fillers"]) for a in p_analyses)
        print(f"  REMAINING FILLERS IN OUTPUT: Script={s_rem}, Prompt={p_rem}")

        s_rem_examples = []
        p_rem_examples = []
        for a, s in zip(s_analyses, s_sampled):
            if a["remaining_fillers"]:
                s_rem_examples.append((a["remaining_fillers"], truncate(s["output"], 100)))
        for a, s in zip(p_analyses, p_sampled):
            if a["remaining_fillers"]:
                p_rem_examples.append((a["remaining_fillers"], truncate(s["output"], 100)))
        for fillers, out in s_rem_examples[:2]:
            print(f"    Script leftover [{', '.join(fillers)}]: \"{out}\"")
        for fillers, out in p_rem_examples[:2]:
            print(f"    Prompt leftover [{', '.join(fillers)}]: \"{out}\"")

        # Content preservation
        s_pres = sum(a["content_preservation"] for a in s_analyses) / len(s_analyses)
        p_pres = sum(a["content_preservation"] for a in p_analyses) / len(p_analyses)
        print(f"\n  CONTENT PRESERVATION: Script={s_pres:.1%}, Prompt={p_pres:.1%}")

        # Hallucination check
        s_hall_count = sum(1 for a in s_analyses if a["hallucinated_words"])
        p_hall_count = sum(1 for a in p_analyses if a["hallucinated_words"])
        print(f"\n  POTENTIAL HALLUCINATIONS: Script={s_hall_count}/10, Prompt={p_hall_count}/10")
        for a, s in zip(s_analyses, s_sampled):
            if a["hallucinated_words"]:
                print(f"    Script: words [{', '.join(a['hallucinated_words'][:3])}] in output not clearly from input")
                print(f"      IN: \"{truncate(s['input'], 100)}\"")
                print(f"      OUT: \"{truncate(s['output'], 100)}\"")
                break
        for a, s in zip(p_analyses, p_sampled):
            if a["hallucinated_words"]:
                print(f"    Prompt: words [{', '.join(a['hallucinated_words'][:3])}] in output not clearly from input")
                print(f"      IN: \"{truncate(s['input'], 100)}\"")
                print(f"      OUT: \"{truncate(s['output'], 100)}\"")
                break

        # Number conversion
        s_num_inputs = sum(1 for a in s_analyses if a["number_conversion"]["spoken_in_input"] > 0)
        p_num_inputs = sum(1 for a in p_analyses if a["number_conversion"]["spoken_in_input"] > 0)
        s_num_formatted = sum(1 for a in s_analyses if a["number_conversion"]["formatted_in_output"])
        p_num_formatted = sum(1 for a in p_analyses if a["number_conversion"]["formatted_in_output"])
        print(f"\n  NUMBER CONVERSION:")
        print(f"    Inputs w/ spoken numbers: Script={s_num_inputs}/10, Prompt={p_num_inputs}/10")
        print(f"    Outputs w/ formatted numbers: Script={s_num_formatted}/10, Prompt={p_num_formatted}/10")

        # Formatting
        s_caps_out = sum(1 for a in s_analyses if a["formatting"]["starts_capitalized"])
        p_caps_out = sum(1 for a in p_analyses if a["formatting"]["starts_capitalized"])
        s_punct_out = sum(1 for a in s_analyses if a["formatting"]["ends_punctuation"])
        p_punct_out = sum(1 for a in p_analyses if a["formatting"]["ends_punctuation"])
        print(f"\n  OUTPUT FORMATTING:")
        print(f"    Starts capitalized: Script={s_caps_out}/10, Prompt={p_caps_out}/10")
        print(f"    Ends with punctuation: Script={s_punct_out}/10, Prompt={p_punct_out}/10")

        # --- Section 4: Category-specific ---
        if cat in ("self_corrections_heavy", "shopping_lists", "technical_code",
                    "financial_business", "meeting_notes"):
            print_subheader(f"CATEGORY-SPECIFIC DEEP CHECK: {cat}")
            deep = category_specific_check(cat, s_sampled, p_sampled)
            if deep:
                print(deep)

        # --- Section 5: Verdict ---
        verdict, reason = render_verdict(cat, s_analyses, p_analyses, s_stats, p_stats,
                                          s_sampled, p_sampled)
        verdicts[cat] = (verdict, reason)
        print_subheader("VERDICT")
        print(f"  >>> {verdict} <<<")
        print(f"  {reason}")

    # ========== OVERALL SUMMARY ==========
    print_header("OVERALL SUMMARY")

    # Count verdicts
    vc = Counter(v for v, _ in verdicts.values())
    print(f"  SCRIPT_BETTER: {vc.get('SCRIPT_BETTER', 0)}/12 categories")
    print(f"  PROMPT_BETTER: {vc.get('PROMPT_BETTER', 0)}/12 categories")
    print(f"  TIE:           {vc.get('TIE', 0)}/12 categories")
    print()

    print(f"  {'Category':<28} {'Verdict':<16} {'Key Reason'}")
    print(f"  {'-'*28} {'-'*16} {'-'*50}")
    for cat in CATEGORIES:
        v, r = verdicts[cat]
        # First reason
        first_reason = r.split(". ", 1)[1].split(";")[0] if ". " in r else r
        print(f"  {cat:<28} {v:<16} {truncate(first_reason, 55)}")

    # Global stats comparison
    print_subheader("GLOBAL STATISTICS")
    g_script_words = []
    g_prompt_words = []
    g_script_fd = []
    g_prompt_fd = []
    for cat in CATEGORIES:
        g_script_words.append(all_script_stats[cat].get("avg_input_words", 0))
        g_prompt_words.append(all_prompt_stats[cat].get("avg_input_words", 0))
        g_script_fd.append(all_script_stats[cat].get("avg_filler_density", 0))
        g_prompt_fd.append(all_prompt_stats[cat].get("avg_filler_density", 0))

    print(f"  Avg input words across cats: Script={sum(g_script_words)/12:.1f}, Prompt={sum(g_prompt_words)/12:.1f}")
    print(f"  Avg filler density across cats: Script={sum(g_script_fd)/12:.1f}%, Prompt={sum(g_prompt_fd)/12:.1f}%")
    print(f"  Total volume: Script={total_script} samples, Prompt={total_prompt} samples")

    # Key qualitative findings
    print_subheader("KEY QUALITATIVE FINDINGS")

    # Compute overall remaining fillers across all sampled
    print("  1. INPUT REALISM: Which dataset sounds more like real speech transcripts?")
    s_avg_len = sum(s.get("avg_input_words", 0) for s in all_script_stats.values()) / 12
    p_avg_len = sum(s.get("avg_input_words", 0) for s in all_prompt_stats.values()) / 12
    print(f"     - Prompt inputs average {p_avg_len:.0f} words vs Script's {s_avg_len:.0f} words")
    print(f"     - Longer inputs tend to capture more realistic speech patterns")

    s_avg_fd_all = sum(s.get("avg_filler_density", 0) for s in all_script_stats.values()) / 12
    p_avg_fd_all = sum(s.get("avg_filler_density", 0) for s in all_prompt_stats.values()) / 12
    print(f"     - Filler density: Script={s_avg_fd_all:.1f}%, Prompt={p_avg_fd_all:.1f}%")
    print(f"       (Natural speech ~3-8% filler rate)")

    print("\n  2. OUTPUT QUALITY: Which dataset produces better training targets?")
    print("     - See per-category breakdowns above for specifics")

    print("\n  3. VOLUME vs QUALITY TRADEOFF:")
    print(f"     - Script has {total_script} samples (volume advantage)")
    print(f"     - Prompt has {total_prompt} samples (potentially higher quality)")

    print("\n  4. RECOMMENDATION:")
    if vc.get('PROMPT_BETTER', 0) > vc.get('SCRIPT_BETTER', 0):
        print("     Prompt dataset is higher quality in more categories.")
        print("     Consider using prompt data as-is and augmenting with script data")
        print("     for categories where script is better or for volume.")
    elif vc.get('SCRIPT_BETTER', 0) > vc.get('PROMPT_BETTER', 0):
        print("     Script dataset is higher quality in more categories.")
        print("     Consider using script data as primary and cherry-picking from prompt.")
    else:
        print("     Datasets are roughly equivalent. Consider combining both,")
        print("     using prompt data for weak categories and script for volume.")

    print()
    print_divider("=")
    print("  END OF REPORT")
    print_divider("=")


if __name__ == "__main__":
    main()
