#!/usr/bin/env python3
"""Validate a batch of transcript-cleanup pairs using any LLM.

Standalone validation script intended to be called by coding agents during
agent-based data generation.  Uses the same ``llm_client.py`` infrastructure
as the rest of the pipeline, so it supports Anthropic, OpenAI, and any
OpenAI-compatible provider (OpenRouter, Ollama, etc.).

Usage examples::

    # Validate using config.yaml's validation section (default)
    python scripts/validate_batch.py --input batch.jsonl

    # Override model/provider on the command line
    python scripts/validate_batch.py --input batch.jsonl \\
        --provider openai_compatible \\
        --model openai/gpt-5.4 \\
        --api-key-env OPENROUTER_API_KEY \\
        --base-url https://openrouter.ai/api/v1

    # Read pairs from stdin (pipe from agent)
    cat batch.jsonl | python scripts/validate_batch.py --input -

Output is a JSON object printed to stdout with the same schema used by
``prompts/agent/validate.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Allow running from project root: ``python scripts/validate_batch.py``
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CONFIG_PATH, setup_logging
from llm_client import create_client_from_config

logger = logging.getLogger("aawaaz.validate_batch")

# ── Judge prompt ──────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating a synthetic training pair for a speech transcript cleanup model.
Your job is to ensure the training data is high quality. Be strict but fair.

INPUT (raw transcript):
{transcript}

OUTPUT (cleaned text):
{output}

Score this pair on 4 criteria. Answer ONLY with a JSON object, no other text.

1. "input_realistic": Is the input realistic speech-to-text (ASR) output?
   Modern ASR like Whisper MAY produce some punctuation and capitalization — that is normal.
   The key question: does this sound like someone SPEAKING, or like someone carefully TYPING?
   It should have:
   - Natural speech patterns: fillers, run-on sentences, topic changes, hesitations
   - Inconsistent formatting (some punctuation present, some missing — not perfectly formatted)
   - Numbers mostly as spoken words (but some digit usage is acceptable)
   FAIL if it reads like carefully composed written text with a few "um"s mechanically inserted.
   FAIL if it has perfect, consistent formatting throughout that no ASR engine would produce.
   PASS if it sounds like natural speech even if the ASR has added some punctuation/capitalization.

2. "content_preserved": Does the output preserve ALL substantive information from the input?
   Compare carefully — no facts, names, numbers, instructions, or meaning should be dropped.
   Minor rewording is fine. Dropping an entire sentence or fact is a FAIL.
   IMPORTANT — the following are NOT substantive content and may be removed:
   - Filler words: "um", "uh", "like", "you know", "basically", "honestly", "literally"
   - Discourse markers: "yeah no", "oh my god", "so basically"
   - Conversational padding: "that's the thing", "and stuff", "and everything"
   - Compositional meta-talk: "I want to say", "how do I put this", "what's the word"
   - Softeners: "kind of", "sort of", "a little bit", "pretty much"
   Removing these is CORRECT cleanup, not content loss.

3. "no_hallucination": Does the output contain ONLY information present in the input?
   NOT hallucination: structural formatting (headers, bullets, numbered lists), punctuation,
   number conversion ("twenty five" → 25), minor grammatical rewording.
   IS hallucination: adding facts, conclusions, opinions, clarifying phrases with new meaning,
   inventing numbers or names not in the input.

4. "corrections_applied": Is the cleanup done correctly?
   - Fillers removed, self-corrections applied, numbers/dates formatted
   - Punctuation and capitalization added consistently
   FAIL if obvious cleanup was missed or done incorrectly.

{{"input_realistic": true/false, "content_preserved": true/false, \
"no_hallucination": true/false, "corrections_applied": true/false}}"""


# ── Pair loading ──────────────────────────────────────────────────────────


def load_pairs(source: str) -> list[dict[str, str]]:
    """Load JSONL pairs from a file path or stdin (``-``)."""
    lines: list[str] = []
    if source == "-":
        lines = sys.stdin.readlines()
    else:
        path = Path(source)
        if not path.exists():
            logger.error("Input file not found: %s", path)
            sys.exit(1)
        lines = path.read_text(encoding="utf-8").splitlines()

    pairs: list[dict[str, str]] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSON on line %d: %s", i + 1, exc)
            continue
        if not isinstance(obj.get("input"), str) or not isinstance(
            obj.get("output"), str
        ):
            logger.warning("Skipping line %d: missing input/output strings", i + 1)
            continue
        pairs.append(obj)
    return pairs


# ── Evaluation ────────────────────────────────────────────────────────────


CRITERIA = ["input_realistic", "content_preserved", "no_hallucination", "corrections_applied"]


def evaluate_pair(
    client: Any, pair: dict[str, str], pair_index: int
) -> dict[str, Any]:
    """Evaluate a single pair. Returns a per-pair result dict."""
    prompt = JUDGE_PROMPT.format(
        transcript=pair["input"],
        output=pair["output"],
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        raw = client.generate(
            messages, max_tokens=1024, temperature=0.0, json_mode=True
        )
    except Exception as exc:
        logger.error("API call failed for pair %d: %s", pair_index, exc)
        return {
            "pair_index": pair_index,
            "pass": False,
            "error": str(exc),
            "criteria": {
                c: {"pass": False, "note": "evaluation failed"} for c in CRITERIA
            },
        }

    # Parse the JSON response
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        scores = json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning(
            "Pair %d: could not parse judge response as JSON: %s",
            pair_index,
            raw[:200],
        )
        return {
            "pair_index": pair_index,
            "pass": False,
            "error": "unparseable judge response",
            "criteria": {
                c: {"pass": False, "note": "unparseable response"} for c in CRITERIA
            },
        }

    # Build structured result
    criteria_results: dict[str, dict[str, Any]] = {}
    all_pass = True
    for c in CRITERIA:
        passed = bool(scores.get(c, False))
        criteria_results[c] = {"pass": passed, "note": ""}
        if not passed:
            all_pass = False

    return {
        "pair_index": pair_index,
        "pass": all_pass,
        "criteria": criteria_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate transcript-cleanup pairs using an LLM judge."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL file with pairs, or '-' for stdin.",
    )
    parser.add_argument(
        "--provider",
        help="LLM provider (anthropic, openai, openai_compatible). "
        "Defaults to config.yaml validation section.",
    )
    parser.add_argument("--model", help="Model name. Defaults to config.yaml.")
    parser.add_argument(
        "--api-key-env",
        help="Environment variable for API key. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--base-url", help="Base URL for API. Defaults to config.yaml."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between API calls in seconds (default: 0.2).",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Resolve model config: CLI args override config.yaml
    import yaml

    provider = args.provider
    model = args.model
    api_key_env = args.api_key_env
    base_url = args.base_url

    if not all([provider, model, api_key_env]):
        # Fall back to config.yaml validation section
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            val_cfg = cfg.get("dataset", {}).get("synthetic", {}).get("validation", {})
            provider = provider or val_cfg.get("provider")
            model = model or val_cfg.get("model")
            api_key_env = api_key_env or val_cfg.get("api_key_env")
            base_url = base_url or val_cfg.get("base_url")
        except Exception as exc:
            logger.error("Could not read config.yaml for defaults: %s", exc)

    if not all([provider, model, api_key_env]):
        logger.error(
            "Must specify --provider, --model, and --api-key-env "
            "(or configure in config.yaml dataset.synthetic.validation)."
        )
        sys.exit(1)

    # Load pairs
    pairs = load_pairs(args.input)
    if not pairs:
        logger.error("No valid pairs found in input.")
        sys.exit(1)

    logger.info("Validating %d pairs with %s/%s", len(pairs), provider, model)

    # Create client
    client = create_client_from_config(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
    )

    # Evaluate each pair
    evaluations: list[dict[str, Any]] = []
    passed_count = 0
    failure_reasons: dict[str, int] = {c: 0 for c in CRITERIA}

    for i, pair in enumerate(pairs):
        result = evaluate_pair(client, pair, i)
        evaluations.append(result)
        if result["pass"]:
            passed_count += 1
        else:
            for c in CRITERIA:
                if not result["criteria"][c]["pass"]:
                    failure_reasons[c] += 1

        if i < len(pairs) - 1 and args.delay > 0:
            time.sleep(args.delay)

    # Find most common failure
    most_common = max(failure_reasons, key=failure_reasons.get) if any(failure_reasons.values()) else ""

    # Build output
    output = {
        "evaluations": evaluations,
        "summary": {
            "total": len(pairs),
            "passed": passed_count,
            "failed": len(pairs) - passed_count,
            "most_common_failure": most_common,
        },
    }

    # Print to stdout (this is what the agent reads)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
