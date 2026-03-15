# AGENTS.md — Standing Instructions for AI Coding Agents

This file contains rules and workflow instructions for any AI coding agent working on this project. Read this file in full before starting any task. These rules apply to every task, every session, without exception.

---

## 1. Spec-First Workflow

This project has a detailed specification: `aawaaz-finetune-spec.md`. It is the source of truth.

### Before Writing Any Code
- Read the **entire** spec, not just the section you think is relevant. Cross-cutting concerns (like Qwen3 thinking mode, chat template handling, and mask_prompt) appear in the "Critical Implementation Notes" section and affect nearly every script.
- Create an implementation plan as a markdown checklist before coding. Each work unit should be one script or one major component. The plan should include:
  - Which spec sections it implements (by header name)
  - Specific technical decisions (libraries, data structures, error handling approach)
  - Dependencies on other work units
  - Any ambiguities or open questions for the human to resolve
- **Do NOT start coding until the plan is reviewed and approved by the human.** If continuing a previous session where the plan was already approved, you may proceed.

### Before Marking Any Work Unit Complete
- Re-read the corresponding spec section(s) in `aawaaz-finetune-spec.md`
- Verify every requirement, flag, edge case, and "CRITICAL" note is addressed
- If the spec says "log X" — make sure you log X. If it says "support --flag" — make sure that flag exists and works.
- Check the "Critical Implementation Notes" section for anything that applies to the code you just wrote

### If Something in the Spec Seems Wrong
- Do NOT silently ignore it or "fix" it in code
- Flag it explicitly: "SPEC QUESTION: Section X says Y, but I think Z because [reason]. Should I follow the spec or change it?"
- Wait for the human's answer before proceeding on that point

---

## 2. Code Quality Rules

### General
- use `uv` with a `venv` to run/use/install any internal or external python scripts/packages 
- Every script must be runnable standalone AND as part of the orchestrated pipeline
- Every script reads from `config.yaml` — do not hardcode values that are in the config
- Use `argparse` for CLI arguments, even if the primary interface is config-driven
- Every script must support `--verbose` and `--dry-run` flags
- Idempotency: re-running a script should not duplicate work or corrupt state
- Use type hints on all function signatures
- Use `pathlib.Path` not string concatenation for file paths
- Use `logging` module, not `print()`, for operational output (print is fine for `--verbose` debug info or interactive prompts)
- Never stage or commit changes to git yourself. Developer will do it themselves.

### Error Handling
- Catch specific exceptions, not bare `except:`
- On failure: log the error clearly, including what was being attempted and what the user should do to fix it
- For long-running operations (API calls, training): save progress incrementally so crashes don't lose all work
- Never silently swallow errors — if something fails and you continue, log a WARNING at minimum

### Dependencies
- Check that required tools/packages are available before starting work, not halfway through
- If a step requires a previous step's output, check that the output exists and looks valid (not just that the file exists — check it's non-empty and parseable)

### File I/O
- Use UTF-8 encoding explicitly on all file reads/writes
- When writing JSONL: one JSON object per line, no trailing comma, newline at end of file
- When writing to output directories: create them if they don't exist (`mkdir -p` / `Path.mkdir(parents=True, exist_ok=True)`)
- Never overwrite output files without warning. If output already exists, either skip (with log) or prompt the user. Support `--force` to override.

---

## 3. Project-Specific Gotchas

These are things that are easy to get wrong in this project. Check these proactively.

### Qwen3 Thinking Mode
Qwen3 models generate `<think>...</think>` blocks by default before responding. This MUST be disabled everywhere in this project — training data, fine-tuning, evaluation, and inference.
- When using `tokenizer.apply_chat_template()`: pass `enable_thinking=False`
- The system prompt includes `/no_think` as a belt-and-suspenders measure
- During evaluation: if output contains `<think>` tags, strip them and log a WARNING
- **Test this early.** If you're not sure how a specific Qwen3 model version handles the thinking disable, write a quick test before building on assumptions.

### Chat Template Format
- Qwen3 uses ChatML format: `<|im_start|>role\ncontent<|im_end|>`
- MLX LoRA training and Unsloth/HF training handle chat templates differently:
  - MLX: expects `{"messages": [...]}` JSONL, applies template internally
  - Unsloth/HF: expects you to apply the template into a `text` field
- **Write a validation function** that round-trips a sample through the tokenizer and verifies the format is correct. Call this in `04_prepare_data.py` before saving, and in `06_finetune.py` before training starts.

### mask_prompt
- We only want loss on the assistant's response (the cleanup), not the system prompt or user input
- MLX: `--mask-prompt` flag
- Unsloth/TRL SFTTrainer: should handle this automatically with messages format, but VERIFY by checking that the loss is reasonable (if loss starts near 0 on first step, the prompt is probably not being masked and the model is memorizing the system prompt)

### The Two Models
- This pipeline targets BOTH Qwen3-0.6B and Qwen3-1.7B
- Every script that operates on a model must accept `--model qwen3-0.6b` or `--model qwen3-1.7b` (or run both if `--model all`)
- The config has different hyperparameters for different model sizes — make sure you're reading the right ones
- Don't assume both models have identical tokenizer behavior — test separately

---

## 4. Communication Rules

### Progress Updates
- When starting a work unit, state what you're building and which spec sections it covers
- After completing a work unit, provide a brief summary: what was built, what decisions were made, and any caveats

### When You're Unsure
- Ask. Don't guess. Especially about:
  - API behavior you haven't verified (e.g., "does Unsloth's SFTTrainer mask prompts automatically?")
  - Library version compatibility (e.g., "does mlx-lm 0.21 support this flag?")
  - Design decisions the spec doesn't cover
- Frame questions with your best guess + reasoning: "I think X because Y — should I proceed with that assumption?"

### When You Find a Better Approach
- Don't silently deviate from the spec
- Say: "The spec says X, but I think Y would be better because [reason]. Want me to go with Y?"
- If it's a minor improvement (e.g., better error message, additional logging), just do it and mention it in the summary

### Scope Discipline
- Do NOT add features, optimizations, or "nice to haves" that aren't in the spec unless the human asks
- The stretch goals section exists — don't implement those unless explicitly asked
- If you notice something the spec missed, flag it as a suggestion rather than implementing it

---

## 5. Testing Expectations

### Every Script Should Have
- A basic smoke test: can it run with minimal input and produce expected output?
- At least one edge case test: empty input, missing file, malformed data
- Verification that it works with both `--platform linux` and `--platform mac` code paths (where applicable)

### Before Declaring a Script "Done"
Run this mental checklist:
- [ ] Does it read all relevant config values from `config.yaml`?
- [ ] Does it support `--verbose` and `--dry-run`?
- [ ] Does it check prerequisites (input files exist, dependencies installed)?
- [ ] Does it create output directories if they don't exist?
- [ ] Does it handle the case where output already exists?
- [ ] Does it log what it's doing at each major step?
- [ ] Does it report a summary at the end (counts, timing, any warnings)?
- [ ] Have I re-read the spec section to make sure nothing was missed?

---

## 6. Session Continuity

If this is a multi-session project (which it will be):

### At the Start of a New Session
- Re-read this AGENTS.md
- Check the implementation plan for which work units are completed vs remaining
- Read the spec sections relevant to the next work unit
- State what you plan to work on and ask if priorities have changed

### At the End of a Session
- Summarize what was completed
- Note any work in progress, open questions, or known issues
- Update the implementation plan checklist if possible

### State Tracking
- The pipeline orchestrator uses `.pipeline_state.json` to track completed steps
- Your implementation plan (the checklist) serves as the development-time equivalent
- Keep both in sync

---

## 7. File Naming and Location Conventions

| What | Where |
|------|-------|
| Spec document | `aawaaz-finetune-spec.md` (project root) |
| This file | `AGENTS.md` (project root) |
| All scripts | `scripts/` with numeric prefix for ordering |
| Config | `config.yaml` (project root) |
| Raw downloaded data | `data/raw/` |
| Synthetic data | `data/synthetic/` |
| Rejected synthetic data | `data/synthetic/rejected/` |
| Final training data | `data/combined/{train,valid,test}.jsonl` |
| Base models | `models/base/` |
| LoRA adapters | `models/adapters/{model_name}/` |
| Fused models | `models/fused/{model_name}/` |
| Quantized models | `models/quantized/{model_name}-4bit/` |
| Eval results | `eval_results/` |
| System prompt | `prompts/system_prompt.txt` |
| Pipeline state | `.pipeline_state.json` (git-ignored) |
