# Category: Technical / Code

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions and
> workflow. Read `prompts/agent/validate.md` when spawning validation sub-agents.

## Description

Someone dictating code, CLI commands, error messages, technical documentation, or
debugging steps OUT LOUD. They're speaking to a transcription system, not typing.
This covers: code dictation, error description, command sequences, config
explanations, and technical troubleshooting.

## Speech Patterns for This Category

- Spelling out identifiers: "process underscore data" or "get underscore user
  underscore by underscore id"
- Describing syntax: "open paren", "close bracket", "arrow function", "equals equals
  equals", "curly brace"
- File paths spoken: "slash user slash local slash bin slash python three"
- Technical jargon mixed with casual speech: "so basically the uh the api endpoint is
  returning a four oh four"
- Error messages read aloud: "it says uh module not found error no module named pandas"
- Numbers as words: "port eight zero eight zero", "version three point twelve"
- Code is described haltingly — NOT rattled off fluently

## Example Pair

**Input:**
```
okay so the error I'm getting is um module not found error no module named uh numpy
and I've tried running pip install numpy but it says uh permission denied so I think I
need to do sudo pip install or actually wait no the better way is to use a virtual
environment so um python three dash m venv dot venv and then source dot venv slash bin
slash activate and then pip install numpy equals equals one dot twenty four
```

**Output:**
```
The error I'm getting is: ModuleNotFoundError: No module named numpy. I've tried
running `pip install numpy` but it says "Permission denied." I think I need to do
`sudo pip install`, or actually the better way is to use a virtual environment:
`python3 -m venv .venv`, then `source .venv/bin/activate`, then
`pip install numpy==1.24`.
```

## Common Pitfalls

- **Code formatting in output**: Use backticks for inline code, proper syntax
  formatting. "pip install numpy" → `pip install numpy`.
- **Spoken identifiers → actual code**: "underscore" → _, "dash" → -, "dot" → .,
  "slash" → /, "equals equals" → ==
- **Self-correction is critical**: "sudo pip install or actually wait no the better
  way" → only the better way
- **Don't over-structure**: The output should be a cleaned-up version of what they
  said, NOT a tutorial, README, or documentation
- **Don't hallucinate code details**: If the speaker said "import react", do NOT
  expand to `import React from "react"` unless they specifically said that
- **Port/version numbers**: "eight zero eight zero" → 8080, "three point twelve" → 3.12
