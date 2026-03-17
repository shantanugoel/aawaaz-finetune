# Category: Casual Conversation

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions.

> **Type**: Core

## Description

Friend-to-friend voice messages, voice notes about daily life, casual updates about
weekend plans, stories about what happened, recommendations, rants, and casual
check-ins. Someone talking to someone they know well, with zero filter on their
speech.

## Speech Patterns for This Category

- Heavy contractions: "gonna", "wanna", "kinda", "gotta", "y'know", "dunno"
- Lots of filler cascades: "like", "so basically", "you know what I mean", "oh my god"
- Topic changes mid-thought: starts about dinner plans, veers into a story about work,
  comes back to dinner
- Trailing off: "and then we were going to... anyway so..."
- Emotional interjections: "oh man", "dude", "seriously though", "I can't even"
- Very run-on: everything connected with "and" and "so" and "but"
- Informal references: "that one place", "the thing", "you know who I mean"
- Stories within stories: "so I told her about— oh wait first I should mention that..."

## Example Pair

**Input:**
```
oh my god so like I finally went to that new thai place you were telling me about um
the one on like oak street or whatever and dude the pad thai was insane like seriously
it was so good um and then I ran into jake there you know jake from from the thing uh
from sarah's birthday party and he was like oh hey we should all hang out sometime and
I was like yeah totally um so I think we might do something this weekend maybe saturday
or sunday I'm not sure yet but like let me know if you're free and oh also I forgot to
tell you I got the tickets for the concert it's on march twenty eighth at uh the civic
center so yeah that's exciting
```

**Output:**
```
I finally went to that new Thai place you were telling me about, the one on Oak Street.
The pad thai was insane, it was so good. I ran into Jake there — Jake from Sarah's
birthday party — and he said we should all hang out sometime. I said yeah, so I think
we might do something this weekend, maybe Saturday or Sunday. Let me know if you're
free. Also, I forgot to tell you, I got the tickets for the concert. It's on March
28th at the Civic Center!
```

## Common Pitfalls

- **Don't make the output formal**: Casual conversation stays casual after cleanup.
  "I was so excited" is correct — don't change it to "I felt a great deal of
  excitement."
- **Preserve emotional content**: "dude that was amazing" → "That was amazing" — the
  sentiment stays, just the filler-words go.
- **The input must RAMBLE**: Real casual speech is disorganized with topic jumps. If
  your input reads like a structured paragraph with fillers, redo it.
- **"Like" disambiguation**: "like the one on Oak Street" = comparison (keep).
  "it was like so good" = filler (remove). Get this right.
- **Don't drop afterthoughts**: "oh also I forgot to tell you I got the tickets" —
  the ticket information must appear in the output.
