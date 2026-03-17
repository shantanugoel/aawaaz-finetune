# Category: Creative Writing

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions and
> workflow. Read `prompts/agent/validate.md` when spawning validation sub-agents.

## Description

Writers dictating story passages, poetry, blog posts, personal essays, or journal
entries. The key insight: the spoken content is someone COMPOSING in real time — they
think about word choices, try different phrasings, change direction. The transcript
is messy; the beauty emerges in the output.

## Speech Patterns for This Category

- Composing aloud: "so the scene opens with... no wait let me start differently"
- Word choice deliberation: "the sky was red... or maybe more like crimson yeah
  crimson", "she walked... no she strode into the room"
- Dialogue within dictation: "and then she says quote I never asked for this end quote"
- Spoken formatting: "new paragraph", "dot dot dot" for ellipsis, "em dash" for —,
  "open quote", "close quote"
- Tense shifts as they compose: present then past, trying things out
- Meta-commentary: "actually no let me start this part differently", "hmm that
  doesn't sound right"
- Ideas flowing into each other without clean sentence boundaries

## Example Pair

**Input:**
```
okay so chapter three um the morning after new paragraph the first thing eleanor
noticed when she woke up was the silence period it wasn't the um the peaceful kind of
silence you get in the countryside but more like the the oppressive kind like the world
was holding its breath period uh she reached for her phone on the nightstand and um saw
three missed calls from david all between two and three am period em dash that couldn't
be good period new paragraph um she sat up slowly pulling the blanket around her
shoulders and like tried to remember what had happened last night but everything after
the restaurant was just a blur you know question mark or maybe not a question mark
actually just period everything after the restaurant was a blur period
```

**Output:**
```
Chapter Three — The Morning After

The first thing Eleanor noticed when she woke up was the silence. It wasn't the
peaceful kind of silence you get in the countryside, but the oppressive kind — like
the world was holding its breath.

She reached for her phone on the nightstand and saw three missed calls from David, all
between 2 and 3 AM — that couldn't be good.

She sat up slowly, pulling the blanket around her shoulders, and tried to remember what
had happened last night. Everything after the restaurant was a blur.
```

## Common Pitfalls

- **THE WORST CATEGORY for realism**: 76% of script-generated inputs fail because they
  sound like polished prose, not dictation. "She walked into the room and the light was
  like gold spilling across the floor" is beautiful writing — but NO ONE speaks that
  fluently while composing. Real dictation is halting, with thinking and revisions.
- **Self-editing is the input's defining feature**: "red... or maybe crimson yeah
  crimson" → output uses only "crimson"
- **Spoken formatting must be converted**: "em dash" → —, "dot dot dot" → ...,
  "new paragraph" → paragraph break, "open quote"/"close quote" → quotation marks
- **Meta-commentary removed**: "or maybe not a question mark actually just period" →
  resolve to the author's final choice (period in this case)
- **Preserve the writer's VOICE**: Don't flatten literary language. If they chose
  evocative metaphors, keep them. The cleaning removes filler and resolves edits,
  not style.
- **Dialogue punctuation**: "she says quote I never asked for this end quote" → She
  said, "I never asked for this."
