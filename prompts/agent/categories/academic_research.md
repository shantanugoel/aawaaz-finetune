# Category: Academic / Research

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions.

> **Type**: Core

## Description

Researchers, professors, or students dictating paper abstracts, research notes,
literature review observations, methodology descriptions, or thinking through their
findings aloud. Heavy academic terminology mixed with casual speech and thinking-
out-loud patterns.

## Speech Patterns for This Category

- Citations spoken naturally: "according to smith et al twenty twenty three", "that
  paper by chen and rodriguez from last year"
- Statistical language: "p value less than point zero five", "n equals three hundred",
  "r squared was like point eight three"
- Hedging language: "our findings suggest", "this appears to indicate", "there seems
  to be a correlation"
- Method description: "we used a between subjects design with uh three conditions"
- Thinking through: "wait what was the sample size... I think it was forty two"
- No formatted citations in transcript — just spoken: "chen and rodriguez twenty
  twenty three"
- Years spoken: "twenty twenty three" not "2023"

## Example Pair

**Input:**
```
so the uh abstract for the paper would be something like um this study examines the
relationship between social media usage and sleep quality among college students aged
eighteen to twenty four uh using a cross sectional survey design we collected data from
um three hundred and forty two participants across four universities period uh results
indicate a significant negative correlation between daily screen time and sleep quality
as measured by the pittsburgh sleep quality index parentheses psqi close parentheses
with um r equals negative point three eight p less than point zero zero one period uh
these findings suggest that targeted interventions to reduce evening screen time could
improve sleep outcomes in this population
```

**Output:**
```
This study examines the relationship between social media usage and sleep quality among
college students aged 18-24. Using a cross-sectional survey design, we collected data
from 342 participants across four universities. Results indicate a significant negative
correlation between daily screen time and sleep quality, as measured by the Pittsburgh
Sleep Quality Index (PSQI), with r = -0.38, p < 0.001. These findings suggest that
targeted interventions to reduce evening screen time could improve sleep outcomes in
this population.
```

## Common Pitfalls

- **Statistical notation must be exact**: "r equals negative point three eight" →
  "r = -0.38", "p less than point zero zero one" → "p < 0.001", "n equals three
  hundred" → "n = 300"
- **Preserve hedging language**: "suggest" and "appear to indicate" are meaningful
  academic language, NOT filler words. Do not remove them.
- **Numbers**: "three hundred and forty two" → "342", "eighteen to twenty four" →
  "18-24"
- **Parenthetical formatting**: "parentheses psqi close parentheses" → "(PSQI)"
- **Citations**: "chen and rodriguez twenty twenty three" → "Chen and Rodriguez (2023)"
  or "Chen & Rodriguez, 2023" — use standard academic format
- **Don't formalize the speaker's voice**: "it's been studied to death" stays as
  "it's been studied to death", NOT "has been extensively studied". "we still get it
  wrong" stays, NOT "errors persist". Clean the fillers, not the personality. The
  speaker chose those words — preserve them.
- **THE #1 FAILURE**: Making the input sound like a written paper. "The study by Chen
  and Rodriguez (2023) examined..." is written text. "so um that paper by chen and
  rodriguez from twenty twenty three looked at..." is speech.
