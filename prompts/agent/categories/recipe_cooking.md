# Category: Recipe / Cooking

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions.

> **Type**: Domain Specific

## Description

Someone dictating a recipe from memory, describing cooking steps while in the kitchen,
or talking through meal planning. Casual, instructional speech with measurements,
temperatures, and timing — often with mid-stream corrections as they remember details.

## Speech Patterns for This Category

- Quantities as words: "two cups of flour", "a tablespoon and a half of olive oil",
  "like a pinch of salt"
- Temperatures spoken: "three fifty degrees", "one seventy five celsius"
- Timing as words: "about twenty to twenty five minutes", "like an hour and a half"
- Conversational instruction: "you wanna", "go ahead and", "what I usually do is",
  "the trick is"
- Self-corrections on order: "add the vegetables... actually you should do the sauce
  first before that"
- Tangents: "oh and make sure your butter is room temp that's super important"
- Approximate language: "like a generous cup", "until it looks golden-ish"

## Example Pair

**Input:**
```
okay so for the uh pasta primavera you're gonna need um two cups of penne pasta one
red bell pepper diced uh one zucchini sliced and about a cup of cherry tomatoes halved
um for the sauce it's uh three tablespoons of olive oil two cloves of garlic minced and
um half a cup of vegetable broth period so first boil the pasta in salted water for
about uh ten to twelve minutes until al dente um while that's going saute the garlic in
olive oil for like thirty seconds and then add the vegetables and cook for uh five to
seven minutes until tender and then um add the broth and let it simmer for like two
minutes and toss in the drained pasta and you're done
```

**Output:**
```
Pasta Primavera

Ingredients:
- 2 cups penne pasta
- 1 red bell pepper, diced
- 1 zucchini, sliced
- ~1 cup cherry tomatoes, halved
- 3 tbsp olive oil
- 2 cloves garlic, minced
- 1/2 cup vegetable broth

Instructions:
1. Boil the pasta in salted water for 10-12 minutes until al dente.
2. While that's going, sauté the garlic in olive oil for 30 seconds.
3. Add the vegetables and cook for 5-7 minutes until tender.
4. Add the broth and let it simmer for 2 minutes.
5. Toss in the drained pasta and serve.
```

## Common Pitfalls

- **Structure the output**: Recipes should have an ingredients list + numbered
  instructions, even if the speaker mixed them together
- **Measurements converted**: "two cups" → "2 cups", "three tablespoons" → "3 tbsp",
  "half a cup" → "1/2 cup"
- **Self-corrections on step order**: "add the vegetables... actually do the sauce
  first" → correct the order in the output
- **Temperature formatting**: "three fifty degrees" → "350°F" (or °C based on context)
- **Timing**: "twenty to twenty five minutes" → "20-25 minutes"
- **Approximate quantities preserved**: "about a cup" → "~1 cup" — don't remove the
  approximation
- **Don't add ingredients or steps**: Only what the speaker mentioned
