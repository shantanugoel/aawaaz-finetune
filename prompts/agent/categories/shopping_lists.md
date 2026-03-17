# Category: Shopping Lists

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions.

## Description

Someone dictating a shopping list, errand list, to-do list, or packing list. Quick,
staccato items often with quantities and brands, self-corrections as they check what
they need, and afterthoughts as they remember things.

## Speech Patterns for This Category

- Rapid item listing: "eggs milk bread uh oh and butter"
- Quantities: "like two dozen eggs", "three pounds of chicken", "a gallon of milk"
- Self-corrections: "no wait not whole milk get skim", "actually make that two bags"
- Grouping by store or section: "from costco we need... and then from the grocery
  store..."
- Afterthoughts: "oh and I almost forgot we need...", "wait do we have rice... no
  get rice"
- Brands/specifics: "the kirkland one", "get the unsalted butter", "the blue box"
- Checking mentally: "do we need... yeah we're out of that", "hmm what else"

## Example Pair

**Input:**
```
okay so grocery list for this week um we need eggs uh two dozen milk a gallon of whole
actually no wait make that skim milk um bread the sourdough kind uh chicken breasts
about three pounds um oh and rice we're almost out so like a five pound bag of basmati
rice uh what else oh yeah broccoli and um like a bag of spinach for smoothies and uh
also we need laundry detergent the the tide pods and um oh paper towels I keep
forgetting those uh and then from the pharmacy I need to pick up my prescription and
uh some ibuprofen the two hundred milligram ones
```

**Output:**
```
Grocery list:
- 2 dozen eggs
- 1 gallon skim milk
- Sourdough bread
- ~3 lbs chicken breasts
- 5 lb bag basmati rice
- Broccoli
- 1 bag spinach (for smoothies)
- Tide Pods (laundry detergent)
- Paper towels

Pharmacy:
- Pick up prescription
- Ibuprofen (200 mg)
```

## Common Pitfalls

- **Self-corrections**: "whole actually no wait make that skim" → only "skim" in
  output. The original choice disappears.
- **List format**: Output should be a clean bulleted/numbered list, NOT paragraph text.
  Lists are lists.
- **Quantities preserved**: "two dozen", "three pounds", "five pound bag" — all kept
  and formatted
- **Grouping**: If the speaker groups by location/store, maintain that grouping in
  the output
- **Afterthoughts must be included**: "oh and I almost forgot" items are real items —
  they must appear in the output list
- **Checking self-talk removed**: "do we need... yeah we're out of that" → just the
  item, no deliberation process
- **Brand references kept**: "the Tide Pods" stays; "the the" stutter goes
