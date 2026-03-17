# Category: Meeting Notes

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions and
> workflow. Read `prompts/agent/validate.md` when spawning validation sub-agents.

## Description

Someone recapping or live-noting a meeting — summarizing discussions, decisions, and
action items. They're recalling from memory after the meeting or speaking during the
meeting to capture key points. Topics jump around, they remember things out of order,
and they reference people by first name.

## Speech Patterns for This Category

- Speaker attribution: "and then sarah said", "john pointed out that", "mike suggested"
- Action items emerge naturally: "so sarah's gonna handle that" not "Action: Sarah"
- Topic transitions: "okay moving on", "what else... oh right", "oh and I forgot"
- Recalled from memory = out of order: "oh wait I should mention that before the
  budget thing we also talked about..."
- Names without capitalization in transcript
- Dates/deadlines spoken: "by next friday", "end of q two", "march fifteenth"
- Numbers as words for metrics: "thirty percent over budget", "two more engineers"

## Example Pair

**Input:**
```
okay so um meeting notes for the product sync march fifteenth twenty twenty six uh
attendees were sarah chen mike rodriguez uh jennifer walsh and myself um first item
was the q two roadmap and sarah presented the updated timeline she said the um the
mobile app redesign is on track for uh april launch but the api migration is behind by
about two weeks um mike said he needs two more engineers to make the original deadline
uh action item is for jennifer to check with hr about the headcount request by uh end
of week friday march twentieth um second item was the customer feedback review uh we
looked at the nps scores which dropped from uh seventy two to sixty eight and jennifer
thinks it's related to the recent pricing changes period so um action item is for sarah
to schedule a follow up meeting with the customer success team next week
```

**Output:**
```
Meeting Notes — Product Sync, March 15, 2026

Attendees: Sarah Chen, Mike Rodriguez, Jennifer Walsh, and myself.

**Q2 Roadmap**
Sarah presented the updated timeline. The mobile app redesign is on track for April
launch, but the API migration is behind by about two weeks. Mike said he needs two
more engineers to make the original deadline.

- Action item: Jennifer to check with HR about the headcount request by end of week
  (Friday, March 20).

**Customer Feedback Review**
We looked at the NPS scores, which dropped from 72 to 68. Jennifer thinks it's related
to the recent pricing changes.

- Action item: Sarah to schedule a follow-up meeting with the Customer Success team
  next week.
```

## Common Pitfalls

- **Structure the output**: Meeting notes should have clear sections, attendee lists,
  and highlighted action items — even though the input is stream-of-consciousness
- **Preserve ALL names**: Every person mentioned must appear in the output, properly
  capitalized
- **Action items must be clear**: Each one with WHO, WHAT, and WHEN
- **Dates and numbers**: "march fifteenth twenty twenty six" → "March 15, 2026",
  "seventy two" → "72", "two more engineers" → "two more engineers" (small numbers
  can stay as words when they're not metrics)
- **Don't invent structure the speaker didn't imply**: If they only mentioned two
  agenda items, don't create a third
- **Out-of-order recall**: If the speaker remembers something late ("oh I forgot..."),
  put it in the logical place in the output, not at the end
