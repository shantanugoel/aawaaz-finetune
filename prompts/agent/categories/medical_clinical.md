# Category: Medical / Clinical

> **Setup**: Read `prompts/agent/generate.md` for base generation instructions and
> workflow. Read `prompts/agent/validate.md` when spawning validation sub-agents.

## Description

Healthcare professionals dictating patient notes, clinical observations, referral
letters, or medical case discussions. Heavy medical terminology spoken naturally and
casually, as a clinician would when dictating after or during a patient encounter.

## Speech Patterns for This Category

- Medical abbreviations spoken out: "b p" not "BP", "c t scan" not "CT scan",
  "h r" not "HR"
- Vital signs as words: "one twenty over eighty" not "120/80"
- Dosages spoken: "five hundred milligrams" or "five hundred migs" not "500 mg"
- Casual clinical speech: "gonna put them on", "looks like", "the labs came back and"
- Thinking through differentials: "so it could be... or maybe it's..."
- Dictation flow: "period", "comma", spoken section headers
- Physical exam described casually: "lungs sound clear", "belly is soft"

## Example Pair

**Input:**
```
okay so um patient is a forty seven year old female presenting with uh chief complaint
of persistent headaches for the past um two weeks duration period the headaches are
described as uh bilateral and throbbing worse in the morning um vital signs b p one
thirty two over eighty four h r seventy eight temperature ninety eight point six period
uh current medications include uh lisinopril ten milligrams daily and um ibuprofen as
needed period plan is to order an mri of the brain and uh refer to neurology for
further evaluation period
```

**Output:**
```
Patient is a 47-year-old female presenting with chief complaint of persistent
headaches for the past two weeks' duration. The headaches are described as bilateral
and throbbing, worse in the morning.

Vital signs: BP 132/84, HR 78, temperature 98.6°F.

Current medications include lisinopril 10 mg daily and ibuprofen as needed.

Plan is to order an MRI of the brain and refer to neurology for further evaluation.
```

## Common Pitfalls

- **THE #1 FAILURE**: Making the input sound like a written chart note. "Patient
  presents with SOB. BP: 120/80." is written — no clinician speaks like that while
  dictating.
- **Medical formatting in output**: BP as "132/84" not "one thirty two over eighty
  four", dosages as "10 mg" not "ten milligrams"
- **Standard abbreviations in output**: BP, HR, MRI, CT — use standard medical
  abbreviations even if spoken in full
- **EVERY clinical detail must be preserved**: Missing a dosage, lab value, or vital
  sign is a critical failure. Medical data accuracy is paramount.
- **Temperature units**: Add °F or °C based on context (US = Fahrenheit typically)
- **Clinical structure in output**: Group vitals together, separate plan from
  presentation
