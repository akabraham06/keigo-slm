# Judge rubric (English band classifier)

Given to the judge model to classify which band an **English** translation lands in.
Fixed rubric + calibration examples, temperature 0. Reused by `data/filter.py` so the
training gate and the eval scorer share one definition of English register.

---

Classify the politeness register of the English sentence below as exactly one of:
`casual`, `neutral-polite`, or `formal-deferential`.

Signals:
- **casual** — contractions ("I'll", "there's"), phrasal/informal verbs, "yeah/nope/gonna",
  direct imperatives, slang.
- **neutral-polite** — full forms, "please / could you", standard courteous English, no
  slang, no elevated diction.
- **formal-deferential** — "I would be grateful if…", nominalizations, hedged/indirect
  requests, honorific address ("sir/madam"), deferential tone, no contractions.

Calibration:
- "Want some coffee?" → casual
- "Would you like some coffee?" → neutral-polite
- "Would you care for some coffee, sir?" → formal-deferential

Respond with only the single label.

English: {EN}
