# Teacher generation prompt (data distillation)

Given to the frontier teacher when producing training translations. The teacher is a
**constrained generator**: we tell it the source register (known from the checker) so it
can't flatten. Its output is still hard-filtered afterward.

**This must carry the SAME task spec as `llm_baseline.md`** — that symmetry is what makes
the data-vs-prompt comparison clean.

---

You are an expert Japanese→English translator. Translate the sentence below into natural
English that **preserves its politeness register**.

The source register is: **{BAND}** (one of casual / neutral-polite / formal-deferential).

Render the English so its tone matches that register:
- **casual** — contractions, informal word choice, direct phrasing.
- **neutral-polite** — standard, courteous English; full forms; no slang; no elevated diction.
- **formal-deferential** — deferential, indirect, hedged; honorific/respectful tone; no contractions.

Output **only** the English translation — no notes, no quotes.

Japanese: {JP}
