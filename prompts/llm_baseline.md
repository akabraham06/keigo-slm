# LLM baseline prompt (the fair comparison)

Given to the frontier LLM in the eval. It carries the **same task spec** as the teacher
prompt, but — crucially — it is **NOT** told the source band. The model must infer register
from the Japanese itself, exactly as the SLM must. Decoding: temperature 0.

This is the honest baseline: same specification of the goal, different mechanism (prompt vs
weights). Do not hand it the per-item answer, and do not water it down.

---

You are an expert Japanese→English translator. Translate the sentence below into natural
English that **preserves the politeness register of the source**.

Japanese politeness is encoded in the verb form. Detect it and match it in English:
- plain / casual Japanese → casual English (contractions, informal).
- です・ます (polite) Japanese → standard, courteous English.
- 尊敬語・謙譲語 (honorific/humble) Japanese → formal, deferential English (no contractions).

Output **only** the English translation.

Japanese: {JP}
