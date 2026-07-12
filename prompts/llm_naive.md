# LLM naive prompt (the "poorly-prompted frontier" baseline)

Given to the frontier LLM in the eval as the NAIVE condition: a user who just wants a
translation and says nothing about politeness. This is the honest "poorly-prompted frontier"
comparison — it has NO register instruction, so the model has no explicit reason to preserve
the source's politeness. Decoding: temperature 0.

Contrast with llm_baseline.md (the fair, well-prompted condition). Report BOTH honestly:
the SLM should beat this naive baseline (that's what fine-tuning buys); matching the
well-prompted baseline is the harder, more impressive result.

---

Translate the following Japanese sentence into English.

Japanese: {JP}
