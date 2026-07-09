"""LLM-as-judge: classify which English band a translation lands in.  [STUB]

Given an English translation, return "informal" | "polite" | "formal" using the rubric in
prompts/judge_rubric.md (few-shot anchored, temperature 0). This is the ONLY fuzzy piece of
the eval, so keep it disciplined:
  - fixed rubric + calibration examples
  - a cheap deterministic cross-check (contraction rate, politeness markers) to catch drift

The SAME function is reused by data/filter.py so the training gate and the eval agree on
what English register means, by construction.

TODO: implement classify_english_band(text) against the teacher/judge model.
"""
from __future__ import annotations


def classify_english_band(english_text: str) -> str:
    raise NotImplementedError("Call the judge model with prompts/judge_rubric.md, temp=0.")


def contraction_rate(text: str) -> float:
    """Cheap deterministic signal used to sanity-check the judge (informal English contracts)."""
    contractions = ("'re", "'s", "'ll", "'ve", "'d", "n't", "'m", "gonna", "wanna", "kinda")
    words = max(len(text.split()), 1)
    return sum(text.lower().count(c) for c in contractions) / words
