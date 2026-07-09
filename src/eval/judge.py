"""LLM-as-judge: classify which English band a translation lands in.  [RUNNABLE with a key]

Given an English translation, return "informal" | "polite" | "formal" using the rubric in
prompts/judge_rubric.md (temperature 0). Reused by data/filter.py so the training gate and
the eval scorer share ONE definition of English register, by construction.

A cheap deterministic signal (contraction_rate) is provided to cross-check the judge.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import complete, load_prompt  # noqa: E402

# The judge speaks the rubric's vocabulary; map it back to our band ids.
_LABEL_MAP = {
    "casual": "informal",
    "neutral-polite": "polite",
    "formal-deferential": "formal",
    # tolerate bare forms
    "informal": "informal", "polite": "polite", "formal": "formal",
}


def classify_english_band(english_text: str) -> str:
    rubric = load_prompt("judge_rubric.md").replace("{EN}", english_text)
    raw = complete(system="You are a careful linguistic annotator.", user=rubric).strip().lower()
    for key, band in _LABEL_MAP.items():          # substring match is robust to stray words
        if key in raw:
            return band
    return "polite"  # safe default if the judge is unclear


def contraction_rate(text: str) -> float:
    """Cheap deterministic signal used to sanity-check the judge (casual English contracts)."""
    contractions = ("'re", "'s", "'ll", "'ve", "'d", "n't", "'m", "gonna", "wanna", "kinda", "gimme")
    words = max(len(text.split()), 1)
    return sum(text.lower().count(c) for c in contractions) / words
