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


# --- Free, no-API fallback judge -------------------------------------------------
# A rule-based English-register classifier. Approximate: good at the formal band
# (the key anti-flattening signal), noisier on the informal<->polite boundary.
# Use via run_eval --judge heuristic when you have no API key.
_FORMAL_CUES = (
    "grateful", "kindly", "might i", "may i", "i shall", "would be glad", "would you care",
    "allow me", "i offer", "sincerest", "beg your pardon", "honored", "please do",
    "i would ask", "quite all right", "i share that view", "gratefully", "is that so",
    "call on you", "welcoming you", "do take", "with care", "i am most", "cordially",
    "i would be most", "please accept", "introduce myself", "care to order", "i believe that",
)
_INFORMAL_CUES = (
    "gonna", "wanna", "gimme", "kinda", "yeah", "nope", "hey", "c'mon", "dig in", "hang on",
    " sec", "cool", "real quick", "swing by", "hit you up", "tons of", "got a sec", "you good",
    "lemme", "peek", "'kay", "see ya", "ring you",
)


_POLITE_CUES = ("please", "could you", "thank you", "would you like", "i'd like", "may i help")
_MODALITY = (" i will ", " i would ", " i shall ", " we will ", " may i ", " shall we ")


def classify_english_band_heuristic(text: str) -> str:
    """Rule-based band guess. No API. Approximate — prefer the LLM judge when possible."""
    t = " " + text.lower().strip() + " "
    has_contraction = contraction_rate(text) > 0
    if any(c in t for c in _FORMAL_CUES):
        return "formal"
    if any(c in t for c in _INFORMAL_CUES) or text.strip().endswith("!"):
        return "informal"
    if any(c in t for c in _POLITE_CUES):
        return "polite"
    if not has_contraction and any(m in t for m in _MODALITY):
        return "formal"
    if not has_contraction:               # bare imperative / short statement, no politeness marker
        return "informal"
    return "polite"                        # contracted, neutral, no slang
