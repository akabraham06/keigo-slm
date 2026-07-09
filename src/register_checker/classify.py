"""Deterministic source-side register classifier.

Reads the politeness register off a Japanese sentence's verb morphology and returns one of
three bands: "formal", "polite", "informal".

This is the project's label source: it assigns the ground-truth register to every training
and eval sentence, near-free and near-perfectly (~0.97 F1 on this 3-class task). The eval
scores the model's ENGLISH output against the band this returns for the JAPANESE source.
"""
from __future__ import annotations

from .markers import FORMAL_MARKERS, POLITE_MARKERS

Band = str  # one of: "formal" | "polite" | "informal"


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(m in text for m in markers)


def classify_register(sentence: str) -> Band:
    """Return the register band of a Japanese sentence.

    Order matters: formal (keigo) markers are the most specific, so they are checked first.
    A formal marker such as ございます also contains the polite substring ます, so checking
    polite first would misclassify it.
    """
    if _contains_any(sentence, FORMAL_MARKERS):
        return "formal"
    if _contains_any(sentence, POLITE_MARKERS):
        return "polite"
    return "informal"


def explain(sentence: str) -> dict:
    """Return the band plus which markers fired — useful for the demo and for debugging."""
    formal_hits = [m for m in FORMAL_MARKERS if m in sentence]
    polite_hits = [m for m in POLITE_MARKERS if m in sentence]
    band = "formal" if formal_hits else ("polite" if polite_hits else "informal")
    return {"band": band, "formal_hits": formal_hits, "polite_hits": polite_hits}


if __name__ == "__main__":
    import sys

    for line in sys.stdin:
        line = line.strip()
        if line:
            print(f"{classify_register(line)}\t{line}")
