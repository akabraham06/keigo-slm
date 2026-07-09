"""Register-match scoring.  [REAL LOGIC]

Given rows that each have a known `source_band` and a `predicted_band` (what band the
model's ENGLISH output landed in, per the judge), compute the headline metrics:

  - register_match_accuracy : fraction where predicted_band == source_band
  - flattening_rate         : fraction of NON-informal sources rendered at a LOWER register
                              (the forbidden failure — formal/polite collapsing toward casual)
  - per-band accuracy and a confusion matrix

This is deliberately relational: we never score "how formal is this English" in the
absolute, only whether the output band matches the known source band. That is the only
trustworthy target-side metric (see SPOV 2).
"""
from __future__ import annotations

# Ordered low -> high. Used to detect "flattening" (a drop in register).
ORDER = {"informal": 0, "polite": 1, "formal": 2}


def score(rows: list[dict]) -> dict:
    """rows: [{"source_band": ..., "predicted_band": ...}, ...]"""
    n = len(rows)
    if n == 0:
        return {"n": 0}

    correct = 0
    flattened = 0
    non_informal = 0
    per_band_total: dict[str, int] = {}
    per_band_correct: dict[str, int] = {}
    confusion: dict[str, dict[str, int]] = {}

    for r in rows:
        src, pred = r["source_band"], r["predicted_band"]
        per_band_total[src] = per_band_total.get(src, 0) + 1
        confusion.setdefault(src, {})
        confusion[src][pred] = confusion[src].get(pred, 0) + 1

        if pred == src:
            correct += 1
            per_band_correct[src] = per_band_correct.get(src, 0) + 1

        if src != "informal":
            non_informal += 1
            if ORDER.get(pred, 0) < ORDER[src]:   # output sits at a lower register
                flattened += 1

    return {
        "n": n,
        "register_match_accuracy": round(correct / n, 4),
        "flattening_rate": round(flattened / non_informal, 4) if non_informal else 0.0,
        "per_band_accuracy": {
            b: round(per_band_correct.get(b, 0) / t, 4) for b, t in per_band_total.items()
        },
        "confusion": confusion,
    }


def format_table(results_by_model: dict[str, dict]) -> str:
    """results_by_model: {"base": score(...), "tuned": score(...), "llm": score(...)}"""
    header = f"| {'model':<8} | match acc | flattening | n |\n|---|---|---|---|\n"
    rows = ""
    for name, s in results_by_model.items():
        rows += (
            f"| {name:<8} | {s.get('register_match_accuracy', 0):.2%} "
            f"| {s.get('flattening_rate', 0):.2%} | {s.get('n', 0)} |\n"
        )
    return header + rows


if __name__ == "__main__":
    # Tiny self-check with fake predictions.
    demo = [
        {"source_band": "formal", "predicted_band": "polite"},   # flattened
        {"source_band": "formal", "predicted_band": "formal"},   # correct
        {"source_band": "polite", "predicted_band": "polite"},   # correct
        {"source_band": "informal", "predicted_band": "informal"},
    ]
    print(score(demo))
