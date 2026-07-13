"""Fail fast when the final SLM submission is missing required evidence."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPECTED_MODELS = {"base", "tuned", "llm-naive", "llm-baseline"}
EXPECTED_N = 127


def main() -> None:
    failures: list[str] = []

    web_path = REPO / "app/web/eval_results.json"
    result_path = REPO / "results/eval_results.json"
    table_path = REPO / "results/results_table.md"
    predictions_path = REPO / "results/predictions.jsonl"

    if not web_path.exists() or not result_path.exists():
        failures.append("measured eval_results.json is missing from results/ or app/web/")
        payload = {}
    else:
        payload = json.loads(result_path.read_text())
        if payload != json.loads(web_path.read_text()):
            failures.append("results/eval_results.json and app/web/eval_results.json differ")

    if payload.get("status") != "measured":
        failures.append("evaluation status is not 'measured'")

    models = payload.get("models", [])
    model_ids = {model.get("id") for model in models}
    missing = EXPECTED_MODELS - model_ids
    if missing:
        failures.append("evaluation is missing conditions: " + ", ".join(sorted(missing)))

    counts = {model.get("n") for model in models if model.get("id") in EXPECTED_MODELS}
    if counts != {EXPECTED_N}:
        failures.append(f"all four conditions must use the full {EXPECTED_N}-example golden set")
    if payload.get("evaluation", {}).get("judge") != "llm":
        failures.append("final comparison must use the LLM judge, not the heuristic fallback")

    if not table_path.exists():
        failures.append("results/results_table.md is missing")
    else:
        table = table_path.read_text().lower()
        if "_todo_" in table or "placeholder" in table or "not real" in table:
            failures.append("results/results_table.md still contains placeholder text")

    prediction_counts: Counter[str] = Counter()
    if not predictions_path.exists():
        failures.append("results/predictions.jsonl is missing")
    else:
        for line in predictions_path.read_text().splitlines():
            if line.strip():
                prediction_counts[json.loads(line)["model"]] += 1
        if set(prediction_counts) != EXPECTED_MODELS:
            failures.append("raw predictions do not contain exactly the four required conditions")
        elif set(prediction_counts.values()) != {EXPECTED_N}:
            failures.append(f"each condition must contain exactly {EXPECTED_N} raw predictions")

    brainlift = (REPO / "BRAINLIFT.md").read_text().lower()
    if "fill in with evidence" in brainlift:
        failures.append("BRAINLIFT.md still has its evidence placeholder")

    if failures:
        print("SUBMISSION NOT READY")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    n = next(iter(counts))
    print("SUBMISSION EVIDENCE READY")
    print(f"  four comparison conditions × {n} held-out examples")
    print(f"  raw predictions: {dict(prediction_counts)}")
    print("  public result JSON and report files agree")


if __name__ == "__main__":
    main()
