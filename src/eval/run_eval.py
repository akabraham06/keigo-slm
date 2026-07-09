"""The results table: base vs tuned vs fairly-prompted LLM on the golden set.  [STUB]

This is the single most important script in the repo — it produces the evidence.

Flow:
  1. Load data/golden/eval_set.jsonl (each row has jp + known source_band).
  2. For each model in {base, tuned, llm}:
       - translate every jp  (base/tuned via inference.translate; llm via the teacher client
         with prompts/llm_baseline.md — the SAME task spec, temp 0, no per-item label)
       - judge.classify_english_band(output) -> predicted_band
       - scorer.score(rows)
  3. Print scorer.format_table(...) and write results/results_table.md + charts.

Fairness (see prompts/): every model gets the same task spec, none gets the answer for a
specific item, decoding held constant. Publish the prompts alongside the numbers.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Wire base/tuned/llm translations -> judge -> scorer.score -> results/results_table.md"
    )


if __name__ == "__main__":
    main()
