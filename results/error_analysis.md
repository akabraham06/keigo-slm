# Error analysis

_Auto-populated by `python -m eval.run_eval` with the tuned model's flattening cases
(source rendered at a lower register). Placeholder until the first real run._

For each remaining tuned-model failure:
- Which band boundary is confused? (e.g. formal→polite flattening)
- Is it a **data** problem? (coverage/balance at that boundary) — almost always yes.
  Fix by adding examples / sharper contrastive pairs at the confused boundary, then retrain.
  **Do not** tune hyperparameters to paper over a data gap.
