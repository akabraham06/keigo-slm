# Error analysis

_Fill in after the first base-vs-tuned run (assignment Day 4)._

For each remaining tuned-model failure:
- Which band boundary is confused? (e.g. formal→polite flattening)
- Is it a **data** problem? (coverage/balance at that boundary) — the answer is almost
  always yes. Fix by adding examples / sharper contrastive pairs at the confused boundary,
  then retrain. **Do not** tune hyperparameters to paper over a data gap.
