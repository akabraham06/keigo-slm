"""Load the tuned model and translate.  [STUB]

Used by both run_eval.py (for base + tuned) and app/app.py. Must format the input with the
EXACT template the model was trained on (prompts/slm_inference.txt) — diverging here
silently handicaps the model.

TODO:
  def load(model_id): ...            # FastLanguageModel.from_pretrained(..., inference)
  def translate(model, tok, jp): ... # apply template, generate (greedy/temp 0), return EN
"""
from __future__ import annotations


def translate(jp: str) -> str:
    raise NotImplementedError("Load the tuned model and translate with the training template.")
