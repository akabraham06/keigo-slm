"""Load an SLM (base or tuned) and translate JP->EN with the training template.

Used by eval/run_eval.py (base + tuned models) and app/app.py. The template MUST match the
one used in training (prompts/slm_inference.txt), or the model is silently handicapped.

GPU-only deps (unsloth/torch) are imported lazily so this module imports fine anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import load_prompt  # noqa: E402

_TEMPLATE = None


def _template() -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = load_prompt("slm_inference.txt")
    return _TEMPLATE


def load(model_path: str, max_seq: int = 1024):
    """Load a base model id or a saved/tuned adapter path. Returns (model, tokenizer)."""
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(model_path, max_seq_length=max_seq, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    return model, tok


def translate(model, tok, jp: str, max_new_tokens: int = 64) -> str:
    """Translate one Japanese sentence. Greedy decoding for a reproducible eval."""
    prompt = _template().replace("{JP}", jp)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    # Keep only what the model wrote after the final 'English:' cue, first line.
    tail = text.split("English:")[-1].strip()
    return tail.splitlines()[0].strip() if tail else ""


def make_translator(model_path: str):
    """Convenience: returns a jp->en function with the model held in a closure."""
    model, tok = load(model_path)
    return lambda jp: translate(model, tok, jp)
