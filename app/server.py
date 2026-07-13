"""FastAPI backend for the live demo — runs the tuned SLM and returns register info.

POST /translate {"jp": "..."} -> {jp, source_band, english, output_band, status}
  source_band : register of the JAPANESE source (deterministic checker, ~0.97 F1)
  english     : the SLM's translation
  output_band : register the ENGLISH landed in (free heuristic judge)
  status      : "preserved" | "flattened" | "over_polite"  (the demo's headline)

Run on a GPU box (Colab) and expose via a public tunnel; the SPA (app/web/index.html)
calls it. See app/README.md for the one-cell Colab launcher.

    PYTHONPATH=src uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from register_checker import classify_register                     # noqa: E402
from eval.judge import classify_english_band_heuristic as judge_en  # noqa: E402

MODEL = os.environ.get("KEIGO_MODEL", "akabraham06/keigo-slm-qwen3-1.7b-v2")
ORDER = {"informal": 0, "polite": 1, "formal": 2}

app = FastAPI(title="keigo-slm demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_translate = None   # loaded once at startup


class Req(BaseModel):
    jp: str


@app.on_event("startup")
def _load() -> None:
    global _translate
    from inference.translate import make_translator   # lazy: needs GPU/unsloth
    print(f"loading model: {MODEL} …")
    _translate = make_translator(MODEL)
    print("model ready.")


@app.get("/health")
def health() -> dict:
    return {"ok": _translate is not None, "model": MODEL}


@app.post("/translate")
def translate(r: Req) -> dict:
    jp = r.jp.strip()
    if not jp:
        return {"error": "empty input"}
    src = classify_register(jp)
    english = _translate(jp)
    out = judge_en(english)
    drop = ORDER[src] - ORDER.get(out, 0)
    status = "flattened" if drop > 0 else ("over_polite" if drop < 0 else "preserved")
    return {"jp": jp, "source_band": src, "english": english,
            "output_band": out, "status": status, "model": MODEL}
