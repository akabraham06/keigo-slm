"""Thin OpenAI-compatible client shared by the teacher (generate) and judge (eval/filter).

Reads credentials from the environment (.env). Works with any OpenAI-compatible endpoint
(OpenAI, Together, Fireworks, local vLLM, etc.) via an optional base URL.

Env:
  TEACHER_API_KEY   required to make calls
  TEACHER_MODEL     required, e.g. "gpt-4o" or a provider model id
  TEACHER_BASE_URL  optional, for non-OpenAI OpenAI-compatible endpoints
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text()


@lru_cache(maxsize=1)
def _client():
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install openai") from e

    key = os.environ.get("TEACHER_API_KEY")
    if not key:
        raise RuntimeError("TEACHER_API_KEY not set (copy .env.example to .env and fill it in)")
    kwargs = {"api_key": key}
    if os.environ.get("TEACHER_BASE_URL"):
        kwargs["base_url"] = os.environ["TEACHER_BASE_URL"]
    return OpenAI(**kwargs)


def complete(system: str, user: str, temperature: float = 0.0) -> str:
    """Single-turn completion. temperature=0 for reproducible eval/filtering."""
    model = os.environ.get("TEACHER_MODEL")
    if not model:
        raise RuntimeError("TEACHER_MODEL not set")
    resp = _client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()
