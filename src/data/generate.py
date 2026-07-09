"""Distill training data from a frontier teacher model.  [STUB]

Pipeline per source sentence:
  1. Take a Japanese sentence (from existing corpora and/or teacher-generated).
  2. classify_register(jp) -> known source band.
  3. Ask the teacher to translate, PASSING the known band + the English-band rubric
     (see prompts/teacher_generation.md). The teacher is a *constrained generator*, not an
     oracle — we tell it the answer so it can't flatten.
  4. Emit {jp, en, source_band, group_id, ...}. Filtering happens in filter.py.

For contrastive coverage, generate/collect sentences in matched groups (same meaning,
three registers) and give each group a shared group_id.

TODO: wire up the teacher client (TEACHER_API_KEY / TEACHER_MODEL from .env),
      write to data/generated/raw.jsonl.
"""
from __future__ import annotations

from register_checker import classify_register  # noqa: F401


def main() -> None:
    raise NotImplementedError(
        "Implement teacher distillation. See prompts/teacher_generation.md for the spec, "
        "and remember to over-sample casual + formal so the model can't fall back on the "
        "neutral-polite prior."
    )


if __name__ == "__main__":
    main()
