"""Distill register-matched ENGLISH targets from a frontier teacher.  [RUNNABLE with a key]

Input : harvested Japanese SOURCE sentences (data/harvested/cocoa_ja_sources.jsonl) and/or
        any {jp, source_band, group_id, ...} rows lacking an `en`.
Output: data/generated/raw.jsonl — the same rows with a register-matched `en` filled in.

The teacher is a CONSTRAINED generator: we pass the source band (known from the checker) and
the English-band rubric (prompts/teacher_generation.md), so it can't flatten. Quality is
enforced afterward by filter.py (which re-judges the English band and drops mismatches).

Usage:
  python -m data.generate                       # uses the harvested file
  python -m data.generate --in data/harvested/cocoa_ja_sources.jsonl --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import complete, load_prompt  # noqa: E402

BAND_NAME = {"informal": "casual", "polite": "neutral-polite", "formal": "formal-deferential"}
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def build_prompt(template: str, band: str, jp: str) -> str:
    return template.replace("{BAND}", BAND_NAME.get(band, band)).replace("{JP}", jp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/harvested/cocoa_ja_sources.jsonl")
    ap.add_argument("--out", default="data/generated/raw.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    template = load_prompt("teacher_generation.md")
    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w") as fh:
        for i, r in enumerate(rows):
            prompt = build_prompt(template, r["source_band"], r["jp"])
            # teacher_generation.md is a full instruction; send it as the user message.
            en = complete(system="You are an expert Japanese-to-English translator.", user=prompt)
            out = {k: v for k, v in r.items() if k != "generic_en_ref"}
            out["en"] = en
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"  {written}/{len(rows)}")
    print(f"generated {written} rows -> {out_path}  (now run: python -m data.filter)")


if __name__ == "__main__":
    main()
