"""Hard quality + register gate.  [RUNNABLE with a key]

Keep a generated pair ONLY if the English output's judged band == the source band. This is
why "dataset > prompt" works: expensive per-example curation once, at training time.

Reuses eval.judge.classify_english_band so the training filter and the eval scorer agree on
what English register means. (Adequacy — does the English convey the meaning — is a second
gate you can add with another teacher call; kept optional here to control cost.)

Usage:
  python -m data.filter --in data/generated/raw.jsonl --out data/generated/filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.judge import classify_english_band  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/generated/raw.jsonl")
    ap.add_argument("--out", default="data/generated/filtered.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    kept, dropped = [], 0
    for r in rows:
        if not r.get("en"):
            dropped += 1
            continue
        if classify_english_band(r["en"]) == r["source_band"]:   # register match
            kept.append(r)
        else:
            dropped += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    rate = len(kept) / len(rows) if rows else 0
    print(f"kept {len(kept)}/{len(rows)} ({rate:.0%}), dropped {dropped} -> {out}")


if __name__ == "__main__":
    main()
