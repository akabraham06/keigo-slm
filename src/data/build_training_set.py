"""Merge + rebalance all data sources into one balanced training file.

Why: raw harvested data is severely informal-heavy (5544/3617/154). Training on that base
rate teaches "when unsure, be casual" — flattening in the other direction (SPOV 1). So we
rebalance to a target ratio and prefer REAL data for the informal/polite bands.

Strategy:
  - formal band: keep ALL from every source (it's the scarce, hard band).
  - informal/polite bands: use curated + real (BSD) pairs only; EXCLUDE the templated
    compositional rows (whose near-identical English caused the informal<->polite collapse).
  - downsample informal/polite to `ratio` x formal, curated first (for casual/service
    diversity) then real BSD.
  - dedup by (jp, en); leak-guard every row against the golden set by content stem.

Inputs (skipped if absent):
  data/seed/contrastive_seed.jsonl        (curated, diverse)
  data/seed/generated_seed.jsonl          (compositional/templated — formal kept only)
  data/harvested/parallel_matched.jsonl   (real BSD, run harvest_parallel first)

Output: data/train_full.jsonl   (point sft.py at it; it auto-prefers this file)

Usage:
  python -m data.harvest_parallel --judge heuristic      # produce parallel_matched.jsonl
  python -m data.build_training_set --ratio 2 2 1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.split import _STRIP, _PUNCT  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
SOURCES = [
    "data/seed/contrastive_seed.jsonl",
    "data/seed/generated_seed.jsonl",
    "data/harvested/parallel_matched.jsonl",
]


def stem(jp: str) -> str:
    s = jp
    for m in _STRIP:
        s = s.replace(m, "")
    return s.strip(_PUNCT)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (skip, not found) {path}")
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"  loaded {len(rows):5d} from {path.name}")
    return rows


def _fill_rank(row: dict) -> int:
    """For informal/polite: curated first (diverse), then real BSD."""
    return 0 if row.get("source") == "curated-seed" else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", nargs=3, type=int, default=[2, 2, 1],
                    help="informal polite formal (relative to formal count)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/train_full.jsonl")
    args = ap.parse_args()
    ri, rp, rf = args.ratio
    rng = random.Random(args.seed)

    print("sources:")
    rows = [r for p in SOURCES for r in _load(REPO / p) if r.get("en")]

    golden_stems = {stem(json.loads(l)["jp"])
                    for l in (REPO / "data/golden/eval_set.jsonl").read_text().splitlines() if l.strip()}

    # dedup + leak-guard
    seen, clean, leaked = set(), [], 0
    for r in rows:
        key = (r["jp"], r["en"])
        if key in seen:
            continue
        seen.add(key)
        if stem(r["jp"]) in golden_stems:
            leaked += 1
            continue
        clean.append(r)

    # bucket by band
    bands = {"informal": [], "polite": [], "formal": []}
    for r in clean:
        bands[r["source_band"]].append(r)

    # formal: keep all. informal/polite: exclude templated, cap to ratio x formal.
    n_formal = len(bands["formal"])
    cap = {"informal": ri * n_formal, "polite": rp * n_formal}
    out_rows = list(bands["formal"])
    for band in ("informal", "polite"):
        pool = [r for r in bands[band] if r.get("source") != "seed-compositional"]
        pool.sort(key=_fill_rank)                 # curated (0) before real (1)
        # shuffle within each rank so the BSD fill is representative
        rank0 = [r for r in pool if _fill_rank(r) == 0]
        rank1 = [r for r in pool if _fill_rank(r) == 1]
        rng.shuffle(rank1)
        chosen = (rank0 + rank1)[: cap[band]]
        out_rows.extend(chosen)

    rng.shuffle(out_rows)
    out = REPO / args.out
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))

    final = {b: sum(1 for r in out_rows if r["source_band"] == b) for b in bands}
    by_src = {}
    for r in out_rows:
        by_src[r.get("source", "?")] = by_src.get(r.get("source", "?"), 0) + 1
    print(f"\nleak-guarded out: {leaked}")
    print(f"final training set: {len(out_rows)} rows  bands={final}")
    print(f"by source: {by_src}")
    print(f"-> {out}   (sft.py auto-uses this file if present)")


if __name__ == "__main__":
    main()
