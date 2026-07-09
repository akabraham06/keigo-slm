"""Leak-safe train / dev / test split.

THE RULE: the golden (test) set must contain sentences the model never trained on.

THE PROJECT-SPECIFIC TRAP: register is switchable on the SAME sentence by verb alone, so a
sentence and its other-register twins are near-duplicates. If an informal twin lands in
`train` and its formal twin in `test`, that is leakage and inflates the score. We therefore
split by CONTENT GROUP, never by individual row — every register variant of one underlying
sentence goes to the same split.

Grouping key, in priority order:
  1. explicit `group_id` (set at generation time for contrastive sets) — preferred
  2. else a derived "stem": the sentence with trailing register markers/punctuation stripped

Usage:
  python -m data.split --in data/generated/filtered.jsonl --out-dir data \
      --ratios 0.8 0.1 0.1 --seed 13
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from register_checker.markers import FORMAL_MARKERS, POLITE_MARKERS

# Longest markers first so we strip e.g. ございます before ます.
_STRIP = sorted(set(FORMAL_MARKERS + POLITE_MARKERS), key=len, reverse=True)
_PUNCT = "。．.！!？?、,　 \t\n"


def content_key(rec: dict) -> str:
    """Return a grouping key that is identical across register variants of one sentence."""
    if rec.get("group_id"):
        return f"gid:{rec['group_id']}"
    stem = rec["jp"]
    for m in _STRIP:
        stem = stem.replace(m, "")
    return "stem:" + stem.strip(_PUNCT)


def split(records: list[dict], ratios=(0.8, 0.1, 0.1), seed=13):
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[content_key(r)].append(r)

    keys = sorted(groups)                 # deterministic base order
    random.Random(seed).shuffle(keys)     # seeded shuffle -> reproducible

    n = len(keys)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    buckets = {
        "train": keys[:n_train],
        "dev": keys[n_train : n_train + n_dev],
        "test": keys[n_train + n_dev :],
    }
    return {name: [r for k in ks for r in groups[k]] for name, ks in buckets.items()}


def _band_counts(rows: list[dict]) -> dict:
    counts = {"informal": 0, "polite": 0, "formal": 0}
    for r in rows:
        counts[r["source_band"]] = counts.get(r["source_band"], 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    splits = split(records, tuple(args.ratios), args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{name:5s}: {len(rows):5d} rows  bands={_band_counts(rows)}  -> {path}")

    # Leakage assertion: no content group may appear in more than one split.
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            k = content_key(r)
            assert seen.setdefault(k, name) == name, f"LEAK: group {k} in multiple splits"
    print("OK: no content group spans multiple splits.")


if __name__ == "__main__":
    main()
