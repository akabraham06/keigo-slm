"""Mine register-MATCHED JP->EN pairs from real human parallel corpora.

Unlike compositional/templated data, these are genuine human translations where the
translator *already preserved* the register. We keep only those:

  for each (ja, en) pair:
     src_band = register_checker(ja)          # deterministic, ~0.97 F1
     out_band = judge(en)                      # heuristic (free) or LLM (needs key)
     keep IF src_band == out_band              # translator preserved register
  then drop any pair whose content stem also appears in the golden set (no leakage).

Primary source: BSD (Business Scene Dialogue) — business dialogue, keigo-rich, so it fills
the formal band that casual corpora lack. License: see the BSD repo; attribute in NOTICE.

Optional: any Hugging Face parallel dataset via --hf-dataset (point it at a verified id,
e.g. an OPUS/Tatoeba JA-EN set, with --ja-field/--en-field).

Usage:
  python -m data.harvest_parallel --judge heuristic                     # BSD, free judge
  python -m data.harvest_parallel --judge llm                           # BSD, LLM judge (key)
  python -m data.harvest_parallel --hf-dataset <id> --hf-config <cfg> \
      --ja-field ja --en-field en --judge heuristic
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from register_checker import classify_register            # noqa: E402
from data.split import _STRIP, _PUNCT                      # marker-strip stem  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
BSD_RAW = "https://raw.githubusercontent.com/tsuruoka-lab/BSD/master"
BSD_SPLITS = ("train", "dev", "test")


def stem(jp: str) -> str:
    s = jp
    for m in _STRIP:
        s = s.replace(m, "")
    return s.strip(_PUNCT)


def _fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def bsd_pairs() -> list[dict]:
    """Yield {ja, en, group_id, domain} from every BSD dialogue turn."""
    pairs = []
    for split in BSD_SPLITS:
        data = _fetch_json(f"{BSD_RAW}/{split}.json")
        for dlg in data:
            did = dlg.get("id", "?")
            for turn in dlg.get("conversation", []):
                ja = (turn.get("ja_sentence") or "").strip()
                en = (turn.get("en_sentence") or "").strip()
                if ja and en:
                    pairs.append({"ja": ja, "en": en,
                                  "group_id": f"bsd-{did}-{turn.get('no')}",
                                  "domain": "business"})
        print(f"  BSD {split}: {len(data)} dialogues")
    return pairs


def hf_pairs(dataset: str, config: str | None, split: str, ja_f: str, en_f: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    pairs = []
    for i, row in enumerate(ds):
        ja = str(row.get(ja_f, "")).strip()
        en = str(row.get(en_f, "")).strip()
        if ja and en:
            pairs.append({"ja": ja, "en": en, "group_id": f"hf-{i}", "domain": "mixed"})
    print(f"  HF {dataset}: {len(pairs)} pairs")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--hf-dataset", default="")
    ap.add_argument("--hf-config", default="")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--ja-field", default="ja")
    ap.add_argument("--en-field", default="en")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/harvested/parallel_matched.jsonl")
    args = ap.parse_args()

    if args.judge == "heuristic":
        from eval.judge import classify_english_band_heuristic as judge
    else:
        from eval.judge import classify_english_band as judge

    # 1. gather source pairs
    if args.hf_dataset:
        pairs = hf_pairs(args.hf_dataset, args.hf_config or None, args.hf_split,
                         args.ja_field, args.en_field)
    else:
        pairs = bsd_pairs()
    if args.limit:
        pairs = pairs[: args.limit]

    # 2. leak-guard: golden content stems
    golden = [json.loads(l) for l in (REPO / "data/golden/eval_set.jsonl").read_text().splitlines() if l.strip()]
    golden_stems = {stem(r["jp"]) for r in golden}

    # 3. label -> judge -> keep matched -> leak-filter
    kept, mism, leak, seen = [], 0, 0, set()
    bands = {"informal": 0, "polite": 0, "formal": 0}
    for p in pairs:
        key = (p["ja"], p["en"])
        if key in seen:
            continue
        seen.add(key)
        src = classify_register(p["ja"])
        if judge(p["en"]) != src:          # translator did NOT preserve register
            mism += 1
            continue
        if stem(p["ja"]) in golden_stems:  # would leak into the exam
            leak += 1
            continue
        kept.append({"jp": p["ja"], "source_band": src, "en": p["en"],
                     "group_id": p["group_id"], "domain": p["domain"],
                     "source": "bsd" if not args.hf_dataset else args.hf_dataset})
        bands[src] += 1

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    total = len(seen)
    print(f"\nsource pairs: {total}")
    print(f"kept (register-matched): {len(kept)}  bands={bands}")
    print(f"dropped: mismatch={mism}  leak_vs_golden={leak}")
    print(f"-> {out}   (judge={args.judge})")
    print("Merge into training with:  cp %s data/seed/" % args.out)


if __name__ == "__main__":
    main()
