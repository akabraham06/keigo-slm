"""Compositional seed generator — scales the trainable set WITHOUT a teacher key.

Method (honest augmentation): we cross a small bank of hand-verified register PREDICATES
(each with informal/polite/formal Japanese *and* register-matched English) with a bank of
document OBJECTS. Every generated sentence is then:
  1. VALIDATED — the register checker must agree with the intended band (else dropped), and
  2. LEAK-FILTERED — any sentence whose content stem also appears in the golden set is dropped.

This guarantees label-correct, leak-safe data at scale. The trade-off vs. teacher distillation
is phrasing diversity: the English is templated. For the final dataset, run data/generate.py
(teacher) to diversify. This is a business-document-heavy bank on purpose — it fills the
formal/keigo band that casual corpora (and the CoCoA harvest, formal=15/1980) lack.

Output: data/seed/generated_seed.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from register_checker import classify_register  # noqa: E402
from data.split import _STRIP, _PUNCT  # reuse the marker-strip stem logic  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

# (jp_object, en_object) — document-like nouns that fit transfer/perception predicates.
OBJECTS = [
    ("報告書", "the report"), ("請求書", "the invoice"), ("契約書", "the contract"),
    ("見積書", "the estimate"), ("企画書", "the proposal"), ("議事録", "the minutes"),
    ("領収書", "the receipt"), ("メール", "the email"), ("ファイル", "the file"),
    ("カタログ", "the catalogue"), ("予定表", "the schedule"), ("書類", "the documents"),
]

# Each predicate: jp templates by band (o = object) + matching English (e = en object).
# Formal templates all contain a checker marker (いたします / 拝見 / 頂戴いたします).
PREDICATES = {
    "send":    {"inf": "{o}送るね。",       "pol": "{o}を送ります。",       "for": "{o}をお送りいたします。",
                "en": ("I'll send {e}.", "I'll send {e} over.", "I will send {e} to you.")},
    "confirm": {"inf": "{o}確認しとくね。", "pol": "{o}を確認します。",     "for": "{o}を確認いたします。",
                "en": ("I'll double-check {e}.", "I'll confirm {e}.", "I will confirm {e}.")},
    "prepare": {"inf": "{o}用意するね。",   "pol": "{o}を用意します。",     "for": "{o}をご用意いたします。",
                "en": ("I'll get {e} ready.", "I'll prepare {e}.", "I will prepare {e}.")},
    "bring":   {"inf": "{o}持ってくね。",   "pol": "{o}を持っていきます。", "for": "{o}をお持ちいたします。",
                "en": ("I'll bring {e}.", "I'll bring {e} along.", "I will bring {e}.")},
    "review":  {"inf": "{o}見とくね。",     "pol": "{o}に目を通します。",   "for": "{o}を拝見いたします。",
                "en": ("I'll take a look at {e}.", "I'll look over {e}.", "I would be glad to review {e}.")},
    "submit":  {"inf": "{o}出すね。",       "pol": "{o}を提出します。",     "for": "{o}を提出いたします。",
                "en": ("I'll hand in {e}.", "I'll submit {e}.", "I will submit {e}.")},
    "return":  {"inf": "{o}返すね。",       "pol": "{o}をお返しします。",   "for": "{o}をお返しいたします。",
                "en": ("I'll give {e} back.", "I'll return {e}.", "I will return {e}.")},
    "receive": {"inf": "{o}もらっとくね。", "pol": "{o}を受け取ります。",   "for": "{o}を頂戴いたします。",
                "en": ("Got {e}, thanks.", "I've received {e}.", "I gratefully accept {e}.")},
}

BANDS = [("inf", "informal", 0), ("pol", "polite", 1), ("for", "formal", 2)]


def stem(jp: str) -> str:
    s = jp
    for m in _STRIP:
        s = s.replace(m, "")
    return s.strip(_PUNCT)


def main() -> None:
    golden = [json.loads(l) for l in (REPO / "data/golden/eval_set.jsonl").read_text().splitlines() if l.strip()]
    golden_stems = {stem(r["jp"]) for r in golden}

    rows, dropped_label, dropped_leak = [], 0, 0
    for pred_name, p in PREDICATES.items():
        for oi, (o_jp, o_en) in enumerate(OBJECTS):
            gid = f"seedgen-{pred_name}-{oi}"
            for key, band, idx in BANDS:
                jp = p[key].format(o=o_jp)
                if classify_register(jp) != band:          # (1) label must be checker-verified
                    dropped_label += 1
                    continue
                if stem(jp) in golden_stems:               # (2) no content overlap with the exam
                    dropped_leak += 1
                    continue
                rows.append({
                    "jp": jp,
                    "source_band": band,
                    "en": p["en"][idx].format(e=o_en),
                    "group_id": gid,
                    "domain": "business",
                    "source": "seed-compositional",
                })

    out = REPO / "data/seed/generated_seed.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    bands = {b: sum(1 for r in rows if r["source_band"] == b) for b in ("informal", "polite", "formal")}
    print(f"generated {len(rows)} rows -> {out}")
    print(f"bands: {bands} | dropped(label mismatch)={dropped_label} dropped(leak vs golden)={dropped_leak}")


if __name__ == "__main__":
    main()
