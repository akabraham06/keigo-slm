"""Harvest real register-labeled JAPANESE source sentences from CoCoA-MT.

Source: amazon-science/contrastive-controlled-mt (CoCoA-MT), en-ja.
License: Community Data License Agreement – Sharing 1.0 (CDLA-Sharing-1.0). Redistribution
is permitted provided the license travels with the data — see data/harvested/NOTICE.md.

IMPORTANT — why we only take the JAPANESE side:
    CoCoA-MT is an EN->JA formality-control set: for one English source it gives a `formal`
    and an `informal` JAPANESE translation. The ENGLISH is identical across both registers.
    For our JP->EN register-PRESERVATION task the English must *carry* the register, which
    CoCoA's English does not. Training JP->EN on (formal_ja -> en) and (informal_ja -> en)
    would teach the model to IGNORE register (same English either way) — the exact failure
    we fight. So we harvest the Japanese as register-labeled SOURCE sentences only, and the
    register-matched English target is produced later by src/data/generate.py (teacher).

Output rows (data/harvested/cocoa_ja_sources.jsonl):
    jp                 : the Japanese sentence (our model's input)
    source_band        : our checker's label on THIS Japanese ("informal"|"polite"|"formal")
    cocoa_label        : CoCoA's own intended label ("formal"|"informal") — for corroboration
    group_id           : shared across the formal/informal twins of one English source
    generic_en_ref     : CoCoA's English — a MEANING reference ONLY, NOT a training target
                         (it is register-generic; do not put it in the `en` field)
    domain, source     : provenance
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from register_checker import classify_register  # noqa: E402

RAW = "https://raw.githubusercontent.com/amazon-science/contrastive-controlled-mt/main/CoCoA-MT"

# (split, domain) pairs available for en-ja.
TRAIN = [("train", "telephony"), ("train", "topical_chat")]


def _fetch_lines(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8").splitlines()


def harvest(sources=TRAIN) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()          # dedup identical (jp, group) rows
    for split, domain in sources:
        base = f"{RAW}/{split}/en-ja/formality-control.{split}.{domain}.en-ja"
        en = _fetch_lines(base)
        formal = _fetch_lines(base + ".formal")
        informal = _fetch_lines(base + ".informal")
        assert len(en) == len(formal) == len(informal), f"line mismatch in {domain}"

        for i, (e, f, inf) in enumerate(zip(en, formal, informal)):
            group = f"cocoa-{split}-{domain}-{i}"
            for jp, cocoa_label in ((f, "formal"), (inf, "informal")):
                jp = jp.strip()
                if not jp:
                    continue
                key = (jp, group)
                if key in seen:        # formal == informal (no register-bearing verb) -> keep one
                    continue
                seen.add(key)
                rows.append({
                    "jp": jp,
                    "source_band": classify_register(jp),
                    "cocoa_label": cocoa_label,
                    "group_id": group,
                    "generic_en_ref": e.strip(),
                    "domain": domain,
                    "source": "cocoa-mt",
                })
        print(f"  {split}/{domain}: {len(en)} english lines")
    return rows


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "harvested"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = harvest()

    out = out_dir / "cocoa_ja_sources.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    bands: dict[str, int] = {}
    for r in rows:
        bands[r["source_band"]] = bands.get(r["source_band"], 0) + 1
    print(f"\nharvested {len(rows)} japanese source sentences -> {out}")
    print(f"checker band distribution: {bands}")
    print("NOTE: `en` target is intentionally absent — generate it with generate.py.")


if __name__ == "__main__":
    main()
