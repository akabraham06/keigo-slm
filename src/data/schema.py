"""Dataset schema (shared by all splits).

Every record is one JSON object per line (JSONL). Training and golden share the same
skeleton but use it differently:

  - TRAINING uses `en` as the imitation target (the model learns to produce it).
  - GOLDEN uses `source_band` as the answer key (the model's OWN output is scored against
    it); `en` is optional there and used only as a human reference — never for scoring.

Fields
------
jp           : str   required. The Japanese source sentence (the model's input).
source_band  : str   required. "informal" | "polite" | "formal". From the register checker.
en           : str   training: required (register-faithful target). golden: optional reference.
group_id     : str   optional but recommended. Contrastive-group key: all register variants
                     of the SAME underlying sentence share one group_id so the splitter keeps
                     them together and they can't leak across train/golden.
domain       : str   optional. e.g. "business", "casual", "narration" — for balance reporting.
source       : str   optional. provenance, e.g. "cocoa-mt", "teacher-gen".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

BANDS = ("informal", "polite", "formal")


@dataclass
class Example:
    jp: str
    source_band: str
    en: Optional[str] = None
    group_id: Optional[str] = None
    domain: Optional[str] = None
    source: Optional[str] = None

    def validate(self) -> None:
        assert self.jp, "jp (source sentence) is required"
        assert self.source_band in BANDS, f"source_band must be one of {BANDS}"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
