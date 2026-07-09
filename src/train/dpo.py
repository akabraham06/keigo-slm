"""Preference tuning (DPO) on top of the SFT model.  [STUB — stretch goal]

Register is a contrastive property, so DPO fits naturally. Preference pairs:
  chosen   = register-matched English
  rejected = flattened English  (FREE: it's what the base model already produces)

Measure whether DPO sharpens spec adherence beyond SFT alone.

TODO: build pairs (reuse base-model outputs as `rejected`), run trl.DPOTrainer.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Build preference pairs, run TRL DPOTrainer on the SFT checkpoint.")


if __name__ == "__main__":
    main()
