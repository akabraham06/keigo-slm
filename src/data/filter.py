"""Hard quality + register gate.  [STUB]

Keep a generated pair ONLY if:
  (a) register match — the English output's judged band == the source band, AND
  (b) adequacy — the translation faithfully conveys the meaning (teacher/judge check).

Everything else is discarded. This gate is why "dataset > prompt" works: we spend
expensive per-example curation once, at training time, that we'd never do at inference.

TODO: read data/generated/raw.jsonl -> write data/generated/filtered.jsonl;
      reuse eval.judge for the band check so filter and eval agree by construction.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Implement the filter. Reuse eval.judge.classify_english_band for (a) so the "
        "training filter and the eval scorer use the SAME definition of English register."
    )


if __name__ == "__main__":
    main()
