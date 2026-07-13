# Data

Three kinds of data, three roles. See the root README for the train/dev/golden split rules.

## `seed/` — the trainable set (committed, verified): **333 rows total**

**`contrastive_seed.jsonl`** — 45 hand-authored rows (15 groups × 3 registers), casual /
business / service, each with register-matched English. `source: curated-seed`.

**`generated_seed.jsonl`** — 288 rows from `python -m data.build_seed`: a compositional
generator that crosses verified register *predicates* with document *objects*. Every row is
(1) validated — the checker must agree with the intended band — and (2) leak-filtered against
the golden set. `source: seed-compositional`. Business-document-heavy on purpose, to fill the
formal/keigo band that casual corpora lack. Trade-off: templated English phrasing (diversify
later with the teacher).

Combined, the trainable set is **balanced 111/111/111**, with **0 checker mismatches, 0
corruption, and 0 content-stem leakage into the golden set** (all re-verified).

## `train_full.jsonl` — the final v2 training mixture: **2,260 rows**

The QLoRA v2 run uses the merged, leak-guarded set produced by
`python -m data.build_training_set`: **904 informal / 904 polite / 452 formal** examples.
It combines 1,076 Tatoeba pairs, 1,043 BSD pairs, 96 formal compositional examples, and the
45 curated contrastive examples. The derived file is gitignored because redistribution
terms differ across its source corpora; the committed scripts, seeds, provenance notice, and
source-specific harvest commands reproduce it.

## `harvested/`  — real source sentences, scraped from CoCoA-MT
`python -m data.harvest_cocoa` pulls ~1,980 **real Japanese sentences** with checker labels
and contrastive `group_id`s. **Source side only** — the register-matched English target must
be generated (see below); CoCoA's English does not carry register. See `harvested/NOTICE.md`
for license (CDLA-Sharing-1.0) and the full explanation. Only a 25-row sample is committed.

> The harvest confirms the core problem in numbers: checker bands came out
> `{polite: 995, informal: 970, formal: 15}` — the **formal/keigo band is nearly absent** in
> casual corpora, which is exactly why it must be supplemented (seed + generation) and why
> the training set must be deliberately balanced.

## `generated/`  — teacher-distilled targets (gitignored, reproducible)
`python -m data.generate` → register-matched English for the harvested sources (teacher, a
constrained generator). `python -m data.filter` → keeps only rows whose English band matches
the source band. Then `python -m data.split` produces `train/dev/test.jsonl`.

## `golden/eval_set.jsonl`  — the held-out exam (labels only, never trained on)
**127 rows** across 42 contrastive content groups, balanced at **42 informal / 43 polite /
42 formal** examples and spanning casual, business, and service language.
`source_band` is the answer key; `reference_en` is a human reference, not used for scoring.
Its `group_id`s are disjoint from seed/samples to prevent leakage, and every label agrees
with the register checker.

## Pipeline

```
harvest_cocoa ─┐
               ├─► generate ─► filter ─► split ─► train/dev/test.jsonl ─► build_dataset (Hub)
curated seed ──┘
```
