# Data

Three kinds of data, three roles. See the root README for the train/dev/golden split rules.

## `seed/contrastive_seed.jsonl`  — curated, committed, ready to use
45 rows = **15 contrastive groups × 3 registers** (informal / polite / formal), each with a
**register-matched English** target, spanning casual / business / service domains.
Hand-authored (teacher-authored, `source: curated-seed`) and **validated**: every row's
label equals the register checker's output (0 mismatches), and the bands are perfectly
balanced (15/15/15). This is honest, immediately-trainable data — not scraped.

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
**10 rows** (a starter set — expand toward ~100–150 before your final number) across 3
contrastive groups (`offer-coffee`, `reviewed-docs`, `please-wait`), bands ~3/4/3.
`source_band` is the answer key; `reference_en` is a human reference, not used for scoring.
Its `group_id`s are disjoint from seed/samples to prevent leakage, and every label agrees
with the register checker.

## Pipeline

```
harvest_cocoa ─┐
               ├─► generate ─► filter ─► split ─► train/dev/test.jsonl ─► build_dataset (Hub)
curated seed ──┘
```
