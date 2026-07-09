# Harvested data — provenance & license

`cocoa_ja_sources.jsonl` is derived from **CoCoA-MT** (en-ja), from
[amazon-science/contrastive-controlled-mt](https://github.com/amazon-science/contrastive-controlled-mt).

- **License:** Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0).
  Redistribution is permitted provided this license notice travels with the data.
- **Citation:** Nădejde et al., *CoCoA-MT: A Dataset and Benchmark for Contrastive
  Controlled MT with Application to Formality* (NAACL 2022), https://arxiv.org/abs/2205.04022

## What we took, and what we did NOT

We extracted the **Japanese sentences only**, as register-labeled *source* material. We did
**not** use CoCoA's English as a training target: CoCoA is an EN→JA set where the English is
identical across the formal/informal Japanese, so it does not carry register. Using it as a
JP→EN target would teach register *flattening* — the opposite of this project's goal. The
`generic_en_ref` field is a meaning reference only, never a target.

The full harvest is reproducible with `python -m data.harvest_cocoa`; only a small sample is
committed to git (see `cocoa_ja_sources.sample.jsonl`).
