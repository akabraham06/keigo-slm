# BrainLift — Keigo-preserving JP→EN SLM

**Owner:** Alan Abraham

## Purpose

Prove that a small, open language model can be fine-tuned to do one narrow thing reliably:
translate Japanese into English while faithfully preserving the source's politeness
register (keigo). The bet is **behavior from data** — a curated dataset, not a prompt, is
what makes a tiny model hold a constrained behavior every time.

## Thesis

Did data → behavior hold? _(fill in with evidence from `results/` once the eval runs.)_

## DOK 4 — Spiky Points of View

### SPOV 1 — Register flattening is a data-distribution problem, not a capability problem
Small models *can* represent register; they flatten because the neutral-polite base rate in
their training data swamps the signal. Rebalancing the training data alone (deliberately
over-sampling casual and formal) moves the model to high register fidelity — provable by the
base-vs-tuned delta on a balanced held-out set. The fix is in the data, not the parameters.

### SPOV 2 — Source-side detectability does not imply target-side evaluability
Japanese register is ~0.97 F1 detectable (it's in the verb). English register is *not* — it
is encoded diffusely across lexis and syntax, with no grammatical marker. Therefore the only
trustworthy metric is **relational** (does the output band match the *known* source band),
not an absolute English-formality score. This is also exactly why generic MT metrics (BLEU,
and largely COMET) are blind to the phenomenon: they measure content overlap, not register.

## DOK 3 — Insights (by category)

### Register / keigo (the target behavior)
- Keigo is a bounded, rule-governed system: four classes → three practical bands
  (informal/polite/formal), switchable via the verb alone. Machine-detectable at ~0.97 F1.
  [WMT23]
- The register signal lives in the (usually sentence-final) verb inflection; a small marker
  list reliably identifies the band. This is the exact hook for a cheap deterministic
  checker. [Feely19]
- Fine-tuning on a small labeled contrastive set gives 82% in-domain / 73% out-of-domain
  formality accuracy while maintaining translation quality — direct evidence the
  "dataset, not prompt" bet works. [CoCoA22]

## Experts
- **Maria Nadejde** — controlled/formality-aware MT; CoCoA-MT; IWSLT formality-control task.
- **Weston Feely, Eva Hasler, Adrià de Gispert** — controlling Japanese honorifics in NMT
  (the verb-form heuristics the register checker reuses).
- **Longyue Wang & Siyou Liu** — discourse-aware / zero-pronoun MT.

## Sources
- [WMT23] Better Evaluation for Formality-Controlled En-Ja MT — https://aclanthology.org/2023.wmt-1.49.pdf
- [Feely19] Controlling Japanese Honorifics in En→Ja NMT — https://aclanthology.org/D19-5203.pdf
- [CoCoA22] CoCoA-MT — https://arxiv.org/pdf/2205.04022
