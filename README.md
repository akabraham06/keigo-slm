# keigo-slm — a small model that preserves Japanese politeness register in JP→EN translation

Fine-tune a small open model (Qwen3) to do **one narrow thing reliably**: translate
Japanese to English while faithfully preserving the source's politeness register
(*keigo*) — casual / polite / formal. The thesis is **behavior from data**: a curated
dataset, not a clever prompt, is what makes a tiny model hold this behavior every time.

> Register (politeness) is grammatically obligatory and verb-encoded in Japanese but has
> no grammatical equivalent in English, so general models silently *flatten* it. That is a
> reliability gap a dataset closes and a prompt can't guarantee.

## Links (fill in as you ship)

| Deliverable | Link |
|---|---|
| 📦 Dataset (Hugging Face) | _TODO_ |
| 🤖 Model (Hugging Face) | [akabraham06/keigo-slm-qwen3-1.7b-v2](https://huggingface.co/akabraham06/keigo-slm-qwen3-1.7b-v2) |
| ▶️ Live demo (HF Space) | _TODO_ |
| 🎬 Demo video (3–5 min) | _TODO_ |

## Reproduce in three commands

```bash
cp .env.example .env          # then fill in HF_TOKEN + teacher key
pip install -r requirements.txt
make test                     # 1. the register checker works
make data && make split       # 2. generate → filter → leak-safe split
make train                    # 3. QLoRA SFT on Qwen3
make eval                     # 4. base vs tuned vs LLM → results table
```

## Run the eval suite

Scores every model on the held-out golden set and writes `results/results_table.md`
(one aligned table: `match↑ | flat↓ | severe↓ | per-band`).

**Free — no API key** (grades the SLMs with the rule-based judge):

```bash
PYTHONPATH=src python -m eval.run_eval \
    --tuned akabraham06/keigo-slm-qwen3-1.7b-v2 \
    --base  unsloth/Qwen3-1.7B \
    --no-llm --judge heuristic
```

**Full report** — adds an LLM judge and both frontier baselines (poorly- vs well-prompted):

```bash
PYTHONPATH=src python -m eval.run_eval \
    --tuned akabraham06/keigo-slm-qwen3-1.7b-v2 \
    --base  unsloth/Qwen3-1.7B \
    --llm --llm-prompt llm_naive.md llm_baseline.md --judge llm
```

The LLM judge/baseline read `TEACHER_API_KEY`, `TEACHER_MODEL`, and optional `TEACHER_BASE_URL`
from the environment. If any is unset, `run_eval` **prompts for it at runtime** (key hidden via
`getpass`) — no secret ever lives in the code. Set them ahead of time (or as Colab secrets) to
skip the prompt.

## What lives where

- [`BEHAVIOR_SPEC.md`](BEHAVIOR_SPEC.md) — the falsifiable spec. **Everything serves this.**
- [`BRAINLIFT.md`](BRAINLIFT.md) — thesis, spiky POVs, and the evidence.
- [`prompts/`](prompts/) — **published for fairness.** Exactly what the teacher, the LLM
  baseline, and the judge were each told. Same task spec, no per-item answer leaked.
- [`src/register_checker/`](src/register_checker/) — deterministic source-side classifier
  (reads register off Japanese verb morphology). This is your near-free, trustworthy label
  source. Unit-tested.
- [`src/data/`](src/data/) — distill from the teacher, hard-filter, split leak-safely, push to Hub.
- [`src/train/`](src/train/) — QLoRA SFT (`sft.py`) and preference tuning (`dpo.py`, stretch).
- [`src/eval/`](src/eval/) — LLM-as-judge, band-match scorer, `run_eval.py` (the results table), charts.
- [`src/inference/`](src/inference/) — load the tuned model and translate.
- [`app/`](app/) — Gradio demo (deployable as an HF Space).
- [`data/golden/`](data/golden/) — the held-out **golden set** (labels only; never trained on).
- [`data/samples/`](data/samples/) — a tiny slice so readers see the data shape.
- [`results/`](results/) — the base-vs-tuned-vs-LLM table + charts + error analysis.

## The metric (why not BLEU)

We score **register-band match**: the source's band is known deterministically (the
checker), and we judge which band the model's *English output* lands in, then compare.
A string-overlap metric like BLEU is **blind to register** — two equally faithful
translations can differ only in tone — so it cannot measure the thing this project is about.
The headline numbers are register-match accuracy and **flattening rate** (formal/humble → neutral collapses).

## The three data splits (and why they differ)

| Split | Role | Seen in training? | Notes |
|---|---|---|---|
| train | teach the behavior | yes | bulk, teacher-generated, auto-filtered |
| dev   | iterate / diagnose | no  | beat it up during data iteration |
| golden (test) | the honest number | **never** | frozen; human-spot-checked labels |

Splitting is **by sentence stem**, not by row: because register is switchable on the same
sentence by verb alone, an informal twin in `train` and its formal twin in `golden` is
leakage. `src/data/split.py` keeps every contrastive group entirely on one side.
