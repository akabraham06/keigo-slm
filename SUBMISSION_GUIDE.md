# Final submission guide

## Submission links

- **Deployed project:** https://keigo-slm.vercel.app
- **Training and evaluation notebook:** https://colab.research.google.com/github/akabraham06/keigo-slm/blob/main/notebooks/train_qlora.ipynb
- **Source and supporting material:** https://github.com/akabraham06/keigo-slm
- **Fine-tuned model:** https://huggingface.co/akabraham06/keigo-slm-qwen3-1.7b-v2

## Do not record yet: one required result is still missing

The model and evaluation harness exist, but the committed result files are placeholders.
The assignment explicitly requires a measured base-vs-tuned comparison. Run the full GPU
evaluation before recording or making a claim that fine-tuning improved performance.

In Colab, use a GPU runtime and run notebook cells **1, 2, and 5** in
`notebooks/train_qlora.ipynb`. The model is already trained, so cells 3 and 4 do not need to
be repeated. Cell 5 evaluates four conditions on the same 127-example golden set:

1. Untuned Qwen3-1.7B with the standard task template.
2. Fine-tuned `keigo-slm` with the identical template and decoding.
3. Frontier LLM with a naive translation request.
4. Frontier LLM with an explicit register-preservation prompt.

It produces:

- `results/results_table.md`
- `results/predictions.jsonl`
- `results/error_analysis.md`
- `results/eval_results.json`
- `app/web/eval_results.json` (the public site's comparison table)

Copy those files back into this repository, replace the evidence placeholder in
`BRAINLIFT.md`, then run:

```bash
python scripts/check_submission.py
```

Record only when that command passes.

## 3–5 minute video script

### 0:00–0:30 — What I trained

“I fine-tuned Qwen3-1.7B with QLoRA for Japanese-to-English translation. The model has one
narrow behavior: preserve the source sentence's politeness register—casual, neutral-polite,
or formal-deferential—in natural English.”

Show the deployed site's title and switch one example through all three register tabs.

### 0:30–1:05 — Goal and behavior spec

“Japanese encodes social distance in verb morphology, while English spreads it across word
choice and syntax. General translation tends to flatten that signal toward neutral English.
My pass/fail spec is that the English output must land in the same register band as the
Japanese source, without lowering formal or polite sources.”

Show `BEHAVIOR_SPEC.md` or the site's thesis section.

### 1:05–1:50 — Data and training

“The dataset was the main intervention. I mined real BSD business and Tatoeba conversational
pairs, added curated contrastive triples, classified Japanese register deterministically,
filtered English targets for register agreement, guarded against golden-set leakage, and
rebalanced the final 2,260 examples to 904 casual, 904 polite, and 452 formal examples. I then
trained a QLoRA adapter for Qwen3-1.7B with one fixed inference template.”

Show the four-step pipeline on the site and briefly show `data/README.md`.

Then open the Colab and show the data-building and QLoRA training cells for about 10 seconds.
Do not narrate every line; the notebook is visual proof that the run is reproducible.

### 1:50–2:45 — Evaluation methodology

“I froze a balanced 127-example golden set—42 casual, 43 polite, and 42 formal examples,
grouped by sentence meaning so register variants cannot leak across splits. Every condition
uses greedy decoding at temperature zero. An LLM judge classifies the English output's band.
The headline metric is register-match accuracy. The forbidden-failure metric is flattening:
a polite or formal source translated at a lower register. I compare the untuned base, the
fine-tuned model, a naive frontier translation prompt, and a register-aware frontier prompt.”

Show the evaluation section and its four rows.

Briefly show Colab cell 5 and its printed results table before returning to the cleaner table
on the deployed site. This connects the public numbers to the actual evaluation run.

### 2:45–3:40 — Results

Read the numbers directly from the deployed results table. Use this structure only after the
measured run exists:

“The untuned base achieved **[BASE MATCH]** register match with **[BASE FLATTENING]**
flattening. The fine-tuned model achieved **[TUNED MATCH]**, a **[DELTA] point** change, while
flattening changed to **[TUNED FLATTENING]**. The naive frontier condition scored **[NAIVE]**,
and the explicit register prompt scored **[PROMPTED]**. This means **[state only the conclusion
supported by the measured delta]**.”

Never say the tuned model “won” if the measured table does not support that statement.

### 3:40–4:15 — Error analysis and close

“The remaining failures are concentrated in **[band boundary from error analysis]**. That
points to **[specific missing or noisy data coverage]**, so the next iteration is a data fix,
not a hyperparameter change. The project demonstrates whether controlling a small model's
training distribution can make one socially important behavior more reliable.”

Show one real failure from `results/error_analysis.md`, then end on the model and GitHub links.

## Final checklist

- [ ] `python scripts/check_submission.py` passes.
- [ ] The deployed page shows measured results, not “Measured run pending.”
- [ ] The results table contains all four comparison conditions on the same `n`.
- [ ] `BRAINLIFT.md` states whether data → behavior held, using the measured delta.
- [ ] The video shows one qualitative example and the quantitative table.
- [ ] The final submission contains the video URL and `https://keigo-slm.vercel.app`.
