"""The results table: base vs tuned vs fairly-prompted LLM on the golden set.

This is the evidence. Flow:
  1. Load data/golden/eval_set.jsonl (jp + known source_band).
  2. For each enabled model, translate every jp, judge the English band, score band-match.
  3. Write results/results_table.md, results/predictions.jsonl, and flag flattening cases.

Fairness: SLMs use the training template (inference.translate); the LLM uses the SAME task
spec (prompts/llm_baseline.md) but is NOT told the per-item band; decoding is greedy/temp 0.

Runs fully on Colab (GPU for the SLMs) with a teacher/judge key set. Individual pieces can be
disabled:
  python -m eval.run_eval --tuned outputs/keigo-sft --base unsloth/Qwen3-1.7B --llm
  python -m eval.run_eval --tuned outputs/keigo-sft --no-base --no-llm   # tuned only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import scorer  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent


def evaluate(golden: list[dict], translate_fns: dict, judge_fn):
    """Core, dependency-injected so it is testable offline.

    translate_fns: {model_name: (jp -> english)}   judge_fn: (english -> band)
    Returns (results_by_model, predictions_by_model).
    """
    results, preds_by_model = {}, {}
    for name, translate in translate_fns.items():
        scored, preds = [], []
        for r in golden:
            en = translate(r["jp"])
            band = judge_fn(en)
            scored.append({"source_band": r["source_band"], "predicted_band": band})
            preds.append({"jp": r["jp"], "source_band": r["source_band"],
                          "output_en": en, "predicted_band": band, "model": name})
        results[name] = scorer.score(scored)
        preds_by_model[name] = preds
        print(f"  {name}: match={results[name]['register_match_accuracy']:.2%} "
              f"flattening={results[name]['flattening_rate']:.2%}")
    return results, preds_by_model


def _write_outputs(results, preds_by_model):
    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)

    # results table
    table = scorer.format_table(results)
    per_band = "\n".join(
        f"- **{m}** per-band: {s.get('per_band_accuracy', {})}" for m, s in results.items())
    (out_dir / "results_table.md").write_text(
        f"# Results — base vs tuned vs LLM (golden set)\n\n{table}\n\n{per_band}\n\n"
        "Metric = output English band == known source band. `flattening` = non-casual "
        "sources rendered at a lower register (the forbidden failure).\n")

    # raw predictions (for charts + inspection)
    all_preds = [p for ps in preds_by_model.values() for p in ps]
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in all_preds))

    # flattening examples for the error analysis
    flats = [p for p in all_preds
             if scorer.ORDER.get(p["predicted_band"], 0) < scorer.ORDER[p["source_band"]]]
    lines = [f"- [{p['model']}] {p['source_band']}→{p['predicted_band']}: "
             f"{p['jp']} → {p['output_en']}" for p in flats[:40]]
    (out_dir / "error_analysis.md").write_text(
        "# Error analysis — flattening cases (source rendered at a lower register)\n\n"
        + ("\n".join(lines) if lines else "_No flattening on this run._") + "\n")
    print(f"wrote results/results_table.md, predictions.jsonl, error_analysis.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden/eval_set.jsonl")
    ap.add_argument("--tuned", default="outputs/keigo-sft")
    ap.add_argument("--base", default="unsloth/Qwen3-1.7B")
    ap.add_argument("--no-base", dest="base", action="store_const", const="")
    ap.add_argument("--llm", dest="llm", action="store_true", default=True)
    ap.add_argument("--no-llm", dest="llm", action="store_false")
    ap.add_argument("--judge", choices=["llm", "heuristic"], default="llm",
                    help="'heuristic' = free rule-based band classifier (no API key)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    golden = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()]
    if args.limit:
        golden = golden[: args.limit]

    if args.judge == "heuristic":
        from eval.judge import classify_english_band_heuristic as judge_fn
        print("judge: heuristic (rule-based, no API — approximate)")
    else:
        from eval.judge import classify_english_band as judge_fn
    translate_fns = {}

    if args.base:
        from inference.translate import make_translator
        translate_fns["base"] = make_translator(args.base)
    if args.tuned:
        from inference.translate import make_translator
        translate_fns["tuned"] = make_translator(args.tuned)
    if args.llm:
        from llm_client import complete, load_prompt
        tmpl = load_prompt("llm_baseline.md")
        translate_fns["llm"] = lambda jp: complete(
            system="You are an expert Japanese-to-English translator.",
            user=tmpl.replace("{JP}", jp))

    if not translate_fns:
        raise SystemExit("no models enabled")

    print(f"evaluating {list(translate_fns)} on {len(golden)} golden rows")
    results, preds = evaluate(golden, translate_fns, judge_fn)
    print("\n" + scorer.format_table(results))
    _write_outputs(results, preds)


if __name__ == "__main__":
    main()
