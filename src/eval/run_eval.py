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
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import scorer  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

DISPLAY_NAMES = {
    "base": "Untuned Qwen3-1.7B",
    "tuned": "Fine-tuned keigo-slm",
    "llm-naive": "Frontier LLM · naive prompt",
    "llm-baseline": "Frontier LLM · register prompt",
}


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
              f"flattening={results[name]['flattening_rate']:.2%} "
              f"severe={results[name]['flattening_two_step_rate']:.2%}")
    return results, preds_by_model


def _write_outputs(results, preds_by_model, *, judge: str, golden_path: str,
                   model_ids: dict[str, str]) -> None:
    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)

    # results table
    table = scorer.format_table(results)
    legend = (
        "**Columns** — `match↑`: output register == source register (higher better). "
        "`flat↓`: non-casual sources rendered lower (the forbidden failure; lower better). "
        "`severe↓`: subset of flattening that dropped 2 levels (formal→casual). "
        "`informal/polite/formal`: per-band match accuracy.")
    (out_dir / "results_table.md").write_text(
        f"# Results — base vs tuned vs frontier (golden set)\n\n{table}\n\n{legend}\n")

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

    # One machine-readable source feeds both the public site and the submission checker.
    payload = {
        "status": "measured",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "golden_set": golden_path,
            "n": max((s.get("n", 0) for s in results.values()), default=0),
            "judge": judge,
            "decoding": "greedy / temperature 0",
            "metrics": {
                "register_match_accuracy": "Output English register matches the Japanese source.",
                "flattening_rate": "Non-casual sources rendered at a lower English register.",
                "flattening_two_step_rate": "Formal sources collapsed all the way to casual English.",
            },
        },
        "models": [
            {
                "id": name,
                "label": DISPLAY_NAMES.get(name, name),
                "model": model_ids.get(name, ""),
                **scores,
            }
            for name, scores in results.items()
        ],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "eval_results.json").write_text(rendered)
    web_result = REPO / "app/web/eval_results.json"
    web_result.write_text(rendered)
    print("wrote results/results_table.md, predictions.jsonl, error_analysis.md, "
          "eval_results.json, and app/web/eval_results.json")


def _ensure_llm_credentials() -> None:
    """Prompt for any missing LLM gateway credentials so the file is runnable without
    hardcoding secrets. The API key is read hidden (getpass); the base URL and model are
    plain prompts (they aren't secret). Anything already in the environment is left as-is,
    so setting them ahead of time (Colab secrets, --export) skips the prompt entirely."""
    import getpass
    if not os.environ.get("TEACHER_API_KEY"):
        os.environ["TEACHER_API_KEY"] = getpass.getpass("TEACHER_API_KEY (input hidden): ").strip()
    if not os.environ.get("TEACHER_MODEL"):
        os.environ["TEACHER_MODEL"] = input(
            "TEACHER_MODEL (e.g. gpt-4o or claude-group/claude-sonnet-4-6): ").strip()
    if not os.environ.get("TEACHER_BASE_URL"):
        base = input("TEACHER_BASE_URL (blank = OpenAI direct): ").strip()
        if base:
            os.environ["TEACHER_BASE_URL"] = base


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
    ap.add_argument("--llm-prompt", nargs="+", default=["llm_baseline.md"],
                    help="one or more frontier baseline prompts, each becomes its own row. "
                         "e.g. --llm-prompt llm_naive.md llm_baseline.md  (poorly- vs well-prompted)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    golden = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()]
    if args.limit:
        golden = golden[: args.limit]

    # If the frontier baseline or the LLM judge is in play, make sure we have credentials —
    # prompt for whatever is missing rather than forcing the user to edit code.
    if args.llm or args.judge == "llm":
        _ensure_llm_credentials()

    if args.judge == "heuristic":
        from eval.judge import classify_english_band_heuristic as judge_fn
        print("judge: heuristic (rule-based, no API — approximate)")
    else:
        from eval.judge import classify_english_band as judge_fn
    translate_fns = {}
    model_ids = {}

    if args.base:
        from inference.translate import make_translator
        translate_fns["base"] = make_translator(args.base)
        model_ids["base"] = args.base
    if args.tuned:
        from inference.translate import make_translator
        translate_fns["tuned"] = make_translator(args.tuned)
        model_ids["tuned"] = args.tuned
    if args.llm:
        from llm_client import complete, load_prompt
        for prompt_file in args.llm_prompt:
            tmpl = load_prompt(prompt_file)
            # label the row by the prompt, e.g. llm_naive.md -> "llm-naive"
            label = "llm-" + prompt_file.replace("llm_", "").replace(".md", "")
            print(f"frontier baseline: {label}  (prompt={prompt_file})")
            translate_fns[label] = (lambda t: lambda jp: complete(
                system="You are an expert Japanese-to-English translator.",
                user=t.replace("{JP}", jp)))(tmpl)
            model_ids[label] = f"{os.environ.get('TEACHER_MODEL', 'frontier model')} · {prompt_file}"

    if not translate_fns:
        raise SystemExit("no models enabled")

    print(f"evaluating {list(translate_fns)} on {len(golden)} golden rows")
    results, preds = evaluate(golden, translate_fns, judge_fn)
    print("\n" + scorer.format_table(results))
    _write_outputs(results, preds, judge=args.judge, golden_path=args.golden,
                   model_ids=model_ids)


if __name__ == "__main__":
    main()
