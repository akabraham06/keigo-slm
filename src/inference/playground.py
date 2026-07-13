"""In-notebook playground: type Japanese, see each model's translation + register, live.

Loads the models ONCE, then `compare(jp)` translates your input with every loaded model and
shows, per model: the English, the register the output landed in (heuristic judge), and whether
it PRESERVED or FLATTENED the source register (detected deterministically from the Japanese).

Usage in Colab (GPU runtime):
    from inference.playground import Playground
    pg = Playground(base="unsloth/Qwen3-1.7B",
                    tuned="akabraham06/keigo-slm-qwen3-1.7b-v2")   # loads once (~1-2 min)
    pg.compare("資料を拝見いたします。")     # try any sentence
    pg.widget()                              # optional: a text box + button UI
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from register_checker import classify_register                     # noqa: E402
from eval.judge import classify_english_band_heuristic as judge_en  # noqa: E402

ORDER = {"informal": 0, "polite": 1, "formal": 2}
_BAND = {"informal": "casual  ", "polite": "polite  ", "formal": "formal  "}


def _status(src: str, out: str) -> str:
    d = ORDER[src] - ORDER.get(out, 0)
    if d > 0:
        return "⚠ FLATTENED" + ("  (2-step!)" if d == 2 else "")
    if d < 0:
        return "↑ more formal"
    return "✓ preserved"


class Playground:
    def __init__(self, tuned: str = "akabraham06/keigo-slm-qwen3-1.7b-v2",
                 base: str = "", frontier: bool = False):
        """Load the models once. `base`="" to skip it; `frontier`=True adds a gateway model
        (well-prompted) — needs TEACHER_API_KEY/TEACHER_MODEL[/TEACHER_BASE_URL] in the env."""
        from inference.translate import make_translator
        self.models = {}
        if base:
            print(f"loading base:  {base} …")
            self.models["base"] = make_translator(base)
        print(f"loading tuned: {tuned} …")
        self.models["tuned"] = make_translator(tuned)
        if frontier:
            from llm_client import complete, load_prompt
            tmpl = load_prompt("llm_baseline.md")
            self.models["frontier"] = lambda jp: complete(
                system="You are an expert Japanese-to-English translator.",
                user=tmpl.replace("{JP}", jp))
            print("added frontier (well-prompted, via gateway)")
        print("ready — call pg.compare('日本語')")

    def compare(self, jp: str) -> None:
        jp = jp.strip()
        if not jp:
            return
        src = classify_register(jp)
        print(f"\nJP: {jp}")
        print(f"    source register (from the Japanese): {src.upper()}")
        print("    " + "-" * 66)
        for name, fn in self.models.items():
            try:
                en = fn(jp)
            except Exception as e:                       # frontier/API hiccup shouldn't kill it
                print(f"    {name:<9} [error: {e}]")
                continue
            out = judge_en(en)
            print(f"    {name:<9} {_BAND.get(out, out):<9} {_status(src, out):<16} {en}")

    def widget(self) -> None:
        """Optional text-box + button UI (Colab/Jupyter)."""
        import ipywidgets as widgets
        from IPython.display import display
        box = widgets.Text(placeholder="日本語を入力…", layout=widgets.Layout(width="70%"))
        btn = widgets.Button(description="Translate", button_style="primary")
        out = widgets.Output()

        def run(_):
            with out:
                self.compare(box.value)
        btn.on_click(run)
        box.on_submit(run)
        display(widgets.HBox([box, btn]), out)
