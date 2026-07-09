"""Gradio inference demo — deployable as a Hugging Face Space.  [STUB]

Japanese in -> English out, showing the DETECTED source band (from the register checker)
and the register the output preserved. This is also where the 'collapse' demo and the blind
A/B can live interactively.

TODO:
  import gradio as gr
  from register_checker import explain          # show which markers fired
  from inference.translate import translate
  # build gr.Interface(fn=..., inputs=jp_textbox, outputs=[en, detected_band, markers])
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    raise NotImplementedError(
        "Build the Gradio interface: jp -> (english, detected source band, markers fired)."
    )


if __name__ == "__main__":
    main()
