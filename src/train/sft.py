"""QLoRA supervised fine-tuning on a small Qwen3 base.  [STUB]

Core arc. Uses Unsloth (wraps TRL SFTTrainer + PEFT). The training example's `en` is the
target the model learns to produce from `jp`, using the template in prompts/slim_inference
(the SAME template used at eval/inference — do not diverge).

TODO:
  from unsloth import FastLanguageModel
  model, tok = FastLanguageModel.from_pretrained("unsloth/Qwen3-1.7B", load_in_4bit=True)
  model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, ...)
  # format each row with prompts/slm_inference.txt, then TRL SFTTrainer(...).train()
  # push adapter: model.push_to_hub(os.environ["HF_MODEL_REPO"])

Reminder: fix disappointing results in the DATA (coverage/balance), not hyperparameters.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Implement QLoRA SFT with Unsloth. Start from Qwen3-1.7B-Instruct.")


if __name__ == "__main__":
    main()
