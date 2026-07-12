"""QLoRA supervised fine-tuning on a small Qwen3 base (Unsloth + TRL).  [RUNNABLE on a GPU]

Trains the SLM to translate JP->EN while preserving register. Loads every data/seed/*.jsonl
trainable row, formats it with the EXACT template used at eval/inference
(prompts/slm_inference.txt), holds out a small dev slice, and runs QLoRA SFT.

Run on Colab/RunPod (needs a CUDA GPU + `pip install unsloth trl peft datasets`):
    python -m train.sft --model unsloth/Qwen3-1.7B --epochs 3 \
        --push-repo akabraham06/keigo-slm-qwen3-1.7b     # push optional

Fix disappointing results in the DATA (coverage/balance), not the hyperparameters.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import load_prompt  # reads prompts/  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent


def load_rows() -> list[dict]:
    # Prefer the merged+balanced set if it exists (data.build_training_set), else the seed.
    full = REPO / "data/train_full.jsonl"
    if full.exists():
        rows = [json.loads(l) for l in open(full) if l.strip()]
        print(f"using balanced training set: {full.name}")
    else:
        rows = []
        for f in sorted(glob.glob(str(REPO / "data/seed/*.jsonl"))):
            rows += [json.loads(l) for l in open(f) if l.strip()]
        print("using data/seed/*.jsonl (run build_training_set for the balanced set)")
    rows = [r for r in rows if r.get("en")]           # need a target
    if not rows:
        raise SystemExit("no trainable rows with `en`")
    return rows


def build_dataset(rows, tokenizer):
    from datasets import Dataset
    template = load_prompt("slm_inference.txt")       # contains {JP} and ends with 'English:'
    def to_text(r):
        prompt = template.replace("{JP}", r["jp"])
        return {"text": f"{prompt} {r['en']}{tokenizer.eos_token}"}
    return Dataset.from_list([to_text(r) for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-1.7B")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--out", default="outputs/keigo-sft")
    ap.add_argument("--push-repo", default="")        # HF repo id; empty = don't push
    args = ap.parse_args()

    # Imported here so the module imports fine on machines without a GPU / unsloth.
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=args.max_seq, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    rows = load_rows()
    ds = build_dataset(rows, tok).train_test_split(test_size=0.1, seed=13)
    print(f"train={len(ds['train'])}  dev={len(ds['test'])}  (from {len(rows)} rows)")

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        args=SFTConfig(
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            warmup_steps=5, num_train_epochs=args.epochs, learning_rate=args.lr,
            logging_steps=5, optim="adamw_8bit", seed=13,
            output_dir=args.out, dataset_text_field="text",
            report_to="none"),   # no wandb (avoids the interactive login prompt in Colab)
    )
    trainer.train()

    Path(REPO / args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(REPO / args.out))
    tok.save_pretrained(str(REPO / args.out))
    print(f"saved adapter -> {args.out}")

    if args.push_repo:
        token = os.environ.get("HF_TOKEN")
        model.push_to_hub(args.push_repo, token=token)
        tok.push_to_hub(args.push_repo, token=token)
        print(f"pushed -> https://huggingface.co/{args.push_repo}")


if __name__ == "__main__":
    main()
