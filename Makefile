# One-command pipeline stages. Run `make <target>`.
# These call the scripts under src/. Fill in the stubs before the later stages work.

.PHONY: help test data split train eval demo

help:
	@echo "test   - run unit tests (register checker)"
	@echo "data   - generate + filter training data from the teacher"
	@echo "split  - split generated data into train/dev/test (leak-safe, by sentence stem)"
	@echo "train  - QLoRA SFT on the base model"
	@echo "eval   - score base vs tuned vs LLM on the golden set -> results table"
	@echo "demo   - launch the local Gradio inference demo"

test:
	PYTHONPATH=src python -m pytest -q

data:
	PYTHONPATH=src python -m data.generate && PYTHONPATH=src python -m data.filter

split:
	PYTHONPATH=src python -m data.split --in data/generated/filtered.jsonl --out-dir data

train:
	PYTHONPATH=src python -m train.sft

eval:
	PYTHONPATH=src python -m eval.run_eval

demo:
	python app/app.py
