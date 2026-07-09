"""Build the versioned Hugging Face dataset and push it to the Hub.  [STUB]

The dataset is the real deliverable. Publish train/dev/test as splits, keep source_band as
a column, and write a dataset card documenting the bands + the filtering gate.

TODO:
  from datasets import Dataset, DatasetDict
  dd = DatasetDict({split: Dataset.from_list(load(split)) for split in ("train","dev","test")})
  dd.push_to_hub(os.environ["HF_DATASET_REPO"], private=True)
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Load the three splits, build a DatasetDict, push_to_hub.")


if __name__ == "__main__":
    main()
