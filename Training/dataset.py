import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from datasets import load_from_disk
from datasets import Dataset

import yaml
import sys
import pandas as pd

sys.path.append("..")
import warnings

warnings.filterwarnings("ignore")
tqdm.pandas()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    NUM_LABELS = config["NUM_LABELS"]
    EMOTIONS = config["EMOTIONS"]


def prepare_simplified_dataset(tokenizer):
    dataset = load_dataset("go_emotions", "simplified")
    i2s = dataset["train"].features["labels"].feature.int2str
    tokenize_function = lambda examples: tokenizer(
    examples["text"], padding="max_length", truncation=True, return_tensors="pt"
    )
    labels_to_one_hot = lambda examples: {
        "labels": np.sum(np.eye(NUM_LABELS, dtype=np.float16)[examples["labels"]], axis=0)
    }
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(
        function=labels_to_one_hot,
        batched=False,
    )

    # label & id mapping
    id2label = {k: i2s(k) for k in range(NUM_LABELS)}
    label2id = {v: k for k, v in id2label.items()}

    return dataset, id2label, label2id

def prepare_local_dataset(tokenizer):
    # store at huggingface
    df = pd.read_csv('cleaning/run1/filtered_run1.csv')
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk("cleaning/run1/filtered_run1")
    dataset = load_from_disk("cleaning/run1/filtered_run1")

    # drop any empty parsed predictions
    dataset = dataset.filter(lambda x: x["parsed_predictions"] != "")

    tokenize_function = lambda examples: tokenizer(
        examples["text"], padding="max_length", truncation=True, return_tensors="pt"
    )
    labels_to_one_hot = lambda examples: {
        "labels": np.sum(np.eye(NUM_LABELS, dtype=np.float16)[examples["labels"]], axis=0)
    }
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(
        function=labels_to_one_hot,
        batched=False,
    )

    return dataset
