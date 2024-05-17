import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import Dataset, load_dataset, DatasetDict


import sys

sys.path.append("..")
import warnings

warnings.filterwarnings("ignore")
tqdm.pandas()
from config import RANDOM_SEED


def prepare_dataset(tokenizer):
    # Load go emotion dataset
    dataset = load_dataset("go_emotions", "raw")
    dataset.set_format(type="pandas")
    df = dataset["train"][:]

    # Train Val Test Split (80-10-10)

    # Combining all features into a string for stratification
    emotions = df.columns[9:]

    df["labels"] = df[emotions].progress_apply(
        lambda row: "".join(str(int(value)) for value in row), axis=1
    )

    X = df[["text"]]
    y = df["labels"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Perform stratified train-test split
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # split validation set into test and validation
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=RANDOM_SEED
    )

    # define DFs for train, val and test
    train_df = df.iloc[X_train.index].copy()
    val_df = df.iloc[X_val.index].copy()
    test_df = df.iloc[X_test.index].copy()

    train_df["labels"] = train_df["labels"].apply(lambda x: [int(i) for i in x])
    val_df["labels"] = val_df["labels"].apply(lambda x: [int(i) for i in x])
    test_df["labels"] = test_df["labels"].apply(lambda x: [int(i) for i in x])

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # define tokenization function
    tokenize_function = lambda examples: tokenizer(
        examples["text"], padding="longest", truncation=True
    )

    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=512)
    val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=512)
    test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=512)

    train_dataset = train_dataset.select_columns(
        ["input_ids", "attention_mask", "labels"]
    )
    val_dataset = val_dataset.select_columns(["input_ids", "attention_mask", "labels"])
    test_dataset = test_dataset.select_columns(
        ["input_ids", "attention_mask", "labels"]
    )

    dataset_dict = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )
    id2label = {i: label for i, label in enumerate(emotions)}
    label2id = {label: i for i, label in enumerate(emotions)}
    return dataset_dict, id2label, label2id
