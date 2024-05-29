import chromadb
from datasets import load_dataset
from tqdm import tqdm
import sys
import os

tqdm.pandas()
sys.path.append("../..")

import config
from config import RANDOM_SEED


def get_dataset_collection(vector_db_path):
    """
    Returns a collection from the ChromaDB client.

    Args:
        vector_db_path (str): The path to the vector database.

    Returns:
        chromadb.Collection: The collection object.
    """
    chroma_client = chromadb.PersistentClient(vector_db_path)
    collection = chroma_client.get_or_create_collection(
        name="goemotions_train", metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_dataframe(split="train"):
    """
    Loads the 'go_emotions' dataset, converts it to a pandas DataFrame, and adds a 'labels_text' column.

    Args:
        split (str, optional): The split of the dataset to load. Defaults to "train".

    Returns:
        pandas.DataFrame: The loaded dataset with an additional 'labels_text' column.
    """
    dataset = load_dataset("go_emotions", "simplified")
    dataset.set_format(type="pandas")
    df_train = dataset[split][:]
    int2label = lambda x: dataset[split].features["labels"].feature.int2str(x)
    df_train["labels_text"] = df_train["labels"].progress_apply(int2label)
    return df_train


def get_embeddings(df, collection, idx=0):
    """
    Retrieves the embeddings for a given index from the collection.

    Args:
        df (pandas.DataFrame): The dataset.
        collection (chromadb.Collection): The collection object.
        idx (int, optional): The index of the sample. Defaults to 0.

    Returns:
        list: The embeddings for the sample.
    """
    return collection.get(ids=[str(idx)], include=["embeddings"])["embeddings"]


def get_similars(df, collection, idx=0, n_results=5, verbose=False):
    """
    Retrieves similar samples from the collection for a given sample.

    Args:
        df (pandas.DataFrame): The dataset.
        collection (chromadb.Collection): The collection object.
        idx (int, optional): The index of the sample. Defaults to 0.
        n_results (int, optional): The number of similar samples to retrieve. Defaults to 5.
        verbose (bool, optional): Whether to print the results. Defaults to False.

    Returns:
        list: A list of dictionaries, each containing the index, text, labels, and distance for a similar sample.
    """
    # get example
    sample = df.iloc[idx]
    text, labels = sample["text"], sample["labels_text"]
    # get embeddings
    embeddings = get_embeddings(df, collection, idx)
    results = collection.query(embeddings, n_results=5)
    # get ids and distances
    ids = [int(_id) for _id in results["ids"][0]]
    distances = [float(d) for d in results["distances"][0]]
    matches = []
    for idx in ids:
        sample = df.iloc[idx]
        text = sample["text"]
        labels = sample["labels_text"]
        distance = distances[ids.index(idx)]
        matches.append(
            {
                "idx": idx,
                "text": text,
                "labels": labels,
                "distance": distance,
            }
        )
        if verbose:
            print(f"text: {text}\nlabels: {labels}\n distance: {distance:0.2f}\n")
    return matches
