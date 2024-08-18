import os
import sys

import chromadb
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

tqdm.pandas()
sys.path.append("../..")

import yaml
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)
definitions, EMOTIONS, RANDOM_SEED = (
    config["definitions"],
    config["EMOTIONS"],
    config["RANDOM_SEED"],
)

id2label = {str(id): label for id, label in enumerate(EMOTIONS)}
label2id = {v: k for k, v in id2label.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    sample = df.loc[idx]
    text, labels = sample["text"], sample["labels_text"]
    # get embeddings
    embeddings = get_embeddings(df, collection, idx)
    results = collection.query(embeddings, n_results=5)
    # get ids and distances
    ids = [int(_id) for _id in results["ids"][0]]
    distances = [float(d) for d in results["distances"][0]]
    matches = []
    for idx in ids:
        if idx in df.index:
            sample = df.loc[idx]
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


# LEGACY CODE
def get_semantic_statstical_distance(emotions1, emotions2, verbose=False):
    """
    Get the semantic statistical distance between two sets of emotions.

    Args:
        emotions1 (list): The first set of emotions.
        emotions2 (list): The second set of emotions.
        verbose (bool): Whether to print the distances.

    Returns:
        float: The semantic statistical distance between the two sets of emotions.
    """
    # Load emotion distances from CSV file
    distances = pd.read_csv("combined_emotion_distances.csv")

    # make emotions1 the largest of the two
    if len(emotions1) < len(emotions2):
        emotions1, emotions2 = emotions2, emotions1

    accumulation = 0

    # set for unique pairs
    traversed = set()
    for emotion1 in emotions1:
        for emotion2 in emotions2:
            # if the two emotions have been found before, skip this step
            if (emotion1, emotion2) in traversed or (emotion2, emotion1) in traversed:
                continue
            else:
                traversed.add((emotion1, emotion2))
                traversed.add((emotion2, emotion1))

            # Get distance from emotion_distances dictionary
            distance = 100000
            for index, row in distances.iterrows():
                if (row["Emotion_1"] == emotion1 and row["Emotion_2"] == emotion2) or (
                    row["Emotion_1"] == emotion2 and row["Emotion_2"] == emotion1
                ):
                    distance = row["Distance"]
                    break
            if distance == 100000:
                print(
                    f"Distance between {emotion1} and {emotion2} not found in the CSV file"
                )

            if verbose:
                print(f"{emotion1} -> {emotion2}: {distance}")
            accumulation += distance

    return accumulation


# Create Embeddings Database code
def init_model():
    # TODO replace with Ollama or fireworks embeddings
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )
    model.eval()
    model = model.to(device)
    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_text_embeddings(text, model, tokenizer, norm=False):
    """
    Gets the text embeddings for the given text.

    Args:
        text (str): The text to get embeddings for.

    Returns:
        list: The text embeddings as a list.
    """
    encoded_input = tokenizer(
        [text], padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    if norm:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().squeeze(0).numpy().tolist()


def init_collection():
    chroma_client = chromadb.PersistentClient("db/")
    collection = chroma_client.get_or_create_collection(
        name="emotions_cosine", metadata={"hnsw:space": "cosine"}
    )
    return collection


def insert_emotion_embeddings(model, tokenizer, definitions, label2id, collection):
    """
    Inserts the emotion embeddings into the collection.
    """
    for emotion in EMOTIONS:
        embedding = get_text_embeddings(definitions[emotion], model, tokenizer)
        collection.upsert(
            ids=label2id[emotion],
            embeddings=embedding,
        )