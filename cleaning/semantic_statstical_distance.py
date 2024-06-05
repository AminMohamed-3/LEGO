import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from emotion_tree import get_distance
import chromadb
import sys
import yaml
import os

config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
# Load the configuration file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
definitions, EMOTIONS, RANDOM_SEED = (
    config["definitions"],
    config["EMOTIONS"],
    config["RANDOM_SEED"],
)

id2label = {str(id): label for id, label in enumerate(EMOTIONS)}
label2id = {v: k for k, v in id2label.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    distances = pd.read_csv("cleaning/combined_emotion_distances.csv")

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


def init_model():
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
    chroma_client = chromadb.PersistentClient("cleaning/db/")
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


def calculate_combined_distances(emotions, collection, label2id, beta=0.25):
    """
    Calculate the combined distances between emotions using both the emotion tree and the embeddings.

    Args:
        emotions (list): The list of emotions.
        collection (chromadb.Collection): The collection of embeddings.
        label2id (dict): The label to id mapping.
        beta (float): The weight for the embeddings.

    Returns:
        np.ndarray: The combined distances.
    """
    distances_embed = np.zeros((len(emotions), len(emotions)))
    counter = 0
    for emotion in emotions:
        embedding = collection.get(ids=label2id[emotion], include=["embeddings"])[
            "embeddings"
        ][0]
        result = collection.query(
            embedding, n_results=28, include=["distances", "documents"]
        )
        ids = result["ids"][0]
        indexes = [(pos, int(idx)) for pos, idx in enumerate(ids)]
        distances_inorder = []
        for i in range(len(indexes)):
            pos_in_order = list(filter(lambda x: x[1] == i, indexes))[0][0]
            distances_inorder.append(result["distances"][0][pos_in_order])
        distances_embed[counter] = np.round(np.array([distances_inorder]), 2)
        counter += 1

    average = (distances_embed.min() + distances_embed.max()) / 2
    np.fill_diagonal(distances_embed, average)
    distances_embed = (distances_embed - distances_embed.min()) / (
        distances_embed.max() - distances_embed.min()
    )

    distances_tree = np.zeros((len(emotions), len(emotions)))
    for i, emotion1 in enumerate(emotions):
        for j, emotion2 in enumerate(emotions):
            distances_tree[i, j] = get_distance([emotion1], [emotion2])
    # normalize
    combined = (distances_tree * beta) + (distances_embed * (1 - beta))
    combined = (combined - combined.min()) / (combined.max() - combined.min())

    combined = combined * 1.5 - 0.5  # scale values to be between -0.5 and 1

    np.fill_diagonal(combined, -1)  # set diagonal to -1

    return combined


def save_emotion_distances(emotions, combined):
    # make a dictionary of each emotions pair and their distance
    emotion_pairs = {}
    for i, emotion1 in enumerate(emotions):
        for j, emotion2 in enumerate(emotions):
            emotion_pairs[(emotion1, emotion2)] = combined[i, j]

    # make it into a pd dataframe with column for emotion 1 and emotion 2 and distance
    df = pd.DataFrame(emotion_pairs.items(), columns=["Emotion Pair", "Distance"])
    df["Emotion_1"] = df["Emotion Pair"].apply(lambda x: x[0])
    df["Emotion_2"] = df["Emotion Pair"].apply(lambda x: x[1])
    df = df.drop(columns=["Emotion Pair"])
    df = df[["Emotion_1", "Emotion_2", "Distance"]]
    df = df.sort_values(by="Distance")
    df = df.reset_index(drop=True)

    # save
    df.to_csv("cleaning/combined_emotion_distances.csv", index=False)


def main():
    tokenizer, model = init_model()
    collection = init_collection()
    insert_emotion_embeddings(model, tokenizer, definitions, label2id, collection)
    combined = calculate_combined_distances(EMOTIONS, collection, label2id)
    save_emotion_distances(EMOTIONS, combined)


if __name__ == "__main__":
    main()
