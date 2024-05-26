from tqdm import tqdm
import os
from dotenv import load_dotenv
from datasets import load_dataset
import chromadb
import requests
import json

import sys

load_dotenv()
# sys.path.append("..")
# from config import *


dataset = load_dataset("go_emotions", "simplified")
dataset.set_format(type="pandas")
df_train = dataset["train"][:]
df_train = df_train.drop(columns=["id"])

chroma_client = chromadb.PersistentClient("db/")
collection = chroma_client.get_or_create_collection(
    name="goemotions_train", metadata={"hnsw:space": "cosine"}
)


def get_embeddings(prompt):
    url = "http://localhost:11434/api/embeddings"
    data = {
        "model": "mxbai-embed-large",
        "prompt": prompt,
    }

    response = requests.post(url, data=json.dumps(data))
    return response.json()["embedding"]


for index, sample in tqdm(df_train.iterrows(), total=len(df_train)):
    embeddings = get_embeddings(sample["text"])
    collection.add(
        ids=str(index),
        embeddings=embeddings,
    )
