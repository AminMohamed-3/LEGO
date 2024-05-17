import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_checkpoint = "distilbert/distilroberta-base"

num_labels = 28
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
