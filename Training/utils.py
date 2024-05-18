from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import numpy as np
from transformers import EvalPrediction, Trainer
from config import NUM_LABELS

def compute_metrics(p: EvalPrediction, NUM_LABELS=28, threshold=0.5):
    """
    Compute metrics for multi-label classification

    Args:

    p: EvalPrediction object
    NUM_LABELS: number of labels
    threshold: threshold for classification

    Returns:
    metrics: dictionary of metrics
    """
    label_ids = p.label_ids[:, :NUM_LABELS] # get rid of padding, should be fixed in the future
    preds = p.predictions

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    # convert all to numpy
    probs = probs.cpu().detach().numpy()
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_pred[np.where(probs < threshold)] = 0 
    y_true = label_ids
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    metrics = {
        "f1": f1_macro_average,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    return metrics

class MultiLabelTrainer(Trainer):
    """
    Trainer for multi-label classification

    Args:
    Trainer: Trainer class from transformers

    Returns:
    MultiLabelTrainer: Trainer class for multi-label classification
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels = labels[:, :NUM_LABELS].float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
    

