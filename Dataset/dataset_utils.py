import ast
import re
import torch
import torch.nn.functional as F
import pandas as pd

def extract_emotions(text):
    """
    Extracts a list of emotions from the given text enclosed in <answer> tags.

    Parameters:
    text (str): The input text containing the emotions.

    Returns:
    list: A list of emotions found between <answer> and </answer> tags.
    """
    # Use regular expression to find the emotions between <answer> and </answer>
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text)

    if match:
        # Extract the list of emotions and split by comma
        emotions = match.group(1).strip().split(",")
        # Clean each emotion by removing unwanted characters and stripping whitespace
        emotions = [re.sub(r'[()]', '', emotion).strip() for emotion in emotions]
        return emotions
    else:
        return []


def parse_llama_output(output):
    parsed_output = []
    for instance in output:
        # Convert string representation of list to actual list
        labels_list = ast.literal_eval(instance)
        # Clean each label by removing unwanted characters and stripping whitespace
        cleaned_labels = [re.sub(r'[()]', '', label).strip() for label in labels_list]
        parsed_output.append(cleaned_labels)
    return parsed_output

def filter_invalid_labels(predicted_labels, valid_labels):
    return [label for label in predicted_labels if label in valid_labels]

def calculate_bce_loss(predictions, ground_truth):
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)
    loss = F.binary_cross_entropy(predictions_tensor, ground_truth_tensor, reduction='none')
    return loss

def normalize_bce_loss(loss, min_loss=0, max_loss=1):
    return (loss - min_loss) / (max_loss - min_loss)

def calculate_metrics(predicted_labels, ground_truth_labels, all_labels):
    scores = {}
    total_loss = 0
    total_samples = len(predicted_labels)
    
    for label in all_labels:
        predicted = [1 if label in instance else 0 for instance in predicted_labels]
        ground_truth = [1 if label in instance else 0 for instance in ground_truth_labels]
        
        tp = sum(1 for p, g in zip(predicted, ground_truth) if p == g == 1)
        fp = sum(1 for p, g in zip(predicted, ground_truth) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(predicted, ground_truth) if p == 0 and g == 1)
        tn = sum(1 for p, g in zip(predicted, ground_truth) if p == g == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        loss = calculate_bce_loss(predicted, ground_truth)
        normalized_loss = normalize_bce_loss(loss.mean().item())
        
        scores[label] = {
            'correct': tp,
            'incorrect': fp + fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'bce_loss': normalized_loss
        }
        
        total_loss += normalized_loss
    
    average_bce = total_loss / total_samples
    return scores, average_bce

def process_and_score(model_output, ground_truth, valid_labels):
    parsed_output = parse_llama_output(model_output)
    cleaned_filtered_output = [filter_invalid_labels(instance, valid_labels) for instance in parsed_output]
    scores, average_bce = calculate_metrics(cleaned_filtered_output, ground_truth, valid_labels)
    return scores, average_bce

def save_results_to_csv(scores, filepath):
    df = pd.DataFrame(scores).T  # Transpose to have labels as rows
    df.to_csv(filepath, index_label='label')