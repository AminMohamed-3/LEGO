import ast
import re
import torch
import torch.nn.functional as F
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

def parse_llama_output(output):
    parsed_output = []
    for instance in output:
        # Convert string representation of list to actual list
        if type(instance) == str:
            labels_list = ast.literal_eval(instance)
        else:
            labels_list = instance
        parsed_output.append(labels_list)
    return parsed_output

def filter_invalid_labels(predicted_labels, valid_labels):
    return [label for label in predicted_labels if label in valid_labels]


def calculate_bce_loss(predictions, ground_truth):
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)
    loss = F.binary_cross_entropy(
        predictions_tensor, ground_truth_tensor, reduction="none"
    )
    return loss


def normalize_bce_loss(loss, min_loss=0, max_loss=1):
    return (loss - min_loss) / (max_loss - min_loss)


def calculate_metrics(predicted_labels, ground_truth_labels, all_labels):
    scores = {}
    total_loss = 0
    total_samples = len(predicted_labels)

    for label in all_labels:
        predicted = [1 if label in instance else 0 for instance in predicted_labels]
        ground_truth = [
            1 if label in instance else 0 for instance in ground_truth_labels
        ]

        tp = sum(1 for p, g in zip(predicted, ground_truth) if p == g == 1)
        fp = sum(1 for p, g in zip(predicted, ground_truth) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(predicted, ground_truth) if p == 0 and g == 1)
        tn = sum(1 for p, g in zip(predicted, ground_truth) if p == g == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        accuracy = (tp) / total_samples  # Calculate accuracy for each label

        loss = calculate_bce_loss(predicted, ground_truth)
        normalized_loss = normalize_bce_loss(loss.mean().item())

        scores[label] = {
            "correct": tp,
            "incorrect": fp + fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "bce_loss": normalized_loss,
            "accuracy": accuracy,
        }

        total_loss += normalized_loss

    average_bce = total_loss / total_samples
    average_f1 = sum(scores[label]["f1_score"] for label in all_labels) / len(
        all_labels
    )
    average_precision = sum(scores[label]["precision"] for label in all_labels) / len(
        all_labels
    )
    average_recall = sum(scores[label]["recall"] for label in all_labels) / len(
        all_labels
    )
    average_accuracy = sum(scores[label]["accuracy"] for label in all_labels) / len(
        all_labels
    )  # Calculate average accuracy
    scores["average"] = {
        "precision": average_precision,
        "recall": average_recall,
        "f1": average_f1,
        "bce": average_bce,
        "accuracy": average_accuracy,
    }
    # make average the first element
    scores = {k: scores[k] for k in ["average"] + all_labels}
    return scores


def append_to_json_file(data, filename="results.json"):
    try:
        with open(filename, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.append(data)

    with open(filename, "w") as f:
        json.dump(existing_data, f)

    return "results saved in results.json"


import pandas as pd
from semantic_statstical_distance import get_semantic_statstical_distance

import pandas as pd


def process_save_results(
    model_name,
    system_message,
    prompt_template,
    model_output,
    ground_truth,
    valid_labels,
    text,
    verbose=True,
    trial_num=0,
    distances_save_name="distance.csv",
    **kwargs,
):

    filtered_predictions = [
        filter_invalid_labels(instance, valid_labels) for instance in model_output
    ]
    ground_truth = parse_llama_output(ground_truth)

    # Calculate metrics
    scores = calculate_metrics(filtered_predictions, ground_truth, valid_labels)

    # Print results
    print(f"Results for {model_name}:")
    print(f"Average BCE Loss: {scores['average']['bce']:.4f}")
    print(f"Average F1 Score: {scores['average']['f1']:.4f}")
    print(f"Average Precision: {scores['average']['precision']:.4f}")
    print(f"Average Recall: {scores['average']['recall']:.4f}")

    # Calculate distances

    distances = [
        get_semantic_statstical_distance(prediction, gt)
        for prediction, gt in zip(
            filtered_predictions, ground_truth
        )
    ]

    avg_distance = sum(distances) / len(distances)
    total_distance = sum(distances)
    print(f"Average Distance: {avg_distance:.4f}")
    print(f"Total Distance: {total_distance:.4f}")

    # Save results to CSV
    df = pd.DataFrame(
        {
            "text": text,
            "predictions": filtered_predictions,
            "ground_truth": ground_truth,
            "distance": distances,
        }
    )
    df.to_csv(distances_save_name, index=False)

    append_to_json_file(
        {
            "model_name": model_name,
            "system_message": system_message,
            "prompt_template": prompt_template,
            **kwargs,
            "scores": scores,
            "distances": {
                "average": avg_distance,
                "total": total_distance,
            },
        }
    )

    # Optionally visualize data
    if verbose:
        visualize_emotion_data("results.json", trial_num=trial_num)
        plot_histogram(distances)  # Call the plotting function here

# Example plotting function
import matplotlib.pyplot as plt


def plot_histogram(distances):
    plt.hist(distances, bins=20, edgecolor="black")
    plt.title("Distribution of Semantic Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()


def visualize_emotion_data(json_path, trial_num=0):
    # Load the JSON data from the file
    with open(json_path, "r") as file:
        data = json.load(file)

    # Initialize an empty dictionary to store the results
    emotion_data = {}
    precision_data = {}
    recall_data = {}
    f1_data = {}
    bce_data = {}

    # Extract the relevant information for each emotion
    scores = data[trial_num]["scores"]  # Access the scores part of the JSON
    for emotion, metrics in scores.items():
        if emotion != "average":  # Skip the average key
            correct = metrics.get("correct", 0)
            incorrect = metrics.get("incorrect", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1_score = metrics.get("f1_score", 0)
            bce_loss = metrics.get("bce_loss", 0)
            emotion_data[emotion] = [correct, incorrect]
            precision_data[emotion] = precision
            recall_data[emotion] = recall
            f1_data[emotion] = f1_score
            bce_data[emotion] = bce_loss

    # Convert the dictionaries to DataFrames
    df_correct_incorrect = pd.DataFrame.from_dict(
        emotion_data, orient="index", columns=["Correct", "Incorrect"]
    )
    df_precision = pd.DataFrame.from_dict(
        precision_data, orient="index", columns=["Precision"]
    )
    df_recall = pd.DataFrame.from_dict(recall_data, orient="index", columns=["Recall"])
    df_f1 = pd.DataFrame.from_dict(f1_data, orient="index", columns=["F1 Score"])
    df_bce = pd.DataFrame.from_dict(bce_data, orient="index", columns=["BCE Loss"])

    # Plot the Correct vs Incorrect Predictions heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        df_correct_incorrect,
        annot=True,
        cmap="YlGnBu",
        fmt="d",
        annot_kws={"fontsize": 12},
    )
    ax.set_title("Correct vs Incorrect Predictions for Each Emotion", fontsize=18)
    ax.set_xlabel("Prediction", fontsize=14)
    ax.set_ylabel("Emotion", fontsize=14)
    plt.show()

    # Create a figure with subplots for the remaining plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))

    # Plot the Precision bar chart
    sns.barplot(
        x=df_precision.index,
        y="Precision",
        data=df_precision,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Precision for Each Emotion", fontsize=18)
    axes[0, 0].set_xlabel("Emotion", fontsize=14)
    axes[0, 0].set_ylabel("Precision", fontsize=14)
    axes[0, 0].set_xticks(range(len(df_precision.index)))
    axes[0, 0].set_xticklabels(df_precision.index, rotation=90, fontsize=12)
    axes[0, 0].grid(axis="y", linestyle="--")

    # Plot the Recall bar chart
    sns.barplot(
        x=df_recall.index, y="Recall", data=df_recall, legend=False, ax=axes[0, 1]
    )
    axes[0, 1].set_title("Recall for Each Emotion", fontsize=18)
    axes[0, 1].set_xlabel("Emotion", fontsize=14)
    axes[0, 1].set_ylabel("Recall", fontsize=14)
    axes[0, 1].set_xticks(range(len(df_recall.index)))
    axes[0, 1].set_xticklabels(df_recall.index, rotation=90, fontsize=12)
    axes[0, 1].grid(axis="y", linestyle="--")

    # Plot the F1 Score bar chart
    sns.barplot(x=df_f1.index, y="F1 Score", data=df_f1, legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("F1 Score for Each Emotion", fontsize=18)
    axes[1, 0].set_xlabel("Emotion", fontsize=14)
    axes[1, 0].set_ylabel("F1 Score", fontsize=14)
    axes[1, 0].set_xticks(range(len(df_f1.index)))
    axes[1, 0].set_xticklabels(df_f1.index, rotation=90, fontsize=12)
    axes[1, 0].grid(axis="y", linestyle="--")

    # Plot the BCE Loss bar chart
    sns.barplot(x=df_bce.index, y="BCE Loss", data=df_bce, legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("BCE Loss for Each Emotion", fontsize=18)
    axes[1, 1].set_xlabel("Emotion", fontsize=14)
    axes[1, 1].set_ylabel("BCE Loss", fontsize=14)
    axes[1, 1].set_xticks(range(len(df_bce.index)))
    axes[1, 1].set_xticklabels(df_bce.index, rotation=90, fontsize=12)
    axes[1, 1].grid(axis="y", linestyle="--")

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    # Display the figure
    plt.show()
