import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from collections import Counter
from itertools import combinations
import seaborn as sns
from tabulate import tabulate
import yaml
import os
import torch
import torch.nn.functional as F

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
        ground_truth = [1 if label in instance else 0 for instance in ground_truth_labels]

        matching = sum(1 for p, g in zip(predicted, ground_truth) if p == g == 1)
        unique_predicted = sum(1 for p, g in zip(predicted, ground_truth) if p == 1 and g == 0)
        unique_ground_truth = sum(1 for p, g in zip(predicted, ground_truth) if p == 0 and g == 1)

        tp = matching
        fp = unique_predicted
        fn = unique_ground_truth
        tn = sum(1 for p, g in zip(predicted, ground_truth) if p == g == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        accuracy = (tp + tn) / total_samples  # Calculate accuracy for each label

        loss = calculate_bce_loss(predicted, ground_truth)
        normalized_loss = normalize_bce_loss(loss.mean().item())

        scores[label] = {
            "matching": matching,
            "unique_predicted": unique_predicted,
            "unique_ground_truth": unique_ground_truth,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "bce_loss": normalized_loss,
            "accuracy": accuracy,
        }

        total_loss += normalized_loss

    average_bce = total_loss / len(all_labels)
    average_f1 = sum(scores[label]["f1_score"] for label in all_labels) / len(all_labels)
    average_precision = sum(scores[label]["precision"] for label in all_labels) / len(all_labels)
    average_recall = sum(scores[label]["recall"] for label in all_labels) / len(all_labels)
    average_accuracy = sum(scores[label]["accuracy"] for label in all_labels) / len(all_labels)
    
    total_matching = sum(scores[label]["matching"] for label in all_labels)
    total_unique_predicted = sum(scores[label]["unique_predicted"] for label in all_labels)
    total_unique_ground_truth = sum(scores[label]["unique_ground_truth"] for label in all_labels)

    scores["average"] = {
        "matching": total_matching,
        "unique_predicted": total_unique_predicted,
        "unique_ground_truth": total_unique_ground_truth,
        "precision": average_precision,
        "recall": average_recall,
        "f1": average_f1,
        "bce": average_bce,
        "accuracy": average_accuracy,
    }
    # make average the first element
    scores = {k: scores[k] for k in ["average"] + all_labels}
    return scores



def visualize_emotion_data(scores, save, path, percentage=False, run_numbers=None):
    """
    Visualizes emotion data metrics using heatmaps and bar charts.

    Parameters:
    scores (dict): A dictionary containing emotion metrics with the updated structure.
    save (bool): If True, saves the plots to the specified path. If False, displays the plots.
    path (str): The file path to save the plots if save is True.
    percentage (bool, optional): If True, displays metrics as percentages. Defaults to False.

    Returns:
    None
    """
    # Initialize dictionaries to store the results
    emotion_data = {}
    precision_data = {}
    recall_data = {}
    f1_data = {}
    bce_data = {}
    
    # Extract relevant information for each emotion
    for emotion, metrics in scores.items():
        if emotion != "average":  # Skip the average key
            unique_ground_truth = metrics.get("unique_ground_truth", 0)
            matching = metrics.get("matching", 0)
            unique_predicted = metrics.get("unique_predicted", 0)
            total = unique_ground_truth + matching + unique_predicted
            
            if percentage and total > 0:
                unique_ground_truth_pct = (unique_ground_truth / total) * 100
                matching_pct = (matching / total) * 100
                unique_predicted_pct = (unique_predicted / total) * 100
                emotion_data[emotion] = [unique_ground_truth_pct, matching_pct, unique_predicted_pct]
            else:
                emotion_data[emotion] = [unique_ground_truth, matching, unique_predicted]
            
            precision_data[emotion] = metrics.get("precision", 0)
            recall_data[emotion] = metrics.get("recall", 0)
            f1_data[emotion] = metrics.get("f1_score", 0)
            bce_data[emotion] = metrics.get("bce_loss", 0)
    
    # Convert the dictionaries to DataFrames
    df_prediction_matching = pd.DataFrame.from_dict(
        emotion_data, orient="index", columns=[f"Unique Run {run_numbers[0]}", "Matching", f"Unique Run {run_numbers[1]}"]
    )
    df_precision = pd.DataFrame.from_dict(precision_data, orient="index", columns=["Precision"])
    df_recall = pd.DataFrame.from_dict(recall_data, orient="index", columns=["Recall"])
    df_f1 = pd.DataFrame.from_dict(f1_data, orient="index", columns=["F1 Score"])
    df_bce = pd.DataFrame.from_dict(bce_data, orient="index", columns=["BCE Loss"])
    
    # Plot the Prediction matching heatmap
    plt.figure(figsize=(12, 8))
    if percentage:
        ax = sns.heatmap(
            df_prediction_matching,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",  # Format for percentages
            annot_kws={"fontsize": 10},
        )
        ax.set_title("Prediction Matching for Each Emotion (%)", fontsize=18)
    else:
        ax = sns.heatmap(
            df_prediction_matching,
            annot=True,
            cmap="YlGnBu",
            fmt="d",  # Format for counts
            annot_kws={"fontsize": 10},
        )
        ax.set_title("Prediction Matching for Each Emotion", fontsize=18)
    
    ax.set_xlabel("Prediction Category", fontsize=14)
    ax.set_ylabel("Emotion", fontsize=14)
    
    if save:
        plt.savefig(f"{path}_prediction_matching.png")
    else:
        plt.show()

    # Create a figure with subplots for the remaining plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))

    # Plot the Precision bar chart
    sns.barplot(
        x=df_precision.index,
        y="Precision",
        data=df_precision,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Precision for Each Emotion", fontsize=18)
    axes[0, 0].set_xlabel("Emotion", fontsize=14)
    axes[0, 0].set_ylabel("Precision", fontsize=14)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=90, fontsize=10)
    axes[0, 0].grid(axis="y", linestyle="--")

    # Plot the Recall bar chart
    sns.barplot(
        x=df_recall.index, y="Recall", data=df_recall, ax=axes[0, 1]
    )
    axes[0, 1].set_title("Recall for Each Emotion", fontsize=18)
    axes[0, 1].set_xlabel("Emotion", fontsize=14)
    axes[0, 1].set_ylabel("Recall", fontsize=14)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=90, fontsize=10)
    axes[0, 1].grid(axis="y", linestyle="--")

    # Plot the F1 Score bar chart
    sns.barplot(x=df_f1.index, y="F1 Score", data=df_f1, ax=axes[1, 0])
    axes[1, 0].set_title("F1 Score for Each Emotion", fontsize=18)
    axes[1, 0].set_xlabel("Emotion", fontsize=14)
    axes[1, 0].set_ylabel("F1 Score", fontsize=14)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=90, fontsize=10)
    axes[1, 0].grid(axis="y", linestyle="--")

    # Plot the BCE Loss bar chart
    sns.barplot(x=df_bce.index, y="BCE Loss", data=df_bce, ax=axes[1, 1])
    axes[1, 1].set_title("BCE Loss for Each Emotion", fontsize=18)
    axes[1, 1].set_xlabel("Emotion", fontsize=14)
    axes[1, 1].set_ylabel("BCE Loss", fontsize=14)
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=90, fontsize=10)
    axes[1, 1].grid(axis="y", linestyle="--")

    # Adjust the spacing between subplots
    plt.tight_layout()

    if save:
        plt.savefig(f"{path}_metrics.png")
    else:
        plt.show()

# Emotion Analysis Comparison Pipeline
def emotion_analysis_comparison_pipeline(run_number1, run_number2, set_, EMOTIONS, plot_percentage = False, save=False):
    def load_and_preprocess(run_number):
        file_path = f'run{run_number}/{set_}_filtered.csv'
        df = pd.read_csv(file_path)
        df.drop_duplicates(subset="text", inplace=True)
        return df.sort_values("text")

    def parse_predictions(predictions):
        if isinstance(predictions, str):
            try:
                return ast.literal_eval(predictions)
            except:
                return predictions
        return predictions

    run1 = load_and_preprocess(run_number1)
    run2 = load_and_preprocess(run_number2)

    # Keep only common texts
    common_texts = set(run1["text"]) & set(run2["text"])
    run1 = run1[run1["text"].isin(common_texts)]
    run2 = run2[run2["text"].isin(common_texts)]

    labels1 = run1["parsed_predictions"].apply(parse_predictions).tolist()
    labels2 = run2["parsed_predictions"].apply(parse_predictions).tolist()

    scores = calculate_metrics(labels2, labels1, EMOTIONS)

    print(f"Comparison between Run {run_number1} and Run {run_number2}:")
    print(f"Average BCE Loss: {scores['average']['bce']:.4f}")
    print(f"Average F1 Score: {scores['average']['f1']:.4f}")
    print(f"Average Precision: {scores['average']['precision']:.4f}")
    print(f"Average Recall: {scores['average']['recall']:.4f}")

    visualize_emotion_data(scores, save, f"plots/emotion_metrics_run{run_number1}_vs_run{run_number2}", plot_percentage, run_numbers=[run_number1, run_number2])

    def count_labels(labels):
        label_counts = {}
        for label_set in labels:
            for label in label_set:
                label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

    label_counts_run1 = count_labels(labels1)
    label_counts_run2 = count_labels(labels2)

    all_labels = sorted(set(list(label_counts_run1.keys()) + list(label_counts_run2.keys())))
    counts_run1 = [label_counts_run1.get(label, 0) for label in all_labels]
    counts_run2 = [label_counts_run2.get(label, 0) for label in all_labels]

    plt.figure(figsize=(12, 8), dpi=200)
    bar_width = 0.35
    index = range(len(all_labels))
    plt.bar(index, counts_run1, bar_width, label=f'Run {run_number1}', color='blue', alpha=0.7)
    plt.bar([i + bar_width for i in index], counts_run2, bar_width, label=f'Run {run_number2}', color='red', alpha=0.7)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(f'Label Distribution Comparison (Run {run_number1} vs Run {run_number2})')
    plt.xticks([i + bar_width/2 for i in index], all_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"plots/label_distribution_run{run_number1}_vs_run{run_number2}.png")
    plt.show()

    def calculate_shannon_entropy(label_counts):
        total = sum(label_counts.values())
        probabilities = [count / total for count in label_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        max_entropy = np.log2(len(label_counts))  # Maximum possible entropy
        return entropy, max_entropy

    def calculate_gini_coefficient(label_counts):
        total = sum(label_counts.values())
        proportions = sorted([count / total for count in label_counts.values()])
        cumulative_proportions = np.cumsum(proportions)
        n = len(proportions)
        gini = (n + 1 - 2 * np.sum(cumulative_proportions)) / n
        return gini

    entropy_run1, max_entropy = calculate_shannon_entropy(label_counts_run1)
    entropy_run2, max_entropy = calculate_shannon_entropy(label_counts_run2)
    gini_run1 = calculate_gini_coefficient(label_counts_run1)
    gini_run2 = calculate_gini_coefficient(label_counts_run2)

    print(f"Shannon Entropy - Run {run_number1}: {entropy_run1:.4f}")
    print(f"Shannon Entropy - Run {run_number2}: {entropy_run2:.4f}")
    print(f"Gini Coefficient - Run {run_number1}: {gini_run1:.4f}")
    print(f"Gini Coefficient - Run {run_number2}: {gini_run2:.4f}")

    run1_values = [entropy_run1, gini_run1]
    run2_values = [entropy_run2, gini_run2]

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Shannon Entropy subplot
    ax1.bar([f'Run {run_number1}', f'Run {run_number2}'], [run1_values[0], run2_values[0]], color=['blue', 'red'])
    ax1.set_ylabel('Shannon Entropy')
    ax1.set_title('Shannon Entropy Comparison')
    ax1.set_ylim(0, max_entropy)
    
    # Add value labels on top of bars
    for i, v in enumerate([run1_values[0], run2_values[0]]):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Gini Coefficient subplot
    ax2.bar([f'Run {run_number1}', f'Run {run_number2}'], [run1_values[1], run2_values[1]], color=['blue', 'red'])
    ax2.set_ylabel('Gini Coefficient')
    ax2.set_title('Gini Coefficient Comparison')
    ax2.set_ylim(0, 1)  # Gini coefficient is always between 0 and 1
    
    # Add value labels on top of bars
    for i, v in enumerate([run1_values[1], run2_values[1]]):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.suptitle(f'Diversity Measures Comparison (Run {run_number1} vs Run {run_number2})')
    plt.tight_layout()
    
    if save:
        plt.savefig(f"plots/diversity_measures_run{run_number1}_vs_run{run_number2}.png")
    
    plt.show()

    average_labels_run1 = sum(label_counts_run1.values()) / len(labels1)
    average_labels_run2 = sum(label_counts_run2.values()) / len(labels2)

    print(f"Average number of labels per comment - Run {run_number1}: {average_labels_run1:.4f}")
    print(f"Average number of labels per comment - Run {run_number2}: {average_labels_run2:.4f}")

def emotion_pair_analysis_pipeline(run_number, set_, config_path, number=10, table=False, save_visualizations=False):
    def load_config(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config["EMOTIONS"], config["RANDOM_SEED"]

    def safe_eval(s):
        try:
            s = s.strip().strip("'\"")
            match = re.match(r'\[(.*)\]', s)
            if match:
                content = match.group(1)
                return [item.strip().strip("'\"") for item in content.split(',') if item.strip()]
            else:
                return []
        except:
            print(f"Warning: Could not parse '{s}'. Returning empty list.")
            return []

    def process_csv(file_path):
        data = []
        empty_predictions = 0
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                predictions = safe_eval(row['parsed_predictions'])
                if not predictions:
                    empty_predictions += 1
                data.append(predictions)
        print(f"Number of empty predictions: {empty_predictions}")
        return data

    def get_top_pairs(data, n=15):
        pair_counter = Counter(tuple(sorted(pred)) for pred in data if pred)
        return pair_counter.most_common(n)

    def get_all_emotion_pairs(data, emotions, n=15):
        all_pairs = list(combinations(emotions, 2))
        pair_counter = Counter()
        
        for pred in data:
            pred_set = set(pred)
            for pair in all_pairs:
                if set(pair).issubset(pred_set):
                    pair_counter[pair] += 1
        
        return pair_counter.most_common(n)

    def print_table(data, title):
        df = pd.DataFrame(data, columns=['Emotions', 'Count'])
        df['Emotions'] = df['Emotions'].apply(lambda x: ', '.join(x))
        print(f"\n{title}")
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

    def plot_top_pairs(data, title, run_number, save=False, set_="train"):
        emotions, counts = zip(*data)
        emotions = [', '.join(e) for e in emotions]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(range(len(emotions)), counts)
        plt.title(title)
        plt.xlabel('Emotion Pairs')
        plt.ylabel('Frequency')
        plt.xticks(range(len(emotions)), emotions, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'run{run_number}/plots/{title}.png')
        plt.show()

    def plot_heatmap(data, emotions, run_number, save=False, set_="train"):
        df = pd.DataFrame(0, index=emotions, columns=emotions)
        
        for (emotion1, emotion2), count in data:
            df.at[emotion1, emotion2] = count
            df.at[emotion2, emotion1] = count
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='d')
        plt.title('Emotion Co-occurrence Heatmap')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'run{run_number}/plots/{set_}_heatmap_run_{run_number}.png')
        plt.show()

    def visualize_data(df, emotions, save=False):
        fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

        # Bar chart of label distribution
        label_counts = {}
        for labels in df["parsed_predictions"]:
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

        filtered_dict = {k: v for k, v in label_counts.items() if k in emotions}
        sorted_filtered_dict = {k: v for k, v in sorted(filtered_dict.items(), key=lambda item: item[1])}

        axs[0].barh(list(sorted_filtered_dict.keys()), list(sorted_filtered_dict.values()))
        axs[0].set_title('Label Distribution')

        # Histogram of number of labels per comment
        count = df["parsed_predictions"].apply(lambda x: len(x))
        axs[1].hist(count, bins=10)
        # Plot mean and median
        axs[1].axvline(count.mean(), color='k', linestyle='dashed', linewidth=1, label=f'Mean: {count.mean():.2f}')
        axs[1].legend()
        axs[1].set_title('Number of Labels per Comment')

        plt.tight_layout()
        plt.show()

        print(f"Total number of labels counted: {sum(sorted_filtered_dict.values())}")
        print(f"Total number of labels in the dataset: {len(df)}")

        if save:
            fig.savefig(f"run{run_number}/{set_}_visualizations.png")

    # Load configuration
    emotions, _ = load_config(config_path)

    # Process CSV
    file_path = f'run{run_number}/{set_}_filtered.csv'
    data = process_csv(file_path)

    # Prepare DataFrame for visualization
    df = pd.DataFrame({"parsed_predictions": data})

    # Visualize Data
    visualize_data(df, emotions, save=save_visualizations)

    # Get emotion pairs
    top_pairs = get_top_pairs(data, n=number)
    all_emotion_pairs = get_all_emotion_pairs(data, emotions)

    # Plot and save
    plot_top_pairs(top_pairs, f"Top {number} Common Emotion Pairs", run_number, save=True, set_=set_)
    plot_top_pairs(all_emotion_pairs, f"Top {number} Emotion Pairs (Comprehensive)", run_number, save=True, set_=set_)
    plot_heatmap(get_all_emotion_pairs(data, emotions, n=len(emotions)**2), emotions, run_number, save=True, set_=set_)

    # Print tables if requested
    if table:
        print_table(top_pairs, f"Top {number} Common Pairs")
        print_table(all_emotion_pairs, f"Top {number} Pairs of Emotions (Comprehensive)")

