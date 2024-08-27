import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset_utils import visualize_emotion_data, calculate_metrics
import csv
import re
from collections import Counter
from itertools import combinations
import seaborn as sns
from tabulate import tabulate
import yaml
import os

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

    visualize_emotion_data(scores, save, f"plots/emotion_metrics_run{run_number1}_vs_run{run_number2}", plot_percentage, (run_number1, run_number2))

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

# Emotion Pair Analysis Pipeline
def emotion_pair_analysis_pipeline(run_number, set_, config_path, number=10, print_table=False):
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

    # Load configuration
    emotions, _ = load_config(config_path)
    
    # Process CSV
    file_path = f'run{run_number}/{set_}_filtered.csv'
    data = process_csv(file_path)
    
    # Get emotion pairs
    top_pairs = get_top_pairs(data, n=number)
    all_emotion_pairs = get_all_emotion_pairs(data, emotions)
    
    # Plot and save
    plot_top_pairs(top_pairs, f"Top {number} Common Emotion Pairs", run_number, save=True, set_=set_)
    plot_top_pairs(all_emotion_pairs, f"Top {number} Emotion Pairs (Comprehensive)", run_number, save=True, set_=set_)
    plot_heatmap(get_all_emotion_pairs(data, emotions, n=len(emotions)**2), emotions, run_number, save=True, set_=set_)
    
    # Print tables if requested
    if print_table:
        print_table(top_pairs, f"Top {number} Common Pairs")
        print_table(all_emotion_pairs, f"Top {number} Pairs of Emotions (Comprehensive)")

# Main function to run both pipelines
def run_emotion_analysis(run_number1, run_number2, set_, config_path, number=10, print_table=False, save=False):
    print("Running Emotion Analysis Comparison Pipeline...")
    emotion_analysis_comparison_pipeline(run_number1, run_number2, set_, EMOTIONS, save)
    
    print("\nRunning Emotion Pair Analysis Pipeline for Run", run_number1)
    emotion_pair_analysis_pipeline(run_number1, set_, config_path, number, print_table)
    
    print("\nRunning Emotion Pair Analysis Pipeline for Run", run_number2)
    emotion_pair_analysis_pipeline(run_number2, set_, config_path, number, print_table)

if __name__ == "__main__":
    config_path = os.path.join("..", "config.yaml")
    EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]  # You might want to load this from config
    run_emotion_analysis(
        run_number1=2,
        run_number2=3,
        set_='train',
        config_path=config_path,
        number=10,
        print_table=False,
        save=True
    )