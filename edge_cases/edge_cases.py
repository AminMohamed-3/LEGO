import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
import csv

# Specify the version
VERSION = "v1"

# Load both models
our_tokenizer = AutoTokenizer.from_pretrained("0ssamaak0/roberta-base-LEGO_emotions")
our_model = AutoModelForSequenceClassification.from_pretrained(
    "0ssamaak0/roberta-base-LEGO_emotions"
)

original_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
original_model = AutoModelForSequenceClassification.from_pretrained(
    "SamLowe/roberta-base-go_emotions"
)

# Define the emotion lists for both models
our_EMOTIONS = [
    "amusement",
    "excitement",
    "joy",
    "love",
    "desire",
    "optimism",
    "caring",
    "pride",
    "admiration",
    "gratitude",
    "relief",
    "approval",
    "realization",
    "surprise",
    "curiosity",
    "confusion",
    "fear",
    "nervousness",
    "remorse",
    "embarrassment",
    "disappointment",
    "sadness",
    "grief",
    "disgust",
    "anger",
    "annoyance",
    "disapproval",
    "neutral",
]

original_EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


# Function to analyze emotions using the appropriate model and emotion list
def analyze_emotion(text, tokenizer, model, emotions_list):
    encoded_input = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded_input)

    probabilities = torch.sigmoid(output.logits).squeeze().tolist()
    sorted_emotions = sorted(
        zip(emotions_list, probabilities), key=lambda x: x[1], reverse=True
    )

    df = pd.DataFrame(
        sorted_emotions[:5], columns=["Emotion", "Score"]
    )  # Limit to top 5 emotions
    df["Score"] = df["Score"].apply(lambda x: f"{x:.4f}")

    return df


# Function to save edge cases
def save_edge_case(text, our_df, original_df, save_our, save_original):
    current_path = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(current_path, f"edge_cases_{VERSION}.csv")
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, "a", newline="") as csvfile:
        fieldnames = ["Text"]

        # Add fieldnames for both models
        fieldnames += [f"emotion_1_{i+1}" for i in range(5)] + [
            f"probability_1_{i+1}" for i in range(5)
        ]
        fieldnames += [f"emotion_2_{i+1}" for i in range(5)] + [
            f"probability_2_{i+1}" for i in range(5)
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        row_dict = {"Text": text}

        # Fill "our" model's values or set them to "NONE" if not saving
        if save_our:
            for i, row in our_df.iterrows():
                row_dict[f"emotion_1_{i+1}"] = row["Emotion"]
                row_dict[f"probability_1_{i+1}"] = float(row["Score"])
        else:
            for i in range(5):
                row_dict[f"emotion_1_{i+1}"] = "NONE"
                row_dict[f"probability_1_{i+1}"] = "NONE"

        # Fill "original" model's values or set them to "NONE" if not saving
        if save_original:
            for i, row in original_df.iterrows():
                row_dict[f"emotion_2_{i+1}"] = row["Emotion"]
                row_dict[f"probability_2_{i+1}"] = float(row["Score"])
        else:
            for i in range(5):
                row_dict[f"emotion_2_{i+1}"] = "NONE"
                row_dict[f"probability_2_{i+1}"] = "NONE"

        writer.writerow(row_dict)

    return "Edge case saved successfully!"


# Function to style the dataframe for better readability
def style_df(df):
    def color_score(score):
        score = float(score)
        r = int(255 * (1 - score))
        g = int(255 * score)
        return f"background-color: rgb({r}, {g}, 0); color: white;"

    return df.style.map(color_score, subset=["Score"])


# Define the Gradio interface
with gr.Blocks(title="LEGO Model Arena") as iface:
    gr.Markdown("# LEGO Model Arena")
    gr.Markdown(
        "Compare the performance of GO Emotions vs LEGO Emotions"
    )

    with gr.Row():
        text_input = gr.Textbox(label="Enter text for emotion analysis")
        analyze_btn = gr.Button("Analyze")

    with gr.Row():
        output_our_df = gr.Dataframe(
            headers=["Emotion", "Score"],
            datatype=["str", "str"],
            col_count=(2, "fixed"),
            label="Top 5 Emotions and Scores roberta-base-lego_emotions (preview)",
        )
        output_original_df = gr.Dataframe(
            headers=["Emotion", "Score"],
            datatype=["str", "str"],
            col_count=(2, "fixed"),
            label="Top 5 Emotions and Scores SamLowe/roberta-base-go_emotions",
        )

    with gr.Row():
        save_our = gr.Checkbox(label="Save for Our Model (Model 1)", value=False)
        save_original = gr.Checkbox(
            label="Save for Original Model (Model 2)", value=False
        )
        save_btn = gr.Button("Save as Edge Case")

    save_status = gr.Textbox(label="Save Status")

    # Store the last analyzed text and dataframes
    last_analyzed_text = gr.State("")
    last_our_df = gr.State(pd.DataFrame())
    last_original_df = gr.State(pd.DataFrame())

    # Define the analyze function for both models
    def analyze(text):
        our_df = analyze_emotion(text, our_tokenizer, our_model, our_EMOTIONS)
        original_df = analyze_emotion(
            text, original_tokenizer, original_model, original_EMOTIONS
        )

        styled_our_df = style_df(our_df)
        styled_original_df = style_df(original_df)

        return styled_our_df, styled_original_df, text, our_df, original_df

    # Define the save function
    def save(last_text, our_df, original_df, save_our, save_original):
        if not last_text:
            return "No text has been analyzed yet."
        return save_edge_case(last_text, our_df, original_df, save_our, save_original)

    # Set up event handlers
    analyze_btn.click(
        analyze,
        inputs=text_input,
        outputs=[
            output_our_df,
            output_original_df,
            last_analyzed_text,
            last_our_df,
            last_original_df,
        ],
    )
    save_btn.click(
        save,
        inputs=[
            last_analyzed_text,
            last_our_df,
            last_original_df,
            save_our,
            save_original,
        ],
        outputs=save_status,
    )
    text_input.submit(
        analyze,
        inputs=text_input,
        outputs=[
            output_our_df,
            output_original_df,
            last_analyzed_text,
            last_our_df,
            last_original_df,
        ],
    )

# Launch the interface
iface.launch()
