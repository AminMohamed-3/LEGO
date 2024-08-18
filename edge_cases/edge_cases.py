import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
import csv

# Specify the version
VERSION = "v1"

# Load model
tokenizer = AutoTokenizer.from_pretrained("0ssamaak0/roberta-base-LEGO_emotions")
model = AutoModelForSequenceClassification.from_pretrained("0ssamaak0/roberta-base-LEGO_emotions")

EMOTIONS = [
    "amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride",
    "admiration", "gratitude", "relief", "approval", "realization", "surprise", "curiosity",
    "confusion", "fear", "nervousness", "remorse", "embarrassment", "disappointment",
    "sadness", "grief", "disgust", "anger", "annoyance", "disapproval", "neutral"
]

def analyze_emotion(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Get the probabilities using sigmoid 
    probabilities = torch.sigmoid(output.logits).squeeze().tolist()
    sorted_emotions = sorted(zip(EMOTIONS, probabilities), key=lambda x: x[1], reverse=True)
    
    df = pd.DataFrame(sorted_emotions[:8], columns=['Emotion', 'Score'])
    df['Score'] = df['Score'].apply(lambda x: f"{x:.4f}")
    
    return df, text

def save_edge_case(text, df):
    csv_filename = f'edge_cases_{VERSION}.csv'
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Text'] + EMOTIONS
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row_dict = {emotion: 0 for emotion in EMOTIONS}
        row_dict['Text'] = text
        
        for _, row in df.iterrows():
            row_dict[row['Emotion']] = float(row['Score'])
        
        writer.writerow(row_dict)
    
    return "Edge case saved successfully!"

def style_df(df):
    def color_score(score):
        score = float(score)
        r = int(255 * (1 - score))
        g = int(255 * score)
        return f'background-color: rgb({r}, {g}, 0); color: white;'

    return df.style.applymap(color_score, subset=['Score'])

# Define Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Emotion Analysis with Edge Case Saving")
    gr.Markdown("Analyze the emotions in the given text and save interesting edge cases.")
    
    with gr.Row():
        text_input = gr.Textbox(label="Enter text for emotion analysis")
        analyze_btn = gr.Button("Analyze")
    
    with gr.Row():
        output_df = gr.Dataframe(
            headers=["Emotion", "Score"],
            datatype=["str", "str"],
            col_count=(2, "fixed"),
            label="Top 8 Emotions and Scores"
        )
    
    with gr.Row():
        save_btn = gr.Button("Save as Edge Case")
    
    save_status = gr.Textbox(label="Save Status")
    
    # Store the last analyzed text and dataframe
    last_analyzed_text = gr.State("")
    last_analyzed_df = gr.State(pd.DataFrame())

    # Define the analyze function
    def analyze(text):
        df, analyzed_text = analyze_emotion(text)
        styled_df = style_df(df)
        return styled_df, analyzed_text, df

    # Define the save function
    def save(last_text, df):
        if not last_text:
            return "No text has been analyzed yet."
        return save_edge_case(last_text, df)

    # Set up event handlers
    analyze_btn.click(analyze, inputs=text_input, outputs=[output_df, last_analyzed_text, last_analyzed_df])
    save_btn.click(save, inputs=[last_analyzed_text, last_analyzed_df], outputs=save_status)

# Launch the interface
iface.launch()
