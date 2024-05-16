import re

def extract_emotions(text):
    """
    Extracts a list of emotions from the given text enclosed in <answer> tags.

    Parameters:
    text (str): The input text containing the emotions.

    Returns:
    list: A list of emotions found between <answer> and </answer> tags.
    """
    # Use regular expression to find the emotions between <answer> and </answer>
    pattern = r'<answer>\s*(.*)'
    match = re.search(pattern, text)

    if match:
        # Extract the list of emotions and split by comma
        emotions = match.group(1).strip().split(', ')
        return emotions
    else:
        return []