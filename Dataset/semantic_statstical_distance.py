import pandas as pd

def get_semantic_statstical_distance(emotions1, emotions2, verbose=False):
    # Load emotion distances from CSV file
    distances = pd.read_csv("/mnt/c/Data/LEGO/Dataset/emotion_distances.csv")

    # make emotions1 the largest of the two
    if len(emotions1) < len(emotions2):
        emotions1, emotions2 = emotions2, emotions1

    accumulation = 0

    # set for unique pairs
    traversed = set()
    for emotion1 in emotions1:
        for emotion2 in emotions2:
            # if the two emotions have been found before, skip this step
            if (emotion1, emotion2) in traversed or (emotion2, emotion1) in traversed:
                continue
            else:
                traversed.add((emotion1, emotion2))
                traversed.add((emotion2, emotion1))

            # Get distance from emotion_distances dictionary
            distance = 100000
            for index, row in distances.iterrows():
                if (row["Emotion_1"] == emotion1 and row["Emotion_2"] == emotion2) or (row["Emotion_1"] == emotion2 and row["Emotion_2"] == emotion1):
                    distance = row["Distance"]
                    break
            if distance == 100000:
                print(f"Distance between {emotion1} and {emotion2} not found in the CSV file")

            if verbose:
                print(f"{emotion1} -> {emotion2}: {distance}")
            accumulation += distance

    return accumulation
