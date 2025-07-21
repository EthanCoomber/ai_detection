
import pandas as pd
import os

def load_and_preprocess_hc3(dataset_path='../data/HC3_dataset.json', output_csv='../data/processed_hc3.csv'):
    """
    Load and preprocess the HC3 dataset.
    - Flattens human and chatgpt answers
    - Saves as a clean CSV ready for model training
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please download HC3 dataset from HuggingFace.")

    # Load dataset (expects JSON)
    data = pd.read_json(dataset_path)

    # Flatten human answers
    human = pd.DataFrame({
        'text': sum(data['human_answers'].tolist(), []),
        'label': 0
    })

    # Flatten chatgpt answers
    chatgpt = pd.DataFrame({
        'text': sum(data['chatgpt_answers'].tolist(), []),
        'label': 1
    })

    # Combine and shuffle
    df = pd.concat([human, chatgpt]).sample(frac=1).reset_index(drop=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed dataset saved to {output_csv} with {len(df)} samples.")

if __name__ == "__main__":
    load_and_preprocess_hc3()
