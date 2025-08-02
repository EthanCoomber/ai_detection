# AI-Generated Text Detection

Detect whether a given text was written by a human or an AI model (e.g., GPT) using NLP techniques. This project implements and compares two approaches:

- **TF-IDF + Logistic Regression** (Baseline)
- **Fine-tuned DistilBERT** (Transformer-based)

The dataset is the [HC3 dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3), containing human-written and AI-generated academic responses.

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Setup Instructions](#setup-instructions)  
3. [Usage](#usage)  
4. [Evaluation](#evaluation)  
5. [Results](#results)  
6. [Future Work](#future-work)  
7. [References](#references)  
8. [License](#license)

## Project Structure
```
Final Project
│
├── data/
│   ├── HC3\_dataset.json             # Raw dataset (from HuggingFace)
│   └── processed\_hc3.csv           # Preprocessed dataset
│
├── models/
│   ├── baseline\_model.pkl          # Trained logistic regression model
│   ├── tfidf\_vectorizer.pkl        # Trained TF-IDF vectorizer
│   └── bert\_model/                # Fine-tuned DistilBERT model and tokenizer
│
├── notebooks/
│   ├── 01\_data\_exploration.ipynb
│   ├── 02\_baseline\_tfidf\_model.ipynb
│   └── 03\_fine\_tune\_bert.ipynb
│
├── scripts/
│   ├── data\_preprocessing.py       # Preprocesses the HC3 dataset
│   ├── train\_baseline.py           # Trains TF-IDF + logistic regression model
│   └── train\_bert.py              # Fine-tunes DistilBERT
│
├── reports/
│   └── AI\_Text\_Detection\_Report.pdf  # Final report
│
├── presentation/
│   └── slides.pdf 
│
├── requirements.txt          
└── README.md              

````

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone <repo-url>
   cd AI_Generated_Text_Detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the HC3 dataset**
   Option A: Using the Hugging Face CLI (requires `huggingface_hub` login):

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   python - <<'PY'
   from huggingface_hub import hf_hub_download
   path = hf_hub_download(repo_id="Hello-SimpleAI/HC3", filename="HC3.jsonl")
   print("Downloaded to:", path)
   PY
   ```

   Option B: Manually download from:
   [https://huggingface.co/datasets/Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

   Save the raw file as:

   ```bash
   mkdir -p data
   mv <downloaded-file> data/HC3_dataset.json
   ```

## Usage

### 1. Preprocess the dataset

```bash
python scripts/data_preprocessing.py
```

Output: `data/processed_hc3.csv` (cleaned and formatted examples for modeling).

### 2. Train baseline model (TF-IDF + Logistic Regression)

```bash
python scripts/train_baseline.py
```

Outputs:

* Serialized TF-IDF vectorizer (`tfidf_vectorizer.pkl`)
* Logistic regression model (`baseline_model.pkl`)
* Evaluation metrics (accuracy, F1, confusion matrix)

### 3. Fine-tune DistilBERT

```bash
python scripts/train_bert.py
```

Outputs:

* Fine-tuned transformer saved under `models/bert_model/`
* Evaluation metrics (accuracy, F1, etc.)

## Evaluation

Both training scripts report:

* **Accuracy**
* **F1-score**
* (Baseline script additionally shows a confusion matrix)

## Results

| Model                        | Accuracy | F1-Score |
| ---------------------------- | -------- | -------- |
| TF-IDF + Logistic Regression | 96%      | 0.95     |
| Fine-tuned DistilBERT        | 99%      | 0.99     |

*Results shown are from a sample run; for publication, report averaged metrics over cross-validation or held-out test splits.*

## Future Work

* Fine-tune on the full HC3 dataset and additional benchmarks.
* Add interpretability (e.g., SHAP, LIME) for model explanations.
* Experiment with larger or more recent transformer variants (RoBERTa, GPT-based).
* Build a lightweight web interface for interactive classification.
* Deploy as an API for real-time detection.

## References

* [HC3 Dataset – Hello-SimpleAI](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
* Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
* Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). *GLTR: Statistical Detection and Visualization of Generated Text.*
* Hugging Face Transformers
* scikit-learn
