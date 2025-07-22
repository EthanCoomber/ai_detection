import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

print("Transformers version:", __import__("transformers").__version__)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_bert_model(data_path='../data/processed_hc3.csv', model_dir='../models/bert_model', debug=False):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run data_preprocessing.py first.")

    # Load dataset
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != ""]

    df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    print(f"Training on a subset of {len(df)} samples for faster execution.")

    X_train, X_val, y_train, y_val = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )


    # Debug mode
    if debug:
        X_train, y_train = X_train[:200], y_train[:200]
        X_val, y_val = X_val[:50], y_val[:50]
        print(f"Debug mode enabled: {len(X_train)} training samples, {len(X_val)} validation samples")

    # Tokenizer and datasets
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_dataset = TextDataset(X_val.tolist(), y_val.tolist(), tokenizer)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Training on device: {device}")

    # Model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4 if not debug else 1,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=torch.cuda.is_available()
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"DistilBERT model saved to {model_dir}")

if __name__ == "__main__":
    # Enable debug=True for a quick test run
    # train_bert_model(debug=True)
    train_bert_model(debug=False)
