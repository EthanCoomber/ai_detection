
# AI-Generated Text Detection

This project detects whether a given text was written by a human or an AI model (e.g., GPT) using NLP techniques. It includes a baseline TF-IDF + Logistic Regression classifier and a fine-tuned BERT model.

---

## **Project Structure**
```
AI_Generated_Text_Detection/
│
├── data/
│   ├── HC3_dataset.json         # Raw dataset (downloaded from HuggingFace)
│   └── processed_hc3.csv        # Clean dataset after preprocessing
│
├── models/
│   ├── baseline_model.pkl       # Saved TF-IDF + Logistic Regression model
│   ├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
│   └── bert_model/              # Fine-tuned BERT model and tokenizer
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_tfidf_model.ipynb
│   └── 03_fine_tune_bert.ipynb
│
├── scripts/
│   ├── data_preprocessing.py    # Cleans & prepares the dataset
│   ├── train_baseline.py        # Trains TF-IDF + Logistic Regression model
│   └── train_bert.py            # Fine-tunes BERT on the dataset
│
├── reports/
│   └── final_report.pdf         # Final project report
│
├── presentation/
│   └── slides.pdf               # Presentation slides
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview & instructions
```
---

## **Setup Instructions**
1. Clone the repository or copy the folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the HC3 dataset from HuggingFace:  
   [https://huggingface.co/datasets/Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

4. Save the dataset as `data/HC3_dataset.json`.

---

## **How to Run**

### **1. Preprocess Data**
```bash
python scripts/data_preprocessing.py
```
- This creates `data/processed_hc3.csv` with human/AI-labeled text.

### **2. Train Baseline Model**
```bash
python scripts/train_baseline.py
```
- Trains a TF-IDF + Logistic Regression classifier.
- Saves model to `models/baseline_model.pkl`.

### **3. Fine-tune BERT**
```bash
python scripts/train_bert.py
```
- Fine-tunes BERT on the HC3 dataset.
- Saves model to `models/bert_model/`.

---

## **Evaluation**
- Both training scripts print evaluation metrics (Accuracy, F1-score).
- Confusion matrix is displayed for the baseline model.
- For BERT, metrics are printed after each epoch.

---

## **Next Steps / Future Work**
- Test with more datasets (e.g., other GPT variants).
- Add explainability (e.g., SHAP/LIME) to see which words influence classification.
- Build a simple web app to test live text input.

---

## **References**
- [HC3 Dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [scikit-learn](https://scikit-learn.org/)
