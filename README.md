# Sentiment Analysis Model

A fine-tuned DistilBERT model for binary sentiment classification (Positive / Negative) on product and service reviews.

> **Disclaimer:** This model is built for **educational and prototyping purposes**. It is intended to test sentiment analysis capabilities within an internal app we are developing. It is **not production-grade** and should not be used as the sole decision-making system in any high-stakes application.

---

## Model Details

| Property | Value |
|----------|-------|
| Base Model | `distilbert-base-uncased` |
| Task | Binary Sequence Classification |
| Labels | `0` = Negative, `1` = Positive |
| Max Sequence Length | 128 tokens |
| Framework | PyTorch + Hugging Face Transformers |

## Performance

| Metric | Negative | Positive | Overall |
|--------|----------|----------|---------|
| Precision | 0.92 | 0.97 | — |
| Recall | 0.94 | 0.96 | — |
| F1 Score | 0.93 | 0.97 | — |
| **Accuracy** | — | — | **95%** |

> Evaluated on a held-out test set (10% of total data, stratified split). These numbers reflect a controlled educational dataset and may not generalize to real-world, noisy, or domain-shifted reviews.

## Dataset

- **Source:** `combined_reviews_dataset.csv` (custom combined review dataset)
- **Total Samples:** ~4,450 (after deduplication)
- **Class Distribution:** ~2:1 (Positive : Negative)
- **Split:** 80% Train / 10% Validation / 10% Test (stratified)

## How to Use

### Install Dependencies

```bash
pip install transformers torch emoji
```

### Load from Hugging Face Hub

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

model_name = "MHKaungPyae/sentiment-model"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
model.eval()
```

### Predict Sentiment

```python
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, truncation=True, padding=True,
                       max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    return {
        "label": "Positive" if predicted_class == 1 else "Negative",
        "confidence": confidence
    }

# Example
result = predict_sentiment("Great product, highly recommend!", model, tokenizer)
print(result)
# {'label': 'Positive', 'confidence': 0.99}
```

## Limitations

This model has several known limitations:

1. **Binary only** — Cannot distinguish neutral reviews. Neutral statements (e.g., "It was okay") are forced into Positive or Negative.
2. **Educational dataset** — Trained on a small, curated dataset. Not representative of real-world review noise, sarcasm, or multilingual text.
3. **No domain adaptation** — Performance may degrade significantly on reviews from different domains (e.g., restaurant vs. electronics).
4. **High confidence ≠ correctness** — The model can be confidently wrong, especially on ambiguous or sarcastic text.
5. **Intended for app prototyping** — Designed to test sentiment features in our app, not for standalone production use.

## Project Structure

```
Sentiment_Analysis/
├── Sentiment_Analysis.ipynb   # Full training pipeline
├── README.md                  # This file
└── confusion_matrix.png       # Confusion matrix visualization
```

## Citation

If you use this model in your work, please cite:

```
Base model: DistilBERT (Sanh et al., 2019)
Fine-tuned for sentiment analysis — educational prototype
```

## License

This project is for educational use only.
