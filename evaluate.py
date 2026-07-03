"""Evaluate a trained model on the validation split.

Prints a per-class classification report (precision / recall / F1) and a
confusion matrix so we can see *where* the model fails — not just the headline
accuracy. This is the "measure before optimizing" step: with a heavily
imbalanced dataset (Negative ~6%), macro numbers hide the minority class, and
the confusion matrix shows exactly which classes get confused for which.

Usage:
    # lightweight fallback model (scikit-learn only — no torch needed)
    python evaluate.py --model fallback

    # primary transformer (requires torch + transformers)
    python evaluate.py --model transformer --model_path ./best_model
"""
import argparse
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from src.data_processor import DataProcessor

LABELS = ["Negative", "Average", "Positive"]  # label ids 0, 1, 2


def get_splits(data_path: str):
    dp = DataProcessor(data_path)
    dp.load_and_filter_data()
    return dp.prepare_datasets()  # X_train, X_val, y_train, y_val


def predict_fallback(X_train, y_train, X_val, model_path: str, refit: bool):
    """Fallback = TF-IDF + LogReg pipeline.

    By default we refit on the current (stratified) train split for an unbiased
    estimate, because a previously saved fallback may have been trained on a
    different split. Pass refit=False to score the persisted joblib instead.
    """
    if refit:
        import joblib  # noqa: F401  (kept for parity/inspection)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, C=1.0)),
        ])
        pipeline.fit(X_train, y_train)
    else:
        import joblib
        pipeline = joblib.load(f"{model_path}/fallback_model.joblib")
    return list(pipeline.predict(X_val))


def predict_transformer(X_val, model_path: str, batch_size: int = 16):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            batch = X_val[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=1).tolist())
    return preds


def print_confusion(cm):
    print("confusion matrix (rows = true, cols = predicted):")
    print("            " + "".join(f"{l:>10}" for l in LABELS))
    for i, row in enumerate(cm):
        print(f"{LABELS[i]:>10}  " + "".join(f"{v:>10}" for v in row))


def main():
    ap = argparse.ArgumentParser(description="Evaluate a model: per-class report + confusion matrix.")
    ap.add_argument("--model", choices=["fallback", "transformer"], default="fallback")
    ap.add_argument("--model_path", default="./best_model")
    ap.add_argument("--data_path", default="data/dutch_sentences.csv")
    ap.add_argument("--no-refit", dest="refit", action="store_false",
                    help="(fallback only) score the saved joblib instead of refitting")
    args = ap.parse_args()

    X_train, X_val, y_train, y_val = get_splits(args.data_path)
    print(f"Validation set: {len(y_val)} reviews | class counts: "
          f"{ {LABELS[k]: v for k, v in sorted(Counter(y_val).items())} }\n")

    if args.model == "fallback":
        y_pred = predict_fallback(X_train, y_train, X_val, args.model_path, args.refit)
    else:
        y_pred = predict_transformer(X_val, args.model_path)

    print(f"=== {args.model.upper()} — classification report ===")
    print(classification_report(y_val, y_pred, labels=[0, 1, 2], target_names=LABELS, digits=3, zero_division=0))
    print_confusion(confusion_matrix(y_val, y_pred, labels=[0, 1, 2]))


if __name__ == "__main__":
    main()
