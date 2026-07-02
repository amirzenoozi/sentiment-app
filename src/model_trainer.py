import mlflow
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }


class ModelTrainer:
    """
    Manages dynamic training routines fueled by parameters passed via CLI arguments.
    """

    def __init__(self, model_name: str = "pdelobelle/robbert-v2-dutch-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def train_fallback_model(self, X_train, y_train):
        print("Training lightweight fallback model...")
        fallback_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, C=1.0))
        ])
        fallback_pipeline.fit(X_train, y_train)
        joblib.dump(fallback_pipeline, "./best_model/fallback_model.joblib")

    def train(self, X_train, y_train, X_val, y_val, epochs: int, batch_size: int, output_dir: str = "./results"):
        import os
        os.makedirs("./best_model", exist_ok=True)

        # Train fallback pipeline
        self.train_fallback_model(X_train, y_train)

        # Prepare tokens
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=256)
        val_encodings = self.tokenizer(X_val, truncation=True, padding=True, max_length=256)

        train_dataset = SentimentDataset(train_encodings, y_train)
        val_dataset = SentimentDataset(val_encodings, y_val)

        # Injecting values received from the line argument parsing block
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to="mlflow",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro"
        )

        mlflow.set_experiment("Dutch_Sentiment_Analysis")
        with mlflow.start_run():
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics
            )

            trainer.train()

            # Dynamic logging of custom inputs into MLflow tracker dashboard
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("batch_size", batch_size)

            self.model.save_pretrained("./best_model")
            self.tokenizer.save_pretrained("./best_model")