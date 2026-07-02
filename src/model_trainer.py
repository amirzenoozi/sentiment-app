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
from sklearn.utils.class_weight import compute_class_weight


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


CLASS_NAMES = ["negative", "average", "positive"]  # label ids 0, 1, 2


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    metrics = {
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall,
    }

    # Per-class F1 so minority-class performance (e.g. the rare Negative class)
    # is visible per epoch instead of being hidden inside the macro average.
    per_class_f1 = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )[2]
    for name, score in zip(CLASS_NAMES, per_class_f1):
        metrics[f'f1_{name}'] = float(score)

    return metrics


class WeightedTrainer(Trainer):
    """Trainer that applies per-class weights to the cross-entropy loss.

    The dataset is heavily imbalanced (Negative ~6%), so an unweighted loss lets
    the model score well by ignoring the minority class (Negative recall ~10%).
    Weighting the loss inversely to class frequency makes each Negative example
    count more, trading a little majority-class accuracy for far better minority
    recall / macro-F1.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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

        # Inverse-frequency class weights to counter the Negative-class imbalance.
        classes = np.array([0, 1, 2])
        weights = compute_class_weight("balanced", classes=classes, y=np.array(y_train))
        class_weights = torch.tensor(weights, dtype=torch.float)
        print(f"Class weights (Negative, Average, Positive): {weights.tolist()}")

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
            trainer = WeightedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                class_weights=class_weights
            )

            trainer.train()

            # Dynamic logging of custom inputs into MLflow tracker dashboard
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("class_weights", weights.tolist())

            self.model.save_pretrained("./best_model")
            self.tokenizer.save_pretrained("./best_model")

            # Register this run's model in the MLflow Model Registry so every
            # retrain becomes a new *version* (v1, v2, ...) that can be compared
            # and rolled back to, instead of silently overwriting the last one.
            # Wrapped defensively: the model is already safely on disk, so a
            # registry/tracking-server hiccup must not fail the training run.
            try:
                mlflow.transformers.log_model(
                    transformers_model={"model": self.model, "tokenizer": self.tokenizer},
                    artifact_path="model",
                    task="text-classification",
                    registered_model_name="dutch-sentiment-robbert",
                )
                print("Logged model to MLflow Model Registry as 'dutch-sentiment-robbert'.")
            except Exception as e:  # noqa: BLE001 - never let registry errors break training
                print(f"WARNING: MLflow model registration skipped ({e}).")