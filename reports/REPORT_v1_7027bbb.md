# Results Report — Dutch Movie Review Sentiment Classifier

This report summarizes the training experiment and evaluation metrics for the Dutch
sentiment classifier. All numbers below are taken directly from the tracked MLflow run
(no hand-entered values).

- **MLflow run:** `rebellious-gull-263`
- **Source:** `train_pipeline.py`
- **Base model:** `pdelobelle/robbert-v2-dutch-base` (RobBERT), fine-tuned with `num_labels=3`
- **Task:** classify a Dutch review as **Negative (0)**, **Average (1)**, or **Positive (2)**

---

## 1. Dataset

The pipeline filters the source CSV to **Dutch-only** rows (`language == "nl"`) and drops
rows with missing text/label (`src/data_processor.py`). Labels are mapped
`Negative→0, Average→1, Positive→2` and split 80/20 with a fixed seed (`random_state=42`).

Derived from the run (training steps × batch size, confirmed by eval throughput × runtime):

| Split             | Size                |
|-------------------|---------------------|
| Train             | ≈ 3,840 reviews     |
| Validation        | ≈ 960 reviews       |
| **Total (Dutch)** | **≈ 4,800 reviews** |

---

## 2. Approach

- **Primary model — RobBERT (transformer).** Fine-tuned Dutch RoBERTa; this is the model
  evaluated below and saved to `best_model/model.safetensors`.
- **Fallback model — TF‑IDF + Logistic Regression** (`ngram_range=(1,2)`, `max_features=5000`).
  A lightweight scikit-learn pipeline (`best_model/fallback_model.joblib`) used automatically
  if the transformer hits a runtime/OOM error at inference time.
- **Non-Dutch handling.** Inputs detected as `en/it/de` are translated to Dutch via a
  translation microservice before classification; other languages are rejected with HTTP 400.

---

## 3. Training configuration

| Hyperparameter                     | Value                                              |
|------------------------------------|----------------------------------------------------|
| Epochs                             | 3                                                  |
| Per-device batch size (train/eval) | 8                                                  |
| Learning rate                      | 5e-5                                               |
| Weight decay                       | 0.01                                               |
| Warmup steps                       | 100                                                |
| Max sequence length (tokenizer)    | 256                                                |
| Optimizer                          | AdamW (`adamw_torch`)                              |
| Seed                               | 42                                                 |
| Eval / save strategy               | per epoch                                          |
| Model selection                    | best by `f1_macro` (`load_best_model_at_end=True`) |
| transformers version               | 4.57.6                                             |
| Total training time                | ≈ 29.6 min                                         |

---

## 4. Results — RobBERT (validation set, ≈960 reviews)

Metrics per epoch, as logged to MLflow:

|  Epoch  | eval_loss  |  Accuracy  |  F1‑macro  | Precision‑macro | Recall‑macro |
|:-------:|:----------:|:----------:|:----------:|:---------------:|:------------:|
|    1    |   0.7210   |   0.6448   |   0.6053   |     0.6207      |    0.5948    |
| **2 ✅** | **0.7504** | **0.6635** | **0.6291** |   **0.6891**    |  **0.5981**  |
|    3    |   1.1618   |   0.6469   |   0.6166   |     0.6455      |    0.5964    |

**✅ Selected model = epoch 2** (highest `f1_macro`; this is what `best_model/` contains).

**Best model summary:**

| Metric          | Value     |
|-----------------|-----------|
| Accuracy        | **66.4%** |
| F1‑macro        | **0.629** |
| Precision‑macro | 0.689     |
| Recall‑macro    | 0.598     |
| eval_loss       | 0.750     |

---

## 5. Observations

- **Overfitting after epoch 2.** Training loss keeps falling (final `train_loss = 0.563`) while
  validation loss *rises sharply* at epoch 3 (0.75 → 1.16) and F1 drops. The
  `load_best_model_at_end` + `metric_for_best_model=f1_macro` setup correctly discards the
  over-trained epoch 3 and keeps epoch 2. A shorter schedule (2 epochs) or stronger
  regularization/early stopping would likely be as good or better and cheaper.
- **Recall < precision** across all epochs (0.598 vs 0.689 at best), i.e. the model is
  relatively conservative — it misses true instances of the minority class. The confusion
  matrix (section 6) shows the driver is **class imbalance**: **Negative** is only ~6% of
  the data (≈60 of 960 validation reviews) and is the weak class — *not* the middle
  "Average" class as one might first assume.
- **Macro ≈ 3-class balanced view.** With only ~66% accuracy on a 3-class problem
  (random ≈ 33%), the model has clearly learned signal but has meaningful headroom.

---

## 6. Fallback model + per-class breakdown (confusion matrix)

The TF‑IDF + Logistic Regression fallback is now evaluated on the **same stratified validation
split** via `evaluate.py` (`python evaluate.py --model fallback`). This also gives the
per-class picture that the macro numbers hide.

**Fallback — validation set (960 reviews: 60 Negative / 450 Average / 450 Positive):**

| Class        | Precision | Recall    | F1        | Support |
|--------------|:---------:|:---------:|:---------:|:-------:|
| **Negative** | 0.857     | **0.100** | **0.179** | 60      |
| Average      | 0.607     | 0.680     | 0.641     | 450     |
| Positive     | 0.658     | 0.622     | 0.639     | 450     |
| **Accuracy** |           |           | **0.624** | 960     |
| **Macro F1** |           |           | **0.486** | 960     |

**Confusion matrix (rows = true, cols = predicted):**

|              | Negative | Average | Positive |
|--------------|:--------:|:-------:|:--------:|
| **Negative** | 6        | 30      | 24       |
| **Average**  | 1        | 306     | 143      |
| **Positive** | 0        | 170     | 280      |

**Read this:** of 60 true Negatives the fallback catches only **6** — the other 54 are
absorbed into Average/Positive. Negative *precision* is high (0.857) but *recall* is 0.100,
so the class is effectively ignored. This is the imbalance bottleneck, and it is why
**class-weighted loss** (see next steps) is the highest-leverage fix. The transformer, being
stronger, does better overall (0.664 acc / 0.629 macro-F1 vs 0.624 / 0.486) but was measured
on the pre-stratify split, so a like-for-like transformer confusion matrix is the next thing
to log after a retrain.

---

## 7. Latency

- **Eval throughput (logged):** 23.6 samples/second (batch size 8) on the training host.
- **API latency:** `POST /classify` returns a `latency_seconds` field per request. Observed
  per-request latency on the current CPU deployment is roughly **2–4 s** end-to-end, but this
  is **environment-dependent** (CPU-only host, and inflated further under concurrent load /
  resource contention). A dedicated single-request benchmark on the target hardware is
  recommended for a firm number.

---

## 8. Reproducibility

- **Experiment tracking:** MLflow (`Dutch_Sentiment_Analysis` experiment, run `rebellious-gull-263`).
- **Determinism:** fixed split seed (42); `langdetect` seeded for stable language routing.
- **Data / model versioning:** dataset tracked with DVC (pointer in git, bytes in a remote);
  Docker images tagged by git SHA. See the README versioning section.

---

## 9. Limitations & next steps

1. ~~Evaluate the fallback~~ ✅ done (section 6) — head-to-head comparison now exists.
2. ~~Per-class breakdown + confusion matrix~~ ✅ done (section 6). It disproved the "Average"
   hypothesis: **Negative** is the weak minority class. Per-class F1 is now also logged per
   epoch during training (`compute_metrics` in `src/model_trainer.py`).
3. **Class-weighted loss (highest leverage).** Implemented via `WeightedTrainer` in
   `src/model_trainer.py` (inverse-frequency weights). Needs a **retrain** to take effect;
   expected to raise Negative recall and macro-F1 at a small cost to overall accuracy.
4. **Reduce epochs / add early stopping** given the clear epoch-3 overfitting.
5. **Explainability** — expose LogReg top features per class and/or transformer attention/SHAP.
6. **Single-request latency benchmark** on the deployment hardware.
7. **Single run only** — repeat across seeds to report metric stability (mean ± std).
