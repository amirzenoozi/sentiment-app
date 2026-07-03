# Results Report v2 — Dutch Sentiment Classifier (class-weighted)

Second training experiment. The change under test is a **class-weighted loss** to fix the
minority-class (Negative) collapse that v1 exposed. All numbers below are taken directly from
the tracked MLflow run and a re-run of `evaluate.py` on the same validation split (no
hand-entered values).

- **MLflow run:** `peaceful-crane-69` (experiment `Dutch_Sentiment_Analysis`)
- **Source:** `train_pipeline.py`
- **Base model:** `pdelobelle/robbert-v2-dutch-base` (RobBERT), `num_labels=3`
- **Task:** classify a Dutch review as **Negative (0)**, **Average (1)**, **Positive (2)**
- **Predecessor:** [`REPORT_v1_7027bbb.md`](./REPORT_v1_7027bbb.md) (run `rebellious-gull-263`)

---

## 1. What changed vs v1

| Aspect                | v1 (`rebellious-gull-263`) | v2 (`peaceful-crane-69`)                        |
|-----------------------|----------------------------|-------------------------------------------------|
| Loss                  | plain cross-entropy        | **class-weighted** cross-entropy (`WeightedTrainer`) |
| Class weights         | none (all = 1)             | `[5.333, 0.711, 0.711]` (Negative, Average, Positive) |
| Epochs                | 3                          | 5                                               |
| Batch size            | 8                          | 16                                              |
| Split                 | random 80/20               | **stratified** 80/20 (preserves 6% Negative in val) |
| Per-class F1 logged   | no                         | **yes** (`f1_negative/average/positive` per epoch) |

The weights are inverse-frequency (`compute_class_weight("balanced")`): Negative is ~6% of the
data, so a Negative mistake now costs ~5.3× a Positive/Average mistake.

> ⚠️ **Honest caveat:** four things changed at once (weights, epochs, batch size, stratified
> split), so the gains below are **not cleanly attributable to class weighting alone**. A clean
> ablation would flip one variable at a time. The direction and size of the Negative-class gain,
> however, are exactly what weighting predicts.

---

## 2. Training configuration (v2)

| Hyperparameter                     | Value                                              |
|------------------------------------|----------------------------------------------------|
| Epochs                             | 5                                                  |
| Per-device batch size (train/eval) | 16                                                 |
| Learning rate                      | 5e-5                                               |
| Weight decay                       | 0.01                                               |
| Warmup steps                       | 100                                                |
| Max sequence length                | 256                                                |
| Optimizer                          | AdamW (`adamw_torch`)                              |
| Loss                               | weighted CE, weights `[5.333, 0.711, 0.711]`       |
| Seed                               | 42                                                 |
| Model selection                    | best by `f1_macro` (`load_best_model_at_end=True`) |
| Training host                      | macOS CPU (≈10.9 h wall; 0.49 samples/s)           |

---

## 3. Results — RobBERT v2 (validation set, 960 reviews)

Per-epoch, as logged to MLflow:

|  Epoch  | eval_loss | Accuracy | F1-macro | F1-neg | F1-avg | F1-pos | Prec-macro | Recall-macro |
|:-------:|:---------:|:--------:|:--------:|:------:|:------:|:------:|:----------:|:------------:|
|    1    |  0.8808   |  0.6521  |  0.6300  | 0.5833 | 0.6223 | 0.6843 |   0.6926   |    0.5985    |
|    2    |  0.8362   |  0.6667  |  0.6589  | 0.6415 | 0.6499 | 0.6852 |   0.6886   |    0.6378    |
| **3 ✅** | **0.9733**| **0.6875**| **0.6760**|**0.6491**|**0.6996**|**0.6792**| **0.6888** | **0.6670**  |
|    4    |  1.3245   |  0.6542  |  0.6423  | 0.6207 | 0.6201 | 0.6862 |   0.6547   |    0.6385    |
|    5    |  1.6533   |  0.6688  |  0.6553  | 0.6261 | 0.6469 | 0.6930 |   0.6664   |    0.6489    |

**✅ Selected model = epoch 3** (highest `f1_macro`; this is what `best_model/` now contains).

### Per-class breakdown (epoch-3 model, via `evaluate.py --model transformer`)

| Class        | Precision | Recall    | F1        | Support |
|--------------|:---------:|:---------:|:---------:|:-------:|
| **Negative** | 0.685     | **0.617** | **0.649** | 60      |
| Average      | 0.663     | 0.740     | 0.700     | 450     |
| Positive     | 0.718     | 0.644     | 0.679     | 450     |
| **Accuracy** |           |           | **0.688** | 960     |
| **Macro F1** |           |           | **0.676** | 960     |

### Confusion matrix (rows = true, cols = predicted)

|              | Negative | Average | Positive |
|--------------|:--------:|:-------:|:--------:|
| **Negative** | **37**   | 16      | 7        |
| **Average**  | 10       | 333     | 107      |
| **Positive** | 7        | 153     | 290      |

**Read this:** of 60 true Negatives, the model now catches **37** (recall 0.617). Compare the
v1-era fallback, which caught only **6/60** (recall 0.100). The minority class is no longer
being ignored — that is the class-weighting doing its job.

---

## 4. Head-to-head — v1 vs v2 (best model of each)

| Metric              | v1 (unweighted) | v2 (weighted) |     Δ      |
|---------------------|:---------------:|:-------------:|:----------:|
| Accuracy            |     0.6635      |   **0.6875**  | **+0.0240** |
| F1-macro            |     0.6291      |   **0.6760**  | **+0.0469** |
| Precision-macro     |   **0.6891**    |     0.6888    |   -0.0003  |
| Recall-macro        |     0.5981      |   **0.6670**  | **+0.0689** |
| Negative recall     |  not logged¹    |   **0.617**   |     —      |

¹ v1 did not log per-class metrics (instrumentation was added afterwards). Its macro recall of
0.598 — well below v2's 0.667 — is consistent with a weak Negative class dragging the average.

**Takeaways:**
- **Everything improved except precision, which held flat.** The usual weighted-loss trade
  (recall up, accuracy/precision down) did *not* cost us here: accuracy rose +2.4 pts and
  precision stayed level. The biggest single gain is **macro recall +6.9 pts** — precisely the
  imbalance metric we targeted.
- **Macro-F1 +4.7 pts** is the honest headline: on a 3-class, imbalanced problem, macro-F1 is a
  fairer scorecard than raw accuracy because it weights the rare Negative class equally.
- **eval_loss is not comparable across the two runs.** Weighted CE changes the loss *scale*, so
  v2's higher numeric loss (0.97 vs 0.75) does **not** mean a worse model — the eval metrics do.

---

## 5. Observations & remaining issues

- **Overfitting still sets in after epoch 3.** eval_loss climbs 0.97 → 1.32 → 1.65 at epochs
  4–5 while `f1_macro` falls. `load_best_model_at_end` correctly kept epoch 3, but the 4th/5th
  epochs were wasted compute. **Early stopping (patience 1–2) or a 3-epoch cap** is the next fix.
- **Negative is fixed but still the weakest class** (F1 0.649 vs ~0.68–0.70 for the others),
  and it now mostly confuses with **Average** (16) more than Positive (7) — sentiment-adjacent
  errors, which is the "good" kind of mistake.
- **Average ↔ Positive is now the dominant error mass** (107 + 153 off-diagonal). That's the
  next accuracy ceiling to attack — likely a data/labeling boundary issue more than a model one.

---

## 6. Reproducibility

- **Experiment tracking:** MLflow, run `peaceful-crane-69`. Per-class F1 logged per epoch.
- **Model versioning:** each retrain is registered in the MLflow Model Registry as a new version
  of `dutch-sentiment-robbert`; the v1 model was also snapshotted to `best_model_v1_7027bbb/`.
- **Determinism:** fixed split seed (42), stratified.
- **Eval reproducibility:** `poetry run python evaluate.py --model transformer --model_path ./best_model`.

---

## 7. Next steps (updated)

1. **Early stopping / fewer epochs** — epochs 4–5 only overfit; stop at best.
2. **Attack Average↔Positive confusion** — the new dominant error mass (inspect mislabeled
   boundary reviews; consider label-smoothing or more mid-tone training data).
3. **Clean ablation** — re-run changing *only* the class weights (hold epochs/batch/split fixed)
   to attribute the gain rigorously.
4. **Multiple seeds** — report mean ± std; both runs are single-shot.
5. **Explainability** — LogReg top features per class and/or transformer attention/SHAP.
