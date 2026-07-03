# Dutch Movie Review Sentiment Classifier

Classifies Dutch movie reviews as **Positive**, **Average**, or **Negative**.

- **Primary model:** fine-tuned `pdelobelle/robbert-v2-dutch-base` (RobBERT).
- **Fallback:** a TF-IDF + Logistic Regression baseline, used automatically if the transformer fails to load or errors at runtime.
- **Quantized model:** an INT8 ONNX build of the primary model for faster, lighter CPU inference, served on its own endpoint for comparison.
- **Language handling:** non-Dutch input is auto-detected; supported languages (`en`, `de`, `it`) are translated to Dutch via a LibreTranslate microservice, anything else is rejected with HTTP 400.

Data and model artifacts are versioned with **DVC** (stored on an SSH remote); the app is packaged with **Docker Compose** and images are published to GHCR via GitHub Actions.

---

## Project Structure

```text
sentiment-app/
├── src/
│   ├── data_processor.py   # Data loading & filtering
│   ├── model_trainer.py    # Training + MLflow logging
│   └── predictor.py        # Inference engine (torch + ONNX backends, fallback, language routing)
├── app.py                  # FastAPI app
├── train_pipeline.py       # Training CLI
├── quantize_cli.py         # ONNX INT8 quantization CLI
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── data/dutch_sentences.csv.dvc   # DVC pointer to the dataset
├── best_model.dvc                 # DVC pointer to the trained model
└── quantized_model.dvc            # DVC pointer to the quantized model
```

---

## Setup

Requires **Python 3.12** and **Poetry**.

```bash
pyenv local 3.12.8
poetry env use python3.12
poetry install
```

---

## DVC (Data & Models)

The dataset (`data/dutch_sentences.csv`), the trained model (`best_model/`), and the quantized model (`quantized_model/`) are **not** committed to Git. DVC stores small pointer files in Git and keeps the actual bytes on an SSH remote (defined in `.dvc/config`; credentials live in the git-ignored `.dvc/config.local`).

```bash
# One-time: set your SSH credential (key or password) — written to config.local
poetry run dvc remote modify --local myremote keyfile ~/.ssh/id_rsa
# ...or password auth:
poetry run dvc remote modify --local myremote password 'YOUR_PASSWORD'

# Download the dataset + models
poetry run dvc pull

# After changing a tracked artifact, publish the new version
poetry run dvc add best_model            # or data/dutch_sentences.csv, quantized_model
git add best_model.dvc && git commit -m "update model"
poetry run dvc push
```

Each Git commit maps to an exact dataset/model revision, so `git checkout <commit> && dvc pull` reproduces any version.

---

## Training

```bash
poetry run python train_pipeline.py --data_path data/dutch_sentences.csv --epochs 5 --batch_size 16
poetry run python train_pipeline.py --help   # all options
```

Training writes the model to `best_model/` and logs metrics (loss, F1-macro, precision, recall) to MLflow.

---

## Quantization

Convert a trained model into an INT8 ONNX build (~4× smaller, faster CPU inference):

```bash
# arm64 (Mac/ARM) is the default; use avx2/avx512 for x86 servers
poetry run python quantize_cli.py ./best_model ./quantized_model --arch avx2
```

Output goes to `quantized_model/` (`model_quantized.onnx` + tokenizer). The API serves it via `/classify/quantized`.

---

## Running the API

```bash
poetry run uvicorn app:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs`

### Endpoints

| Method | Path                  | Description                                            |
|--------|-----------------------|--------------------------------------------------------|
| POST   | `/classify`           | Classify with the primary (PyTorch) model              |
| POST   | `/classify/quantized` | Classify with the INT8 ONNX model (503 if not loaded)  |
| POST   | `/compare`            | Run both models, return labels, latencies, and speedup |
| GET    | `/health`             | Liveness probe                                         |
| GET    | `/ready`              | Readiness probe (reports which engines are loaded)     |

**Example**

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"review": "Dit is een absolute topfilm! Geweldig acteerwerk."}'
# -> {"label": "Positive", "latency_seconds": 0.045}
```

Unsupported non-Dutch input returns HTTP 400.

---

## Deployment (Docker Compose)

The stack runs three services: `api` (port 8022), `translation_service` (LibreTranslate), and `mlflow_server` (port 5000). Models are mounted from host directories:

```yaml
volumes:
  - ./model:/app/best_model
  - ./quantized_model:/app/quantized_model
```

```bash
# On the server (image is built & pushed to GHCR by GitHub Actions)
docker login ghcr.io -u YOUR_GITHUB_USERNAME
docker compose pull
docker compose up -d
docker compose logs -f
```

- API: `http://YOUR_SERVER_IP:8022/docs`
- MLflow: `http://YOUR_SERVER_IP:5000`

To update a served model, refresh the mounted directory (e.g. `dvc pull` then copy into `./model` / `./quantized_model`) and restart the `api` service.

---

## Testing

```bash
poetry run python -m unittest tests/test_predictor.py
```
