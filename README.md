# Dutch Movie Review Sentiment Classifier

Classifies Dutch movie reviews as **Positive**, **Average**, or **Negative**.

- **Primary model:** fine-tuned `pdelobelle/robbert-v2-dutch-base` (RobBERT).
- **Fallback:** a TF-IDF + Logistic Regression baseline, used automatically if the transformer fails to load or errors at runtime.
- **Quantized model:** an INT8 ONNX build of the primary model for faster, lighter CPU inference, served on its own endpoint for comparison.
- **Language handling:** non-Dutch input is auto-detected; supported languages (`en`, `de`, `it`) are translated to Dutch via a LibreTranslate microservice, anything else is rejected with HTTP 400.

Data and model artifacts are versioned with **DVC** (stored on an SSH remote); the app is packaged with **Docker Compose** and images are published to GHCR via GitHub Actions.

---

## My Assumptions

- The primary model is fine-tuned on Dutch movie reviews using [RobBERT](https://huggingface.co/pdelobelle/robbert-v2-dutch-base).
- The fallback model is a TF-IDF + Logistic Regression baseline in case the RobBERT model fails to load or errors at runtime.
- Input is a single review per request; batching is out of scope. Non-Dutch text is only supported for the languages the translator handles (e.g. en/de/it) — anything else returns HTTP 400.
  - No need to handle a multi-language in batch review.
  - Keep the latency low for single-review classification.
- We might have some reviews from other languages, especially in the production environment when the system intract with real users. 
  - English is added because it's one of the most popular languages.
  - German is added because it's really close to Dutch.
  - Italian is added because it's for my tests to see how system will be act when we have completely different language.
- Model artifacts (`best_model`, `quantized_model`) are provided at runtime via mounted volumes / DVC, not baked into the image. Readiness reflects the core classification path, not the translation service.
- The dataset is heavily imbalanced (Negative ~6%), handled at training time with class-weighted + contrastive loss rather than resampling. The quantized ONNX model targets x86 CPU servers (avx2 profile).

## Suggested Future Work

- Balance the dataset by resampling Negative reviews.
- Support batch classification to improve throughput for bulk workloads. This would complement the current single-review latency focus.
- Reduce class imbalance via filtered back-translation augmentation (see `augment_cli.py`), A/B-tested in MLflow.
- Add production monitoring for input drift and per-class accuracy, plus a readiness check and retries for the translation service. This closes the loop between deployment and model quality.
- In the real-world scenario, we might need to handle the load according to the number of user that we might have, in that case, it's better to put a Queue manager like Redis or RabbitMQ to handle to load over the API.
- Add a Caching system can be a good idea for the future and after checking the traffic of the service, for instance if we have repetitive requests for the same review, it can be a good idea to cache the results.
- Add a validation monitoring to check the model quality for the live system by validation 1-2 percent of the real-data which are validate using a offline GPT model.


## Need an Investigation for Future
Since the accuracy of the main RobBERT model for [Dutch Book Reviews Dataset](https://github.com/benjaminvdb/DBRD) is **95.1 percent**, I think it is a good idea to use the Binary Classification instead of the Multi-Class Classification.
Then we can handle the `Average` class using a software solution by setting a threshold value on the prediction confidence score, for example everthing under 0.55 will be considered as `Average`.
In that case, theoritacoly we can boost the accuracy of the model much faster than the data Augmentation or data Resampling.


## Limitation
- I have a MacOS with M2 chip, and training the model using this chip is take a long time.
- My Server is Linux but it also has some selfhosted applications running, means it can affect the latency of the model in the prediction.
- I  tried to create a better version of the dataset using `augment_cli.py` but it due to a hardware limitations, it takes a lot to run and generate the new verison of the dataset. 

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

Output goes to `quantized_model/` (`model_quantized.onnx` + tokenizer). The API serves it via `/v2/classify`.

---

## Running the API

```bash
poetry run uvicorn app:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs`

### Endpoints

Classification routes are **versioned by the model that serves them**: `/v1` is the
full-precision PyTorch model, `/v2` is the INT8 ONNX (quantized) model. This lets both
models run side-by-side under stable URLs, so a client can pin to `/v1` or migrate to the
faster `/v2` without either path changing meaning. The ops probes (`/health`, `/ready`)
are unversioned — they describe the process, not a model.

| Method | Path             | Description                                            |
|--------|------------------|--------------------------------------------------------|
| POST   | `/v1/classify`   | Classify with the primary (PyTorch) model              |
| POST   | `/v2/classify`   | Classify with the INT8 ONNX model (503 if not loaded)  |
| POST   | `/v2/compare`    | Run both models, return labels, latencies, and speedup |
| GET    | `/health`        | Liveness probe                                         |
| GET    | `/ready`         | Readiness probe (reports which engines are loaded)     |

**Example**

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"review": "Dit is een absolute topfilm! Geweldig acteerwerk."}'
# -> {"label": "Positive", "score": 0.98, "is_translated": false,
#     "detected_language": "nl", "latency_seconds": 0.045}
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

---

## Load & Performance

Measured with `stress_test.py` against the deployed service (Dutch-only payloads, single
shared CPU server). Each row raises the number of concurrent clients and the request count
(up to 2,000). **Concurrency** = how many requests are in flight simultaneously (not the
total). Latency percentiles:

- **p50** (median): half the requests were at least this fast — the "typical" experience.
- **p95**: 95% finished within this; only the slowest 5% were worse.
- **p99**: 99% finished within this; only the worst 1% took longer (the "tail").

```bash
poetry run python stress_test.py --host https://app365.amirdouzandeh.me --endpoint /classify -n 2000 -c 50
```

**Primary model** (`/v1/classify`, full-precision PyTorch):

| Concurrency | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Success |
|------------:|---------:|-------------------:|---------:|---------:|---------:|:-------:|
| 1           | 200      | 6.4                | 155      | 184      | 207      | 100%    |
| 5           | 500      | 10.2               | 481      | 604      | 697      | 100%    |
| 10          | 1000     | 10.9               | 915      | 1073     | 1134     | 100%    |
| 25          | 1500     | 10.3               | 2396     | 2787     | 3000     | 100%    |
| 50          | 2000     | 10.0               | 4917     | 5790     | 6400     | 100%    |

**Quantized model** (`/v2/classify`, INT8 ONNX):

| Concurrency | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Success |
|------------:|---------:|-------------------:|---------:|---------:|---------:|:-------:|
| 1           | 200      | 12.4               | 80       | 94       | 99       | 100%    |
| 5           | 500      | 28.1               | 175      | 233      | 273      | 100%    |
| 10          | 1000     | 30.4               | 326      | 441      | 511      | 100%    |
| 25          | 1500     | 31.0               | 798      | 999      | 1101     | 100%    |
| 50          | 2000     | 31.9               | 1545     | 1776     | 1891     | 100%    |

**Takeaway:** each model's throughput saturates at a CPU-bound ceiling — ~10 req/s for the
primary model, ~31 req/s for the quantized one (~3× higher, at roughly half the single-request
latency). Beyond that ceiling extra load only queues, so latency grows linearly with
concurrency (`latency ≈ concurrency ÷ throughput`) while throughput stays flat. No requests
fail even at 50 concurrent / 2,000 requests — the service degrades by slowing down, not
dropping. To scale further, run multiple API workers/replicas rather than tuning a single
instance.
