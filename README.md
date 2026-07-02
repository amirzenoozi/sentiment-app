# Dutch Movie Review Sentiment Classifier (sentiment-app)

An enterprise-grade, modular, and production-ready machine learning pipeline and REST API designed to classify the sentiment of Dutch movie reviews into three distinct categories: **Positive**, **Average**, or **Negative**. 

The architecture features a high-performance **Transformer (RobBERT)** as the primary engine, paired with an automated **lightweight machine learning fallback mechanism** to ensure 100% uptime and high availability under computational constraints or unexpected runtime failures.

---

## 🏗️ Architecture & Core Principles

- **Primary Transformer Engine:** Fine-tuned `pdelobelle/robbert-v2-dutch-base` optimized for native Dutch language semantics.
- **Automated Fallback Pipeline:** A highly efficient `TF-IDF + Logistic Regression` baseline that automatically triggers if the primary Transformer model encounters any hardware limitations, GPU Out-Of-Memory (OOM) situations, or runtime errors.
- **Language Boundary Enforcement:** Automated language detection wrapper via `langdetect` ensuring that inputs violating the Dutch language constraint are filtered at the edge with immediate HTTP 400 Bad Request feedback.
- **Experiment Tracking:** Comprehensive academic metric collection logging Loss, F1-Macro, Precision, and Recall dynamically to an **MLflow** dashboard.
- **Package Management:** Native deterministic dependency locking using **Poetry** to prevent environment drift across development platforms (e.g., Apple Silicon) and production environments (Linux AMD64).
- **CI/CD & Virtualization:** Continuous integration workflows via GitHub Actions publishing directly to GitHub Container Registry (GHCR) paired with a `docker-compose` topology.

---

## 🛠️ Project Structure

```text
sentiment-app/
│
├── .github/
│   └── workflows/
│       └── deploy.yml       # Automated GitHub Actions CI/CD pipeline
│
├── src/
│   ├── __init__.py          # Package initialization marker
│   ├── data_processor.py    # Data ingest, filtering (NL language column), and parsing
│   ├── model_trainer.py     # MLflow logging orchestration, RobBERT & Baseline training loops
│   └── predictor.py         # Thread-safe hybrid inference engine with validation wrappers
│
├── tests/
│   ├── __init__.py          # Test suite marker
│   └── test_predictor.py    # Suite testing baseline predictability constraints
│
├── app.py                   # FastAPI application layer exposing the REST schema
├── Dockerfile               # Production multi-stage Docker configuration
├── docker-compose.yml       # Local production orchestration engine (API + MLflow)
├── pyproject.toml           # Poetry manifest declaring dependency trees
└── train_pipeline.py        # CLI entrypoint orchestrating dynamic model preparation
```

## 🚀 Local Development Setup
1. Prerequisite Environment (Apple Silicon / Linux)
Ensure that you are running a stable Python runtime. It is highly recommended to use pyenv to lock the environment to Python 3.12:

```bash
# Install and lock Python 3.12 locally inside the project directory
pyenv install 3.12.8
pyenv local 3.12.8

# Verify your terminal session points to Python 3.12
python --version
```

2. Dependency Management via Poetry
Configure Poetry to prioritize your active isolated Python runtime and install the dependency tree:

```bash
# Tell Poetry to always respect the local pyenv environment wrapper
poetry config virtualenvs.prefer-active-python true

# Force Poetry to build the virtual environment using the explicit Python 3.12 path
poetry env use python3.12

# Install all locked production and evaluation dependencies
poetry install
```

## 🏋️ CLI Training Pipeline
The training framework is wrapped in a robust Command Line Interface (CLI) allowing you to override hyper-parameters dynamically without mutating the code surface.
Execution Commands:

```bash
# View all available parameters and configurations via the help manual
poetry run python train_pipeline.py --help

# Run the training pipeline using a custom file path (e.g., data/my_reviews.csv)
poetry run python train_pipeline.py --data_path data/dutch_sentences.csv

# Run an intensive training cycle overriding default Epochs and Batch Sizes
poetry run python train_pipeline.py --data_path data/dutch_sentences.csv --epochs 5 --batch_size 16
```

## Experiment Monitoring (MLflow Server)
Launch the telemetry tracking interface locally to monitor loss curves, precision, and F1-macro matrices:

```bash
# Boot the MLflow telemetry user interface locally
poetry run mlflow ui
```

Once active, navigate your browser to: `http://localhost:5000`


## 🧪 Evaluation & Unit Testing
Validate the inference layer and guarantee that predictable label sets (Positive, Average, Negative) are properly bound by executing the test runner:

```bash
# Run the complete test suite against your local predictor implementations
poetry run python -m unittest tests/test_predictor.py
```

## 🌐 Running the Web API Locally
Boot the asynchronous FastAPI application layer locally on your workstation:

```bash
# Run the local development server with live reload enabled
poetry run uvicorn app:app --reload --port 8000
```

Access the interactive OpenAPI Swagger documentation by navigating to: `http://localhost:8000/docs`

### API Validation Examples
1. Valid Evaluation Request (Dutch)
```bash
curl -X 'POST' \
  'http://localhost:8000/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "review": "Dit is een absolute topfilm! Geweldig acteerwerk."
}'
```

Expected Response (HTTP 200 OK):
```json
{
  "label": "Positive",
  "latency_seconds": 0.0452
}
```

2. Invalid Evaluation Request (Non-Dutch Violation)
```bash
curl -X 'POST' \
  'http://localhost:8000/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "review": "This movie was incredibly boring and a waste of time."
}'
```

Expected Response (HTTP 400 Bad Request):
```json
{
  "detail": "Input language detected as 'en', but only Dutch ('nl') is supported."
}
```

## 🐳 Production Server Deployment (Docker Compose)
The system utilizes an enterprise-grade dual-container architecture for remote deployment. The app image is automatically built and stored inside the GitHub Container Registry (GHCR) via GitHub Actions.
Deployment Lifecycle on the Linux Server:

Step 1: Remote Container Registry Authentication
Generate a GitHub Personal Access Token (classic) with read:packages scope, then execute authentication on the Linux production host terminal:

```bash
docker login ghcr.io -u YOUR_GITHUB_USERNAME
```

Step 2: Provision Infrastructure Configurations
Create the docker-compose.yml file on your server host. Pull down the newly updated image architectures from the upstream container clouds and spin up the production infrastructure:

```bash
# Force a pull sequence downloading the latest production images from GHCR
docker compose pull

# Spin up the dual-container topology in detached (background) mode
docker compose up -d

# Check live orchestration health logs
docker compose logs -f
```

Server Endpoint Map:
- FastAPI Core Gateway: `http://YOUR_SERVER_IP:8000/docs`
- Centralized MLflow Monitor Console: `http://YOUR_SERVER_IP:5000`