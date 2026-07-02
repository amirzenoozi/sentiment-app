FROM python:3.12-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry config cache-dir /tmp/poetry_cache \
    && poetry install --no-root \
    && rm -rf /tmp/poetry_cache ~/.cache/pip

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]