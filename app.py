from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import time
from src.predictor import SentimentPredictor, InvalidLanguageError

app = FastAPI(title="Dutch Movie Review Sentiment Classifier API")

try:
    predictor = SentimentPredictor(model_path="./best_model")
except Exception:
    predictor = SentimentPredictor(model_path="pdelobelle/robbert-v2-dutch-base")


class ReviewInput(BaseModel):
    review: str


@app.get("/health", status_code=status.HTTP_200_OK, tags=["ops"])
def health():
    """Liveness probe: returns 200 whenever the process is running.

    Intentionally does no dependency checks so orchestrators can distinguish
    'process alive' from 'able to serve traffic' (see /ready).
    """
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness probe: 200 only when the service can actually classify.

    The service can serve as long as at least one engine is loaded — the
    primary transformer or the lightweight fallback. If neither loaded we
    return 503 so load balancers hold traffic instead of routing to a
    container that would only emit meaningless defaults.
    """
    transformer_ready = getattr(predictor, "transformer_ready", False)
    fallback_ready = getattr(predictor, "fallback_ready", False)
    can_serve = transformer_ready or fallback_ready

    payload = {
        "ready": can_serve,
        "transformer_ready": transformer_ready,
        "fallback_ready": fallback_ready,
        "device": str(getattr(predictor, "device", "unknown")),
    }

    if not can_serve:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=payload,
        )
    return payload


@app.post("/classify", status_code=status.HTTP_200_OK)
def classify_review(input_data: ReviewInput):
    start_time = time.time()

    try:
        # Generate target sentiment classification label
        label = predictor.predict(input_data.review)
        latency = time.time() - start_time

        return {
            "label": label,
            "latency_seconds": round(latency, 4)
        }

    except InvalidLanguageError as lang_err:
        # Handle invalid language and return HTTP 400 Bad Request to the client
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(lang_err)
        )
    except Exception as general_err:
        # Fallback for any other unexpected application error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(general_err)}"
        )