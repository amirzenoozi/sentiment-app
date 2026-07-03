from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import time
from src.predictor import SentimentPredictor, InvalidLanguageError

app = FastAPI(title="Dutch Movie Review Sentiment Classifier API")

try:
    predictor = SentimentPredictor(model_path="./best_model")
except Exception:
    predictor = SentimentPredictor(model_path="pdelobelle/robbert-v2-dutch-base")

# Optional INT8-quantized ONNX engine, served side-by-side for latency/quality comparison.
# It is loaded only if the ./quantized_model volume is present; otherwise the dedicated
# endpoints return 503 while the primary /classify route keeps serving normally.
try:
    quantized_predictor = SentimentPredictor(model_path="./quantized_model", backend="onnx")
    if not quantized_predictor.transformer_ready:
        quantized_predictor = None
except Exception:
    quantized_predictor = None


class ReviewInput(BaseModel):
    review: str


def _run_prediction(engine, review: str):
    """Run one prediction and return the response body.

    Shared by the primary and quantized routes so both apply identical language
    validation and error semantics. `is_translated` is True when the input was
    translated to Dutch before inference.
    """
    start_time = time.time()
    details = engine.predict_with_details(review)
    latency = time.time() - start_time
    return {
        "label": details["label"],
        "is_translated": details["is_translated"],
        "detected_language": details["detected_language"],
        "latency_seconds": round(latency, 4),
    }


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
        "quantized_ready": quantized_predictor is not None,
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
    """Classify a review with the primary (full-precision PyTorch) engine."""
    try:
        return _run_prediction(predictor, input_data.review)

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


@app.post("/classify/quantized", status_code=status.HTTP_200_OK)
def classify_review_quantized(input_data: ReviewInput):
    """Classify a review with the INT8-quantized ONNX engine (CPU-optimized)."""
    if quantized_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Quantized model is not loaded. Mount a valid ./quantized_model directory.",
        )

    try:
        return _run_prediction(quantized_predictor, input_data.review)

    except InvalidLanguageError as lang_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(lang_err)
        )
    except Exception as general_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(general_err)}"
        )


@app.post("/compare", status_code=status.HTTP_200_OK)
def compare_models(input_data: ReviewInput):
    """Run both engines on the same input and return their labels and latencies side-by-side."""
    if quantized_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Quantized model is not loaded. Mount a valid ./quantized_model directory.",
        )

    try:
        primary = _run_prediction(predictor, input_data.review)
        quantized = _run_prediction(quantized_predictor, input_data.review)

        speedup = None
        if quantized["latency_seconds"] > 0:
            speedup = round(primary["latency_seconds"] / quantized["latency_seconds"], 2)

        return {
            "primary": primary,
            "quantized": quantized,
            "labels_agree": primary["label"] == quantized["label"],
            "speedup_factor": speedup,
        }

    except InvalidLanguageError as lang_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(lang_err)
        )
    except Exception as general_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(general_err)}"
        )