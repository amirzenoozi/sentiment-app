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