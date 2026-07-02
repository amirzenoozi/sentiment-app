import torch
import joblib
import logging
import os
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, DetectorFactory

# Ensure reproducible language detection results
DetectorFactory.seed = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvalidLanguageError(Exception):
    """Custom exception raised when the input text is not in Dutch."""
    pass


class SentimentPredictor:
    """
    Handles robust, cross-platform inference with automated static fallback triggers
    and incoming language validation checks.
    """

    def __init__(self, model_path: str = "./best_model"):
        self.labels = {0: "Negative", 1: "Average", 2: "Positive"}

        # Get translation endpoint from environment variables (configured in docker-compose)
        self.translation_url = os.getenv("TRANSLATION_API_URL", "http://localhost:5001/translate")

        # Parse the comma-separated language strings from Environment Variables
        env_langs = os.getenv("SUPPORTED_TRANSLATION_LANGS", "en,de,it")
        self.supported_langs = [lang.strip().lower() for lang in env_langs.split(",") if lang.strip()]
        logger.info(f"Loaded allowed non-Dutch translation languages from ENV: {self.supported_langs}")

        # Device selection supporting Apple Silicon (MPS), NVIDIA (CUDA), and CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 1. Attempt loading the main Transformer model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.transformer_ready = True
            logger.info(f"Primary Transformer loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
            self.transformer_ready = False

        # 2. Attempt loading the fallback model
        try:
            self.fallback_model = joblib.load(f"{model_path}/fallback_model.joblib")
            self.fallback_ready = True
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            self.fallback_ready = False


    def _translate_to_dutch(self, text: str, source_lang: str) -> str:
        try:
            payload = {
                "q": text,
                "source": source_lang,
                "target": "nl",
                "format": "text"
            }
            response = requests.post(self.translation_url, json=payload, timeout=3.0)
            if response.status_code == 200:
                translated_text = response.json().get("translatedText", text)
                logger.info(f"Microservice Translation successful: {source_lang} -> nl")
                return translated_text
        except Exception as e:
            logger.warning(f"Translation microservice connection error: {e}")
        return text


    def _validate_language(self, detected_lang: str):
        """
        Validates the incoming language code against allowed global systems.
        Raises InvalidLanguageError if the language cannot be classified or translated.
        """
        if detected_lang == "nl":
            return  # Natively supported language bypasses further checks

        # Check if the foreign language is explicitly registered in our ENV variables whitelist
        if detected_lang not in self.supported_langs:
            raise InvalidLanguageError(
                f"Input language '{detected_lang}' is not supported. "
                f"Supported fallback translation languages are: {self.supported_langs}"
            )


    def _predict_fallback(self, text: str) -> str:
        if not self.fallback_ready:
            return "Average"
        prediction_id = self.fallback_model.predict([text])[0]
        return self.labels[prediction_id]

    def predict(self, text: str) -> str:
        if not text.strip():
            return "Average"

        # Step 1: Detect incoming text language wrapper at the edge
        try:
            detected_lang = detect(text)
        except Exception as detection_err:
            logger.warning(f"Langdetect failed: {detection_err}. Forcing default native routing.")
            detected_lang = "nl"

        # Step 2: Enforce system architectural constraints and environment rules
        self._validate_language(detected_lang)

        # Step 3: Conditional dynamic routing through translation if text is foreign
        if detected_lang != "nl":
            text = self._translate_to_dutch(text, detected_lang)

        # Step 4: Core inference execution block with automated recovery triggers
        if not self.transformer_ready:
            return self._predict_fallback(text)

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
            return self.labels[predicted_class_id]
        except Exception as runtime_error:
            logger.warning(f"Primary Transformer runtime failure: {runtime_error}. Switching to fallback model.")
            return self._predict_fallback(text)