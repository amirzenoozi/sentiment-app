import os
import unittest
from unittest.mock import patch, MagicMock
from src.predictor import SentimentPredictor, InvalidLanguageError


class TestSentimentPredictorMocked(unittest.TestCase):
    """
    Unit test suite utilizing mocking techniques to isolate business logic,
    language validation, and fallback routing from heavy model weights and network dependencies.
    """

    @patch('src.predictor.AutoModelForSequenceClassification')
    @patch('src.predictor.AutoTokenizer')
    @patch('src.predictor.joblib.load')
    def setUp(self, mock_joblib_load, mock_tokenizer, mock_transformer):
        """Sets up the predictor environment by mocking heavy file system and model initializations."""
        # Mocking the primary Transformer loading phase
        self.mock_transformer_instance = MagicMock()
        mock_transformer.from_pretrained.return_value = self.mock_transformer_instance
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        # Mocking the static fallback pipeline loading phase
        self.mock_fallback_instance = MagicMock()
        mock_joblib_load.return_value = self.mock_fallback_instance

        # Force mock environment variables for test reproducibility
        os.environ["SUPPORTED_TRANSLATION_LANGS"] = "en,fr,de"
        os.environ["TRANSLATION_API_URL"] = "http://fake-translation-service/translate"

        # Initialize the predictor inside the mocked bubble
        self.predictor = SentimentPredictor(model_path="./fake_path")

    def tearDown(self):
        """Cleans up system environment side effects after tests complete."""
        if "SUPPORTED_TRANSLATION_LANGS" in os.environ:
            del os.environ["SUPPORTED_TRANSLATION_LANGS"]

    def test_native_dutch_bypass_validation(self):
        """Verifies that native Dutch sentences safely bypass the translation validation logic."""
        try:
            self.predictor._validate_language("nl")
        except InvalidLanguageError:
            self.fail("_validate_language raised InvalidLanguageError unexpectedly for native 'nl' code.")

    def test_allowed_foreign_language_validation(self):
        """Ensures registered translation languages (e.g., 'en') pass validation without exceptions."""
        try:
            self.predictor._validate_language("en")
            self.predictor._validate_language("fr")
        except InvalidLanguageError:
            self.fail("Validation failed for a whitelisted foreign language defined in environment variables.")

    def test_unsupported_foreign_language_exception(self):
        """Confirms that languages not listed in the ENV config (e.g., 'zh' for Chinese) raise an InvalidLanguageError."""
        with self.assertRaises(InvalidLanguageError):
            self.predictor._validate_language("zh")

    @patch('src.predictor.requests.post')
    def test_successful_translation_routing(self, mock_post):
        """Validates that a foreign input triggers the microservice translation interface successfully."""
        # Simulate a perfect HTTP 200 response from the LibreTranslate microservice
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"translatedText": "Dit is een geweldige film"}
        mock_post.return_value = mock_response

        translated_text, was_translated = self.predictor._translate_to_dutch("This is a great movie", "en")

        self.assertEqual(translated_text, "Dit is een geweldige film")
        self.assertTrue(was_translated)
        mock_post.assert_called_once()

    @patch('src.predictor.requests.post')
    def test_failed_translation_returns_original(self, mock_post):
        """A failed translation call returns the original text and reports translated=False."""
        mock_post.side_effect = Exception("connection refused")

        text, was_translated = self.predictor._translate_to_dutch("This is a great movie", "en")

        self.assertEqual(text, "This is a great movie")
        self.assertFalse(was_translated)

    def test_empty_string_prediction_default(self):
        """Guarantees that an empty or whitespace-only input safely short-circuits to an 'Average' sentiment ranking."""
        result_empty = self.predictor.predict("")
        result_spaces = self.predictor.predict("   ")

        self.assertEqual(result_empty, "Average")
        self.assertEqual(result_spaces, "Average")

    def test_predict_fallback_mechanism_mapping(self):
        """Validates that the fallback engine maps integer outputs to labels and returns a score."""
        # Force the mocked fallback pipeline to output index 2 (Positive)
        self.mock_fallback_instance.predict.return_value = [2]
        self.mock_fallback_instance.predict_proba.return_value = [[0.1, 0.2, 0.7]]

        label, score = self.predictor._predict_fallback("Some text")
        self.assertEqual(label, "Positive")
        self.assertAlmostEqual(score, 0.7)

        # Force the mocked fallback pipeline to output index 0 (Negative)
        self.mock_fallback_instance.predict.return_value = [0]
        self.mock_fallback_instance.predict_proba.return_value = [[0.8, 0.1, 0.1]]

        label, score = self.predictor._predict_fallback("Some text")
        self.assertEqual(label, "Negative")
        self.assertAlmostEqual(score, 0.8)

    def test_predict_with_details_empty_input(self):
        """Empty input returns the Average default with a null score and no translation."""
        details = self.predictor.predict_with_details("   ")

        self.assertEqual(details["label"], "Average")
        self.assertIsNone(details["score"])
        self.assertFalse(details["is_translated"])
        self.assertIsNone(details["detected_language"])

    def test_predict_returns_label_string(self):
        """predict() keeps its string contract even though the engine now tracks metadata."""
        self.assertIsInstance(self.predictor.predict("   "), str)


if __name__ == '__main__':
    unittest.main()