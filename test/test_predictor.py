import unittest
from src.predictor import SentimentPredictor


class TestSentimentPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the predictor using the fallback/base model for safe unit testing environment
        cls.predictor = SentimentPredictor(model_path="pdelobelle/robbert-v2-dutch-base")

    def test_predict_returns_valid_label(self):
        sample_review = "Dit is een redelijk goede film voor de hele familie."
        prediction = self.predictor.predict(sample_review)

        # Verify that output strictly adheres to project specifications
        valid_labels = ["Positive", "Average", "Negative"]
        self.assertIn(prediction, valid_labels)


if __name__ == "__main__":
    unittest.main()