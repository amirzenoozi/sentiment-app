import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Handles data loading, filtering for Dutch language, and train-test splitting.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_and_filter_data(self, language_column: str = "language", text_column: str = "Reviews") -> pd.DataFrame:
        # Load the CSV dataset
        self.df = pd.read_csv(self.file_path)

        # Filter strictly for Dutch language reviews
        if language_column in self.df.columns:
            self.df = self.df[self.df[language_column].str.lower() == "nl"]

        # Drop rows with missing text or target labels
        self.df = self.df.dropna(subset=[text_column, "Label"])
        return self.df

    def prepare_datasets(self, text_column: str = "Reviews", label_column: str = "Label", test_size: float = 0.2):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_filter_data() first.")

        # Map categorical labels to integer labels (Negative: 0, Average: 1, Positive: 2)
        label_mapping = {"Negative": 0, "Average": 1, "Positive": 2}
        self.df["encoded_label"] = self.df[label_column].map(label_mapping)

        X = self.df[text_column].tolist()
        y = self.df["encoded_label"].tolist()

        # Split into training (80%) and validation (20%) sets with a fixed seed.
        # stratify=y keeps each class's proportion identical across train/val,
        # which stabilizes both training and the reliability of eval metrics.
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        return X_train, X_val, y_train, y_val