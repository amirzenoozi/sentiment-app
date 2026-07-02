import argparse
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer


def main():
    # Set up the argument parser for CLI execution
    parser = argparse.ArgumentParser(description="CLI Tool for Training the Dutch Sentiment Analysis Model.")

    # Define customizable arguments with defaults
    parser.add_argument( "--data_path", type=str, default="movies_reviews.csv", help="Path to the input CSV dataset file (default: movies_reviews.csv)")
    parser.add_argument( "--epochs", type=int, default=3, help="Number of training epochs for the Transformer model (default: 3)")
    parser.add_argument( "--batch_size", type=int, default=8, help="Batch size for training and evaluation (default: 8)")
    parser.add_argument( "--model_name", type=str, default="pdelobelle/robbert-v2-dutch-base", help="Pre-trained HuggingFace model variant for Dutch (default: pdelobelle/robbert-v2-dutch-base)")

    # Parse command line inputs
    args = parser.parse_args()

    print(f"Starting pipeline with Data: {args.data_path} | Model: {args.model_name}")

    # 1. Initialize data processor with the dynamic dynamic CLI path
    processor = DataProcessor(file_path=args.data_path)
    processor.load_and_filter_data()
    X_train, X_val, y_train, y_val = processor.prepare_datasets()

    # 2. Launch training with customized hyper-parameters
    trainer = ModelTrainer(model_name=args.model_name)
    trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("All processes finished. Check MLflow or './best_model' directory.")


if __name__ == "__main__":
    main()