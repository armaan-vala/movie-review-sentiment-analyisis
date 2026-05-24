"""
Movie Review Sentiment Analysis
================================
IMDB Dataset (50K reviews) pe sentiment analysis
3 ML models train karke best model save karta hai
"""

import os
import sys
from src.preprocess import load_data, preprocess_dataset
from src.visualize import (
    plot_sentiment_distribution,
    plot_review_lengths,
    plot_wordclouds,
    plot_model_comparison,
)
from src.model import prepare_data, train_and_evaluate, save_model
from src.predict import interactive_mode

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IMDB Dataset.csv")


def train():
    print("=" * 50)
    print("  MOVIE REVIEW SENTIMENT ANALYSIS")
    print("=" * 50)

    print("\n[Step 1/5] Loading dataset...")
    df = load_data(DATASET_PATH)

    print("\n[Step 2/5] Generating visualizations...")
    plot_sentiment_distribution(df)

    print("\n[Step 3/5] Preprocessing text...")
    df = preprocess_dataset(df)
    plot_review_lengths(df)
    plot_wordclouds(df)

    print("\n[Step 4/5] Training models...")
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)
    results, best_model, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)
    plot_model_comparison(results)

    print("\n[Step 5/5] Saving best model...")
    save_model(best_model, vectorizer, best_name)

    print("\n" + "=" * 50)
    print("  ALL DONE!")
    print("=" * 50)
    print("Check 'outputs/' folder for all charts and saved model.")


if __name__ == "__main__":
    if "--predict" in sys.argv:
        interactive_mode()
    else:
        train()
