"""Predict sentiment for new reviews"""

from src.preprocess import clean_text
from src.model import load_model


def predict_sentiment(review_text):
    model, vectorizer, model_name = load_model()
    cleaned = clean_text(review_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    return sentiment, model_name


def interactive_mode():
    print("\n" + "=" * 50)
    print("  Movie Review Sentiment Predictor")
    print("=" * 50)
    print("Type a movie review and get sentiment prediction.")
    print("Type 'quit' to exit.\n")

    while True:
        review = input("Enter review: ").strip()
        if review.lower() == "quit":
            print("Bye!")
            break
        if not review:
            continue
        sentiment, model_name = predict_sentiment(review)
        print(f"  -> {sentiment} (by {model_name})\n")
