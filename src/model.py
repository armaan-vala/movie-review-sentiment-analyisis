"""Model training and evaluation"""

import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def prepare_data(df):
    X = df["clean_review"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(max_iter=2000),
    }

    results = {}
    best_accuracy = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])

        results[name] = {"accuracy": acc, "report": report}
        print(f"  Accuracy: {acc:.4f}")
        print(report)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print(f"\nBest Model: {best_name} ({best_accuracy:.4f})")
    return results, best_model, best_name


def save_model(model, vectorizer, model_name):
    data = {"model": model, "vectorizer": vectorizer, "model_name": model_name}
    path = os.path.join(OUTPUT_DIR, "sentiment_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Model saved: outputs/sentiment_model.pkl")


def load_model():
    path = os.path.join(OUTPUT_DIR, "sentiment_model.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["vectorizer"], data["model_name"]
