"""FastAPI app for Movie Review Sentiment Prediction"""

import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "sentiment_model.pkl")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    vectorizer = data["vectorizer"]
    model_name = data["model_name"]


def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(w) for w in text if w not in STOP_WORDS and len(w) > 2]
    return " ".join(text)


app = FastAPI(title="Movie Sentiment Analyzer")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "outputs")), name="static")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Sentiment Analyzer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
        }
        .container {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 90%;
        }
        h1 { text-align: center; margin-bottom: 8px; font-size: 1.8rem; }
        .subtitle { text-align: center; color: #aaa; margin-bottom: 30px; font-size: 0.9rem; }
        textarea {
            width: 100%;
            height: 140px;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.08);
            color: #fff;
            font-size: 1rem;
            resize: vertical;
            outline: none;
        }
        textarea::placeholder { color: #888; }
        textarea:focus { border-color: #7c5cfc; }
        button {
            width: 100%;
            padding: 14px;
            margin-top: 15px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #7c5cfc, #a855f7);
            color: #fff;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.02); }
        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 700;
        }
        .positive {
            background: rgba(34,197,94,0.15);
            border: 1px solid rgba(34,197,94,0.4);
            color: #22c55e;
        }
        .negative {
            background: rgba(239,68,68,0.15);
            border: 1px solid rgba(239,68,68,0.4);
            color: #ef4444;
        }
        .model-info { text-align: center; margin-top: 10px; color: #888; font-size: 0.8rem; }
        .footer { text-align: center; margin-top: 25px; color: #555; font-size: 0.75rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Movie Sentiment Analyzer</h1>
        <p class="subtitle">Paste a movie review and find out if it's positive or negative</p>
        <form method="post" action="/predict">
            <textarea name="review" placeholder="Type or paste a movie review here... &#10;&#10;Example: This movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat." required>{{review_text}}</textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>
        {{result_html}}
        <p class="model-info">Powered by {{model_name}} trained on 50K IMDB reviews</p>
        <p class="footer">Built by Armaan Vala</p>
    </div>
</body>
</html>
"""


def render_page(review_text="", result_html=""):
    return HTML_PAGE.replace("{{review_text}}", review_text).replace(
        "{{result_html}}", result_html
    ).replace("{{model_name}}", model_name)


@app.get("/", response_class=HTMLResponse)
async def home():
    return render_page()


@app.post("/predict", response_class=HTMLResponse)
async def predict(review: str = Form(...)):
    cleaned = clean_text(review)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    if prediction == 1:
        sentiment = "POSITIVE"
        css_class = "positive"
        emoji = "&#127775;"
    else:
        sentiment = "NEGATIVE"
        css_class = "negative"
        emoji = "&#128078;"

    result_html = f'<div class="result {css_class}">{emoji} {sentiment}</div>'
    return render_page(review_text=review, result_html=result_html)


@app.get("/api/predict")
async def api_predict(review: str):
    cleaned = clean_text(review)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return {
        "review": review,
        "sentiment": "POSITIVE" if prediction == 1 else "NEGATIVE",
        "model": model_name,
    }
