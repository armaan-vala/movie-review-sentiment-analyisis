# Movie Review Sentiment Analyzer

A machine learning web app that predicts whether a movie review is **positive** or **negative**. Built using the IMDB 50K dataset, trained 3 ML models (Naive Bayes, Logistic Regression, Linear SVM), picked the best one (Linear SVM), and deployed it as a web app using FastAPI on Render.

**Live Demo:** [https://movie-review-analyisis.onrender.com](https://movie-review-analyisis.onrender.com)

---

## Dataset

- **IMDB 50K Movie Reviews** — 25K positive + 25K negative (balanced)
- Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Models Used

| Model | Description |
|-------|-------------|
| Naive Bayes | Probability-based classifier |
| Logistic Regression | Weight-based linear classifier |
| **Linear SVM** | Best accuracy — used in deployment |

Text features extracted using **TF-IDF Vectorizer** (50K features, unigrams + bigrams).

## Tech Stack

- **Python** — core language
- **scikit-learn** — ML models + TF-IDF
- **NLTK** — text preprocessing (stopwords, lemmatization)
- **FastAPI + Uvicorn** — web app and API
- **Render** — deployment (free tier)
- **Matplotlib / Seaborn / WordCloud** — visualizations

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/armaan-vala/movie-review-sentiment-analyisis.git
cd movie-review-sentiment-analyisis

# Install dependencies
pip install -r requirements.txt

# Train the model (requires IMDB Dataset.csv in root folder)
python main.py

# Run the web app
uvicorn app:app --reload

# Open in browser
# http://localhost:8000
```

## Project Structure

```
├── main.py              # Train models and save best one
├── app.py               # FastAPI web app
├── requirements.txt     # Dependencies
├── src/
│   ├── preprocess.py    # Text cleaning (HTML, stopwords, lemmatization)
│   ├── model.py         # TF-IDF + train 3 models + save/load pickle
│   ├── visualize.py     # Charts (distribution, wordclouds, comparison)
│   └── predict.py       # CLI prediction mode
└── outputs/
    ├── sentiment_model.pkl   # Saved best model
    └── *.png                 # Generated charts
```

## Author

**Armaan Vala** — [GitHub](https://github.com/armaan-vala)
