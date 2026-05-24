"""Data loading and text preprocessing"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    return df


def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)          # HTML tags hatao
    text = re.sub(r"http\S+", "", text)          # URLs hatao
    text = re.sub(r"[^a-zA-Z]", " ", text)       # Sirf letters rakho
    text = text.lower().split()
    text = [lemmatizer.lemmatize(w) for w in text if w not in STOP_WORDS and len(w) > 2]
    return " ".join(text)


def preprocess_dataset(df):
    print("Cleaning reviews...")
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)  # positive=1, negative=0
    print("Done! Sample cleaned review:")
    print(f"  Original:  {df['review'].iloc[0][:100]}...")
    print(f"  Cleaned:   {df['clean_review'].iloc[0][:100]}...")
    return df
