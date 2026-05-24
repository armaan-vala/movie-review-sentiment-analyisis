"""Visualization functions for sentiment analysis"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def plot_sentiment_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x="sentiment", data=df, hue="sentiment", palette=["#e74c3c", "#2ecc71"], legend=False)
    plt.title("Sentiment Distribution in Dataset")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, "sentiment_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/sentiment_distribution.png")


def plot_review_lengths(df):
    df["review_length"] = df["clean_review"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="review_length", hue="sentiment", bins=50, palette=["#e74c3c", "#2ecc71"])
    plt.title("Review Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.xlim(0, 500)
    plt.savefig(os.path.join(OUTPUT_DIR, "review_lengths.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/review_lengths.png")


def plot_wordclouds(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, sentiment in enumerate(["positive", "negative"]):
        text = " ".join(df[df["sentiment"] == sentiment]["clean_review"])
        color = "Greens" if sentiment == "positive" else "Reds"
        wc = WordCloud(width=800, height=400, background_color="white", colormap=color, max_words=100).generate(text)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"{sentiment.upper()} Reviews - Word Cloud", fontsize=14)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "wordclouds.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/wordclouds.png")


def plot_model_comparison(results):
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accuracies, color=["#3498db", "#e67e22", "#2ecc71"])
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{acc:.2%}", ha="center", fontsize=12)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/model_comparison.png")
