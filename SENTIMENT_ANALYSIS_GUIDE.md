# Sentiment Analysis - Complete Learning Guide

## 1. Sentiment Analysis Kya Hai?

Sentiment Analysis ek Natural Language Processing (NLP) technique hai jisme hum text se **emotion/opinion** nikaalte hain — positive, negative, ya neutral.

**Real-life examples:**
- Amazon reviews se pata karna ki product accha hai ya bura
- Twitter pe kisi topic pe logon ka reaction jaanna
- YouTube comments se video ki public opinion samajhna

---

## 2. Sentiment Analysis Ke Types

### Type 1: Fine-Grained Sentiment Analysis
Text ko detailed categories me classify karta hai:
- Very Positive
- Positive
- Neutral
- Negative
- Very Negative

**Example:** Movie review rating 1-5 stars predict karna.

### Type 2: Emotion Detection
Specific emotions detect karta hai:
- Happy, Sad, Angry, Fear, Surprise, Disgust

**Example:** Customer support messages me frustration detect karna.

### Type 3: Aspect-Based Sentiment Analysis
Ek sentence me alag-alag cheezon ka alag sentiment nikalta hai.

**Example:**
> "Phone ki camera bahut acchi hai but battery bahut jaldi khatam hoti hai"
- Camera -> Positive
- Battery -> Negative

### Type 4: Intent-Based Analysis
User ka intention samajhna — complaint hai, suggestion hai, ya praise hai.

### Type 5: Multilingual Sentiment Analysis
Multiple languages me sentiment detect karna (Hindi, English, Hinglish, etc.)

---

## 3. Sentiment Analysis Kaise Kaam Karta Hai?

### Approach 1: Rule-Based (Lexicon-Based)
- Ek dictionary hoti hai jisme har word ka score hota hai
- "good" = +1, "bad" = -1, "excellent" = +2
- Saare words ke scores add karke final sentiment nikaalte hain

**Pros:** Simple, fast, no training needed
**Cons:** Sarcasm nahi samajhta, context miss karta hai

### Approach 2: Machine Learning Based
- Labelled data se model train karte hain
- Algorithms: Naive Bayes, SVM, Logistic Regression, Random Forest
- Model khud features seekhta hai

**Pros:** Better accuracy, context samajhta hai
**Cons:** Labelled data chahiye, training time lagta hai

### Approach 3: Deep Learning Based
- Neural networks use karte hain (LSTM, BERT, Transformers)
- Bahut zyada accurate results
- Large datasets pe best kaam karta hai

**Pros:** State-of-the-art accuracy, complex patterns samajhta hai
**Cons:** Heavy computation, bahut data chahiye

### Approach 4: Pre-trained Models (Sabse Easy)
- Already trained models use karte hain (VADER, TextBlob, HuggingFace)
- Bas import karo aur use karo
- Beginners ke liye best starting point

---

## 4. Essential Tools & Libraries

### Python Libraries

| Library | Kya Karta Hai | Difficulty |
|---------|---------------|------------|
| **TextBlob** | Simple sentiment analysis | Beginner |
| **VADER** | Social media text ke liye best | Beginner |
| **NLTK** | Full NLP toolkit | Beginner-Intermediate |
| **spaCy** | Industrial-strength NLP | Intermediate |
| **scikit-learn** | ML models banana | Intermediate |
| **HuggingFace Transformers** | Pre-trained deep learning models | Intermediate-Advanced |
| **TensorFlow / PyTorch** | Custom deep learning models | Advanced |

### Data Collection Tools

| Tool | Kya Karta Hai |
|------|---------------|
| **YouTube Data API** | YouTube comments fetch karna |
| **Tweepy** | Twitter data collect karna |
| **BeautifulSoup** | Web scraping |
| **Selenium** | Dynamic websites se data lena |
| **PRAW** | Reddit data collect karna |

---

## 5. Step-by-Step Learning Path

### Phase 1: Python Basics (1-2 weeks)
Agar Python nahi aati to pehle ye seekho:
- Variables, loops, functions
- Lists, dictionaries
- File handling
- pip se libraries install karna

### Phase 2: Basic NLP Concepts (1 week)
```
Ye concepts samajhna zaroori hai:
- Tokenization: Sentence ko words me todna
- Stop Words: "is", "the", "a" jaise common words hatana
- Stemming: "running" -> "run" banana
- Lemmatization: Words ko base form me laana
- Bag of Words: Text ko numbers me convert karna
- TF-IDF: Important words ko zyada weight dena
```

### Phase 3: Sentiment Analysis with Pre-trained Tools (1 week)

#### Step 1: TextBlob se start karo
```python
from textblob import TextBlob

text = "I love this product! It's amazing"
blob = TextBlob(text)

print(blob.sentiment)
# Output: Sentiment(polarity=0.625, subjectivity=0.6)
# polarity: -1 (negative) to +1 (positive)
# subjectivity: 0 (objective) to 1 (subjective)
```

#### Step 2: VADER seekho (Social Media ke liye best)
```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
text = "This movie was absolutely fantastic! 🔥🔥"

scores = sia.polarity_scores(text)
print(scores)
# Output: {'neg': 0.0, 'neu': 0.38, 'pos': 0.62, 'compound': 0.74}
# compound > 0.05  -> Positive
# compound < -0.05 -> Negative
# else             -> Neutral
```

#### Step 3: HuggingFace Transformers
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I really enjoyed this video!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Phase 4: Machine Learning Approach (2 weeks)
```
1. Dataset load karo (CSV file with text + label)
2. Text preprocessing karo (cleaning, tokenization)
3. Text ko numbers me convert karo (TF-IDF / CountVectorizer)
4. Model train karo (Naive Bayes / Logistic Regression)
5. Model test karo (accuracy, precision, recall)
6. Predictions karo
```

```python
# Basic ML Pipeline Example
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Data split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Text ko numbers me convert karo
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model train karo
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Accuracy check karo
accuracy = model.score(X_test_tfidf, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### Phase 5: Deep Learning (Advanced - 2-3 weeks)
- LSTM / GRU models
- BERT fine-tuning
- Custom transformer models

---

## 6. Project Ideas (Easy to Advanced)

### Project 1: Basic - Movie Review Sentiment Analyzer
**Difficulty:** Beginner
**Dataset:** IMDB Movie Reviews (Kaggle se download)
**Goal:** Reviews ko positive/negative classify karna
**Tools:** TextBlob / VADER + Python
```
Steps:
1. Kaggle se IMDB dataset download karo
2. CSV load karo pandas se
3. TextBlob/VADER se har review ka sentiment nikalo
4. Accuracy calculate karo
5. Results visualize karo (matplotlib/seaborn)
```

### Project 2: Intermediate - YouTube Comments Sentiment Analyzer ⭐ (RECOMMENDED)
**Difficulty:** Intermediate
**Goal:** Kisi bhi YouTube video ke comments ka sentiment analyze karna
**Tools:** YouTube Data API + VADER/TextBlob + Matplotlib

```
Steps:
1. Google Cloud Console pe jaake YouTube Data API v3 enable karo
2. API key generate karo
3. Video ID se comments fetch karo (API se)
4. Comments ko clean karo (emojis, links, special chars hatao)
5. Har comment ka sentiment nikalo
6. Results dikhao:
   - Pie chart: Positive vs Negative vs Neutral percentage
   - Word cloud: Most common positive/negative words
   - Time-based analysis: Sentiment over time
   - Top positive aur top negative comments
```

**YouTube API se comments fetch karne ka basic code:**
```python
from googleapiclient.discovery import build

API_KEY = "YOUR_API_KEY"
VIDEO_ID = "dQw4w9WgXcQ"  # Video ID from URL

youtube = build("youtube", "v3", developerKey=API_KEY)

def get_comments(video_id, max_results=100):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()
    
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        likes = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
        comments.append({"text": comment, "likes": likes})
    
    return comments

comments = get_comments(VIDEO_ID)
print(f"Fetched {len(comments)} comments")
```

### Project 3: Intermediate - Product Review Analyzer
**Difficulty:** Intermediate
**Dataset:** Amazon Reviews
**Goal:** Product reviews me aspect-based sentiment nikalna
**Tools:** spaCy + scikit-learn

### Project 4: Advanced - Real-time Twitter Sentiment Dashboard
**Difficulty:** Advanced
**Goal:** Kisi topic pe live tweets ka sentiment track karna
**Tools:** Tweepy + Transformers + Streamlit/Dash

### Project 5: Advanced - Hinglish Sentiment Analyzer
**Difficulty:** Advanced
**Goal:** Hindi-English mixed text ka sentiment analyze karna
**Tools:** HuggingFace multilingual models + Custom dataset

---

## 7. YouTube Comments Project - Detailed Roadmap

Ye project recommend karta hun kyunki:
- Real-world data use hota hai
- API integration seekhoge
- Data cleaning seekhoge
- Visualization seekhoge
- Portfolio me accha lagega

### Folder Structure
```
sentiment-analysis/
├── config/
│   └── config.py          # API keys and settings
├── data/
│   └── comments.csv       # Fetched comments stored here
├── src/
│   ├── fetch_comments.py  # YouTube API se comments laana
│   ├── preprocess.py      # Text cleaning and preprocessing
│   ├── analyze.py         # Sentiment analysis logic
│   └── visualize.py       # Charts and graphs
├── notebooks/
│   └── exploration.ipynb  # Jupyter notebook for experimentation
├── requirements.txt       # Project dependencies
├── main.py                # Main entry point
└── README.md              # Project documentation
```

### Requirements
```
google-api-python-client
nltk
textblob
vaderSentiment
pandas
matplotlib
seaborn
wordcloud
```

### Complete Pipeline
```
[YouTube Video URL]
        |
        v
[Fetch Comments via API]
        |
        v
[Store in CSV/DataFrame]
        |
        v
[Clean & Preprocess Text]
  - Remove URLs, emojis, special chars
  - Lowercase
  - Remove stop words
        |
        v
[Sentiment Analysis]
  - VADER for social media text
  - TextBlob for comparison
  - (Optional) HuggingFace for accuracy
        |
        v
[Visualization & Insights]
  - Pie chart (pos/neg/neutral %)
  - Bar chart (sentiment distribution)
  - Word cloud
  - Top comments by sentiment
  - Sentiment vs Likes correlation
```

---

## 8. Useful Datasets for Practice

| Dataset | Description | Link |
|---------|-------------|------|
| IMDB Reviews | 50K movie reviews | Kaggle |
| Twitter Sentiment | Tweets with labels | Kaggle |
| Amazon Reviews | Product reviews | Kaggle |
| Yelp Reviews | Restaurant reviews | Kaggle |
| Reddit Comments | Subreddit comments | Kaggle |

---

## 9. Common Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Sarcasm detection | Advanced models (BERT) use karo |
| Emojis handling | Emoji library se text me convert karo |
| Hinglish text | Multilingual models (mBERT, XLM-R) |
| Imbalanced data | Oversampling / class weights use karo |
| Negation ("not good") | VADER handles this well |
| Short text (comments) | VADER > TextBlob for short text |

---

## 10. Next Steps - Kaise Shuru Karein?

### Week 1-2: Basics
- [ ] Python basics strong karo
- [ ] pip install textblob nltk pandas matplotlib
- [ ] TextBlob aur VADER se simple examples run karo
- [ ] IMDB dataset pe basic sentiment analysis karo

### Week 3-4: YouTube Project
- [ ] Google Cloud Console pe account banao
- [ ] YouTube Data API enable karo aur API key lo
- [ ] Comments fetch karne ka code likho
- [ ] Preprocessing pipeline banao
- [ ] Sentiment analysis run karo
- [ ] Visualizations banao

### Week 5-6: Improve
- [ ] ML approach try karo (Naive Bayes, Logistic Regression)
- [ ] HuggingFace transformers try karo
- [ ] Streamlit se web app banao
- [ ] GitHub pe project upload karo

---

> **Tip:** Pehle Phase 1-3 pe focus karo. Simple tools (TextBlob, VADER) se start karo,
> phir gradually advanced cheezein seekhna. Overthink mat karo, code likhna shuru karo! 🚀
