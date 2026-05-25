# Movie Review Sentiment Analysis — Interview Prep Guide

Ye guide padhne ke baad tujhe project ke andar-bahar sab samajh aa jayega.
Interview me confidently baat kar payega.

---

## 1. PROJECT KA COMPLETE FLOW (Step by Step)

Jab `main.py` run hota hai, ye 5 steps hote hain:

```
CSV Load → Visualize → Preprocess → Train 3 Models → Save Best Model
```

### Step 1: Data Loading (`src/preprocess.py → load_data`)
- `IMDB Dataset.csv` load hota hai — isme 50,000 movie reviews hain
- Har review ke saath ek label hai: "positive" ya "negative"
- Dataset **balanced** hai — 25K positive, 25K negative (ye important hai interview me bolna)

### Step 2: Visualization (`src/visualize.py`)
4 charts banate hain:
- **Sentiment Distribution** — bar chart dikhata hai kitne positive, kitne negative (balanced proof)
- **Review Lengths** — histogram dikhata hai reviews kitne lambe hain
- **Word Clouds** — positive me kaunse words zyada aate hain, negative me kaunse
- **Model Comparison** — teeno models ki accuracy ka bar chart

### Step 3: Text Preprocessing (`src/preprocess.py → clean_text`)
Raw review ko ML-ready banata hai. Ye steps hote hain:

```
Original:  "<br>This movie was AMAZING!!! I loved it :) http://imdb.com"
                          ↓
Step 1: HTML tags hatao    → "This movie was AMAZING!!! I loved it :) http://imdb.com"
Step 2: URLs hatao         → "This movie was AMAZING!!! I loved it :)"
Step 3: Sirf letters rakho → "This movie was AMAZING   I loved it"
Step 4: Lowercase karo     → "this movie was amazing i loved it"
Step 5: Stopwords hatao    → "movie amazing loved"  (the, was, I, it = faaltu words)
Step 6: Lemmatization      → "movie amazing loved"  (loved → love nahi hua kyunki WordNet)
```

**Interview me ye bolna:**
- "Stopwords hataye kyunki 'the', 'is', 'was' jaise words sentiment nahi batate, sirf noise add karte hain"
- "Lemmatization kiya taaki 'running', 'runs', 'ran' sab 'run' ban jaye — isse vocabulary chhoti hoti hai aur model better seekhta hai"

### Step 4: Feature Extraction + Training (`src/model.py`)

**TF-IDF Vectorization:**
- Text ko numbers me convert karta hai (ML models text nahi samajhte, sirf numbers)
- `max_features=50000` — top 50K important words rakhe
- `ngram_range=(1,2)` — single words ("good") AUR word pairs ("not good") dono consider kiye
  - Ye important hai! "not good" ka matlab "good" se bilkul alag hai

**Train-Test Split:**
- 80% data training ke liye (model seekhega)
- 20% data testing ke liye (model ki exam)
- `random_state=42` — har baar same split aaye (reproducibility)

**3 Models train kiye:**

| Model | Kya karta hai | Layman explanation |
|-------|--------------|-------------------|
| **Naive Bayes** | Word frequency se probability calculate karta hai | "agar review me 'terrible' hai toh 80% chance negative hai" |
| **Logistic Regression** | Har word ko ek weight deta hai, sab add karke decide karta hai | "terrible = -5, amazing = +4, total negative toh NEGATIVE" |
| **Linear SVM** | Data ke beech ek line kheenchta hai — ek side positive, dusri negative | "is line ke upar positive, neeche negative" |

### Step 5: Save Best Model (`outputs/sentiment_model.pkl`)
- Jo model sabse zyada accurate hota hai, usse pickle file me save karta hai
- Pickle = Python object ko file me save karna (serialize karna)
- Model + Vectorizer dono save hote hain (dono chahiye prediction ke liye)

---

## 2. HAR TECHNICAL CHOICE KA REASON (Interviewer yahi puchega)

### "TF-IDF kyu use kiya? Bag of Words kyu nahi?"

**Answer:** "Bag of Words sirf count karta hai ki word kitni baar aaya. Par TF-IDF ye bhi dekhta hai ki word kitna RARE hai. Agar 'the' har review me hai toh uski value kam ho jati hai. Par 'masterpiece' kam reviews me hai toh uski value zyada hoti hai. Isse important words ko zyada weight milta hai."

**TF-IDF ka full form:** Term Frequency - Inverse Document Frequency
- **TF** = ye word is review me kitni baar aaya
- **IDF** = ye word kitni kam reviews me aaya (rare = important)
- **TF × IDF** = final score

### "SVM kyu jeeta? Naive Bayes kyu haara?"

**Answer:** "Naive Bayes assume karta hai ki sab words independent hain — 'not' aur 'good' ko alag dekhta hai. Par SVM word combinations ka pattern pakadta hai, isliye zyada accurate hota hai. Especially TF-IDF features ke saath SVM bahut acha kaam karta hai text classification me."

### "Lemmatization kyu kiya? Stemming kyu nahi?"

**Answer:** "Stemming word ko roughly chop karta hai — 'studies' se 'studi' bana deta hai jo actual word nahi hai. Lemmatization dictionary use karke proper root word nikalta hai — 'studies' se 'study'. Isse readable bhi rehta hai aur accurate bhi."

### "50,000 max features kyu rakhe?"

**Answer:** "Agar sab words rakhenge toh bahut zyada features ho jayenge — model slow hoga aur overfitting ka risk badhega. 50K enough hai ki important words capture ho jayein par noise na aaye."

### "ngram_range=(1,2) kyu?"

**Answer:** "Sirf unigrams (single words) se 'not good' ka matlab samajh nahi aata — 'not' alag, 'good' alag process hota. Bigrams (word pairs) rakhne se 'not good' ek feature ban jata hai, jo actually negative hai. Isse negation handle hoti hai."

### "FastAPI kyu use kiya deployment ke liye? Flask kyu nahi?"

**Answer:** "FastAPI modern hai, Flask se fast hai, automatic API documentation deta hai /docs pe, aur type hints support karta hai. Code almost same lagta hai par FastAPI production-ready features free me deta hai."

### "Pickle kyu use kiya model save karne ke liye?"

**Answer:** "Pickle Python ka built-in serialization hai — model object ko directly file me save karta hai. Simple use case ke liye best hai. Alternatives jaise ONNX ya joblib bhi hain par is project ke scope me pickle sufficient tha."

---

## 3. KEY ML CONCEPTS (Simple Explanation)

### Overfitting vs Underfitting
- **Overfitting** = Model ne training data ratt liya, naye data pe fail karta hai (jaise sirf textbook ratte exam me fail)
- **Underfitting** = Model ne kuch seekha hi nahi (jaise bina padhe exam de diya)
- **Good fit** = Training pe bhi acha, test pe bhi acha

### Accuracy, Precision, Recall, F1-Score
- **Accuracy** = total me se kitne sahi the (90% accuracy = 100 me se 90 sahi)
- **Precision** = jitne POSITIVE bola unme se kitne sach me positive the
  - "Maine 10 reviews ko positive bola, unme se 8 sach me positive the = 80% precision"
- **Recall** = jo sach me positive the, unme se kitne pakde
  - "Total 10 positive reviews the, maine 8 dhundhe = 80% recall"
- **F1-Score** = Precision aur Recall ka balance (dono achhe chahiye)

### Train-Test Split
- Model ko training data pe seekhna hai, test data pe prove karna hai
- Agar test pe bhi accuracy achi hai = model generalize kar raha hai
- 80-20 split standard hai

### TF-IDF (Term Frequency - Inverse Document Frequency)
- Har word ko ek importance score deta hai
- Agar word ek review me bahut aaya par doosri reviews me kam aaya = HIGH score (important word)
- Agar word har jagah aaya (the, is, a) = LOW score (faaltu word)

### Pickle / Serialization
- Python object (model) ko file me save karna = serialization (pickle.dump)
- File se wapas object banana = deserialization (pickle.load)
- Isse model dobara train nahi karna padta — ek baar train karo, hamesha use karo

---

## 4. TOP 20 INTERVIEW QUESTIONS WITH ANSWERS

### Project Overview Questions

**Q1: "Apna project briefly explain karo."**
> "Maine ek sentiment analysis system banaya hai jo movie reviews ko positive ya negative classify karta hai. Maine IMDB ke 50K reviews ka dataset use kiya, text preprocessing kiya with NLTK, TF-IDF se features extract kiye, aur 3 ML models compare kiye — Naive Bayes, Logistic Regression, aur Linear SVM. SVM ne best accuracy di, toh usse pickle me save karke FastAPI se deploy kiya Render.com pe. Ab koi bhi browser se review daal ke sentiment check kar sakta hai."

**Q2: "Ye project kyu banaya?"**
> "NLP aur ML ka practical application seekhna tha. Sentiment analysis ek classic NLP problem hai jo real world me bahut use hota hai — product reviews, social media monitoring, brand analysis me. Isse end-to-end ML pipeline seekhi — data loading se deployment tak."

**Q3: "Dataset ke baare me batao."**
> "IMDB 50K dataset use kiya — 25K positive aur 25K negative reviews. Balanced dataset hai isliye accuracy metric reliable hai. Agar imbalanced hota toh F1-score dekhna padta."

### Technical Questions

**Q4: "Preprocessing me kya kya kiya aur kyu?"**
> "6 steps: HTML tags remove, URLs remove, non-alphabetic characters remove, lowercase, stopwords remove, lemmatization. Har step ka goal tha noise reduce karna aur sirf meaningful words rakhna taaki model better patterns seekhe."

**Q5: "TF-IDF kya hai? Simple me samjhao."**
> "Ye har word ko ek importance score deta hai. Agar word ek document me zyada aaya par overall kam documents me hai, toh wo word important hai. Jaise 'masterpiece' kam reviews me aata hai par jab aata hai toh strong positive signal hai — TF-IDF isse high score dega."

**Q6: "Teeno models me se SVM kyu best raha?"**
> "SVM high-dimensional data me acha kaam karta hai — aur TF-IDF 50K features create karta hai jo high-dimensional hai. SVM ek optimal hyperplane dhundhta hai jo classes ko best separate kare. Naive Bayes word independence assume karta hai jo text me sahi nahi hota, aur Logistic Regression bhi acha tha par SVM ne margins maximize karke thodi better accuracy di."

**Q7: "Agar accuracy 89% hai toh baaki 11% kyu fail hue?"**
> "Kuch reviews me sarcasm hota hai jaise 'Oh what a GREAT movie, I fell asleep' — ye actually negative hai par words positive hain. Aur kuch mixed reviews hain jaise 'acting achi thi par story bekaar' — ye tricky cases hain jo simple bag-of-words models ke liye mushkil hain."

**Q8: "Model ko kaise save kiya aur kyu?"**
> "Pickle library se model aur vectorizer dono ek file me save kiye. Dono zaroori hain — vectorizer text ko same format ke numbers me convert karta hai jaise training me tha, aur model un numbers pe prediction karta hai. Bina vectorizer ke model kaam nahi karega."

**Q9: "FastAPI me deployment kaise kaam karta hai?"**
> "App start hote hi pickle file load hoti hai — model aur vectorizer memory me aa jaate hain. Jab user review submit karta hai, same preprocessing pipeline run hota hai (clean_text), phir vectorizer text ko TF-IDF features me convert karta hai, phir model predict karta hai. Response HTML ya JSON me milta hai."

**Q10: "Train aur test data kyu alag kiye?"**
> "Agar model ko same data pe test karein jis pe train kiya toh cheating hogi — model ne answers ratte honge. Test data wo hai jo model ne kabhi dekha nahi, toh usse pata chalta hai ki model naye data pe kitna acha kaam karega. Ye generalization test hai."

### Deep Dive Questions

**Q11: "Overfitting kaise check kiya?"**
> "Train accuracy aur test accuracy compare ki. Agar train pe 99% aur test pe 60% hoti toh overfitting hoti. Par hamare case me dono close hain, matlab model generalize kar raha hai. Aur TF-IDF ka max_features=50K rakhna bhi regularization ka kaam karta hai — sab words nahi, sirf top 50K."

**Q12: "Agar dataset imbalanced hota toh kya karte?"**
> "Tab accuracy misleading hoti — 95% negative me agar model sab ko negative bole toh bhi 95% accuracy dikhegi. Toh F1-score dekhte, SMOTE ya undersampling use karte, aur class_weight='balanced' parameter use karte models me."

**Q13: "Ye model real-time me kitna fast hai?"**
> "Bahut fast — milliseconds me predict karta hai. SVM ek simple linear model hai, prediction sirf dot product hai. Heavy deep learning models ki tarah GPU nahi chahiye. Free tier Render pe bhi smoothly chalta hai."

**Q14: "Kya ye model Hindi reviews pe kaam karega?"**
> "Nahi, kyunki training data sirf English tha. Hindi ke liye ya toh Hindi dataset se retrain karna hoga, ya multilingual model use karna hoga jaise mBERT. Par architecture same rehta — sirf data aur model change hoga."

**Q15: "Pickle ke security risks kya hain?"**
> "Pickle files untrusted source se load karna dangerous hai — malicious code execute ho sakta hai. Isliye production me sirf apni generated pickle files use karni chahiye, ya ONNX/joblib jaisi safer alternatives consider karni chahiye."

### Improvement Questions (YE 100% PUCHENGE)

**Q16: "Is project me kya improve kar sakte ho?"**
> "3 cheezein:
> 1. Deep learning models try karna — BERT ya DistilBERT se accuracy 93-95% tak ja sakti hai
> 2. Confidence score add karna — sirf positive/negative nahi, kitna confident hai ye bhi dikhana
> 3. More categories — sirf positive/negative nahi, 1-5 star rating predict karna"

**Q17: "BERT kyu nahi use kiya?"**
> "BERT heavy model hai — training me GPU chahiye, deployment me zyada RAM. Free tier pe fit nahi hota. Par TF-IDF + SVM ne 89% accuracy di jo is use case ke liye sufficient hai. Production me agar resources hon toh BERT definitely better choice hai."

**Q18: "API ke alawa aur kaise deploy kar sakte the?"**
> "Docker container me wrap karke AWS/GCP pe deploy kar sakte the. Ya Streamlit se aur rich UI bana sakte the. Ya mobile app me integrate kar sakte the REST API ke through."

**Q19: "Agar 10 lakh reviews process karne hon toh?"**
> "Batch processing karna padega — sab ek saath nahi, chunks me. Celery jaise task queue use karna padega. Aur model ko optimize karna padega — sparse matrix operations already fast hain par infrastructure scale karna padega."

**Q20: "Is project se kya seekha?"**
> "End-to-end ML pipeline seekhi — data loading, EDA, preprocessing, feature engineering, model training, evaluation, saving, aur deployment. Seekha ki simple models (SVM) bhi bahut powerful hote hain agar preprocessing acha ho. Aur deployment seekhi — model banana alag hai, usse production me laana alag skill hai."

---

## 5. WEAK POINTS — HONEST ANSWERS

Interviewer impressed hota hai jab tum khud weaknesses bolo aur solution bhi batao:

| Weak Point | Honest Answer |
|-----------|--------------|
| "Sirf binary classification hai" | "Haan, par foundation strong hai. Multi-class ke liye same pipeline me sirf labels change karne hain" |
| "Deep learning nahi use kiya" | "Consciously choose kiya — TF-IDF + SVM lightweight hai, free tier pe deployable hai, aur 89% accuracy is use case ke liye sufficient hai" |
| "Sirf English support hai" | "Correct, multilingual ke liye mBERT use karna hoga — ye next improvement me plan hai" |
| "No confidence score" | "SVM me decision_function se distance milta hai — usse confidence derive kar sakte hain, ye add kar sakte hain" |
| "Sarcasm detect nahi hota" | "Ye NLP ka open problem hai — sarcasm ke liye context understanding chahiye jo bag-of-words models me nahi hoti. Transformer models isme better hain" |

---

## 6. PROJECT STRUCTURE — Quick Reference

```
sentiment-analysis/
├── main.py              → Entry point: train ya predict mode
├── app.py               → FastAPI web app (deployment ke liye)
├── requirements.txt     → Sab Python libraries listed
├── IMDB Dataset.csv     → 50K reviews (GitHub pe nahi, bada file hai)
├── src/
│   ├── preprocess.py    → Data loading + text cleaning
│   ├── model.py         → TF-IDF + 3 models train + save/load
│   ├── visualize.py     → 4 charts generate
│   └── predict.py       → Terminal se prediction
└── outputs/
    ├── sentiment_model.pkl          → Saved best model
    ├── sentiment_distribution.png   → Bar chart
    ├── review_lengths.png           → Histogram
    ├── wordclouds.png              → Word clouds
    └── model_comparison.png         → Accuracy comparison
```

---

## 7. QUICK TIPS FOR INTERVIEW

1. **"Maine kiya" bolo, "AI ne kiya" mat bolo** — tum explain kar paa rahe ho matlab tumne seekha hai
2. **Flow samjho, ratta mat maaro** — CSV → Clean → TF-IDF → Train → Predict, bas ye flow yaad rakho
3. **Tradeoffs bolo** — "maine SVM choose kiya kyunki..." ye sunke interviewer khush hota hai
4. **Limitations khud batao** — "ye model sarcasm nahi samajhta" — honesty impress karti hai
5. **Next steps batao** — "aage BERT try karunga" — ye growth mindset dikhata hai
6. **Live demo ke liye ready raho** — link kaam karta hai, browser me kholke dikhao
