import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

print("Creating improved TF-IDF vectors (with bigrams)...")

# Load dataset
df = pd.read_csv("dataset/SMSSpamCollection_processed.csv")

X = df["message"]
y = df["label_num"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Improved TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save upgraded vectors
with open("models/tfidf_vectorizer_v2.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/X_train_tfidf_v2.pkl", "wb") as f:
    pickle.dump(X_train_tfidf, f)

with open("models/X_test_tfidf_v2.pkl", "wb") as f:
    pickle.dump(X_test_tfidf, f)

with open("models/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("models/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("Improved TF-IDF vectors saved successfully!")
