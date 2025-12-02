# step3_vectorize.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# -----------------------------
# 1. Load the processed datasets
# -----------------------------
train_path = "dataset/train.csv"
test_path = "dataset/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df.dropna(subset=["message"], inplace=True)
test_df.dropna(subset=["message"], inplace=True)

train_df = train_df[train_df["message"].str.strip() != ""]
test_df = test_df[test_df["message"].str.strip() != ""]

X_train = train_df['message']
y_train = train_df['label']

X_test = test_df['message']
y_test = test_df['label']

print("Loaded train and test datasets successfully.")

# -----------------------------
# 2. Create the TF-IDF vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=3000,       # limit features to avoid huge model
    stop_words='english',    # remove common English stopwords
    ngram_range=(1, 2)       # use unigrams + bigrams (improves accuracy)
)

# Fit on training data only
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF vectorization complete.")
print("Training vector shape:", X_train_tfidf.shape)
print("Testing vector shape:", X_test_tfidf.shape)

# -----------------------------
# 3. Save the vectorizer and vectors
# -----------------------------
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(X_train_tfidf, "models/X_train_tfidf.pkl")
joblib.dump(X_test_tfidf, "models/X_test_tfidf.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("TF-IDF vectorizer and data saved in /models folder.")
