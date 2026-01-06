# unified_spam_detector.py

import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ---------------------------------
# 1. LOAD EMAIL DATASET (KAGGLE)
# ---------------------------------
print("Loading email dataset...")

email_df = pd.read_csv("dataset/emails.csv")

# Separate labels and features
email_labels = email_df["Prediction"]
email_features = email_df.drop(columns=["Email No.", "Prediction"])


# ---------------------------------
# 2. RECONSTRUCT EMAIL TEXT
# ---------------------------------
print("Reconstructing email text from word counts...")

def reconstruct_email_text(row):
    words = []
    for word, count in row.items():
        if count > 0:
            words.extend([word] * int(count))
    return " ".join(words)

email_df["text"] = email_features.apply(reconstruct_email_text, axis=1)
email_df["label"] = email_labels


# ---------------------------------
# 3. LOAD SMS DATASET
# ---------------------------------
print("Loading SMS dataset...")

sms_df = pd.read_csv("dataset/SMSSpamCollection_processed.csv")

# Handle common SMS dataset formats safely
if {"label", "message"}.issubset(sms_df.columns):
    sms_df["label"] = sms_df["label"].map({"spam": 1, "ham": 0})
    sms_df.rename(columns={"message": "text"}, inplace=True)
elif {"v1", "v2"}.issubset(sms_df.columns):
    sms_df.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
    sms_df["label"] = sms_df["label"].map({"spam": 1, "ham": 0})
else:
    raise ValueError("Unknown SMS dataset column format")


# ---------------------------------
# 4. TEXT PREPROCESSING
# ---------------------------------
print("Preprocessing text...")

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

sms_df["clean_text"] = sms_df["text"].apply(preprocess)
email_df["clean_text"] = email_df["text"].apply(preprocess)


# ---------------------------------
# 5. COMBINE SMS + EMAIL
# ---------------------------------
print("Combining datasets...")

combined_df = pd.concat(
    [
        sms_df[["clean_text", "label"]],
        email_df[["clean_text", "label"]],
    ],
    ignore_index=True,
)

print(f"Total samples: {combined_df.shape[0]}")


# ---------------------------------
# 6. TF-IDF VECTORIZATION
# ---------------------------------
print("Vectorizing text using TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2),
    min_df=2,
)

X = vectorizer.fit_transform(combined_df["clean_text"])
y = combined_df["label"]


# ---------------------------------
# 7. TRAIN / TEST SPLIT
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ---------------------------------
# 8. TRAIN LINEAR SVM
# ---------------------------------
print("\nTraining Unified Linear SVM...")

svm = LinearSVC()
svm.fit(X_train, y_train)

svm_preds = svm.predict(X_test)
print("\nLinear SVM Results:")
print(classification_report(y_test, svm_preds))


# ---------------------------------
# 9. TRAIN LOGISTIC REGRESSION
# ---------------------------------
print("Training Unified Logistic Regression...")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)
print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_preds))


print("\nUnified spam detection model training complete.")
