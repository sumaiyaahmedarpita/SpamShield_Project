# roc_cv_evaluation.py
# --------------------------------------------------
# ROC–AUC and Cross‑Validation for SpamShield
# Models: Logistic Regression, Multinomial Naïve Bayes
# --------------------------------------------------

import pandas as pd
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve

# --------------------------------------------------
# 1. PREPROCESSING
# --------------------------------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
sms_df = pd.read_csv("dataset/SMSSpamCollection_processed.csv")
sms_df["label"] = sms_df["label"].map({"spam": 1, "ham": 0})
sms_df.rename(columns={"message": "text"}, inplace=True)
sms_df["clean_text"] = sms_df["text"].apply(preprocess)

email_df = pd.read_csv("dataset/emails.csv")
email_labels = email_df["Prediction"]
email_features = email_df.drop(columns=["Email No.", "Prediction"])

def reconstruct_email_text(row):
    words = []
    for word, count in row.items():
        if count > 0:
            words.extend([word] * int(count))
    return " ".join(words)

email_df["text"] = email_features.apply(reconstruct_email_text, axis=1)
email_df["label"] = email_labels
email_df["clean_text"] = email_df["text"].apply(preprocess)

combined_df = pd.concat(
    [
        sms_df[["clean_text", "label"]],
        email_df[["clean_text", "label"]],
    ],
    ignore_index=True
)

# --------------------------------------------------
# 3. TF‑IDF
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(combined_df["clean_text"])
y = combined_df["label"]

# --------------------------------------------------
# 4. TRAIN / TEST SPLIT (ROC–AUC)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------------------
# 5. ROC–AUC EVALUATION
# --------------------------------------------------
print("\n================ ROC–AUC SCORES ================\n")

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print(f"Logistic Regression ROC–AUC: {lr_auc:.4f}")

# Multinomial Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_probs = nb.predict_proba(X_test)[:, 1]
nb_auc = roc_auc_score(y_test, nb_probs)
print(f"Naïve Bayes ROC–AUC: {nb_auc:.4f}")

# --------------------------------------------------
# 6. 5‑FOLD CROSS‑VALIDATION
# --------------------------------------------------
print("\n=========== 5‑Fold Cross‑Validation (Accuracy) ===========\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_cv = cross_val_score(lr, X, y, cv=cv, scoring="accuracy")
nb_cv = cross_val_score(nb, X, y, cv=cv, scoring="accuracy")

print(f"Logistic Regression CV Accuracy: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"Naïve Bayes CV Accuracy: {nb_cv.mean():.4f} ± {nb_cv.std():.4f}")

print("\nROC–AUC and Cross‑Validation complete.")
