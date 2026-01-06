# extended_evaluation.py
# --------------------------------------------------
# Extended Evaluation for SpamShield Project
# Models: Multinomial Naïve Bayes, Random Forest
# Evaluations: SMS vs Email, Confusion Matrix
# --------------------------------------------------

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# 1. TEXT PREPROCESSING FUNCTION
# --------------------------------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# 2. LOAD SMS DATASET
# --------------------------------------------------
print("Loading SMS dataset...")

sms_df = pd.read_csv("dataset/SMSSpamCollection_processed.csv")
sms_df["label"] = sms_df["label"].map({"spam": 1, "ham": 0})
sms_df.rename(columns={"message": "text"}, inplace=True)
sms_df["clean_text"] = sms_df["text"].apply(preprocess)
sms_df["source"] = "SMS"


# --------------------------------------------------
# 3. LOAD EMAIL DATASET
# --------------------------------------------------
print("Loading Email dataset...")

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
email_df["source"] = "Email"


# --------------------------------------------------
# 4. COMBINE SMS + EMAIL DATASETS
# --------------------------------------------------
print("Combining datasets...")

combined_df = pd.concat(
    [
        sms_df[["clean_text", "label", "source"]],
        email_df[["clean_text", "label", "source"]],
    ],
    ignore_index=True
)

print(f"Total samples: {combined_df.shape[0]}")


# --------------------------------------------------
# 5. TF-IDF VECTORIZATION
# --------------------------------------------------
print("Vectorizing text using TF-IDF...")
start = time.time()

vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(combined_df["clean_text"])
y = combined_df["label"]

end = time.time()
print(f"TF-IDF completed in {end - start:.2f} seconds")
print("TF-IDF shape:", X.shape)
sources = combined_df["source"]


# --------------------------------------------------
# 6. TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X,
    y,
    sources,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 7. LINEAR SVM EVALUATION
# --------------------------------------------------
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

print("\n================ LINEAR SVM EVALUATION ================\n")

svm = LinearSVC()
svm.fit(X_train, y_train)

svm_preds = svm.predict(X_test)

# Overall performance
print("Linear SVM - Overall Results:")
print(classification_report(y_test, svm_preds))

# Confusion Matrix (Overall)
cm_svm = confusion_matrix(y_test, svm_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"])
plt.title("Linear SVM – Confusion Matrix (Overall)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# SMS-only evaluation
sms_mask = src_test == "SMS"
print("\nLinear SVM – SMS Only:")
print(classification_report(y_test[sms_mask], svm_preds[sms_mask]))

# Email-only evaluation
email_mask = src_test == "Email"
print("\nLinear SVM – Email Only:")
print(classification_report(y_test[email_mask], svm_preds[email_mask]))


# --------------------------------------------------
# 8. LOGISTIC REGRESSION EVALUATION
# --------------------------------------------------
print("\n================ LOGISTIC REGRESSION EVALUATION ================\n")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)

# Overall performance
print("Logistic Regression – Overall Results:")
print(classification_report(y_test, lr_preds))

# Confusion Matrix (Overall)
cm_lr = confusion_matrix(y_test, lr_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"])
plt.title("Logistic Regression – Confusion Matrix (Overall)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# SMS-only evaluation
print("\nLogistic Regression – SMS Only:")
print(classification_report(y_test[sms_mask], lr_preds[sms_mask]))

# Email-only evaluation
print("\nLogistic Regression – Email Only:")
print(classification_report(y_test[email_mask], lr_preds[email_mask]))

# ----------------------------
# --------------------------------------------------
# 7. TRAIN MULTINOMIAL NAÏVE BAYES
# --------------------------------------------------
print("\nTraining Multinomial Naïve Bayes...")

nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)

print("\nMultinomial Naïve Bayes Results:")
print(classification_report(y_test, nb_preds))

# --------------------------------------------------
# 8. TRAIN RANDOM FOREST (OPTIONAL / SLOWER)
# --------------------------------------------------
print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_preds))

# --------------------------------------------------
# 9. CONFUSION MATRICES (OVERALL)
# --------------------------------------------------
models = {
    "Naive Bayes": nb_preds,
    "Random Forest": rf_preds
}

for name, preds in models.items():
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# --------------------------------------------------
# 10. SEPARATE EVALUATION: SMS vs EMAIL
# --------------------------------------------------
print("\nSeparate Evaluation: SMS vs Email")

for source in ["SMS", "Email"]:
    print(f"\n--- {source} Samples ---")
    idx = src_test == source

    if idx.sum() == 0:
        print("No samples found.")
        continue

    print("\nNaive Bayes:")
    print(classification_report(y_test[idx], nb_preds[idx]))

    print("\nRandom Forest:")
    print(classification_report(y_test[idx], rf_preds[idx]))

# --------------------------------------------------
# 11. SIMPLE ERROR ANALYSIS
# --------------------------------------------------
print("\nError Analysis (Naive Bayes):")

errors = combined_df.iloc[y_test.index] if hasattr(y_test, "index") else None

fp = ((y_test == 0) & (nb_preds == 1)).sum()
fn = ((y_test == 1) & (nb_preds == 0)).sum()

print(f"False Positives (ham → spam): {fp}")
print(f"False Negatives (spam → ham): {fn}")

print("\nExtended evaluation complete.")
