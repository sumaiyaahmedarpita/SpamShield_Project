import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

print("Recreating TF-IDF vectors...")

# Load dataset
df = pd.read_csv("dataset/SMSSpamCollection_processed.csv")

# Extract important columns
X = df["message"]
y = df["label_num"]  # use numerical labels (0 = ham, 1 = spam)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save all files
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/X_train_tfidf.pkl", "wb") as f:
    pickle.dump(X_train_tfidf, f)

with open("models/X_test_tfidf.pkl", "wb") as f:
    pickle.dump(X_test_tfidf, f)

with open("models/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("models/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("Recreated and saved TF-IDF data successfully!")
