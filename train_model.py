import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading TF-IDF vectors and labels...")

# Load features and labels
with open("models/X_train_tfidf.pkl", "rb") as f:
    X_train_tfidf = pickle.load(f)

with open("models/X_test_tfidf.pkl", "rb") as f:
    X_test_tfidf = pickle.load(f)

with open("models/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("models/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print("Training Logistic Regression model...")

# Create and train classifier
model = LogisticRegression(max_iter=3000)
model.fit(X_train_tfidf, y_train)

# Predict test data
y_pred = model.predict(X_test_tfidf)

# Show accuracy
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Detailed performance
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save trained model
with open("models/spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel training complete! Model saved as spam_classifier.pkl")
