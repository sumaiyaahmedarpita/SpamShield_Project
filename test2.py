
import pickle
# Load TF-IDF data
with open("models/X_train_tfidf.pkl", "rb") as f:
    X_train_tfidf = pickle.load(f)

with open("models/X_test_tfidf.pkl", "rb") as f:
    X_test_tfidf = pickle.load(f)

with open("models/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("models/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print("Loaded all TF-IDF vectors and labels!")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score



# Train the model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
with open("models/text_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete and saved!")
