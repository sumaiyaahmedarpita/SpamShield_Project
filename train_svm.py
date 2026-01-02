import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("Loading improved TF-IDF vectors...")

with open("models/X_train_tfidf_v2.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("models/X_test_tfidf_v2.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("models/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("models/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print("Training Linear SVM model...")

model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
with open("models/svm_spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nSVM model saved successfully!")
