import pickle

# Load trained model
with open("models/spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_message(text):
    # Convert text into TF-IDF vector
    text_tfidf = vectorizer.transform([text])
    
    # Prediction
    pred = model.predict(text_tfidf)[0]
    
    return "SPAM" if pred == 1 else "HAM"

# Test the function
if __name__ == "__main__":
    print("\n--- SpamShield Message Tester ---\n")
    while True:
        msg = input("Enter a message (or 'exit' to quit): ")
        if msg.lower() == "exit":
            break
        
        result = predict_message(msg)
        print(f" Prediction → {result}\n")
