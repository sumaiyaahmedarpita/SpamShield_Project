# step2_preprocess.py
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Optional NLP tools
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# If running for the first time, uncomment these:
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# ---------- CONFIG ----------
DATASET_PATH = "dataset/SMSSpamCollection"   # update if your filename differs
RANDOM_STATE = 42
TEST_SIZE = 0.20
REMOVE_STOPWORDS = False    # change to True to remove stopwords
LEMMATIZE = False           # change to True to lemmatize
SAVE_PROCESSED = True
PROCESSED_PATH = "dataset/SMSSpamCollection_processed.csv"
# ----------------------------

# 1) Load raw dataset
# The UCI file has no header and uses tab between label and message
try:
    df = pd.read_csv(DATASET_PATH, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
except FileNotFoundError:
    raise SystemExit(f"Dataset not found at {DATASET_PATH}. Put the file there or update DATASET_PATH.")

print("Initial shape:", df.shape)
print(df.head(5))

# 2) Basic cleaning: drop NaNs, duplicates
df.dropna(subset=['message'], inplace=True)        # remove rows where message is missing
df.drop_duplicates(inplace=True)                   # drop exact duplicates (label+message)
print("After dropping NaNs and duplicates:", df.shape)

# 3) Normalize labels if necessary (lowercase)
df['label'] = df['label'].str.strip().str.lower()  # expect 'ham' or 'spam'

# 4) Simple text cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower()                       # lowercase
    text = re.sub(r'\[.*?\]', ' ', text)      # remove text inside brackets
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text) # remove punctuation (keep letters/numbers)
    text = re.sub(r'\s+', ' ', text).strip()  # collapse extra spaces
    return text

# 5) Optional: stopword removal and lemmatization
stop_words = set(stopwords.words('english')) if REMOVE_STOPWORDS else set()
lemmatizer = WordNetLemmatizer() if LEMMATIZE else None

def advanced_clean(text):
    t = clean_text(text)
    if REMOVE_STOPWORDS:
        tokens = [w for w in t.split() if w not in stop_words]
    else:
        tokens = t.split()
    if LEMMATIZE:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Apply preprocessing
df['message_clean'] = df['message'].apply(advanced_clean)

# Quick sanity checks
print("\nSample original vs cleaned messages:")
for i, row in df.sample(5, random_state=RANDOM_STATE).iterrows():
    print("Original:", row['message'])
    print("Cleaned :", row['message_clean'])
    print("---")

# 6) Check class distribution
print("\nLabel distribution:")
print(df['label'].value_counts())

# 7) Encode label to numeric for modeling (ham=0, spam=1)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
if df['label_num'].isnull().any():
    raise SystemExit("Found label values other than 'ham' or 'spam'. Please inspect dataset labels.")

# 8) Train-test split (stratify to keep class proportion)
X = df['message_clean']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

print(f"\nTrain size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
print("Train label distribution:\n", y_train.value_counts(normalize=True))
print("Test label distribution:\n", y_test.value_counts(normalize=True))

# 9) Optional: save a processed CSV for later steps
if SAVE_PROCESSED:
    df.to_csv(PROCESSED_PATH, index=False, encoding='utf-8')
    print(f"\nProcessed dataset saved to: {PROCESSED_PATH}")

# You can also save train/test splits to CSV if you prefer
train_df = pd.DataFrame({'message': X_train, 'label': y_train})
test_df  = pd.DataFrame({'message': X_test,  'label': y_test})
train_df.to_csv('dataset/train.csv', index=False)
test_df.to_csv('dataset/test.csv', index=False)
print("Saved dataset/train.csv and dataset/test.csv")
