"""
train_model.py
==============
Trains a Multinomial Naive Bayes classifier on the SMS Spam Collection
dataset and saves the trained model and TF-IDF vectorizer.

Usage:
    python train_model.py
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -------------------------------
# File Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "spam.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

# -------------------------------
# Load Dataset
# -------------------------------
def load_dataset():
    df = pd.read_csv(DATASET_PATH, encoding="latin-1")

    # Keep only first two columns
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]

    # Convert labels
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    df.dropna(inplace=True)

    return df


# -------------------------------
# Train Model
# -------------------------------
def train():

    print("Loading dataset...")

    df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Model saved as model.pkl")
    print("Vectorizer saved as vectorizer.pkl")


if __name__ == "__main__":
    train()
