"""
train_model.py
==============
Trains a Multinomial Naive Bayes classifier on the SMS Spam Collection
dataset, evaluates it, and saves the model + vectorizer to disk.

Usage:  python train_model.py
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

# ── Configuration ───────────────────────────────────────────────────
DATASET_PATH = os.path.join("dataset", "spam.csv")
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(path: str) -> pd.DataFrame:
    """Load the SMS Spam Collection CSV and return a clean DataFrame."""
    df = pd.read_csv(path, encoding="latin-1")
    # The raw file often has extra unnamed columns — keep only the first two
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df.dropna(inplace=True)
    return df


def train():
    """Full training pipeline: load → split → vectorize → train → evaluate → save."""
    # 1. Load data
    print("📂 Loading dataset …")
    df = load_dataset(DATASET_PATH)
    print(f"   {len(df)} samples loaded  (spam: {df['label'].sum()}, "
          f"ham: {len(df) - df['label'].sum()})")

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # 3. TF-IDF vectorization (English stop words removed)
    print("🔤 Vectorizing with TF-IDF …")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Train Multinomial Naive Bayes
    print("🧠 Training Multinomial Naive Bayes …")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n📊 Evaluation Metrics")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"\n   Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['ham', 'spam'])}")

    # 6. Save model & vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"💾 Model saved to      {MODEL_PATH}")
    print(f"💾 Vectorizer saved to {VECTORIZER_PATH}")
    print("✅ Training complete!")


if __name__ == "__main__":
    train()