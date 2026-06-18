"""
predict.py
==========
Loads the trained model + vectorizer and exposes a simple
`predict_email(text)` function used by the Streamlit app.
"""

import os
import joblib

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def predict_email(text: str):
    """
    Classify a single email text.

    Returns
    -------
    tuple  (label: str, confidence: float)
        label is 'spam' or 'ham'; confidence is the max predicted probability.
    None
        If the model files are missing.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = max(probabilities)
    label = "spam" if prediction == 1 else "ham"

    return label, confidence