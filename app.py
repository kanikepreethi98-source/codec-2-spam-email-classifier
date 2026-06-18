import joblib
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.error(
        "Model files not found. Please run `train_model.py` first to generate "
        "`model.pkl` and `vectorizer.pkl`."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

st.title("📧 Spam Email Classifier")

message = st.text_area("Enter a message")

if st.button("Predict"):
    if message.strip():
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("Please enter a message.")
