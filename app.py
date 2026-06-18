import joblib
import streamlit as st
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Streamlit UI
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")

st.title("📧 Spam Email Classifier")
st.write("Enter a message below to check whether it is **Spam** or **Not Spam**.")

message = st.text_area("Message")

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
