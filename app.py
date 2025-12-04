# app.py
import streamlit as st
import joblib
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
# Load model and vectorizer
model = joblib.load("ai_model_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Known AI models
ai_models = {"GPT-4", "Claude-3", "Gemini-1.5", "LLaMA-3", "Mistral-7B"}

# Title
st.title("🧠 AI Text Detector")
st.markdown("**Detect AI, Human, or Mixed Text with AI source identification**")

# Text input
user_text = st.text_area("📄 Paste your text here:", height=200)

# Predict button
if st.button("🔍 Analyze"):
    if user_text.strip():
        sentences = sent_tokenize(user_text)
        predictions = []
        
        for sentence in sentences:
            vect = vectorizer.transform([sentence])
            pred = model.predict(vect)[0]
            # If prediction is not one of the known AI models, mark as Human
            label = pred if pred in ai_models else "Human"
            predictions.append((sentence, label))

        # Classify overall result
        ai_sentences = [p for s, p in predictions if p != "Human"]
        human_sentences = [p for s, p in predictions if p == "Human"]

        if ai_sentences and human_sentences:
            st.warning("🔄 **Mixed Content Detected**")
        elif ai_sentences:
            st.error(f"🤖 AI-Generated Text (Likely {ai_sentences[0]})")
        else:
            st.success("✅ This is likely Human-Written.")

        # Sentence-wise breakdown
        st.markdown("---")
        st.markdown("### 🧩 Sentence Breakdown:")

        for i, (sentence, label) in enumerate(predictions):
            if label == "Human":
                st.markdown(f"✅ **[Human]** {sentence}")
            else:
                st.markdown(f"🟥 **[{label}]** {sentence}")
