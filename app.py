import streamlit as st
import pickle
import os
import re
import numpy as np
from scipy.sparse import hstack


st.markdown(
    """
    <h2 style='text-align: center;'>
        ⚠️ Emotion Escalation Early Warning System
    </h2>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.markdown("## 📌 Project Summary")

    st.markdown(
        """
        **Emotion Escalation Early Warning System**

        This application predicts whether a conversation is likely
        to **emotionally escalate in the next turn**, based on recent
        conversational context.
        """
    )

    st.markdown("### 🎯 Goal")
    st.write(
        "Detect rising emotional intensity early and raise a warning "
        "before conversations turn hostile."
    )

    st.markdown("### ⚙️ How it works")
    st.write(
        "- Analyzes the **last two messages** in a conversation\n"
        "- Extracts text features using **TF-IDF**\n"
        "- Estimates emotional intensity from language\n"
        "- Combines ML prediction with escalation rules"
    )

    st.markdown("### 🧪 Risk Levels")
    st.write(
        "- 🟢 **LOW** – Stable or calm conversation\n"
        "- 🟠 **MEDIUM** – Growing frustration\n"
        "- 🔴 **HIGH** – Strong anger or loss of control"
    )

    st.markdown("### 🛠️ Tech Stack")
    st.write(
        "- Python\n"
        "- Scikit-learn\n"
        "- NLP (TF-IDF)\n"
        "- Streamlit"
    )

    st.markdown("---")
    st.caption("Built as an ML + NLP project for early warning analysis.")


st.caption(
    "This system monitors conversations and raises an early warning "
    "when emotional escalation is likely in the next turn."
)

st.divider()

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error("Model or vectorizer file not found in app folder.")
        st.stop()

    if os.path.getsize(MODEL_PATH) == 0 or os.path.getsize(VECTORIZER_PATH) == 0:
        st.error("Model or vectorizer file is empty.")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_artifacts()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def estimate_emotion_score(text):
    text = text.lower()

    if any(word in text for word in [
        "furious", "extremely angry", "cannot tolerate",
        "unbearable", "hate", "lose control"
    ]):
        return 3

    elif any(word in text for word in [
        "angry", "frustrated", "frustrating",
        "irritating", "annoying", "upsetting",
        "stressed", "stress", "overwhelmed",
        "fed up", "tired", "pressure"
    ]):
        return 2

    elif any(word in text for word in [
        "upset", "sad", "disappointed", "confused"
    ]):
        return 1

    else:
        return 0



def risk_level(prob, emotion_score):
    if emotion_score >= 3 or prob >= 0.85:
        return "HIGH"

    elif emotion_score == 2:
        return "MEDIUM"

    else:
        return "LOW"

def risk_color(level):
    if level == "HIGH":
        return "🔴 HIGH RISK"
    elif level == "MEDIUM":
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

st.markdown("### 💬 Conversation Input")
st.write("Enter one message per line. The last message is analyzed with context.")

conversation = st.text_area(
    "Conversation",
    height=300,
    placeholder="Why are you late?\nYou never listen to me.\nI am furious right now."
)

if st.button("🚨 Predict Escalation Risk", use_container_width=True):
    lines = [line.strip() for line in conversation.split("\n") if line.strip()]

    if len(lines) < 2:
        st.warning("Please enter at least two messages to analyze escalation.")
    else:
        prev_msg = clean_text(lines[-2])
        curr_msg = clean_text(lines[-1])

        combined_text = prev_msg + " " + curr_msg
        X_text = vectorizer.transform([combined_text])

        emotion_score_value = estimate_emotion_score(curr_msg)
        emotion_score = np.array([[emotion_score_value]])

        X = hstack([X_text, emotion_score])

        prob = model.predict_proba(X)[0][1]
        level = risk_level(prob, emotion_score_value)

        st.divider()

        st.markdown("### 📊 Prediction Result")

        st.markdown(
            f"### {risk_color(level)}"
        )

        st.write(f"**Model Confidence:** {prob:.2f}")

        st.markdown("**Why this warning was triggered:**")
        if level == "HIGH":
            st.write("- Strong anger or loss of emotional control detected")
        elif level == "MEDIUM":
            st.write("- Growing frustration or irritation detected")
        else:
            st.write("- Conversation tone appears stable")

        st.caption(
            "This is an early warning signal, not a final judgment. "
            "The system prioritizes catching escalation early."
        )
