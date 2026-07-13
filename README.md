# Emotion Escalation Early Warning System

An NLP-based application that predicts whether a conversation is likely to **emotionally escalate in the next turn**, enabling early intervention before emotions become hostile or intense.

---

## 📌 Problem Statement

In customer support, online discussions, and conflict scenarios, emotional escalation is often identified only after it has occurred. This project aims to detect early signs of escalation by analyzing recent conversation context and predicting future emotional intensity.

---

## 🎯 Project Objectives

- Monitor multi-turn conversations
- Detect increasing emotional intensity
- Predict escalation before peak negativity
- Classify conversations into **LOW**, **MEDIUM**, or **HIGH** risk levels

---

## 🧠 How It Works

1. Enter a conversation (one message per line).
2. The system analyzes the **last two messages** as context.
3. Text features are extracted using **TF-IDF**.
4. A trained **Random Forest** model predicts the probability of emotional escalation.
5. The prediction is combined with emotion-related signals to generate an early warning level.

---

## 🚦 Risk Levels

| Risk Level | Meaning |
|------------|---------|
| 🟢 LOW | Conversation is stable |
| 🟠 MEDIUM | Signs of increasing frustration |
| 🔴 HIGH | High likelihood of emotional escalation |

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- TF-IDF (NLP)
- Random Forest Classifier
- Streamlit

---

## 🌐 Live Demo

**Streamlit App:**  
https://machine-learning-emotion-escalation-early-warning-system-frecx.streamlit.app/

---

## ▶️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Tanishka666/Machine-learning-emotion-escalation-early-warning-system.git
cd Machine-learning-emotion-escalation-early-warning-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## 📊 Model

- **Algorithm:** Random Forest Classifier
- **Feature Extraction:** TF-IDF Vectorization
- **Task:** Binary classification for emotion escalation prediction

---

## 📄 Dataset

The model is trained on the **DailyDialog** dataset, a benchmark dataset containing multi-turn human conversations with emotion annotations.

---

## 👩‍💻 Author

**Tanishka Gandhi**
