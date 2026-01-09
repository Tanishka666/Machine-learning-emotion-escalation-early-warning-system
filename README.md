# Emotion Escalation Early Warning System

An NLP-based system that predicts whether a conversation is likely to **emotionally escalate in the next turn**.  
The goal is to raise an early warning before emotions turn hostile or intense.

---

## 📌 Problem Statement

In many real-world conversations such as customer support chats, online discussions, or conflict scenarios, emotional escalation is often detected **after** it happens.

This project focuses on **early detection**, by analyzing recent conversation context and predicting whether escalation is likely in upcoming turns.

---

## 🎯 Project Objective

- Monitor multi-turn conversations
- Detect rising emotional intensity
- Predict escalation **before peak negativity**
- Provide clear risk levels: **LOW / MEDIUM / HIGH**

---

## 🧠 How the System Works

1. The user inputs a conversation (one message per line)
2. The system analyzes the **last two messages** as context
3. Text features are extracted using **TF-IDF**
4. Emotional intensity is estimated from language cues
5. A trained **Random Forest model** predicts escalation probability
6. Model output and emotion signals are combined to generate an early warning level

---

## 🚦 Risk Levels

| Risk Level | Meaning |
|----------|--------|
| 🟢 LOW | Conversation is stable |
| 🟠 MEDIUM | Growing frustration detected |
| 🔴 HIGH | Strong anger or loss of emotional control |

The system prioritizes **early warnings** rather than reacting after escalation occurs.

---

## 🛠 Tech Stack

- Python  
- Scikit-learn  
- NLP (TF-IDF)  
- Random Forest Classifier  
- Streamlit  

---

## ▶️ Run the Project Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
