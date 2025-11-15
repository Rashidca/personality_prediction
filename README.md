
# 🧠 MBTI Personality Prediction Web App | Streamlit

This project is a **text-based personality prediction application** built using **Machine Learning and NLP**.
Users can freely write about themselves — interests, thoughts, emotions, habits, or even just vent — and the app predicts which of the **16 MBTI (Myers-Briggs Type Indicator) personality types** they belong to.

The app also displays an **in-depth explanation** of the predicted personality type and allows users to **visualize their writing through a word cloud**.

---

## 🚀 Live Features

| Feature                      | Description                               |
| ---------------------------- | ----------------------------------------- |
| Text input / journaling box  | Users can write anything about themselves |
| Personality prediction       | ML model predicts 1 of the 16 MBTI types  |
| Word cloud                   | Visual representation of the user's words |
| Detailed personality profile | Explanation of the MBTI type              |
| Learning section             | Info about all 16 personality types       |
| About section                | Project purpose, dataset, and model info  |

---

## 🧬 Technology Stack

* **Python**
* **Streamlit**
* **Scikit-Learn**
* **NLTK**
* **TF-IDF Vectorization**
* **Linear SVC Classifier**
* **Joblib (Model Loading)**
* **Matplotlib & WordCloud for visualization**

---

## 🗂 Files Used

| File             | Purpose                       |
| ---------------- | ----------------------------- |
| `app.py`         | Streamlit web app             |
| `vectorizer.pkl` | Trained TF-IDF vectorizer     |
| `le.pkl`         | Trained LabelEncoder          |
| `linear_svc.pkl` | Trained MBTI prediction model |

⚠ The app **does NOT train the model at runtime** — it only loads pretrained `.pkl` files for fast prediction.

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```
git clone <repository-link>
cd <project-folder>
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the application

```
streamlit run app.py
```

### Requirements

```
streamlit
scikit-learn
nltk
tqdm
wordcloud
joblib
matplotlib
seaborn
plotly
numpy
```

---

## 🧩 How It Works (Pipeline)

1. User enters text
2. Text cleaning → tokenization → stopword removal → lemmatization
3. Convert text to vector using pretrained **TF-IDF Vectorizer**
4. Model predicts personality using **LinearSVC**
5. Personality label mapped using **LabelEncoder**
6. Web app displays:

   * MBTI type
   * Full personality description
   * Optional word cloud

---

## 📚 Dataset

The model was trained on a publicly available MBTI dataset consisting of user-generated text labeled with personality types.

---

## 🎯 Purpose & Impact

This project aims to explore the link between **language patterns and personality traits**.
It helps users reflect on:

* communication style
* thinking and emotional patterns
* interpersonal preferences

🔹 *Note: The MBTI model is for self-reflection only — not psychological diagnosis or clinical evaluation.*

---

## 👨‍💻 Developer

**Muhammed Rashid**
📩 *Add contact info if you like*
💙 Contributions, suggestions & feedback are welcome!

---

## ⭐ Future Enhancements

* Show prediction confidence scores
* Display bar charts for I/E, S/N, T/F, J/P
* Downloadable PDF personality report
* Dark / Light theme toggle
* User profile & saved history

