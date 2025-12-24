# 🚀 AutoJudge — Programming Problem Difficulty Predictor

AutoJudge is an **end-to-end Machine Learning system** that predicts the **difficulty of programming problems** using only their textual descriptions.

### 🔍 What it predicts
- **Difficulty Class**: `Easy / Medium / Hard`
- **Difficulty Score**: A continuous numeric value *(≈ 1–10)*

The project demonstrates a **complete ML workflow** — from data preprocessing and model experimentation to deployment via a **Flask web application**.

---

## 🎯 Project Motivation

Difficulty estimation of programming problems is inherently **subjective** and often **inconsistent across platforms**.

AutoJudge explores **how far textual information alone**  
(problem statements, descriptions, and I/O specifications) can be used to estimate difficulty.

### This project focuses on:
- Practical ML decision-making  
- Evidence-based model selection  
- Clean engineering practices  
- End-to-end usability  

---

## 🧠 Approach Overview

### 1️⃣ Data Processing
- Combined problem **title**, **description**, **input format**, and **output format** into a single text field
- Normalized text *(lowercasing, whitespace cleanup)*
- Added an auxiliary numeric feature: `text_length`

---

### 2️⃣ Feature Engineering
- **TF-IDF Vectorization**
  - n-grams *(1–3)*
  - Vocabulary pruning (`min_df`, `max_df`)
  - Sublinear term-frequency scaling

---

### 3️⃣ Difficulty Classification *(Easy / Medium / Hard)*

#### Models Tried
- Logistic Regression *(baseline)*
- Linear SVM *(final choice)*

#### ✅ Final Model
- **Linear SVM with improved TF-IDF**
- Selected based on better **F1-score** on the dominant **hard** class

---

### 4️⃣ Difficulty Score Regression

#### 🎯 Target
- `problem_score` *(continuous difficulty value)*

#### Models Evaluated
- Ridge Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

#### ✅ Final Model
- **Ridge Regression**

Tree-based models underperformed due to **sparse, high-dimensional TF-IDF features**.  
Model selection was based on **empirical evaluation**, not complexity.

---

### 5️⃣ Web Application (Flask)

A simple **Flask web app** allows users to:

1. Paste a programming problem description  
2. Click **Predict**  
3. Receive:
   - Predicted difficulty class
   - Predicted difficulty score

---

## 📊 Results Summary

### 🔹 Classification
- **Accuracy ≈ 49%**
- Close to strong baselines given:
  - Subjective labels
  - Text-only features
  - Class imbalance

### 🔹 Regression
- **Mean Absolute Error (MAE) ≈ 1.6**
- Predictions typically within **~1.5 difficulty points**

These results are **realistic and defensible** for text-only difficulty estimation.

---

## 🛠 Tech Stack

- **Language**: Python  
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `scipy`  
- **NLP**: TF-IDF  
- **Models**:
  - Linear SVM *(classification)*
  - Ridge Regression *(regression)*
- **Web Framework**: Flask  
- **Version Control**: Git & GitHub  

---

## 📂 Project Structure
### AutoJudge/
- data/ 
- models/ 
- notebooks/data_exploration.ipynb
- templates/index.html 
- .gitignore
- app.py 
- README.md
- requirements.txt

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
git clone <your-repo-url>
cd AutoJudge

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Run the Flask app
python app.py

### 4️⃣ Open in browser
http://127.0.0.1:5000

---

## ⚠️ Limitations
- **Uses only textual information**
- **Does not include:**
    - Editorial solutions
    - Code submissions
    - Acceptance rates
    - Problem tags
    - Including these features would likely improve performance.

    ---

## 🔮 Future Improvements
- Incorporate problem constraints and tags
- Use transformer-based embeddings (e.g., BERT)
- Add confidence estimates for predictions
- Improve UI styling
- Deploy the application online (Render / Railway)

## 👤 Author
Bhoomika Chourasiya
B.Tech Computer Science & Engineering
Indian Institute of Technology, Roorkee

## ⭐ Final Note
This project prioritizes methodology, reasoning, and completeness over chasing unrealistic accuracy numbers, reflecting real-world ML workflows and engineering best practices.