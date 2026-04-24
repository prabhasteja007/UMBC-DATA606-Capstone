# 🏥 Patient Churn Predictor (Streamlit App)

An interactive **machine learning web application** that predicts patient churn risk and provides actionable insights to improve retention in healthcare systems.

Built as part of a **UMBC Data Science Capstone Project**, this app demonstrates end-to-end ML deployment — from model training to real-time prediction.

---

## 🚀 Overview

Patient churn is a critical problem in healthcare, impacting both **patient outcomes** and **revenue stability**. This application enables healthcare providers to:

- Predict whether a patient is likely to churn
- Understand key risk factors
- Take proactive intervention steps

The app uses a **Random Forest model** trained on structured patient data and provides a clean, intuitive UI for decision-making.

---

## 🎯 Key Features

### 🔮 Prediction
- Real-time churn probability prediction
- Risk categorization: **Low / Medium / High**

### 📊 Visualization
- Risk gauge chart
- Feature impact analysis
- Patient engagement metrics

### 💡 Decision Support
- Automated intervention recommendations
- Behavioral + satisfaction-based insights

### 📂 Batch Processing
- Upload CSV files for bulk predictions

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest Classifier  
- **Training Data:** 2,000 patient records  
- **Evaluation Metric:** ROC-AUC ≈ 0.647  
- **Target Variable:** `Churned` (0 = No, 1 = Yes)

### 🔑 Important Features
- Days Since Last Visit
- Overall Satisfaction
- Missed Appointments
- Tenure
- Cost per Visit
- Distance to Facility

The model also includes **feature engineering**, such as:
- Engagement Score = Visits − Missed Appointments
- Cost Efficiency metrics
- Average Satisfaction score

---

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo/app
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Running the App
```bash
streamlit run app.py
```
```bash
http://localhost:8501
```
