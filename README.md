# 🏥 Patient Churn Prediction & Healthcare Retention Analytics

AI-powered healthcare analytics system for predicting patient churn and supporting proactive retention strategies.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

Patient churn is a major challenge in healthcare systems because losing patients affects continuity of care, operational efficiency, and revenue. This project uses machine learning and healthcare analytics to predict patient churn risk using demographic, behavioral, satisfaction, engagement, and financial factors.

The project includes:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Training
- Model Evaluation
- Streamlit Web Application Deployment
- Patient Risk Assessment Dashboard

The deployed application allows healthcare organizations to identify high-risk patients and take proactive retention actions.

---

## 🚀 Live Demo

### 🌐 Streamlit App
👉 https://patientchurnpredictor.streamlit.app/

---

## 📂 Repository Structure

```bash
UMBC-DATA606-Capstone/
│
├── app.py                         # Streamlit web application
├── churn_analysis.py              # Model training and ML pipeline
├── eda.py                         # Exploratory data analysis
├── requirements.txt              # Python dependencies
│
├── model/
│   ├── churn_model.pkl           # Trained Random Forest model
│   ├── model_columns.pkl         # Encoded feature columns
│   └── best_threshold.pkl        # Tuned classification threshold
│
├── docs/
│   ├── report.md                 # Final project report
│   └── README.md                 # Docs folder README
│
├── data/
│   ├── patient_churn_main.csv
│   ├── patient_churn_validation.csv
│   └── patient_conversion_marketing.csv
│
└── README.md
```

---

## 🎯 Objectives

The primary objectives of this project are:

- Predict patient churn using machine learning
- Identify behavioral indicators of disengagement
- Analyze healthcare retention patterns
- Build an interactive healthcare analytics dashboard
- Provide actionable intervention recommendations

---

## 📊 Dataset Information

### Main Dataset
- **Rows:** 2,000
- **Columns:** 21

### Validation Dataset
- **Rows:** 500
- **Columns:** 11

### Features Include
- Demographics
- Insurance details
- Patient engagement metrics
- Satisfaction scores
- Billing issues
- Digital portal usage
- Visit frequency
- Distance to facility

### Target Variable
- `Churned`
  - `1 = Churned`
  - `0 = Retained`

---

## 🔍 Exploratory Data Analysis (EDA)

The project includes extensive EDA using:
- Plotly
- Pandas
- Seaborn

### Key Insights
- Lower satisfaction strongly correlates with churn
- Missed appointments increase churn risk
- Digital engagement improves retention
- Billing issues contribute to patient loss
- Early-stage patient experience is critical

### Visualizations Included
- Churn Distribution
- Satisfaction vs Churn
- Correlation Heatmap
- Risk Segmentation
- Marketing Conversion Analysis
- Behavioral Interaction Plots

---

## ⚙️ Feature Engineering

Custom engineered features include:

### Engagement Score
```python
Visits_Last_Year - Missed_Appointments
```

### Cost Per Visit
```python
Avg_Out_Of_Pocket_Cost / (Visits_Last_Year + 1)
```

### Satisfaction Average
```python
(
    Overall_Satisfaction +
    Wait_Time_Satisfaction +
    Staff_Satisfaction
) / 3
```

### Risk Segmentation
Patients are categorized into:
- Low Risk
- Medium Risk
- High Risk

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

| Model | ROC-AUC |
|---|---|
| Random Forest | 0.6467 |
| XGBoost | 0.6318 |
| Logistic Regression | 0.6141 |

### ✅ Selected Model
**Random Forest Classifier**

Reason:
- Best ROC-AUC score
- Better stability for deployment
- Improved threshold calibration

---

## 📈 Model Performance

### Best Model Metrics
- **ROC-AUC:** 0.647
- **Optimized Threshold:** 0.457
- **Balanced Classification Performance**

### Important Predictive Features
- Days Since Last Visit
- Overall Satisfaction
- Distance to Facility
- Out-of-Pocket Cost
- Patient Tenure
- Engagement Score

---

## 🖥️ Streamlit Application Features

### Interactive Risk Assessment
Users can input:
- Demographics
- Clinical details
- Satisfaction scores
- Engagement metrics
- Financial information

### Dashboard Features
- Churn probability prediction
- Risk categorization
- Feature contribution analysis
- Risk gauge visualization
- Recommended interventions
- Batch prediction upload support

### Recommended Actions
- Patient outreach
- Telehealth recommendations
- Financial counseling
- Portal enrollment encouragement
- Patient advocacy support

---

## 🛠️ Technologies Used

### Programming & Data Science
- Python
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Visualization
- Plotly
- Matplotlib
- Seaborn

### Deployment
- Streamlit

---

## 📦 Installation & Setup

### Clone Repository

```bash
git clone https://github.com/prabhasteja007/UMBC-DATA606-Capstone.git
cd UMBC-DATA606-Capstone
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 📸 Application Preview

### Features Included
- Real-time patient churn prediction
- Healthcare retention analytics
- Interactive visual dashboards
- Intervention recommendation engine

---

## 📚 Academic Context

This project was completed as part of the:

### UMBC DATA606 Capstone Project
Master of Professional Studies in Data Science  
University of Maryland, Baltimore County (UMBC)

---

## 🔮 Future Improvements

Potential future enhancements include:
- Real-world healthcare datasets
- Deep learning models
- Explainable AI (SHAP/LIME)
- Real-time API integration
- Time-series patient analytics
- Cloud deployment optimization

---

## 👤 Author

### Prabhas Teja

- GitHub: https://github.com/prabhasteja007
- LinkedIn: https://www.linkedin.com/in/prabhas-teja/

---

## 📄 License

This project is developed for academic and educational purposes as part of the UMBC Data Science Capstone program.

---
