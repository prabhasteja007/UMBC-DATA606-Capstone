# Notebooks Folder

This folder contains the Jupyter notebooks used for exploratory data analysis, feature engineering, machine learning model development, and evaluation for the Patient Churn Prediction and Healthcare Retention Analytics project.

---

## Contents

### `EDA.ipynb`
This notebook focuses on exploratory data analysis (EDA) and visualization of patient churn and marketing conversion datasets.

#### Key Components
- Data cleaning and preprocessing
- Missing value analysis
- Descriptive statistics
- Churn distribution analysis
- Correlation analysis
- Behavioral pattern analysis
- Risk segmentation
- Marketing conversion analysis
- Interactive visualizations using Plotly

#### Major Insights
- Lower patient satisfaction is associated with higher churn
- Missed appointments strongly correlate with churn risk
- Digital engagement improves retention
- Billing issues contribute to patient disengagement

---

### `Churn_analysis.ipynb`
This notebook contains the machine learning workflow for patient churn prediction.

#### Key Components
- Feature engineering
- Data preprocessing
- Train-test splitting
- Model training and evaluation
- Threshold tuning
- Feature importance analysis
- External validation testing
- Model serialization for deployment

#### Machine Learning Models
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

#### Evaluation Metrics
- ROC-AUC
- Precision
- Recall
- F1-Score
- Classification Report

#### Final Selected Model
- Random Forest Classifier

---

## Datasets Used

The notebooks use the following datasets:

- `patient_churn_main.csv`
- `patient_churn_validation.csv`
- `patient_conversion_marketing.csv`

---

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- Matplotlib
- Seaborn
- Joblib

---

## Purpose

The notebooks support:
- Exploratory data analysis
- Predictive modeling
- Healthcare analytics research
- Patient retention analysis
- Development of the deployed Streamlit application

---

## Academic Context

These notebooks were developed as part of the DATA606 Capstone Project for the:

**Master of Professional Studies in Data Science**  
University of Maryland, Baltimore County (UMBC)
