# Patient Churn Prediction and Healthcare Retention Analytics

A machine learning–based healthcare analytics project developed to predict patient churn risk and support proactive patient retention strategies through predictive modeling and interactive analytics.

---

## Project Overview

Patient retention is an important challenge in healthcare systems because patient disengagement can negatively affect continuity of care, operational efficiency, patient outcomes, and organizational revenue. This project applies machine learning techniques to identify patients at risk of churn using demographic, behavioral, financial, satisfaction, and engagement-related variables.

The project follows a complete end-to-end data science workflow, including:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model development
- Model evaluation and validation
- Streamlit-based web application deployment

The final system provides an interactive interface for healthcare analytics and patient risk assessment.

---

## Live Application

**Streamlit Deployment:**  
https://patientchurnpredictor.streamlit.app/

---

## Repository Structure

```bash
UMBC-DATA606-Capstone/
│
├── app.py
├── churn_analysis.py
├── eda.py
├── requirements.txt
│
├── model/
│   ├── churn_model.pkl
│   ├── model_columns.pkl
│   └── best_threshold.pkl
│
├── docs/
│   ├── report.md
│   └── README.md
│
├── data/
│   ├── patient_churn_main.csv
│   ├── patient_churn_validation.csv
│   └── patient_conversion_marketing.csv
│
└── README.md
```

---

## Objectives

The primary objectives of this project are:

1. Analyze factors influencing patient churn
2. Develop predictive machine learning models for churn prediction
3. Identify behavioral indicators associated with disengagement
4. Evaluate model performance using appropriate classification metrics
5. Build an interactive web application for healthcare analytics and risk assessment

---

## Dataset Description

The project utilizes three datasets:

### 1. Main Patient Churn Dataset
- **Rows:** 2,000
- **Columns:** 21

### 2. Validation Dataset
- **Rows:** 500
- **Columns:** 11

### 3. Marketing Conversion Dataset
Used for marketing and engagement analysis.

### Features Include
- Demographic information
- Insurance type
- Visit frequency
- Missed appointments
- Satisfaction metrics
- Provider ratings
- Billing issues
- Portal usage
- Distance to healthcare facility

### Target Variable
- `Churned`
  - `1 = Patient churned`
  - `0 = Patient retained`

---

## Exploratory Data Analysis

Exploratory analysis was conducted using Pandas, Plotly, and Seaborn to identify patterns and relationships within the data.

### Key Findings
- Lower satisfaction scores were associated with higher churn probability
- Missed appointments were strong indicators of patient disengagement
- Patients with shorter tenure demonstrated higher churn rates
- Portal usage was associated with improved retention
- Billing-related issues contributed to increased churn risk

### Analytical Components
- Distribution analysis
- Correlation analysis
- Risk segmentation
- Behavioral pattern analysis
- Marketing conversion analysis

---

## Feature Engineering

Several derived features were created to improve predictive performance.

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
Patients were categorized into:
- Low Risk
- Medium Risk
- High Risk

---

## Machine Learning Models

The following classification models were implemented and evaluated:

| Model | ROC-AUC Score |
|---|---|
| Random Forest | 0.6467 |
| XGBoost | 0.6318 |
| Logistic Regression | 0.6141 |

### Selected Model
The Random Forest classifier achieved the highest ROC-AUC score and was selected for deployment within the application.

### Important Predictive Features
- Days Since Last Visit
- Overall Satisfaction
- Distance to Facility
- Out-of-Pocket Cost
- Patient Tenure
- Satisfaction Average
- Engagement Metrics

---

## Web Application

A Streamlit-based web application was developed to provide an interactive interface for patient churn prediction.

### Application Features
- Real-time churn prediction
- Patient risk categorization
- Interactive visual analytics
- Feature contribution analysis
- Intervention recommendations
- Batch prediction functionality

### Intervention Recommendations
The application generates actionable recommendations such as:
- Proactive patient outreach
- Telehealth engagement
- Financial counseling
- Portal enrollment encouragement
- Patient advocacy support

---

## Technologies Used

### Programming and Data Analysis
- Python
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Data Visualization
- Plotly
- Matplotlib
- Seaborn

### Deployment
- Streamlit

---

## Installation and Execution

### Clone Repository

```bash
git clone https://github.com/prabhasteja007/UMBC-DATA606-Capstone.git
cd UMBC-DATA606-Capstone
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Academic Context

This project was completed as part of the DATA606 Capstone Project requirement for the:

**Master of Professional Studies in Data Science**  
University of Maryland, Baltimore County (UMBC)

---

## Future Work

Potential future enhancements include:
- Integration with real-world healthcare datasets
- Explainable AI techniques such as SHAP and LIME
- Time-series patient behavior modeling
- Deep learning approaches
- API integration with healthcare systems
- Improved external validation and calibration

---

## Author

**Prabhas Teja**  
Master of Professional Studies in Data Science  
University of Maryland, Baltimore County (UMBC)

GitHub:  
https://github.com/prabhasteja007

---

## License

This repository is intended for academic and educational purposes.
