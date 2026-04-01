# UMBC-DATA606-Capstone

# Patient Churn & Marketing Conversion Predictive Analytics  
### Healthcare Retention Intelligence System  

Prepared for UMBC Data Science Master Degree Capstone  
Dr. Chaojie (Jay) Wang  

**Author:** Prabhas Teja Penugonda  
GitHub Repository: (https://github.com/prabhasteja007/UMBC-DATA606-Capstone)  
LinkedIn: (https://www.linkedin.com/in/prabhas-teja/)

---

# 1. Background

## What is this project about?

This project focuses on building predictive machine learning models to:

1. Predict patient churn in healthcare systems  
2. Predict marketing campaign conversion outcomes  

Healthcare providers face major financial and operational challenges when patients discontinue services (churn). Simultaneously, marketing campaigns aim to acquire and convert new patients efficiently. This project combines both operational analytics and marketing intelligence into a unified predictive framework.

---

## Why does it matter?

- Patient retention is more cost-effective than acquisition.
- Early churn detection enables proactive intervention.
- Marketing conversion prediction improves campaign ROI.
- Data-driven decision systems enhance healthcare sustainability.

---

## Research Questions

1. Can we accurately predict which patients are likely to churn?
2. Which factors contribute most to churn behavior?
3. Can marketing data predict customer conversion effectively?
4. Which marketing features drive higher conversion probability?
5. How can predictive models be deployed into an interactive application?

---

# 2. Data

This project uses three structured CSV datasets:

- `patient_churn_main.csv`
- `patient_conversion_marketing.csv`
- `patient_churn_validation.csv`

---

## Dataset 1: Patient Churn Main

- **Rows:** 2,000  
- **Columns:** 21  
- **Unit of Observation:** One unique patient  

### Columns

- PatientID
- Age
- Gender
- State
- Tenure_Months
- Specialty
- Insurance_Type
- Visits_Last_Year
- Missed_Appointments
- Days_Since_Last_Visit
- Last_Interaction_Date
- Overall_Satisfaction
- Wait_Time_Satisfaction
- Staff_Satisfaction
- Provider_Rating
- Avg_Out_Of_Pocket_Cost
- Billing_Issues
- Portal_Usage
- Referrals_Made
- Distance_To_Facility_Miles
- **Churned (Target Variable)**

### Target Variable

- **Churned**
  - 0 = Active Patient
  - 1 = Churned Patient

---

## Dataset 2: Marketing Conversion Data

- **Rows:** 8,000  
- **Columns:** 20  
- **Unit of Observation:** One customer exposed to a marketing campaign  

### Columns

- CustomerID
- Age
- Gender
- Income
- CampaignChannel
- CampaignType
- AdSpend
- ClickThroughRate
- ConversionRate
- WebsiteVisits
- PagesPerVisit
- TimeOnSite
- SocialShares
- EmailOpens
- EmailClicks
- PreviousPurchases
- LoyaltyPoints
- AdvertisingPlatform
- AdvertisingTool
- **Conversion (Target Variable)**

### Target Variable

- **Conversion**
  - 0 = Not Converted
  - 1 = Converted

---

## Dataset 3: Churn Validation Dataset

- **Rows:** 500  
- **Columns:** 11  
- Used for independent validation testing  

### Columns

- Patient_ID
- Age
- Gender
- Tenure_Months
- Visits_Last_Year
- Chronic_Disease
- Insurance_Type
- Satisfaction_Score
- Total_Bill_Amount
- Missed_Appointments
- **Churn (Target Variable)**

---

# 3. Data Characteristics

## Data Type Summary

The datasets contain:

- Numerical features (Age, Tenure_Months, AdSpend, etc.)
- Categorical features (Gender, Insurance_Type, CampaignChannel)
- Behavioral features (Portal_Usage, EmailClicks, WebsiteVisits)
- Satisfaction metrics
- Financial metrics

---

## Feature Selection Strategy

### For Churn Prediction

Potential predictors:

- Tenure_Months
- Visits_Last_Year
- Missed_Appointments
- Satisfaction scores
- Billing_Issues
- Portal_Usage
- Distance_To_Facility_Miles
- Insurance_Type
- Provider_Rating
- Avg_Out_Of_Pocket_Cost

Target:

- Churned

---

### For Conversion Prediction

Potential predictors:

- CampaignChannel
- CampaignType
- AdSpend
- ClickThroughRate
- WebsiteVisits
- EmailOpens
- EmailClicks
- TimeOnSite
- LoyaltyPoints
- PreviousPurchases
- AdvertisingPlatform

Target:

- Conversion

---

## 4. Exploratory Data Analysis (EDA)

EDA was performed using Python and Plotly Express to understand data distributions, relationships, and key drivers of churn and conversion.

---

### 4.1 Data Cleaning and Preparation

The following preprocessing steps were applied:

- Removed duplicate records to ensure data integrity  
- Handled missing values using forward-fill imputation  
- Converted date columns into proper datetime format  
- Verified data consistency across datasets  
- Ensured tidy data structure (each row = one observation, each column = one feature)  

---

### 4.2 Univariate Analysis

#### Churn Distribution

The churn distribution provides a baseline understanding of churn prevalence and helps guide model evaluation strategies.

#### Age Distribution

Most patients fall within the middle-age group, indicating that this demographic dominates healthcare utilization.

---

### 4.3 Bivariate Analysis

#### Satisfaction vs Churn

Patients with lower satisfaction scores exhibit significantly higher churn rates, confirming that service quality is a critical retention factor.

#### Missed Appointments vs Churn

Higher missed appointments strongly correlate with churn, indicating disengagement behavior.

#### Tenure vs Churn

Patients with shorter tenure are more likely to churn, highlighting the importance of early-stage experience.

#### Portal Usage vs Churn

Higher portal usage is associated with lower churn, suggesting digital engagement improves retention.

#### Insurance Type vs Churn

Certain insurance categories show higher churn rates, indicating cost or access-related differences.

#### Billing Issues vs Churn

Although only a small proportion of patients experience billing issues, those who do show higher churn rates. This indicates that financial friction significantly impacts retention.

---

### 4.4 Correlation Analysis

Correlation analysis shows strong relationships between:

- Satisfaction metrics and churn  
- Missed appointments and churn  
- Tenure and churn (negative correlation)  

These variables are likely strong predictors in machine learning models.

---

### 4.5 Feature Engineering

A **Churn Risk Score** was created using:

- Satisfaction  
- Missed appointments  
- Tenure  

Patients were segmented into:
- Low Risk  
- Medium Risk  
- High Risk  

High-risk patients showed significantly higher churn rates, validating the effectiveness of the risk segmentation approach.

---

### 4.6 Marketing Analysis

#### Engagement Funnel

Higher website visits and interaction levels lead to increased conversion rates.

#### Cost Efficiency

Marketing channels vary in cost efficiency, indicating opportunities for budget optimization.

#### Diminishing Returns

Increasing ad spend improves conversion but at a decreasing rate, suggesting diminishing returns.

---

### 4.7 Key Insights

- Patient churn is driven by satisfaction, engagement, and tenure  
- Missed appointments act as early warning signals  
- Billing issues, though rare, significantly increase churn risk  
- Digital engagement improves retention  
- Marketing effectiveness depends on engagement depth  
- Increasing ad spend alone is not sufficient due to diminishing returns  
# 5. Model Training

## Models for Churn Prediction

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine

## Models for Conversion Prediction

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting

---

## Training Strategy

- Train/Test Split: 80/20
- 5-Fold Cross Validation
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis

---

## Evaluation Metrics

Since this is a classification problem:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Primary evaluation focus:
- Recall (to minimize false negatives in churn)
- ROC-AUC (overall discrimination power)

---

## Python Libraries

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- plotly
- joblib

---

# 6. Application Development

An interactive web application will be developed using:

## Streamlit (Primary Framework)

The application will allow users to:

- Input patient attributes
- Predict churn probability
- Predict marketing conversion probability
- View feature importance visualizations
- Support decision-making in real-time

---

# 7. Conclusion

## Summary

This capstone project integrates healthcare analytics and marketing intelligence to:

- Predict patient churn
- Predict marketing campaign conversion
- Improve operational efficiency
- Enhance data-driven healthcare strategy

---

## Limitations

- Synthetic/structured dataset constraints
- Potential class imbalance
- Limited temporal variables
- External socioeconomic factors not included

---

## Lessons Learned

- Feature engineering significantly impacts performance
- Behavioral variables strongly influence churn prediction
- Marketing engagement metrics are key drivers of conversion
- Model interpretability is critical in healthcare applications

---

## Future Work

- Incorporate time-series modeling
- Integrate real EMR systems
- Use SHAP for advanced model explainability
- Deploy production-level API
- Explore deep learning architectures

---

# 8. References

- Scikit-learn Documentation
- XGBoost Documentation
- Healthcare churn research literature
- Marketing analytics research papers

---

# Project Status

✔ Proposal Completed  
⬜ EDA In Progress  
⬜ Model Training  
⬜ Web App Deployment  
⬜ Final Report Submission  
