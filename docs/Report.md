# Patient Churn Prediction and Healthcare Retention Analytics  
### Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

**Author:** Prabhas Teja  
**GitHub Repository:** *Add your GitHub repository link here*  
**LinkedIn Profile:** *Add your LinkedIn profile link here*  
**PowerPoint Presentation:** *Add your presentation link here*  
**YouTube Presentation Video:** *Add your YouTube video link here*  

---

# 1. Background

Healthcare organizations face significant challenges in retaining patients and maintaining long-term engagement. Patient churn occurs when patients stop visiting a healthcare provider or discontinue care services. High churn rates can negatively impact continuity of care, operational efficiency, patient outcomes, and revenue generation.

This project focuses on predicting patient churn using machine learning techniques and developing an interactive web application that enables healthcare administrators to proactively identify high-risk patients. By leveraging patient demographics, engagement metrics, satisfaction scores, financial factors, and behavioral indicators, this project aims to support data-driven patient retention strategies.

The project also includes marketing conversion analytics to understand how digital engagement and marketing effectiveness influence patient acquisition and retention.

## Research Questions

1. What factors contribute most significantly to patient churn?
2. Can machine learning models accurately predict patient churn risk?
3. Which patient behaviors indicate early signs of disengagement?
4. How can healthcare organizations use predictive analytics to improve retention strategies?
5. How does digital engagement and marketing performance affect patient conversion and retention?

The project follows an end-to-end data science workflow including exploratory data analysis (EDA), feature engineering, model development, evaluation, and deployment through a Streamlit web application.

---

# 2. Data

The project uses three datasets:

1. `patient_churn_main.csv` – Primary dataset used for training and analysis  
2. `patient_churn_validation.csv` – External validation dataset  
3. `patient_conversion_marketing.csv` – Marketing and conversion analytics dataset  

The datasets were loaded and analyzed using Python, Pandas, Plotly, and Scikit-learn libraries.

## Dataset Size and Shape

### Main Churn Dataset
- Shape: **2000 rows × 21 columns**
- Each row represents **one patient record**

### Validation Dataset
- Shape: **500 rows × 11 columns**

### Marketing Dataset
- Used for conversion and campaign analysis

## Data Features

The main dataset contains patient demographic, behavioral, financial, and engagement-related variables.

| Column Name | Data Type | Description |
|---|---|---|
| PatientID | Object | Unique patient identifier |
| Age | Integer | Patient age |
| Gender | Object | Patient gender |
| State | Object | Patient state |
| Tenure_Months | Integer | Duration of relationship with provider |
| Specialty | Object | Medical specialty |
| Insurance_Type | Object | Insurance category |
| Visits_Last_Year | Integer | Number of visits in the previous year |
| Missed_Appointments | Integer | Number of missed appointments |
| Days_Since_Last_Visit | Integer | Days since last interaction |
| Overall_Satisfaction | Float | Patient satisfaction score |
| Wait_Time_Satisfaction | Float | Wait time satisfaction score |
| Staff_Satisfaction | Float | Staff satisfaction score |
| Provider_Rating | Float | Provider quality rating |
| Avg_Out_Of_Pocket_Cost | Integer | Average patient healthcare cost |
| Billing_Issues | Integer | Indicates billing-related problems |
| Portal_Usage | Integer | Indicates patient portal usage |
| Referrals_Made | Integer | Number of referrals |
| Distance_To_Facility_Miles | Float | Distance to healthcare facility |
| Churned | Integer | Target variable indicating churn |

## Target Variable

The target variable for machine learning is:

- **Churned**
  - `1 = Patient churned`
  - `0 = Patient retained`

## Selected Features for Modeling

The following variables were used as predictors:

- Demographic variables
- Satisfaction metrics
- Engagement metrics
- Financial indicators
- Digital engagement metrics
- Behavioral indicators
- Engineered features such as:
  - `Engagement_Score`
  - `Cost_Per_Visit`
  - `Satisfaction_Avg`

---

# 3. Exploratory Data Analysis (EDA)

EDA was performed using Jupyter Notebook, Pandas, Plotly Express, and Seaborn. The analysis focused on understanding patient behavior, churn patterns, and feature relationships.

## Data Cleaning

The following preprocessing steps were performed:

- Removed duplicate records
- Checked for missing values
- Converted date columns to datetime format
- Forward-filled missing values where appropriate
- Created readable labels for visualization

The dataset contained no missing values after cleaning.

## Key Findings

### Churn Distribution

The dataset showed a relatively high churn rate:

- **Churn Rate: 68.35%**

This indicated significant patient disengagement within the dataset.

### Satisfaction vs Churn

Patients with lower satisfaction scores showed substantially higher churn rates. Satisfaction metrics emerged as strong indicators of retention behavior.

### Missed Appointments vs Churn

Higher numbers of missed appointments strongly correlated with churn, suggesting disengagement behavior prior to patient loss.

### Tenure vs Churn

Patients with shorter tenure were more likely to churn. This highlighted the importance of early-stage patient experience and onboarding quality.

### Portal Usage vs Churn

Patients actively using the healthcare portal demonstrated lower churn rates, indicating that digital engagement positively impacts retention.

### Billing Issues vs Churn

Patients experiencing billing issues had significantly higher churn rates compared to those without billing problems. Financial friction was identified as an important churn driver.

## Correlation Analysis

A correlation heatmap was generated to analyze feature relationships. The analysis showed:

- Satisfaction variables negatively correlated with churn
- Missed appointments positively correlated with churn
- Tenure negatively correlated with churn

These features were later confirmed as important predictors in machine learning models.

## Feature Engineering

Several new features were created.

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

### Risk Score

A custom churn risk score was engineered using:
- Satisfaction
- Missed appointments
- Patient tenure

Patients were segmented into:
- Low Risk
- Medium Risk
- High Risk

---

# 4. Model Training

The project implemented multiple machine learning models to predict patient churn.

## Development Environment

The models were developed using:
- Google Colab
- Jupyter Notebook
- Python

## Python Libraries Used

The following packages were used:

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- Joblib
- Streamlit

## Data Preparation

### Train-Test Split

The dataset was split using:
- 80% training data
- 20% testing data

Stratified sampling was used to preserve churn distribution.

### Feature Encoding

Categorical variables were encoded using:

```python
pd.get_dummies()
```

### Scaling

StandardScaler was used for Logistic Regression.

```python
StandardScaler()
```

## Machine Learning Models

### Logistic Regression
- Balanced class weights
- Max iterations: 1000

### Random Forest Classifier
- 300 estimators
- Balanced class weights
- Parallel processing enabled

### XGBoost Classifier
- 500 estimators
- Learning rate: 0.03
- Max depth: 4

## Model Performance

| Model | ROC-AUC Score |
|---|---|
| Random Forest | 0.6467 |
| XGBoost | 0.6318 |
| Logistic Regression | 0.6141 |

The **Random Forest model** achieved the highest ROC-AUC score and was selected for deployment.

## Feature Importance

Top predictive features included:

1. Days_Since_Last_Visit
2. Overall_Satisfaction
3. Distance_To_Facility_Miles
4. Avg_Out_Of_Pocket_Cost
5. Tenure_Months
6. Age
7. Satisfaction_Avg
8. Cost_Per_Visit

## Validation Performance

A shared-feature XGBoost model was validated using the external validation dataset.

### Validation ROC-AUC
- **0.5131**

The lower performance on external validation indicated possible dataset distribution differences and highlighted opportunities for future improvement.

---

# 5. Application of the Trained Models

A Streamlit-based web application was developed to allow users to interact with the trained machine learning model and generate patient churn predictions in real time.

## Features of the Web Application

### Patient Risk Assessment

Users can input:
- Demographics
- Clinical information
- Engagement metrics
- Satisfaction scores
- Financial information

The application predicts:
- Churn probability
- Risk category
- Intervention recommendations

### Risk Categories
- Low Risk
- Medium Risk
- High Risk

### Interactive Visualizations

The application includes:
- Risk gauge charts
- Feature contribution analysis
- Behavioral analytics
- Batch prediction support

### Recommended Interventions

The app provides personalized intervention recommendations such as:
- Proactive outreach
- Patient advocacy
- Financial counseling
- Telehealth suggestions
- Portal enrollment promotion

---

# 6. Conclusion

This project successfully demonstrated the application of machine learning and healthcare analytics to predict patient churn and identify high-risk patients.

The analysis revealed that:
- Patient satisfaction strongly impacts retention
- Missed appointments are early indicators of churn
- Digital engagement improves patient loyalty
- Financial issues increase churn risk
- Early-stage patient experience is critical

Among the evaluated models, Random Forest achieved the best predictive performance and was integrated into a Streamlit application for real-time prediction and decision support.

## Limitations

Several limitations were identified:

- Moderate predictive performance
- Limited dataset size
- Potential synthetic nature of data
- External validation performance drop
- Limited temporal and longitudinal features

## Lessons Learned

This project provided experience in:
- Healthcare analytics
- Exploratory data analysis
- Feature engineering
- Machine learning model evaluation
- Streamlit application development
- Model deployment workflows

## Future Work

Future improvements may include:
- Larger real-world healthcare datasets
- Deep learning approaches
- Time-series modeling
- Explainable AI techniques
- Real-time healthcare integration
- Improved external validation

---

# 7. References

1. Scikit-learn Documentation  
   https://scikit-learn.org/

2. XGBoost Documentation  
   https://xgboost.readthedocs.io/

3. Streamlit Documentation  
   https://streamlit.io/

4. Plotly Documentation  
   https://plotly.com/python/

5. Pandas Documentation  
   https://pandas.pydata.org/

6. NumPy Documentation  
   https://numpy.org/

7. Project EDA Notebook

8. Project Model Training Notebook

9. Streamlit Application Source Code

10. Final Report Template
