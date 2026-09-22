
🔗 SDAIA Academy on GitHub

# Diabetes Classification

This is my training project for the **SDAIA Academy** program. The goal is to build a machine learning model that predicts whether a patient has diabetes based on their health data.

## About the project

The dataset (`diabetes_raw.csv`) has 1545 rows with columns like Age, Gender, BMI, Glucose, BloodPressure, Insulin, FamilyHistory, SmokingStatus, ActivityLevel, and the target column Diabetes (0 = no diabetes, 1 = diabetes).

What I did in the notebook:
- Explored the data (EDA) — checked distributions, class balance, and noticed some data issues (invalid zero values in Glucose/BloodPressure, inconsistent Gender labels like "F", "female", "Female ").
- Cleaned the data — handled missing values (filled with median/mode), removed duplicate rows, standardized the Gender column, converted the invalid zeros to NaN and imputed them.
- Encoded categorical columns and checked correlation with the target.
- Split the data (80/20, stratified) and scaled the numerical columns.
- Trained two models: **XGBoost** and **LightGBM**.
- Evaluated both with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and confusion matrices.
- Used **SHAP** to explain the models' predictions and see which features matter most.

## How to run it

Requirements:
```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap jupyter
```

Then just open the notebook and run the cells in order:
```
jupyter notebook Diabetes_Classification_Lesson.ipynb
```

Make sure `diabetes_raw.csv` is in the same folder as the notebook.

## Tools used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM, SHAP
