# Credit Card Fraud Detection

## About
Built an end-to-end machine learning pipeline to detect fraudulent 
credit card transactions using Random Forest Classifier.
Proper train-test split done before scaling and SMOTE 
to ensure zero data leakage.

## Dataset
Download from Kaggle: 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SMOTE

## Results (No Data Leakage)
- Precision (Fraud): 0.91 (91%)
- Recall (Fraud): 0.83 (83%)
- F1-Score (Fraud): 0.87 (87%)
- ROC-AUC Score: 0.9917
- 81 out of 98 fraud cases detected in test data

## Pipeline (Correct Order)
1. Exploratory Data Analysis (EDA)
2. Train-Test Split (80-20) 
3. Feature Scaling (fit only on train)
4. SMOTE only on Training Data
5. Random Forest Model Training
6. Evaluation

## Key Insights
- Dataset highly imbalanced: only 0.17% fraud cases
- V14 most important feature for fraud detection
- Fraud transactions show random time patterns
- Zero data leakage in entire pipeline
