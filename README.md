# Credit Card Fraud Detection

## About
Built a machine learning model to detect fraudulent credit card 
transactions using Random Forest Classifier with proper 
train-test split before SMOTE to prevent data leakage.

## Dataset
Download from Kaggle: 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SMOTE

## Results
- Precision (Fraud): 0.87 (87%)
- Recall (Fraud): 0.85 (85%)
- F1-Score (Fraud): 0.86 (86%)
- ROC-AUC Score: 0.9849
- 83 out of 98 fraud cases detected in test data

## Steps
1. Exploratory Data Analysis (EDA)
2. Feature Scaling
3. Train-Test Split (80-20)
4. SMOTE only on Training Data (to prevent data leakage)
5. Random Forest Model Training
6. Evaluation
