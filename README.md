# Credit Card Fraud Detection

Out of 283,726 card transactions, only 473 are fraud. That's about 1 in 600. A model that says "everything is fine" every single time would still score 99.8% accuracy, and it would be completely useless.

This project is about doing better than that. I built a fraud detection pipeline on the well-known European cardholder dataset, compared three models with and without SMOTE, tuned the decision threshold, and ended up with a Random Forest that catches **77% of fraud while being right 91% of the time when it raises an alarm**.

---

## The Problem

Fraud is rare, expensive, and sneaky. Two things make it hard:

- **Extreme imbalance.** Fraud is just 0.17% of the data, so plain accuracy is a misleading metric.
- **Anonymised features.** V1–V28 are PCA components, so I can't just read a column and say "this looks suspicious". The model has to find the pattern.

So I judged every model on **Precision, Recall, F1 and PR-AUC**, not accuracy.

## Dataset

- Source: [Credit Card Fraud Detection (Kaggle / ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions over two days, 31 columns
- `Time`, `Amount`, `V1`–`V28` (PCA-transformed) and `Class` (0 = normal, 1 = fraud)
- After removing 1,081 duplicate rows: **283,726 rows** (283,253 normal, 473 fraud)
- No missing values

The CSV is not included in this repo because of its size. Download `creditcard.csv` from the link above and put it in the project folder.

## What I Did

1. **Cleaning.** Removed duplicates, checked for nulls, constant columns and duplicate features.
2. **EDA.** Looked at class balance, transaction amounts, fraud over time, outliers, and which features move with fraud.
3. **Split.** 80/20 train/test, stratified so both sets keep the same fraud ratio (378 fraud in train, 95 in test).
4. **Scaling.** `StandardScaler` fitted on the training data only, to avoid leakage.
5. **Baselines.** Logistic Regression, Decision Tree and Random Forest on the original imbalanced data.
6. **SMOTE.** Balanced the *training set only* (never the test set) and re-ran all three models.
7. **Threshold tuning.** Carved a validation set out of the training data and tested thresholds from 0.1 to 0.9, so the test set stayed untouched.
8. **Final model.** Retrained on full training data, evaluated once on the test set, looked at feature importance, saved with `joblib`.

## What the EDA Showed

- Fraud transactions have a **lower median amount** (about $9.82) than normal ones (about $22), but the average is higher because a few fraud cases are large.
- Fraud rate is not flat through the day. It spikes in the early hours of each day (0.54% in the 4–8 hour window versus 0.07% in the quietest window).
- Features most linked to fraud: **V17, V14, V12, V10, V16, V3, V7** (negative) and **V11, V4** (positive).

## Results

All numbers below are on the held-out test set (56,746 transactions, 95 of them fraud).

| Model | Sampling | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | No SMOTE | 0.846 | 0.579 | 0.688 | 0.956 | 0.692 |
| Logistic Regression | SMOTE | 0.053 | 0.874 | 0.100 | 0.963 | 0.675 |
| Decision Tree | No SMOTE | 0.720 | 0.705 | 0.713 | 0.852 | 0.509 |
| Decision Tree | SMOTE | 0.351 | 0.642 | 0.454 | 0.820 | 0.226 |
| Random Forest | No SMOTE | 0.972 | 0.726 | 0.831 | 0.924 | 0.788 |
| **Random Forest** | **SMOTE** | **0.913** | **0.768** | **0.834** | **0.969** | **0.811** |

A few honest observations:

- **SMOTE is not magic.** It pushed Logistic Regression's recall up to 87%, but precision collapsed to 5%. That means roughly 18 false alarms for every real fraud caught, which is not usable. It also hurt the Decision Tree.
- **Random Forest won.** With SMOTE it had the best ROC-AUC and PR-AUC, and the best balance between catching fraud and not annoying genuine customers.
- **Without SMOTE, Random Forest had higher precision** (0.972) but missed more fraud. Which one is "better" depends on whether a missed fraud or a false alarm costs more.

### Final Model: Random Forest + SMOTE (threshold 0.5)

|  | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 56,644 | 7 |
| **Actual Fraud** | 22 | 73 |

- 73 of 95 frauds caught
- Only 7 genuine transactions wrongly flagged out of 56,651
- 22 frauds still slipped through

On the validation set, threshold 0.5 gave the best F1, so I kept it. Lowering it to 0.1 catches more fraud (recall 0.89) but precision drops to 0.36, so the right threshold in a real system is a business decision.

### Top Features by Importance

V14, V10, V12, V17 and V4 carry most of the signal. This lines up with the correlation analysis from the EDA.

## Limitations

- Only 473 fraud examples, so results can shift with a different split. The test set has just 95 frauds.
- V1–V28 are anonymised, so I can't explain *why* a transaction was flagged in business terms.
- SMOTE creates synthetic samples that may not match real fraud behaviour.
- The model still misses about 23% of fraud.
- Data covers only two days, so it says nothing about long-term or seasonal patterns.

## Future Work

- Try XGBoost / LightGBM and cost-sensitive learning
- Use cross-validation for more stable numbers
- Add SHAP for per-transaction explanations
- Wrap the model in a FastAPI endpoint for real-time scoring

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · imbalanced-learn · joblib

## How to Run

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
```

1. Download `creditcard.csv` and place it in the project folder.
2. Open the notebook with `jupyter notebook` and run all cells from top to bottom.
3. The trained model and scaler are saved as `random_forest_fraud_model.pkl` and `fraud_scaler.pkl`.

To use the saved model:

```python
import joblib

model = joblib.load("random_forest_fraud_model.pkl")
scaler = joblib.load("fraud_scaler.pkl")

# X_new must have the same 30 columns: Time, V1..V28, Amount
fraud_prob = model.predict_proba(scaler.transform(X_new))[:, 1]
is_fraud = fraud_prob >= 0.5
```

## Author

**Rahul Kumar Mahto**
Final-year B.Tech ECE, NIT Silchar
[LinkedIn](#) · [GitHub](#)
