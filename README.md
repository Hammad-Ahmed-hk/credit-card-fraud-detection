# 🛡️ Fraud Shield: Credit Card Fraud Detection

A Streamlit app and training notebook that classify credit card transactions as legitimate or fraudulent with a Random Forest model.

> **Read this first:** the headline metrics in the notebook (Random Forest AUC 1.0000, accuracy 99.91%) are **inflated by a data-leakage problem** explained in [Known issues](#known-issues). Do not treat this model as production-ready, and do not quote those numbers as real-world performance.

## What's inside

| File | Purpose |
|---|---|
| `CreditCardFraudDetection.ipynb` | Exploration, balancing, training, evaluation and model export |
| `app.py` | Streamlit app that loads the saved model and scores a transaction |
| `fraud_detection_rf_model.pkl` | Trained Random Forest |
| `scaler.pkl` | Saved `StandardScaler` (see known issues) |
| `*.png` | Charts exported from the notebook |
| `requirements.txt` | Python dependencies |

## Data

[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle: 284,807 European card transactions, of which 492 are fraud. After removing duplicate rows the notebook uses 283,726 rows, of which 473 are fraud (a 599:1 imbalance). Features are `V1` to `V28` (anonymised PCA components) plus `Amount` and `Time`. **The CSV is not included in this repository.** Download `creditcard.csv` from Kaggle and place it next to the notebook to re-run it.

## Run the app

```bash
git clone https://github.com/Hammad-Ahmed-hk/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
streamlit run app.py
```

The app expects **30 already-scaled features** (`V1`–`V28`, `scaled_Amount`, `scaled_Time`). It offers presets or lets you paste values.

## Results as reported by the notebook

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Random Forest (saved) | 99.91% | 1.0000 |
| Decision Tree | 98.66% | 0.9973 |
| Logistic Regression | 94.68% | 0.9890 |

These come from a test set that contains synthetic rows (see below), so they overstate real performance.

## Known issues

1. **SMOTE is applied before the train/test split (data leakage).** The notebook oversamples the whole dataset, then splits it. Synthetic fraud rows are interpolated from real ones, so near-copies of training rows land in the test set. That is why accuracy is near 100% and the test set is 50% fraud, when real fraud is about 0.17% of transactions.
   **Fix:** split first (stratified), apply SMOTE to the training set only, and evaluate on the untouched, imbalanced test set.
2. **Wrong metrics for imbalanced data.** Accuracy is misleading here. Report precision, recall and PR-AUC (average precision) on the untouched test set.
3. **One scaler fitted twice.** `scaler.pkl` was fitted on `Time` after being fitted on `Amount`, so it can only reliably scale `Time`. The app therefore asks for pre-scaled inputs rather than raw amounts. Use two scalers, or a `ColumnTransformer` inside a `Pipeline`, so raw values can be scored.
4. **Scaling before splitting.** The scaler sees the test data. It is a smaller effect than item 1 but worth fixing in the same pass.
5. **Unused imports** (`GradientBoostingClassifier`, `RandomUnderSampler`).

## Roadmap

- [ ] Re-train with split-then-SMOTE inside a `Pipeline` (imbalanced-learn)
- [ ] Report precision, recall and PR-AUC on an untouched test set
- [ ] Accept raw `Amount` and `Time` in the app
- [ ] Add a short model card with limitations

## Author

Hammad Ahmed, [@Hammad-Ahmed-hk](https://github.com/Hammad-Ahmed-hk)
