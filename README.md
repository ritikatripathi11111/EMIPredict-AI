# EMIPredict AI — Intelligent Financial Risk Assessment Platform

A FinTech ML platform that predicts EMI loan eligibility (classification) and
the maximum monthly EMI a customer can safely afford (regression), backed by
MLflow experiment tracking and served through a multi-page Streamlit app.

## Architecture

```
Dataset (400K records) → Preprocessing → Feature Engineering
        → Classification Pipeline (Logistic Regression / Random Forest / XGBoost)
        → Regression Pipeline (Linear Regression / Random Forest / XGBoost)
        → MLflow Tracking & Model Selection
        → Streamlit App (Dashboard, Prediction, Data Explorer, Model Performance, Admin)
        → Streamlit Cloud Deployment
```

## Project Structure

```
EMIPredict-AI/
├── data/                       # raw + processed datasets (not committed if large)
├── src/
│   ├── preprocessing.py        # cleaning, validation, train/val/test split
│   ├── feature_engineering.py  # derived ratios, encoding, scaling
│   ├── train_classification.py # 3 classifiers + MLflow logging
│   ├── train_regression.py     # 3 regressors + MLflow logging
│   ├── predict.py              # inference pipeline used by the app
│   └── database.py             # SQLite CRUD for customer records
├── models/                     # saved best models + comparison tables (generated)
├── mlruns/                     # MLflow tracking store (generated)
├── app.py                      # Streamlit application
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the Full Pipeline

```bash
# 1. Clean and split the raw dataset
python src/preprocessing.py

# 2. Engineer features (ratios, encoding, scaling)
python src/feature_engineering.py

# 3. Train and compare classification models (logged to MLflow)
python src/train_classification.py

# 4. Train and compare regression models (logged to MLflow)
python src/train_regression.py

# 5. Launch the app
streamlit run app.py

# Optional: inspect MLflow experiment comparisons
mlflow ui --backend-store-uri mlruns
```

## Data Quality Notes

The raw dataset required cleaning before use:

- **Malformed numerics**: `age`, `monthly_salary`, and `bank_balance` contained
  values like `'23400.0.0'` (repeated decimal-suffix typo) — fixed via regex
  normalization before casting to float.
- **Missing values**: ~0.6% nulls each in `education`, `monthly_rent`,
  `credit_score`, `bank_balance`, `emergency_fund` — imputed with
  median (numeric) / mode (categorical).
- **Out-of-spec ranges**: `credit_score` had values outside the documented
  300–850 range (up to 1200) — clipped. `max_monthly_emi` had 458 rows above
  the documented ₹50,000 cap — clipped rather than dropped, to preserve
  legitimate high-income records.
- **Class imbalance**: `emi_eligibility` is heavily skewed
  (Not_Eligible ≈77%, Eligible ≈18%, High_Risk ≈4%). Models use
  `class_weight="balanced"`, and **macro F1 / ROC-AUC** — not raw accuracy —
  drive model selection, since a naive classifier could hit >90% accuracy by
  never predicting High_Risk.

## Feature Engineering

Derived features include: `debt_to_income`, `expense_to_income`,
`emi_to_income`, `loan_to_income`, `financial_buffer_ratio`,
`emergency_fund_months`, `credit_risk_score` (binned), employment stability
indicators, and interaction terms (`income_x_credit`, `debt_x_dependents`).

## Model Selection

- **Classification**: selected by macro F1 (robust to class imbalance)
- **Regression**: selected by RMSE (target: < ₹2,000 per project spec)

Final selected models and comparison metrics are written to
`models/classification_comparison.csv` and `models/regression_comparison.csv`,
and surfaced on the app's Model Performance page.

## Deployment

1. Push this repository to GitHub (exclude `data/*.csv` and `mlruns/` if large
   — add to `.gitignore`, or use Git LFS)
2. Connect the repo on [Streamlit Cloud](https://streamlit.io/cloud)
3. Set `app.py` as the entry point
4. Ensure `models/*.pkl` are committed or regenerated via a build step, since
   the app requires trained models to serve predictions

## Known Limitations / Next Steps

- Hyperparameters are lightly tuned defaults, not exhaustively searched
- MLflow model registry (versioning) is not yet wired up — runs are logged
  but not promoted to registry stages
- Admin CRUD uses SQLite for simplicity; not intended for concurrent
  production writes
