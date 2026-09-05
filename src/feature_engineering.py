"""
EMIPredict AI - Feature Engineering
Builds derived financial ratios, risk-scoring features, and interaction
features, then encodes categoricals and scales numerics.
Fit encoders/scalers on TRAIN only, then reuse (transform) on val/test/inference
to avoid data leakage.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

CATEGORICAL_COLS = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
]

EPS = 1e-6  # avoid division-by-zero on ratio features


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Total monthly expenses ---
    df["total_monthly_expenses"] = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"]
        + df["travel_expenses"] + df["groceries_utilities"]
        + df["other_monthly_expenses"] + df["current_emi_amount"]
    )

    # --- Core affordability ratios ---
    df["debt_to_income"] = df["current_emi_amount"] / (df["monthly_salary"] + EPS)
    df["expense_to_income"] = df["total_monthly_expenses"] / (df["monthly_salary"] + EPS)
    df["remaining_income"] = df["monthly_salary"] - df["total_monthly_expenses"]
    df["financial_buffer_ratio"] = df["remaining_income"] / (df["monthly_salary"] + EPS)

    # --- Requested loan affordability ---
    df["requested_emi_estimate"] = df["requested_amount"] / (df["requested_tenure"] + EPS)
    df["emi_to_income"] = df["requested_emi_estimate"] / (df["monthly_salary"] + EPS)
    df["loan_to_income"] = df["requested_amount"] / (df["monthly_salary"] * 12 + EPS)

    # --- Savings / safety-net features ---
    df["emergency_fund_months"] = df["emergency_fund"] / (df["total_monthly_expenses"] + EPS)
    df["bank_balance_to_income"] = df["bank_balance"] / (df["monthly_salary"] + EPS)

    # --- Employment stability ---
    df["is_stable_employment"] = (df["years_of_employment"] >= 2).astype(int)
    df["employment_stability_score"] = np.clip(df["years_of_employment"] / 10, 0, 1)

    # --- Credit risk bucket (numeric, ordinal) ---
    df["credit_risk_score"] = pd.cut(
        df["credit_score"], bins=[0, 580, 670, 740, 800, 850],
        labels=[0, 1, 2, 3, 4]  # 0=poor risk .. 4=excellent
    ).astype(int)

    # --- Dependents burden ---
    df["dependents_ratio"] = df["dependents"] / (df["family_size"] + EPS)

    # --- Interaction features ---
    df["income_x_credit"] = df["monthly_salary"] * df["credit_risk_score"]
    df["debt_x_dependents"] = df["debt_to_income"] * (df["dependents"] + 1)

    return df


def encode_and_scale(train_df, val_df=None, test_df=None, artifacts_dir="models"):
    """
    Fits LabelEncoders + StandardScaler on train_df, applies to val/test.
    Saves fitted encoders/scaler to artifacts_dir for reuse at inference time.
    Returns transformed dataframes.
    """
    train_df = train_df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["emi_eligibility_encoded", "max_monthly_emi"]  # never scale targets
    scale_cols = [c for c in numeric_cols if c not in exclude and c != "max_monthly_emi"]

    scaler = StandardScaler()
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])

    joblib.dump(encoders, f"{artifacts_dir}/label_encoders.pkl")
    joblib.dump(scaler, f"{artifacts_dir}/scaler.pkl")
    joblib.dump(scale_cols, f"{artifacts_dir}/scale_cols.pkl")

    def transform(df):
        df = df.copy()
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            # unseen categories at inference -> map to a new 'unknown' code
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        df[scale_cols] = scaler.transform(df[scale_cols])
        return df

    val_out = transform(val_df) if val_df is not None else None
    test_out = transform(test_df) if test_df is not None else None

    return train_df, val_out, test_out, encoders, scaler


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    val = pd.read_csv("data/val.csv")
    test = pd.read_csv("data/test.csv")

    train_fe = add_derived_features(train)
    val_fe = add_derived_features(val)
    test_fe = add_derived_features(test)
    print(f"Added derived features. New shape: {train_fe.shape}")
    print("New columns:", [c for c in train_fe.columns if c not in train.columns])

    train_enc, val_enc, test_enc, encoders, scaler = encode_and_scale(train_fe, val_fe, test_fe)

    train_enc.to_csv("data/train_features.csv", index=False)
    val_enc.to_csv("data/val_features.csv", index=False)
    test_enc.to_csv("data/test_features.csv", index=False)
    print("Saved engineered + encoded feature sets to data/")
