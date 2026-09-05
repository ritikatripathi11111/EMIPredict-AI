"""
EMIPredict AI - Data Preprocessing Pipeline
Cleans the raw EMI dataset: fixes malformed numerics, handles missing values,
validates ranges against the data dictionary, and produces a clean dataframe
ready for feature engineering.
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_PATH = "data/emi_prediction_dataset.csv"

NUMERIC_COLS = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure", "max_monthly_emi",
]

CATEGORICAL_COLS = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
]

TARGET_CLASSIFICATION = "emi_eligibility"
TARGET_REGRESSION = "max_monthly_emi"


def fix_malformed_numeric(series: pd.Series) -> pd.Series:
    """
    Fixes values like '58.0.0' or '23400.0.0.0' (repeated '.0' suffix typo)
    by collapsing to a single decimal point, then casts to float.
    Already-clean numeric values pass through unchanged.
    """
    def clean(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        # collapse repeated ".0" suffixes, e.g. "23400.0.0.0" -> "23400.0"
        s = re.sub(r"(\.0){2,}$", ".0", s)
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(clean)


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- 1. Fix malformed numeric strings (age, monthly_salary, bank_balance) ---
    for col in ["age", "monthly_salary", "bank_balance"]:
        df[col] = fix_malformed_numeric(df[col])

    # --- 2. Cast remaining numeric columns properly ---
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 3. Drop exact duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    # --- 4. Range validation against the data dictionary ---
    # credit_score should be 300-850 per spec; real data has 0-1200 outliers.
    invalid_credit = ((df["credit_score"] < 300) | (df["credit_score"] > 850)).sum()
    print(f"credit_score out-of-spec rows: {invalid_credit} -> clipping to [300, 850]")
    df["credit_score"] = df["credit_score"].clip(lower=300, upper=850)

    # age should be 25-60 per spec
    df["age"] = df["age"].clip(lower=18, upper=75)

    # max_monthly_emi spec range is 500-50000; a small number of rows exceed this.
    # We cap rather than drop, to avoid losing legitimate high-income records.
    over_cap = (df[TARGET_REGRESSION] > 50000).sum()
    print(f"max_monthly_emi rows above 50,000: {over_cap} -> capping at 50,000")
    df[TARGET_REGRESSION] = df[TARGET_REGRESSION].clip(upper=50000)

    # negative/zero values that shouldn't exist for these fields
    non_negative_cols = [
        "monthly_salary", "monthly_rent", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "current_emi_amount", "bank_balance", "emergency_fund",
        "requested_amount", "requested_tenure",
    ]
    for col in non_negative_cols:
        df[col] = df[col].clip(lower=0)

    # --- 5. Handle missing values ---
    # Numeric: median imputation (robust to outliers, doesn't distort ratios)
    numeric_missing = ["monthly_rent", "credit_score", "bank_balance", "emergency_fund"]
    for col in numeric_missing:
        median_val = df[col].median()
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed {n_missing} missing values in '{col}' with median={median_val:.1f}")

    # Categorical: mode imputation
    if df["education"].isna().sum() > 0:
        mode_val = df["education"].mode()[0]
        n_missing = df["education"].isna().sum()
        df["education"] = df["education"].fillna(mode_val)
        print(f"Imputed {n_missing} missing values in 'education' with mode='{mode_val}'")

    # --- 6. Drop rows still missing a target (can't train/evaluate on these) ---
    before = len(df)
    df = df.dropna(subset=[TARGET_CLASSIFICATION, TARGET_REGRESSION])
    print(f"Dropped {before - len(df)} rows with missing target values")

    # --- 7. Final sanity check: any remaining nulls ---
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls) > 0:
        print("WARNING: columns still containing nulls after cleaning:")
        print(remaining_nulls)
    else:
        print("No remaining nulls.")

    return df.reset_index(drop=True)


def train_test_val_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits into train / validation / test, stratified on the classification
    target so the (imbalanced) eligibility classes are represented proportionally
    in every split.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_CLASSIFICATION], random_state=random_state
    )
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val[TARGET_CLASSIFICATION],
        random_state=random_state
    )
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    df_raw = load_raw()
    print(f"Loaded raw data: {df_raw.shape}")

    df_clean = clean_data(df_raw)
    print(f"Cleaned data: {df_clean.shape}")

    train, val, test = train_test_val_split(df_clean)

    df_clean.to_csv("data/emi_cleaned.csv", index=False)
    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    print("Saved cleaned data and splits to data/")
