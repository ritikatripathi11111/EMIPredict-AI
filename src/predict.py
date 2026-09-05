"""
EMIPredict AI - Prediction Pipeline
Takes a single customer's raw financial profile (as a dict), applies the same
feature engineering + encoding + scaling used in training, runs both the
classification and regression models, and returns a structured result
including a plain-language risk explanation.
"""

import pandas as pd
import numpy as np
import joblib
import os

from feature_engineering import add_derived_features, CATEGORICAL_COLS

MODELS_DIR = "models"

_cache = {}


def _load_artifacts():
    """Lazy-load and cache all model/encoder artifacts (avoids reloading per request)."""
    if _cache:
        return _cache

    _cache["clf_model"] = joblib.load(f"{MODELS_DIR}/best_classification_model.pkl")
    _cache["clf_target_encoder"] = joblib.load(f"{MODELS_DIR}/classification_target_encoder.pkl")
    _cache["reg_model"] = joblib.load(f"{MODELS_DIR}/best_regression_model.pkl")
    _cache["label_encoders"] = joblib.load(f"{MODELS_DIR}/label_encoders.pkl")
    _cache["scaler"] = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    _cache["scale_cols"] = joblib.load(f"{MODELS_DIR}/scale_cols.pkl")

    try:
        _cache["clf_name"] = joblib.load(f"{MODELS_DIR}/best_classification_model_name.pkl")
        _cache["reg_name"] = joblib.load(f"{MODELS_DIR}/best_regression_model_name.pkl")
    except FileNotFoundError:
        _cache["clf_name"] = "unknown"
        _cache["reg_name"] = "unknown"

    return _cache


def _prepare_input(customer: dict) -> pd.DataFrame:
    """Turns a raw customer dict into a single-row, fully engineered/encoded/scaled dataframe."""
    artifacts = _load_artifacts()
    df = pd.DataFrame([customer])

    df = add_derived_features(df)

    encoders = artifacts["label_encoders"]
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        val = str(df.at[0, col])
        df[col] = le.transform([val])[0] if val in le.classes_ else -1

    scaler = artifacts["scaler"]
    scale_cols = artifacts["scale_cols"]
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df


def _risk_explanation(customer: dict, eligibility: str, max_emi: float, requested_emi: float) -> dict:
    """Rule-based explanation layer -- summarizes the key financial signals in plain language."""
    checks = []

    credit_score = customer["credit_score"]
    checks.append({
        "label": "Credit Score",
        "status": "good" if credit_score >= 700 else ("warning" if credit_score >= 600 else "poor"),
        "detail": f"{credit_score:.0f}"
    })

    years_emp = customer["years_of_employment"]
    checks.append({
        "label": "Employment Stability",
        "status": "good" if years_emp >= 3 else ("warning" if years_emp >= 1 else "poor"),
        "detail": f"{years_emp:.1f} years"
    })

    salary = customer["monthly_salary"]
    dti = customer["current_emi_amount"] / max(salary, 1)
    checks.append({
        "label": "Debt-to-Income",
        "status": "good" if dti < 0.2 else ("warning" if dti < 0.4 else "poor"),
        "detail": f"{dti*100:.1f}%"
    })

    ef_months = customer["emergency_fund"] / max(salary, 1)
    checks.append({
        "label": "Emergency Fund",
        "status": "good" if ef_months >= 3 else ("warning" if ef_months >= 1 else "poor"),
        "detail": f"{ef_months:.1f}x monthly salary"
    })

    if requested_emi <= max_emi:
        affordability = "COMFORTABLE"
        recommendation = f"Customer appears {eligibility.lower().replace('_', ' ')} for the requested EMI."
    else:
        affordability = "STRETCHED"
        recommendation = (
            f"Requested EMI (₹{requested_emi:,.0f}) exceeds the recommended safe maximum "
            f"(₹{max_emi:,.0f}). Consider a lower amount or longer tenure."
        )

    return {
        "checks": checks,
        "affordability_status": affordability,
        "recommendation": recommendation,
    }


def predict(customer: dict) -> dict:
    """
    customer: dict with all 22 raw input fields (see data dictionary / app form).
    Returns eligibility, risk level, max safe EMI, requested EMI, and explanation.
    """
    artifacts = _load_artifacts()

    X = _prepare_input(customer)
    drop_cols = [c for c in ["emi_eligibility", "max_monthly_emi"] if c in X.columns]
    X_model = X.drop(columns=drop_cols, errors="ignore")

    clf_pred_idx = artifacts["clf_model"].predict(X_model)[0]
    clf_proba = artifacts["clf_model"].predict_proba(X_model)[0]
    eligibility = artifacts["clf_target_encoder"].inverse_transform([clf_pred_idx])[0]
    confidence = float(np.max(clf_proba))

    # Keep probabilities aligned with the original class labels for UI visualization.
    class_labels = artifacts["clf_target_encoder"].inverse_transform(
        np.arange(len(clf_proba))
    )
    probabilities = {
        str(label): float(prob)
        for label, prob in zip(class_labels, clf_proba)
    }

    max_emi = float(artifacts["reg_model"].predict(X_model)[0])
    max_emi = float(np.clip(max_emi, 500, 50000))

    requested_emi = customer["requested_amount"] / max(customer["requested_tenure"], 1)

    risk_level = {"Eligible": "LOW", "High_Risk": "MEDIUM", "Not_Eligible": "HIGH"}.get(eligibility, "UNKNOWN")

    explanation = _risk_explanation(customer, eligibility, max_emi, requested_emi)

    return {
        "eligibility": eligibility,
        "risk_level": risk_level,
        "confidence": confidence,
        "probabilities": probabilities,
        "max_safe_emi": round(max_emi, 2),
        "requested_emi": round(requested_emi, 2),
        "classification_model": artifacts["clf_name"],
        "regression_model": artifacts["reg_name"],
        **explanation,
    }


if __name__ == "__main__":
    # quick manual smoke test
    sample_customer = {
        "age": 32, "gender": "Male", "marital_status": "Married", "education": "Graduate",
        "monthly_salary": 65000, "employment_type": "Private", "years_of_employment": 4.5,
        "company_type": "MNC", "house_type": "Rented", "monthly_rent": 15000,
        "family_size": 3, "dependents": 1, "school_fees": 0, "college_fees": 0,
        "travel_expenses": 3000, "groceries_utilities": 8000, "other_monthly_expenses": 2000,
        "existing_loans": "No", "current_emi_amount": 0, "credit_score": 740,
        "bank_balance": 150000, "emergency_fund": 100000, "emi_scenario": "Vehicle EMI",
        "requested_amount": 500000, "requested_tenure": 36,
    }
    result = predict(sample_customer)
    import json
    print(json.dumps(result, indent=2))
