"""
EMIPredict AI - Regression Model Training
Trains Linear Regression, Random Forest, and XGBoost to predict
max_monthly_emi (continuous, INR 500-50,000), logs every run to MLflow,
and saves the best-performing model to models/.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

TARGET = "max_monthly_emi"
DROP_COLS = ["emi_eligibility", "max_monthly_emi"]  # emi_eligibility is the classification target, not a feature

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("EMI_Regression")


def load_data():
    train = pd.read_csv("data/train_features.csv")
    val = pd.read_csv("data/val_features.csv")
    test = pd.read_csv("data/test_features.csv")
    return train, val, test


def prep_xy(df):
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET]
    return X, y


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def train_all():
    train, val, test = load_data()

    X_train, y_train = prep_xy(train)
    X_val, y_val = prep_xy(val)
    X_test, y_test = prep_xy(test)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            objective="reg:squarederror", random_state=42, n_jobs=-1
        ),
    }

    results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            mlflow.log_params(model.get_params())

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            metrics = evaluate(y_val, y_pred)

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                                    model,
                                    name="model",
                                    skops_trusted_types=[
                                        "xgboost.core.Booster",
                                        "xgboost.sklearn.XGBRegressor"
                                    ]
                                )

            print(f"{name} validation metrics: {metrics}")
            results[name] = {"model": model, "metrics": metrics}

    # --- Select best model by RMSE (lower is better; project target is < 2000 INR) ---
    best_name = min(results, key=lambda n: results[n]["metrics"]["rmse"])
    best_model = results[best_name]["model"]
    print(f"\nBest regression model: {best_name} (RMSE = {results[best_name]['metrics']['rmse']:.2f})")

    # --- Final evaluation on held-out test set ---
    y_test_pred = best_model.predict(X_test)
    test_metrics = evaluate(y_test, y_test_pred)
    print(f"Best model test-set metrics: {test_metrics}")

    joblib.dump(best_model, "models/best_regression_model.pkl")
    joblib.dump(best_name, "models/best_regression_model_name.pkl")

    comparison = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T
    comparison.to_csv("models/regression_comparison.csv")
    print(f"\nSaved best model ({best_name}) and comparison table to models/")

    return results, best_name, test_metrics


if __name__ == "__main__":
    train_all()
