"""
EMIPredict AI - Classification Model Training
Trains Logistic Regression, Random Forest, and XGBoost to predict
emi_eligibility (Eligible / High_Risk / Not_Eligible), logs every run to
MLflow, and saves the best-performing model to models/.

Note: classes are imbalanced (Not_Eligible ~77%, High_Risk ~4%), so we use
class_weight='balanced' (or scale_pos_weight equivalent for XGBoost) and
report F1/ROC-AUC alongside accuracy -- accuracy alone is misleading here.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

TARGET = "emi_eligibility"
DROP_COLS = ["emi_eligibility", "max_monthly_emi"]  # max_monthly_emi is the regression target, not a feature

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("EMI_Classification")


def load_data():
    train = pd.read_csv("data/train_features.csv")
    val = pd.read_csv("data/val_features.csv")
    test = pd.read_csv("data/test_features.csv")
    return train, val, test


def prep_xy(df, target_encoder):
    X = df.drop(columns=DROP_COLS)
    y = target_encoder.transform(df[TARGET])
    return X, y


def evaluate(y_true, y_pred, y_proba, classes):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        metrics["roc_auc_ovr"] = np.nan
    return metrics


def train_all():
    train, val, test = load_data()

    target_encoder = LabelEncoder()
    target_encoder.fit(train[TARGET])
    joblib.dump(target_encoder, "models/classification_target_encoder.pkl")

    X_train, y_train = prep_xy(train, target_encoder)
    X_val, y_val = prep_xy(val, target_encoder)
    X_test, y_test = prep_xy(test, target_encoder)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=-1
        ),
    }

    results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            mlflow.log_params(model.get_params())

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            metrics = evaluate(y_val, y_pred, y_proba, target_encoder.classes_)

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                                model,
                                name="model",
                                skops_trusted_types=[
                                    "xgboost.core.Booster",
                                    "xgboost.sklearn.XGBClassifier"
                                ]
                            )

            print(f"{name} validation metrics: {metrics}")
            results[name] = {"model": model, "metrics": metrics}

    # --- Select best model by macro F1 (robust to class imbalance) ---
    best_name = max(results, key=lambda n: results[n]["metrics"]["f1_macro"])
    best_model = results[best_name]["model"]
    print(f"\nBest classification model: {best_name} (F1 macro = {results[best_name]['metrics']['f1_macro']:.4f})")

    # --- Final evaluation on held-out test set ---
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    test_metrics = evaluate(y_test, y_test_pred, y_test_proba, target_encoder.classes_)
    print(f"Best model test-set metrics: {test_metrics}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_test_pred, target_names=target_encoder.classes_))
    print("Confusion matrix (test set):")
    print(confusion_matrix(y_test, y_test_pred))

    joblib.dump(best_model, "models/best_classification_model.pkl")
    joblib.dump(best_name, "models/best_classification_model_name.pkl")

    comparison = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T
    comparison.to_csv("models/classification_comparison.csv")
    print(f"\nSaved best model ({best_name}) and comparison table to models/")

    return results, best_name, test_metrics


if __name__ == "__main__":
    train_all()
