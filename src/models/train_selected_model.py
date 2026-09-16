from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

DATA_PATH = Path("data/processed/ai4i2020_model.csv")

REGISTERED_MODEL_NAME = "predictive-maintenance-gradient-boosting"

RANDOM_STATE = 42

SELECTED_THRESHOLD = 0.07


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("predictive-maintenance")


# ---------------------------------------------------------
# Load processed data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]


# ---------------------------------------------------------
# Reproduce the train / validation / test split used
# during threshold selection
# ---------------------------------------------------------

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    stratify=y_dev,
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------
# Locked Gradient Boosting configuration
# ---------------------------------------------------------

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE,
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Validation evaluation
# No tuning occurs here.
# We are evaluating the already-selected threshold.
# ---------------------------------------------------------

val_prob = model.predict_proba(X_val)[:, 1]

val_pred = (val_prob >= SELECTED_THRESHOLD).astype(int)

val_precision = precision_score(y_val, val_pred)
val_recall = recall_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred)
val_pr_auc = average_precision_score(y_val, val_prob)


# ---------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------

test_prob = model.predict_proba(X_test)[:, 1]

test_pred = (test_prob >= SELECTED_THRESHOLD).astype(int)

test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
test_pr_auc = average_precision_score(y_test, test_prob)


# ---------------------------------------------------------
# MLflow run
# ---------------------------------------------------------

with mlflow.start_run(
    run_name="gradient_boosting_selected_model"
):

    # Parameters
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("random_state", RANDOM_STATE)

    mlflow.log_param(
        "selected_threshold",
        SELECTED_THRESHOLD,
    )

    mlflow.log_param("training_rows", len(X_train))
    mlflow.log_param("validation_rows", len(X_val))
    mlflow.log_param("test_rows", len(X_test))

    # Validation metrics
    mlflow.log_metric(
        "validation_precision",
        val_precision,
    )

    mlflow.log_metric(
        "validation_recall",
        val_recall,
    )

    mlflow.log_metric(
        "validation_f1",
        val_f1,
    )

    mlflow.log_metric(
        "validation_pr_auc",
        val_pr_auc,
    )

    # Test metrics
    mlflow.log_metric(
        "test_precision",
        test_precision,
    )

    mlflow.log_metric(
        "test_recall",
        test_recall,
    )

    mlflow.log_metric(
        "test_f1",
        test_f1,
    )

    mlflow.log_metric(
        "test_pr_auc",
        test_pr_auc,
    )

    # Useful traceability tags
    mlflow.set_tag(
        "model_status",
        "selected_candidate",
    )

    mlflow.set_tag(
        "selection_basis",
        "validation_performance",
    )

    mlflow.set_tag(
        "purpose",
        "predictive_maintenance_failure_detection",
    )

    # Add input/output schema information
    signature = infer_signature(
        X_train,
        model.predict(X_train),
    )

    input_example = X_train.head(5)

    # Log model and automatically create the next
    # version in the existing Model Registry entry
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME,
    )


# ---------------------------------------------------------
# Console summary
# ---------------------------------------------------------

print()
print("SELECTED MODEL ARTIFACT CREATED")
print("-------------------------------")
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")
print()
print(f"Decision threshold: {SELECTED_THRESHOLD}")
print()
print("Validation:")
print(f"Precision: {val_precision:.3f}")
print(f"Recall:    {val_recall:.3f}")
print(f"F1:        {val_f1:.3f}")
print(f"PR-AUC:    {val_pr_auc:.3f}")
print()
print("Test:")
print(f"Precision: {test_precision:.3f}")
print(f"Recall:    {test_recall:.3f}")
print(f"F1:        {test_f1:.3f}")
print(f"PR-AUC:    {test_pr_auc:.3f}")
print()
print(
    "Registered model:",
    REGISTERED_MODEL_NAME,
)