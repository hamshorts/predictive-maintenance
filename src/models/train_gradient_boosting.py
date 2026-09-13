from pathlib import Path
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
)

DATA_PATH = Path("data/processed/ai4i2020_model.csv")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("predictive-maintenance")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

with mlflow.start_run(run_name="gradient_boosting_baseline"):

    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 3)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_prob)

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("training_rows", len(X_train))
    mlflow.log_metric("test_rows", len(X_test))
    mlflow.log_metric("test_failures", int(y_test.sum()))

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model"
    )

    print("\nGRADIENT BOOSTING BASELINE")
    print("--------------------------")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Test failures: {y_test.sum()}")

    print("\nPROJECT METRICS")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 score: {f1:.3f}")
    print(f"Precision-Recall AUC: {pr_auc:.3f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, digits=3))