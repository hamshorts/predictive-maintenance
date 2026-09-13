from pathlib import Path
import numpy as np
import pandas as pd

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

DATA_PATH = Path("data/processed/ai4i2020_model.csv")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("predictive-maintenance")

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# First hold out 20% as the final test set.
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Split the remaining 80% into:
# 60% training
# 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev,
)

print("\nDATA SPLIT")
print("----------")
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")

# --------------------------------------------------
# Candidate models
# --------------------------------------------------

models = {
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

# We decided recall is especially important.
MINIMUM_RECALL = 0.80

thresholds = np.arange(0.05, 0.96, 0.01)

# --------------------------------------------------
# Tune each candidate
# --------------------------------------------------

for model_name, model in models.items():

    print(f"\n{'=' * 60}")
    print(model_name.upper())
    print("=" * 60)

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    val_pr_auc = average_precision_score(
    y_val,
    val_prob
)

    threshold_results = []

    for threshold in thresholds:

        val_pred = (val_prob >= threshold).astype(int)

        precision = precision_score(
            y_val,
            val_pred,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            val_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            val_pred,
            zero_division=0
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results_df = pd.DataFrame(threshold_results)

    # Only consider thresholds that meet our recall requirement.
    eligible = results_df[
        results_df["recall"] >= MINIMUM_RECALL
    ]

    if eligible.empty:
        print(
            f"No threshold achieved recall >= "
            f"{MINIMUM_RECALL:.2f}"
        )
        continue

    # Among thresholds meeting recall goal,
    # choose the one with highest precision.
    best_row = eligible.sort_values(
        by=["precision", "f1"],
        ascending=False
    ).iloc[0]

    best_threshold = float(best_row["threshold"])

    print("\nBEST VALIDATION THRESHOLD")
    print("-------------------------")
    print(f"Threshold: {best_threshold:.2f}")
    print(
        f"Validation precision: "
        f"{best_row['precision']:.3f}"
    )
    print(
        f"Validation recall: "
        f"{best_row['recall']:.3f}"
    )
    print(
        f"Validation F1: "
        f"{best_row['f1']:.3f}"
    )
    print(
    f"Validation PR-AUC: "
    f"{val_pr_auc:.3f}"
)

    # --------------------------------------------------
    # Evaluate chosen threshold ONCE on final test set
    # --------------------------------------------------

    test_prob = model.predict_proba(X_test)[:, 1]

    test_pred = (
        test_prob >= best_threshold
    ).astype(int)

    test_precision = precision_score(
        y_test,
        test_pred,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_pred,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_prob
    )

    print("\nFINAL TEST PERFORMANCE")
    print("----------------------")
    print(f"Precision: {test_precision:.3f}")
    print(f"Recall: {test_recall:.3f}")
    print(f"F1 score: {test_f1:.3f}")
    print(f"PR-AUC: {test_pr_auc:.3f}")

    # --------------------------------------------------
    # Log threshold experiment in MLflow
    # --------------------------------------------------

    with mlflow.start_run(
        run_name=f"{model_name}_threshold_tuning"
    ):

        mlflow.log_param("model_type", model_name)
        mlflow.log_param(
            "minimum_required_recall",
            MINIMUM_RECALL
        )
        mlflow.log_param(
            "selected_threshold",
            best_threshold
        )
        mlflow.log_param(
            "train_rows",
            len(X_train)
        )
        mlflow.log_param(
            "validation_rows",
            len(X_val)
        )
        mlflow.log_param(
            "test_rows",
            len(X_test)
        )

        mlflow.log_metric(
            "test_precision",
            test_precision
        )
        mlflow.log_metric(
            "test_recall",
            test_recall
        )
        mlflow.log_metric(
            "test_f1",
            test_f1
        )
        mlflow.log_metric(
            "test_pr_auc",
            test_pr_auc
        )
        mlflow.log_metric(
            "validation_pr_auc",
            val_pr_auc
        )