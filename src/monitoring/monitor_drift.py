from pathlib import Path

import pandas as pd


REFERENCE_DATA_PATH = Path(
    "data/processed/ai4i2020_model.csv"
)

PREDICTION_LOG_PATH = Path(
    "logs/predictions.csv"
)

DRIFT_THRESHOLD_STD = 1.0
MIN_MONITORING_ROWS = 30


FEATURE_MAP = {
    "air_temperature": "Air temperature",
    "process_temperature": "Process temperature",
    "rotational_speed": "Rotational speed",
    "torque": "Torque",
    "tool_wear": "Tool wear",
}


def main():

    if not REFERENCE_DATA_PATH.exists():
        print("Reference dataset not found.")
        print(f"Expected location: {REFERENCE_DATA_PATH}")
        return

    if not PREDICTION_LOG_PATH.exists():
        print("Prediction log not found.")
        print(f"Expected location: {PREDICTION_LOG_PATH}")
        return

    reference_df = pd.read_csv(
        REFERENCE_DATA_PATH
    )

    incoming_df = pd.read_csv(
        PREDICTION_LOG_PATH
    )

    if incoming_df.empty:
        print("Prediction log contains no rows.")
        return

    print()
    print("DATA DRIFT MONITORING SUMMARY")
    print("-----------------------------")
    print(
        f"Incoming predictions analyzed: "
        f"{len(incoming_df)}"
    )
    print(
        f"Drift threshold: "
        f"{DRIFT_THRESHOLD_STD:.1f} "
        "reference standard deviations"
    )

    if len(incoming_df) < MIN_MONITORING_ROWS:
        print()
        print(
            "WARNING: Monitoring sample is small."
        )
        print(
            f"At least {MIN_MONITORING_ROWS} "
            "incoming rows are recommended before "
            "treating these drift results as meaningful."
        )

    print()
    print("Numeric feature comparison:")
    print()

    drifted_features = []

    for log_column, reference_column in FEATURE_MAP.items():

        reference_mean = (
            reference_df[reference_column].mean()
        )

        reference_std = (
            reference_df[reference_column].std()
        )

        incoming_mean = (
            incoming_df[log_column].mean()
        )

        if reference_std == 0:
            standardized_shift = 0.0
        else:
            standardized_shift = abs(
                incoming_mean - reference_mean
            ) / reference_std

        drift_detected = (
            standardized_shift >= DRIFT_THRESHOLD_STD
        )

        status = (
            "DRIFT FLAG"
            if drift_detected
            else "within range"
        )

        print(reference_column)
        print(
            f"  Reference mean: "
            f"{reference_mean:.2f}"
        )
        print(
            f"  Incoming mean:  "
            f"{incoming_mean:.2f}"
        )
        print(
            f"  Standardized shift: "
            f"{standardized_shift:.2f}"
        )
        print(
            f"  Status: {status}"
        )
        print()

        if drift_detected:
            drifted_features.append(
                reference_column
            )

    print("Overall result:")
    print("---------------")

    if drifted_features:
        print(
            "Potential drift detected in:"
        )

        for feature in drifted_features:
            print(f"- {feature}")
    else:
        print(
            "No numeric features exceeded "
            "the drift threshold."
        )

    print()


if __name__ == "__main__":
    main()