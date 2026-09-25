from pathlib import Path

import pandas as pd


LOG_PATH = Path("logs/predictions.csv")


def main():
    if not LOG_PATH.exists():
        print("No prediction log found.")
        print(f"Expected location: {LOG_PATH}")
        return

    df = pd.read_csv(LOG_PATH)

    if df.empty:
        print("Prediction log exists but contains no predictions.")
        return

    total_predictions = len(df)

    predicted_failures = int(
        df["predicted_failure"].sum()
    )

    predicted_failure_rate = (
        predicted_failures / total_predictions
    )

    average_probability = (
        df["failure_probability"].mean()
    )

    print()
    print("PREDICTION MONITORING SUMMARY")
    print("-----------------------------")
    print(f"Total predictions: {total_predictions}")
    print(
        f"Predicted failures: {predicted_failures}"
    )
    print(
        "Predicted failure rate: "
        f"{predicted_failure_rate:.2%}"
    )
    print(
        "Average failure probability: "
        f"{average_probability:.4f}"
    )

    print()
    print("Prediction probability range:")
    print(
        f"Minimum: {df['failure_probability'].min():.4f}"
    )
    print(
        f"Maximum: {df['failure_probability'].max():.4f}"
    )

    print()
    print("Machine type counts:")
    print(
        df["machine_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Average input values:")
    print(
        f"Air temperature: "
        f"{df['air_temperature'].mean():.2f}"
    )
    print(
        f"Process temperature: "
        f"{df['process_temperature'].mean():.2f}"
    )
    print(
        f"Rotational speed: "
        f"{df['rotational_speed'].mean():.2f}"
    )
    print(
        f"Torque: "
        f"{df['torque'].mean():.2f}"
    )
    print(
        f"Tool wear: "
        f"{df['tool_wear'].mean():.2f}"
    )

    print()


if __name__ == "__main__":
    main()