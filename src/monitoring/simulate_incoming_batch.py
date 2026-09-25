import json
from pathlib import Path
from urllib import request

import pandas as pd


DATA_PATH = Path(
    "data/processed/ai4i2020_model.csv"
)

API_URL = "http://127.0.0.1:8000/predict"

BATCH_SIZE = 50

RANDOM_STATE = 42


def get_machine_type(row):
    if row["Type_H"] == 1:
        return "H"
    if row["Type_L"] == 1:
        return "L"
    return "M"


def send_prediction(payload):
    body = json.dumps(payload).encode("utf-8")

    api_request = request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with request.urlopen(
        api_request,
        timeout=10,
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def main():

    if not DATA_PATH.exists():
        print("Processed dataset not found.")
        print(f"Expected location: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    sample = df.sample(
        n=BATCH_SIZE,
        random_state=RANDOM_STATE,
    )

    successful_requests = 0
    failed_requests = 0

    print()
    print("SIMULATING INCOMING BATCH")
    print("-------------------------")
    print(f"Rows to send: {BATCH_SIZE}")
    print(f"API endpoint: {API_URL}")
    print()

    for number, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):

        payload = {
            "machine_type": get_machine_type(row),
            "air_temperature": float(
                row["Air temperature"]
            ),
            "process_temperature": float(
                row["Process temperature"]
            ),
            "rotational_speed": float(
                row["Rotational speed"]
            ),
            "torque": float(
                row["Torque"]
            ),
            "tool_wear": float(
                row["Tool wear"]
            ),
        }

        try:
            send_prediction(payload)
            successful_requests += 1

            if number % 10 == 0:
                print(
                    f"Processed {number} of "
                    f"{BATCH_SIZE} rows."
                )

        except Exception as error:
            failed_requests += 1
            print(
                f"Request {number} failed: "
                f"{error}"
            )

    print()
    print("BATCH SIMULATION COMPLETE")
    print("-------------------------")
    print(
        f"Successful requests: "
        f"{successful_requests}"
    )
    print(
        f"Failed requests: "
        f"{failed_requests}"
    )
    print()


if __name__ == "__main__":
    main()