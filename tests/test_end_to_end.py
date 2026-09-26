import json
from pathlib import Path
from urllib import request


API_ROOT_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_ROOT_URL}/predict"

LOG_PATH = Path("logs/predictions.csv")


LOW_RISK_INPUT = {
    "machine_type": "L",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500.0,
    "torque": 40.0,
    "tool_wear": 120.0,
}


HIGH_RISK_INPUT = {
    "machine_type": "L",
    "air_temperature": 301.7,
    "process_temperature": 309.5,
    "rotational_speed": 1298.0,
    "torque": 65.5,
    "tool_wear": 229.0,
}


def get_json(url):
    with request.urlopen(
        url,
        timeout=10,
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def post_json(url, payload):
    body = json.dumps(payload).encode("utf-8")

    api_request = request.Request(
        url,
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


def count_log_rows():
    if not LOG_PATH.exists():
        return 0

    with LOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as log_file:
        lines = log_file.readlines()

    if not lines:
        return 0

    return len(lines) - 1


def main():

    print()
    print("END-TO-END SYSTEM TEST")
    print("----------------------")

    # -----------------------------------------------------
    # 1. Test API health
    # -----------------------------------------------------

    health = get_json(API_ROOT_URL)

    assert health["status"] == "running"
    assert health["alias"] == "candidate"
    assert health["threshold"] == 0.07

    print("PASS: API health endpoint")

    # -----------------------------------------------------
    # 2. Record prediction-log size before testing
    # -----------------------------------------------------

    rows_before = count_log_rows()

    # -----------------------------------------------------
    # 3. Test low-risk prediction
    # -----------------------------------------------------

    low_result = post_json(
        PREDICT_URL,
        LOW_RISK_INPUT,
    )

    assert low_result["predicted_failure"] == 0
    assert low_result["decision_threshold"] == 0.07
    assert low_result["failure_probability"] < 0.07

    print("PASS: Low-risk prediction")

    # -----------------------------------------------------
    # 4. Test high-risk prediction
    # -----------------------------------------------------

    high_result = post_json(
        PREDICT_URL,
        HIGH_RISK_INPUT,
    )

    assert high_result["predicted_failure"] == 1
    assert high_result["decision_threshold"] == 0.07
    assert high_result["failure_probability"] >= 0.07

    print("PASS: High-risk prediction")

    # -----------------------------------------------------
    # 5. Verify prediction logging
    # -----------------------------------------------------

    rows_after = count_log_rows()

    assert rows_after == rows_before + 2

    print("PASS: Prediction logging")

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("ALL END-TO-END TESTS PASSED")
    print("---------------------------")
    print(
        f"Log rows before test: {rows_before}"
    )
    print(
        f"Log rows after test:  {rows_after}"
    )
    print()


if __name__ == "__main__":
    main()