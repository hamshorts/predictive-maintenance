from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

REGISTERED_MODEL_URI = (
    "models:/predictive-maintenance-gradient-boosting@candidate"
)

SELECTED_THRESHOLD = 0.07

mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ---------------------------------------------------------
# Load registered candidate model
# ---------------------------------------------------------

model = mlflow.sklearn.load_model(REGISTERED_MODEL_URI)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Predicts machine-failure risk using the registered "
        "Gradient Boosting candidate model."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    machine_type: Literal["H", "L", "M"] = Field(
        description="Machine type: H, L, or M"
    )
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "model": "predictive-maintenance-gradient-boosting",
        "alias": "candidate",
        "threshold": SELECTED_THRESHOLD,
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert machine type to the same one-hot representation
    # used during model training.
    type_h = 1 if request.machine_type == "H" else 0
    type_l = 1 if request.machine_type == "L" else 0
    type_m = 1 if request.machine_type == "M" else 0

    # Create the model input using the exact feature names
    # and order used during training.
    model_input = pd.DataFrame(
        [
            {
                "Air temperature": request.air_temperature,
                "Process temperature": request.process_temperature,
                "Rotational speed": request.rotational_speed,
                "Torque": request.torque,
                "Tool wear": request.tool_wear,
                "Type_H": type_h,
                "Type_L": type_l,
                "Type_M": type_m,
            }
        ]
    )

    # Probability that Machine failure = 1
    failure_probability = float(
        model.predict_proba(model_input)[0, 1]
    )

    predicted_failure = int(
        failure_probability >= SELECTED_THRESHOLD
    )

    return {
        "failure_probability": round(failure_probability, 4),
        "predicted_failure": predicted_failure,
        "decision_threshold": SELECTED_THRESHOLD,
        "machine_type": request.machine_type,
    }