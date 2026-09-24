# Predictive Maintenance ML System

## Project Description

This project develops a machine learning system for predictive maintenance using the UCI AI4I 2020 Predictive Maintenance dataset.

The goal is to estimate machine-failure risk from operating conditions such as temperature, rotational speed, torque, tool wear, and machine type. The system is designed to support maintenance decisions by identifying equipment that may have a higher risk of failure.

The project incorporates DataOps and ModelOps practices including:

- Batch data ingestion
- Data validation and processing
- DVC data versioning
- Model experimentation
- MLflow experiment tracking
- Model versioning with the MLflow Model Registry

The models evaluated include Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. Gradient Boosting is the current selected candidate model.

## Project Structure

```text
predictive-maintenance/
├── data/
├── reports/
├── src/
│   ├── data/
│   └── models/
├── requirements.txt
└── README.md
```

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Example Usage

Validate the dataset:

```bash
python src/data/validate_data.py
```

Process the dataset:

```bash
python src/data/process_data.py
```

Run the selected model:

```bash
python src/models/train_selected_model.py
```

Start MLflow:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

## Contributors

- Aryssa Lane
- Phillips Medley
- Katy Schultz
- Hamilton Schwartz
- Sashaank Suresh

## Project Status

The data pipeline, experiment tracking, threshold tuning, and model versioning portions of the project are complete.

Deployment, monitoring, and final system integration are currently in progress.