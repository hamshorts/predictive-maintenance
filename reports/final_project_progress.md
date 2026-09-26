# Final Project Progress

## Project

Predictive Maintenance ML System using the UCI AI4I 2020 Predictive Maintenance dataset.

## Current System Status

### DataOps

Completed:

- Batch ingestion of the AI4I 2020 dataset
- Raw data storage
- Data validation
  - schema
  - missing values
  - duplicate rows
  - categorical values
  - target values
  - project specific numerical ranges
- Removal of target leakage variables
- Machine type one-hot encoding
- Processed model ready dataset
- Raw and processed dataset versioning with DVC

### Model Development and ModelOps

Completed:

- Logistic regression baseline
- Decision tree baseline
- Random forest baseline
- Gradient boosting baseline
- MLflow experiment tracking
- Model comparison using precision, recall, F1, and PR-AUC
- Validation based threshold tuning
- Selected gradient boosting candidate
- Selected decision threshold: 0.07
- MLflow model registry
  - version 1: @baseline
  - version 2: @candidate

### Deployment

Completed:

- Local FastAPI application
- Model loaded from MLflow model registry using the @candidate alias
- POST /predict endpoint
- Readable API inputs
- Internal one-hot encoding of machine type
- Failure risk probability returned
- Decision threshold applied
- Predicted failure class returned
- Low risk and high risk API tests completed
- FastAPI interactive documentation available through /docs

### Monitoring

Completed:

- Prediction logging to CSV
- Prediction timestamp logging
- Input value logging
- Failure probability logging
- Predicted class logging
- Decision threshold logging
- Prediction behavior summary script
- Numeric feature drift monitoring script
- Minimum monitoring sample warning
- Simulated normal incoming batch
- Simulated shifted incoming batch

Normal batch test:

- 50 simulated API requests
- 50 successful requests
- 0 failed requests
- No numeric drift flags

Shifted batch test:

- 50 simulated API requests
- Artificial torque shift of +25
- 50 successful requests
- 0 failed requests
- Torque drift successfully detected

The shifted batch test is a simulated monitoring test and does not represent actual production drift.

## Current Architecture

UCI AI4I 2020 dataset  
→ batch ingestion  
→ raw data storage  
→ validation  
→ processing/feature engineering  
→ processed data storage  
→ DVC versioning  
→ model experimentation  
→ MLflow experiment tracking  
→ MLflow model registry  
→ FastAPI scoring  
→ prediction logging  
→ prediction monitoring  
→ drift monitoring  
→ future feedback and retraining

## Remaining Work

### System Integration and Testing

- Perform an end-to-end system test
- Verify API behavior for invalid inputs
- Document API startup and usage
- Confirm reproducibility from the repository
- Update system diagram to show implemented FastAPI deployment and monitoring

### Monitoring Refinement

- Clearly distinguish simulated monitoring from real production monitoring
- Document how labeled outcomes would later support precision and recall monitoring
- Explain how meaningful drift or performance degradation could trigger review and retraining

### Final Report

- Problem and business purpose
- Dataset and limitations
- DataOps pipeline
- Feature engineering and leakage prevention
- Model experimentation
- Model comparison
- Threshold selection
- Model versioning
- Deployment architecture
- Monitoring design and demonstration
- Challenges and limitations
- Future improvements

### Final Presentation

- Updated architecture diagram
- DataOps demonstration
- MLflow experiment comparison
- Model Registry demonstration
- FastAPI prediction demonstration
- Monitoring demonstration
- Normal versus shifted drift example
- Challenges, limitations, and next steps