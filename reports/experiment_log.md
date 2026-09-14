# Predictive Maintenance Experiment Log

## Project Objective

Build and evaluate a machine learning system that predicts industrial equipment failure risk and supports earlier maintenance decisions.

## Dataset

AI4I 2020 Predictive Maintenance Dataset

Target:
- Machine failure

Primary predictors:
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type

Excluded leakage variables:
- TWF
- HDF
- PWF
- OSF
- RNF

## Data Preparation

- Raw dataset preserved and versioned with DVC.
- Schema, missing values, duplicates, machine type categories, target values, and numerical ranges validated.
- Machine type one-hot encoded.
- Failure-mode variables removed to prevent target leakage.
- Processed/model-ready dataset versioned with DVC.

## Experiment 1 — Logistic Regression Baseline

### Objective
Establish an interpretable baseline model for predicting machine failure and determine how well a simple linear classification approach can identify the minority failure class.

### Data and Preprocessing
- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Target: Machine failure
- Leakage variables excluded: TWF, HDF, PWF, OSF, RNF
- Machine Type was one-hot encoded.
- Numerical features were standardized using StandardScaler within the model pipeline.
- Data was split using stratified sampling so that the rare failure cases were represented in both training and test data.
- Training rows: 8,000
- Test rows: 2,000

### Model Configuration
Model: Logistic Regression

Key settings:
- class_weight = "balanced"
- max_iter = 1000
- random_state = 42
- StandardScaler applied before Logistic Regression

MLflow run name:
- logistic_regression_baseline

### Results
- Precision: 0.142
- Recall: 0.824
- F1 Score: 0.242
- PR-AUC: 0.382
- Accuracy: 0.825

### Interpretation
The logistic regression baseline identified approximately 82% of actual machine failures, which is useful because missed failures are costly. However, its low precision indicates that many of the generated failure alerts were false positives. Accuracy was not treated as the primary metric because only about 3.4% of the dataset represents machine failures.

### Decision
Retain Logistic Regression as the interpretable baseline, but test more flexible models to determine whether precision, F1, and PR-AUC can be improved while maintaining strong recall.

## Experiment 2 — Decision Tree Baseline

### Objective
Evaluate whether a simple non-linear tree-based model can improve machine-failure detection compared with the logistic regression baseline.

### Data and Preprocessing
- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Target: Machine failure
- Leakage variables excluded: TWF, HDF, PWF, OSF, RNF
- Machine Type was one-hot encoded.
- Data was split using stratified sampling.
- Training rows: 8,000
- Test rows: 2,000

### Model Configuration
Model: Decision Tree Classifier

Key settings:
- class_weight = "balanced"
- max_depth = 5
- random_state = 42

MLflow run name:
- decision_tree_baseline

### Results
- Precision: 0.311
- Recall: 0.882
- F1 Score: 0.460
- PR-AUC: 0.515
- Accuracy: 0.929

### Interpretation
The decision tree improved substantially over logistic regression. It identified about 88% of actual failures while also producing fewer false alerts, as reflected in the higher precision, F1 score, and PR-AUC.

### Decision
Retain the Decision Tree as a stronger baseline and continue testing more powerful ensemble models to determine whether predictive performance can be improved further.

## Experiment 3 — Random Forest Baseline

### Objective
Evaluate whether an ensemble of decision trees can improve machine-failure prediction compared with the single decision tree while maintaining strong recall.

### Data and Preprocessing
- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Target: Machine failure
- Leakage variables excluded: TWF, HDF, PWF, OSF, RNF
- Machine Type was one-hot encoded.
- Data was split using stratified sampling.
- Training rows: 8,000
- Test rows: 2,000

### Model Configuration
Model: Random Forest Classifier

Key settings:
- n_estimators = 200
- max_depth = 8
- class_weight = "balanced"
- random_state = 42
- n_jobs = -1

MLflow run name:
- random_forest_baseline

### Results
- Precision: 0.420
- Recall: 0.853
- F1 Score: 0.563
- PR-AUC: 0.639
- Accuracy: 0.955

### Interpretation
The Random Forest improved precision, F1 score, PR-AUC, and overall accuracy compared with the Decision Tree while still identifying about 85% of actual failures. This suggests that combining many decision trees provided a better balance between detecting failures and limiting false alerts.

### Decision
Retain the Random Forest as a strong candidate model and compare it with Gradient Boosting to determine whether additional improvements in predictive performance are possible.

## Experiment 4 — Gradient Boosting Baseline

### Objective
Evaluate whether a boosting-based ensemble model can improve predictive performance compared with Logistic Regression, Decision Tree, and Random Forest.

### Data and Preprocessing
- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Target: Machine failure
- Leakage variables excluded: TWF, HDF, PWF, OSF, RNF
- Machine Type was one-hot encoded.
- Data was split using stratified sampling.
- Training rows: 8,000
- Test rows: 2,000

### Model Configuration
Model: Gradient Boosting Classifier

Key settings:
- n_estimators = 150
- learning_rate = 0.05
- max_depth = 3
- random_state = 42

MLflow run name:
- gradient_boosting_baseline

### Results
- Precision: 0.889
- Recall: 0.588
- F1 Score: 0.708
- PR-AUC: 0.788
- Accuracy: 0.984

### Interpretation
Gradient Boosting produced the strongest precision, F1 score, PR-AUC, and accuracy of the baseline models. However, recall dropped to about 59%, meaning the model missed too many actual failures at the default 0.50 classification threshold.

### Decision
Retain Gradient Boosting as a leading candidate, but evaluate lower classification thresholds to determine whether recall can be improved while preserving useful precision.

## Experiment 5 — Threshold Tuning and Candidate Selection

### Objective
Evaluate whether adjusting the classification threshold can improve failure detection for the strongest ensemble models while maintaining a useful level of precision.

Because missed machine failures are costly, the experiment required validation recall of at least 0.80 before considering a threshold acceptable.

### Data Split
- Training rows: 6,000
- Validation rows: 2,000
- Final test rows: 2,000
- Stratified sampling was used to preserve the failure-class distribution.
- Threshold selection was performed using validation data only.
- The final test set was reserved for evaluation after model and threshold decisions were made.

### Candidate Models
- Random Forest
- Gradient Boosting

### Threshold Search
Thresholds from 0.05 through 0.95 were evaluated in 0.01 increments.

For each model:
- Only thresholds achieving validation recall of at least 0.80 were considered.
- Among those thresholds, the threshold with the highest precision was selected.
- F1 score was used as a tie-breaker when needed.

### Random Forest Results
Selected threshold:
- 0.41

Validation results:
- Precision: 0.350
- Recall: 0.824
- F1 Score: 0.491
- PR-AUC: 0.487

Final test results:
- Precision: 0.390
- Recall: 0.838
- F1 Score: 0.533
- PR-AUC: 0.611

### Gradient Boosting Results
Selected threshold:
- 0.07

Validation results:
- Precision: 0.350
- Recall: 0.824
- F1 Score: 0.491
- PR-AUC: 0.723

Final test results:
- Precision: 0.410
- Recall: 0.868
- F1 Score: 0.557
- PR-AUC: 0.773

### Interpretation
Both models achieved the required validation recall at their selected thresholds. Their precision, recall, and F1 scores were identical at the selected validation operating points, but Gradient Boosting had substantially higher validation PR-AUC (0.723 versus 0.487). This indicates that Gradient Boosting provided better overall precision-recall performance across possible thresholds.

### Decision
Gradient Boosting was selected as the preferred model using validation results, with a classification threshold of 0.07.

The final test results showed that the selected model identified approximately 87% of actual machine failures, while approximately 41% of generated failure alerts corresponded to actual failures.

The final test set was not used to select a different model or threshold.