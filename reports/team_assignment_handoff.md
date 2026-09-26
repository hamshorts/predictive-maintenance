# Experiment Tracking Conceptual Design

## Experiment Objective

The experiment objective is to compare multiple classification models for predicting machine failure using the AI4I 2020 Predictive Maintenance Dataset. The experiments are designed to determine which model provides the best balance between detecting actual machine failures and limiting unnecessary maintenance alerts.

The models evaluated are:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Because missed machine failures may be costly, recall is emphasized. Additional metrics include precision, F1 score, and PR-AUC.

## Data and Preprocessing

Target:
- Machine failure

Predictor variables:
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine Type

The following failure-mode variables were excluded because they could leak information about the target:
- TWF
- HDF
- PWF
- OSF
- RNF

Machine Type was one-hot encoded.

StandardScaler was used within the Logistic Regression pipeline.

Stratified sampling was used so the rare failure class remained represented across data splits.

Raw and processed data are versioned with DVC, while code and configuration changes are tracked with Git.

## Experiment Tracking with MLflow

MLflow is used to track model experiments.

Items tracked include:
- Run name
- Model type
- Model hyperparameters
- Random seed
- Training and test sizes
- Validation size for threshold tuning
- Selected classification threshold
- Minimum required recall
- Precision
- Recall
- F1 score
- PR-AUC

The experiment history provides a reproducible record of how different models and configurations performed.

## Baseline Model Results

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.142 | 0.824 | 0.242 | 0.382 |
| Decision Tree | 0.311 | 0.882 | 0.460 | 0.515 |
| Random Forest | 0.420 | 0.853 | 0.563 | 0.639 |
| Gradient Boosting | 0.889 | 0.588 | 0.708 | 0.788 |

Gradient Boosting produced the highest baseline precision, F1 score, and PR-AUC. However, its recall at the default 0.50 classification threshold was only 0.588. This motivated a separate threshold-tuning experiment.

## Threshold Tuning

Random Forest and Gradient Boosting were selected for threshold tuning.

The tuning process used:
- 6,000 training rows
- 2,000 validation rows
- 2,000 final test rows

A minimum validation recall of 0.80 was required.

Among thresholds meeting the recall requirement, the threshold with the highest precision was selected, with F1 used as a tie-breaker.

### Random Forest
Selected threshold:
- 0.41

Validation PR-AUC:
- 0.487

Final test results:
- Precision: 0.390
- Recall: 0.838
- F1: 0.533
- PR-AUC: 0.611

### Gradient Boosting
Selected threshold:
- 0.07

Validation PR-AUC:
- 0.723

Final test results:
- Precision: 0.410
- Recall: 0.868
- F1: 0.557
- PR-AUC: 0.773

## Model Selection

Gradient Boosting was selected as the preferred candidate based on validation results.

Both Random Forest and Gradient Boosting satisfied the minimum validation recall requirement, but Gradient Boosting achieved substantially higher validation PR-AUC.

The selected Gradient Boosting threshold is 0.07.

The final test set was not used to change the selected model or threshold.
