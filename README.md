# Hospital Readmission Risk Prediction

An end-to-end machine learning project focused on predicting 30-day hospital readmission risk using real-world clinical data, with an emphasis on recall-driven evaluation and model selection.

## Problem

Hospital readmissions within 30 days of discharge are costly and often preventable.
The goal of this project is to predict whether a patient will be readmitted within 30 days using information available at discharge time, enabling earlier intervention for high-risk patients.

## Dataset

This project uses a public clinical dataset containing over 100,000 hospital encounters from 130 U.S. hospitals.
The data includes demographic information, hospital utilization metrics, lab procedures, medications, and admission details.

The target variable is a binary indicator of whether a patient was readmitted within 30 days of discharge.

Due to privacy constraints, diagnosis codes and discharge disposition fields were excluded to prevent data leakage.

## Key Challenges

- Strong class imbalance (~11% positive readmissions)
- High cost of false negatives in a clinical setting
- Potential data leakage from post-discharge information
- Balancing recall and precision in model evaluation

## Approach

The project follows a structured machine learning workflow:

1. Exploratory data analysis to assess class imbalance and data quality
2. Explicit handling of data leakage by removing identifiers and post-treatment features
3. Baseline modeling using logistic regression with class balancing
4. Recall-focused evaluation using precision–recall curves and threshold analysis
5. Model comparison to motivate more expressive, non-linear models

## Baseline Results

A logistic regression model was used as an interpretable baseline.

- ROC-AUC: ~0.65
- Recall (positive class): ~0.54 at default threshold
- Recall increased to ~0.98 with lower thresholds, at the cost of precision

These results highlight the tradeoff between recall and precision in imbalanced clinical data and motivate the use of non-linear models.

## Key Takeaways

- Accuracy is misleading for highly imbalanced healthcare data
- Threshold selection has a significant impact on clinical usefulness
- Logistic regression provides a strong baseline but struggles to capture complex feature interactions
- Recall-driven evaluation is essential when false negatives are costly

## Next Steps

- Train and evaluate decision trees and ensemble models
- Compare bias–variance tradeoffs across model families
- Perform subgroup analysis to examine model behavior across patient populations
- Evaluate final model on a held-out test set

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
