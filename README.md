# Hospital Readmission Risk Prediction

An end-to-end machine learning project focused on predicting 30-day hospital readmission risk using real-world clinical data. This project tackles the challenges of **class imbalance (11% positive rate)** and **data leakage** in a healthcare setting.

## 📊 Key Results (Final Test Set)

After training and comparing Logistic Regression, Decision Trees, Random Forest, and XGBoost, the **Tuned Random Forest** was selected as the champion model.

## 📊 Key Results (Final Test Set)

After training and comparing Logistic Regression, Decision Trees, Random Forest, and XGBoost, the **Tuned Random Forest** was selected as the champion model.

| Metric        | Score     | Clinical Interpretation                                                                                                                                    |
| :------------ | :-------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ROC-AUC**   | **0.636** | The model successfully discriminates between high and low risk patients better than random chance.                                                         |
| **PR-AUC**    | **0.188** | Average Precision score, accounting for class imbalance. Lower than ROC-AUC but expected given the 11% positive rate.                                      |
| **Recall**    | **34%**   | At the default threshold, the model captures 34% of readmissions (Sensitivity). _Note: This can be increased to >60% by adjusting the decision threshold._ |
| **Precision** | **22%**   | For every ~5 patients flagged as "High Risk," ~1 is actually readmitted.                                                                                   |
| **F1-Score**  | **0.231** | Harmonic mean of precision and recall, balancing the tradeoff between the two metrics.                                                                     |

**Note:** Model performance was validated via 5-fold cross-validation (CV AUC: 0.639 ± 0.016), confirming robust generalization. |

### 💡 Clinical Insight: The "Frequent Flyer" Effect

Feature Importance analysis revealed that **Historical Utilization** is the dominant driver of risk.

- **Top Predictor:** `number_inpatient` (Inpatient visits in the previous year) - **34% Importance**.
- **Takeaway:** A patient's history of hospital usage is a far stronger predictor of future readmission than their specific medication regimen (Insulin vs. Metformin) or number of diagnoses.

---

## Problem

Hospital readmissions within 30 days of discharge are costly (estimated \$41B+ annually in the US) and often preventable. The goal of this project is to predict readmission risk _at the time of discharge_ to enable targeted interventions.

## Dataset

- **Source:** Diabetes 130-US Hospitals Dataset (1999-2008).
- **Size:** ~100,000 patient records.
- **Features:** Demographics, hospital utilization (prior visits), lab results, and medications.
- **Target:** Binary indicator (1 = Readmitted within 30 days, 0 = Otherwise).

## 🛠 Approach & Methodology

### 1. Data Engineering

- **Leakage Prevention:** Removed post-discharge features (e.g., `discharge_disposition_id` relating to hospice/death) to ensure the model only uses data available _at discharge_.
- **Preprocessing Pipeline:** Used `ColumnTransformer` to apply:
  - `OneHotEncoder` for categorical variables.
  - `StandardScaler` for numerical variables.
  - **Note:** Fit _only_ on the training split to maintain strict separation.

### 2. Modeling Strategy

We compared four distinct model families to analyze the Bias-Variance tradeoff:

1.  **Logistic Regression:** Baseline linear model (High Bias).
2.  **Decision Tree:** Non-linear, unconstrained (High Variance/Overfitting).
3.  **Random Forest:** Ensemble bagging to reduce variance (Best Performance).
4.  **XGBoost:** Gradient boosting with `scale_pos_weight` for imbalance.

### 3. Handling Imbalance

The dataset has a severe imbalance (~11% positive case rate). We addressed this via:

- **Stratified Splitting:** Ensuring Train/Val/Test sets had equal class distribution.
- **Class Weighting:** Used `class_weight='balanced'` (Random Forest) and `scale_pos_weight` (XGBoost) to penalize false negatives heavily.

## 📈 Model Comparison

| Model                        | Val AUC   | Observations                                                      |
| :--------------------------- | :-------- | :---------------------------------------------------------------- |
| **Logistic Regression**      | 0.640     | Good baseline, but fails to capture non-linear interactions.      |
| **Decision Tree (Unpruned)** | 0.531     | Severe overfitting (Perfect Train score, poor Val score).         |
| **XGBoost**                  | 0.647     | Strong performance, effectively tied with Random Forest.          |
| **Random Forest (Tuned)**    | **0.648** | **Selected Champion.** Best balance of stability and performance. |

## Future Improvements

- **Threshold Optimization:** Implement a cost-sensitive threshold (e.g., minimizing financial cost of False Negatives vs. False Positives).
- **Feature Engineering:** Group ICD-9 diagnosis codes into broader "Comorbidity Indices" (e.g., Charlson Index).
- **Deep Learning:** Experiment with Neural Networks (MLP), though tabular performance was capped at AUC ~0.63 in initial tests.

## Tech Stack

- **Python 3.10+**
- **Scikit-Learn:** Pipelines, GridSearch, Evaluation Metrics.
- **Pandas & NumPy:** Data manipulation.
- **Matplotlib:** Visualization (ROC Curves, Confusion Matrices).
- **XGBoost:** Gradient Boosting implementation.
