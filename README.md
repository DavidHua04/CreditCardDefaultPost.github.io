# Credit Card Default Prediction

A machine learning project that predicts the probability of credit card default using historical payment behavior and demographic data.

**Live report:** [GitHub Pages site](https://CreditCardDefaultPost.github.io)

---

## Overview

Given a customer's profile and six months of payment history, can we estimate the likelihood of default next month? This project builds and compares multiple classification models to answer that question, with a focus on interpretability alongside predictive performance.

**Dataset:** [UCI Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) — 30,000 credit card holders in Taiwan (April–September 2005), including demographics, credit limit, repayment history, and bill/payment amounts.

---

## Methods

### Feature Engineering
Three features were engineered from the raw data:

| Feature | Description |
|---|---|
| `AVG_UTILIZATION` | Mean of 6-month bill amounts / credit limit — average credit utilization |
| `DELAY_COUNT` | Number of months with payment delays out of the past 6 months |
| `PAY_STD` | Standard deviation of 6-month payment amounts — payment consistency |

### Preprocessing
- `StandardScaler` on numeric features
- `OneHotEncoder` on categorical features (education, marriage status, sex)
- All steps integrated into a `scikit-learn` `Pipeline`

### Models Compared
| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Dummy Classifier (baseline) | ~0.50 | ~0.50 |
| Logistic Regression | — | — |
| Decision Tree | — | — |
| Random Forest | — | — |
| XGBoost | — | — |
| Gradient Boosting (sklearn) | 0.784 | — |
| HistGradientBoosting | 0.782 | — |
| **LightGBM (tuned)** | **0.820** | **0.780** |

Hyperparameters were tuned using `RandomizedSearchCV` with 5-fold cross-validation. ROC-AUC was the primary evaluation metric.

### Best Model: LightGBM
- **Validation ROC-AUC: 0.82**, **Test ROC-AUC: 0.78**
- Fastest competitive model (0.10s training time vs. 6.3s for sklearn Gradient Boosting)
- Minimal overfitting gap (~0.04)

---

## Key Findings

- **Repayment history** (`DELAY_COUNT`, `PAY_0`) was the strongest predictor of default
- **Engineered features** (`AVG_UTILIZATION`) ranked among the top predictors alongside raw payment history
- **Demographics** (age, education, marriage) contributed modestly but were not decisive
- SHAP analysis confirmed that transactional behavior alone carries strong predictive signal, even without credit scores or employment data

---

## Tech Stack

- **Python** — pandas, numpy, matplotlib
- **scikit-learn** — pipelines, preprocessing, model selection, cross-validation
- **LightGBM**, **XGBoost** — gradient boosting models
- **SHAP** — model interpretability and feature importance
- **Jupyter Notebook** — analysis and experimentation
- **GitHub Pages** — static HTML report

---

## Caveats

- Dataset is from 2005 and covers a single Taiwanese bank — generalizability is limited
- `EDUCATION` field contains undocumented category codes that were not fully cleaned
- Extreme outliers in bill/payment amounts may contribute to validation–test score mismatch

---

## Files

```
├── Prediction on Default of Credit Card Clients.ipynb  # Main analysis notebook
├── index.html                                           # Published report (GitHub Pages)
└── img/                                                 # Figures used in the report
    ├── all_features.png       # Feature distributions
    ├── outliers1/2.png        # Outlier visualizations
    └── feature_value.png      # SHAP feature importance plot
```
