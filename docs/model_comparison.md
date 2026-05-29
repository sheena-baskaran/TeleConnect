# TeleConnect Churn Model — Full Model Comparison Report

Generated: 2026-05-29 17:47 UTC
Dataset: `data/test_datafile.csv` (5,050 raw rows → 5,000 after cleaning)
Test split: 20% holdout, stratified by churn label (seed=42)
Base churn rate: **36.5%**

---

## 1. Overview — Threshold-Independent Metrics

These metrics summarise ranking quality across ALL thresholds.

| Model | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|
| Logistic Regression | 0.7099 | 0.5337 | ✅ **Best** |
| XGBoost | 0.6957 | 0.5267 |  |

> **Why PR-AUC is the primary metric:** at a 36.5% base rate, ROC-AUC is
> optimistic because the large true-negative pool inflates it.
> PR-AUC focuses on the positive (churn) class — the one we actually care about.

---

## 2. Per-Threshold Metrics

> Threshold 0.5 is the default. In production, the operating point should
> be tuned to the economics of your retention offers vs. saved-customer value.

### Logistic Regression

| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 0.515 | 0.4248 | 0.9288 | 0.583 | 339 | 459 | 26 | 176 |
| 0.4 | 0.6 | 0.4726 | 0.8274 | 0.6016 | 302 | 337 | 63 | 298 |
| 0.5 | 0.642 | 0.5065 | 0.7425 | 0.6022 | 271 | 264 | 94 | 371 |

**Metric definitions:**
- **Recall** = TP / (TP + FN) — % of actual churners caught. *Most important for retention.*
- **Precision** = TP / (TP + FP) — % of flagged customers who truly churn.
- **FN (False Negatives)** = missed churners → lost revenue, the costly mistake.
- **FP (False Positives)** = wasted retention offer → cheap mistake.

### XGBoost

| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 0.54 | 0.4362 | 0.8904 | 0.5856 | 325 | 420 | 40 | 215 |
| 0.4 | 0.598 | 0.4697 | 0.7863 | 0.5881 | 287 | 324 | 78 | 311 |
| 0.5 | 0.64 | 0.5053 | 0.6493 | 0.5683 | 237 | 232 | 128 | 403 |

**Metric definitions:**
- **Recall** = TP / (TP + FN) — % of actual churners caught. *Most important for retention.*
- **Precision** = TP / (TP + FP) — % of flagged customers who truly churn.
- **FN (False Negatives)** = missed churners → lost revenue, the costly mistake.
- **FP (False Positives)** = wasted retention offer → cheap mistake.

---

## 3. Confusion Matrices (threshold = 0.5)

### Logistic Regression

```
                    Predicted: Stay   Predicted: Churn
  Actual: Stay                 371               264
  Actual: Churn                 94               271
```

| | Count | % of test |
|---|---|---|
| True Positives (caught churners) | 271 | 27.1% |
| False Negatives (missed churners) ⚠️ | 94 | 9.4% |
| False Positives (wrong alarms) | 264 | 26.4% |
| True Negatives (correct stays) | 371 | 37.1% |

### XGBoost

```
                    Predicted: Stay   Predicted: Churn
  Actual: Stay                 403               232
  Actual: Churn                128               237
```

| | Count | % of test |
|---|---|---|
| True Positives (caught churners) | 237 | 23.7% |
| False Negatives (missed churners) ⚠️ | 128 | 12.8% |
| False Positives (wrong alarms) | 232 | 23.2% |
| True Negatives (correct stays) | 403 | 40.3% |

---

## 4. Feature Importances (Top 15)

Logistic Regression: absolute standardised coefficients (|coef|).
XGBoost: feature gain (total improvement in loss from splits on that feature).

| Feature | Logistic Regression | XGBoost | Notes |
|---|---|---|---|
| `contract_type` | 0.4098 | 0.3026 | Strongest driver — month-to-month vs locked-in |
| `tenure_bucket` | 0.1964 | 0.0663 | Early churn hazard is non-monotonic |
| `satisfaction_score` | 0.0660 | 0.0284 | Direct satisfaction signal |
| `internet_service` | 0.0621 | 0.0897 |  |
| `gender` | 0.0558 | 0.0771 |  |
| `tenure_months` | 0.0321 | 0.0289 | Longer tenure = stickier |
| `payment_method` | 0.0266 | 0.0950 |  |
| `has_bundle` | 0.0264 | 0.0471 |  |
| `phone_service` | 0.0264 | 0.0386 |  |
| `num_support_tickets` | 0.0262 | 0.0210 | Friction proxy |
| `total_charges` | 0.0190 | 0.0235 |  |
| `avg_monthly_gb_used` | 0.0133 | 0.0223 |  |
| `charges_per_tenure_month` | 0.0095 | 0.0231 |  |
| `avg_monthly_minutes` | 0.0080 | 0.0227 |  |
| `age` | 0.0071 | 0.0236 |  |

---

## 5. Which Model to Use and Why

**Winner by PR-AUC: Logistic Regression** (PR-AUC 0.5337 vs 0.5267)

| Dimension | Logistic Regression | XGBoost |
|---|---|---|
| Interpretability | High — coefficients explain each prediction | Lower — requires SHAP |
| Captures interactions | No — linear in log-odds | Yes — tree splits |
| Calibration | Good out-of-box | Needs Platt/isotonic scaling |
| Training speed | Fast | Moderate |
| Overfit risk | Low | Medium (needs tuning) |
| Best for | Baseline, stakeholder trust, small data | Larger data, complex patterns |

**On this dataset**, the churn signal is largely additive
(contract type + tenure + satisfaction dominate), which explains why
Logistic Regression is competitive — there are limited interaction effects
that XGBoost can exploit. With a larger dataset or richer feature engineering,
XGBoost would likely pull ahead.

---

## 6. Class Imbalance Handling

Base churn rate: **36.5%** (36.5% positive, 63.5% negative)

| Approach | Applied to | How |
|---|---|---|
| `class_weight='balanced'` | Logistic Regression | Up-weights minority (churn) in the loss |
| `scale_pos_weight` = 1.74 | XGBoost | Multiplies positive-class gradient |
| SMOTE (not applied) | — | Considered but rejected — see note |

> **Why not SMOTE?** SMOTE synthesises new samples in a mixed continuous+one-hot
> feature space, which can create statistically implausible customers (e.g. a
> 'synthetic' customer with half a contract type). Reweighting is simpler, keeps
> the original feature distribution intact, and avoids data leakage if applied
> inside a cross-validation fold. With more data, SMOTE inside a pipeline CV
> would be worth revisiting.

---

## 7. Data Quality Summary

| Column | Issue | Rows affected | Strategy |
|---|---|---|---|
| `customer_id` | exact duplicate rows | 50 | dropped |
| `gender` | inconsistent encodings (8 distinct raw values) | 1,556 | canonicalized to ['Female', 'Male', 'Unknown']; 147 -> 'Unknown' |
| `phone_service` | inconsistent encodings (6 distinct raw values) | 1,269 | canonicalized to ['No', 'Yes'] |
| `internet_service` | inconsistent encodings (5 distinct raw values) | 1,301 | canonicalized to ['DSL', 'Fiber optic', 'No', 'Unknown']; 497 -> 'Unknown' |
| `payment_method` | inconsistent encodings (8 distinct raw values) | 788 | canonicalized to ['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check', 'Unknown']; 30 -> 'Unknown' |
| `contract_type` | inconsistent encodings (3 distinct raw values) | 0 | canonicalized to ['Month-to-month', 'One year', 'Two year'] |
| `age` | impossible / sentinel values outside [18, 100] | 20 | set to NaN (imputed by pipeline) |
| `age` | missing / blank / 'nan' tokens | 10 | set to NaN (imputed by pipeline) |
| `tenure_months` | impossible / sentinel values outside [0, 120] | 5 | set to NaN (imputed by pipeline) |
| `tenure_months` | missing / blank / 'nan' tokens | 10 | set to NaN (imputed by pipeline) |
| `monthly_charges` | impossible / sentinel values outside [0, 250] | 6 | set to NaN (imputed by pipeline) |
| `monthly_charges` | missing / blank / 'nan' tokens | 12 | set to NaN (imputed by pipeline) |
| `total_charges` | impossible / sentinel values outside [0, 20000] | 4 | set to NaN (imputed by pipeline) |
| `total_charges` | missing / blank / 'nan' tokens | 22 | set to NaN (imputed by pipeline) |
| `avg_monthly_gb_used` | impossible / sentinel values outside [0, 1000] | 10 | set to NaN (imputed by pipeline) |
| `avg_monthly_gb_used` | missing / blank / 'nan' tokens | 15 | set to NaN (imputed by pipeline) |
| `num_support_tickets` | impossible / sentinel values outside [0, 60] | 10 | set to NaN (imputed by pipeline) |
| `avg_monthly_minutes` | missing / blank / 'nan' tokens | 80 | set to NaN (imputed by pipeline) |
| `satisfaction_score` | impossible / sentinel values outside [0, 10] | 140 | set to NaN (imputed by pipeline) |
| `satisfaction_score` | missing / blank / 'nan' tokens | 20 | set to NaN (imputed by pipeline) |
| `total_charges` | decimal-shift error vs tenure*monthly | 50 | divided by 100 to reconcile |

---

## 8. Limitations & What to Do Next

| Limitation | Suggested fix |
|---|---|
| Threshold fixed at 0.5 | Tune to retention economics (offer cost vs. CLV saved) |
| No probability calibration | Apply Platt scaling / isotonic regression |
| No temporal validation | Build a time-based train/test split; add recency features |
| No hyperparameter search | `GridSearchCV` / `Optuna` on XGBoost |
| `top_risk_factors` = approximation | Replace with full SHAP TreeExplainer |
| No drift monitoring | Track feature distributions + prediction scores in prod |
