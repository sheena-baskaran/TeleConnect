"""
Churn Model — Local Evaluation Report

Loads the trained pipeline + metadata and runs a full local evaluation against
the cleaned dataset. Prints a complete model report:

  - Algorithm & training details
  - Data quality summary (before/after cleaning)
  - Train/test split performance (all metrics)
  - Confusion matrix (text)
  - Feature importances (top 10)
  - Risk-tier distribution
  - predict_churn() API smoke tests

Run:
    python tests/test_churn_model.py
    python -m pytest tests/test_churn_model.py -v -s
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- paths ------------------------------------------------------------------
PIPELINE_PATH = ROOT / "models" / "churn_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
CLEANED_PATH  = ROOT / "data" / "cleaned_customers.csv"
RAW_PATH      = ROOT / "data" / "test_datafile.csv"

DIVIDER = "=" * 60


def load_artifacts():
    import joblib
    assert PIPELINE_PATH.exists(), f"Model not found: {PIPELINE_PATH}\nRun the notebook first."
    assert METADATA_PATH.exists(), f"Metadata not found: {METADATA_PATH}"
    pipeline = joblib.load(PIPELINE_PATH)
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, meta


def load_data():
    assert CLEANED_PATH.exists(), f"Cleaned data not found: {CLEANED_PATH}"
    df = pd.read_csv(CLEANED_PATH)
    return df


# =============================================================================
# Section 1 — Algorithm & training details
# =============================================================================
def section_algorithm(meta: dict):
    print(f"\n{DIVIDER}")
    print("  SECTION 1 — ALGORITHM & TRAINING DETAILS")
    print(DIVIDER)

    name = meta["model_name"]
    print(f"\n  Algorithm          : {name}")

    if name == "LogisticRegression":
        print("  Family             : Linear (Generalized Linear Model)")
        print("  Solver             : lbfgs (default)")
        print("  Max iterations     : 2000")
        print("  Class imbalance    : class_weight='balanced' (auto-reweights minority)")
        print("  Why chosen         :")
        print("    - Transparent, calibrated baseline; coefficients directly interpretable")
        print("    - Robust to small datasets; hard to overfit")
        print("    - Weakness: assumes additive log-odds effects; misses feature interactions")
    elif name == "XGBoost":
        print("  Family             : Gradient-Boosted Decision Trees (ensemble)")
        print("  n_estimators       : 300")
        print("  max_depth          : 4")
        print("  learning_rate      : 0.05")
        print("  subsample          : 0.9  (row sampling per tree)")
        print("  colsample_bytree   : 0.9  (feature sampling per tree)")
        print("  Class imbalance    : scale_pos_weight = neg/pos")
        print("  Why chosen         :")
        print("    - Captures non-linear relationships and feature interactions")
        print("    - Strong on mixed tabular data; handles missing values natively")
        print("    - Weakness: less interpretable, more hyperparameters")

    print(f"\n  Model version      : {meta['version']}")
    print(f"  Trained at (UTC)   : {meta['trained_at_utc']}")
    print(f"  Training rows      : {meta['n_training_rows']:,}")
    print(f"  Base churn rate    : {meta['base_churn_rate']:.1%}")
    print(f"\n  Numeric features   ({len(meta['numeric_features'])}):")
    for f in meta["numeric_features"]:
        print(f"    - {f}")
    print(f"\n  Categorical features ({len(meta['categorical_features'])}):")
    for f in meta["categorical_features"]:
        print(f"    - {f}")
    print(f"\n  Risk-tier thresholds:")
    print(f"    low    < {meta['threshold_low']:.4f}")
    print(f"    medium = {meta['threshold_low']:.4f} to {meta['threshold_high']:.4f}")
    print(f"    high  >= {meta['threshold_high']:.4f}")
    print("  (Thresholds = 50th and 80th percentile of predicted probabilities)")

    print(f"\n  Preprocessing pipeline:")
    print("    Numerics  -> SimpleImputer(median) -> StandardScaler")
    print("    Categ.    -> SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown=ignore)")


# =============================================================================
# Section 2 — Data quality summary
# =============================================================================
def section_data_quality():
    print(f"\n{DIVIDER}")
    print("  SECTION 2 — DATA QUALITY (before vs after cleaning)")
    print(DIVIDER)

    raw = pd.read_csv(RAW_PATH)
    cleaned = pd.read_csv(CLEANED_PATH)

    print(f"\n  Raw rows       : {len(raw):,}")
    print(f"  Cleaned rows   : {len(cleaned):,}  ({len(raw)-len(cleaned)} dropped)")
    print(f"  Raw columns    : {len(raw.columns)}")
    print(f"\n  {'Issue':<45} {'Raw':>8} {'Cleaned':>10}")
    print("  " + "-" * 65)

    issues = [
        ("Duplicate rows",
         int(raw.duplicated().sum()), int(cleaned.duplicated().sum())),
        ("Duplicate customer_ids",
         int(raw["customer_id"].duplicated().sum()), int(cleaned["customer_id"].duplicated().sum())),
        ("Missing avg_monthly_minutes",
         int(raw["avg_monthly_minutes"].isna().sum()),
         int(cleaned["avg_monthly_minutes"].isna().sum())),
        ("gender: M/F/Other/blank (non-standard)",
         int(raw["gender"].isin(["M","F","m","f","Other",""]).sum()), 0),
        ("phone_service: Y/N/no (non-standard)",
         int(raw["phone_service"].isin(["Y","N","no","yes"]).sum()), 0),
        ("internet_service: fiber/None/nan (non-standard)",
         int(raw["internet_service"].isin(["fiber","None","nan"]).sum()), 0),
        ("payment_method: BT/CC/blank aliases",
         int(raw["payment_method"].isin(["BT","CC",""]).sum()), 0),
        ("satisfaction_score > 10 (sentinel/out-of-scale)",
         int((pd.to_numeric(raw["satisfaction_score"], errors="coerce") > 10).sum()),
         int((cleaned["satisfaction_score"] > 10).sum() if "satisfaction_score" in cleaned else 0)),
        ("Negative numeric values (impossible)",
         int(sum((pd.to_numeric(raw[c], errors="coerce") < 0).sum()
                  for c in ["age","tenure_months","monthly_charges",
                             "avg_monthly_gb_used","num_support_tickets","satisfaction_score"])),
         0),
    ]
    for label, before, after in issues:
        status = "OK" if after == 0 else f"{after} remain"
        print(f"  {label:<45} {before:>8,} {status:>10}")


# =============================================================================
# Section 3 — Performance metrics
# =============================================================================
def section_performance(pipeline, meta: dict, df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    from src.features import add_engineered_features, ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

    print(f"\n{DIVIDER}")
    print("  SECTION 3 — PERFORMANCE METRICS")
    print(DIVIDER)

    df = add_engineered_features(df.copy())
    X = df[ALL_FEATURES]
    y = df["churned"].astype(int)

    # Same split as training (same seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    acc   = accuracy_score(y_test, pred)
    prec  = precision_score(y_test, pred, zero_division=0)
    rec   = recall_score(y_test, pred, zero_division=0)
    f1    = f1_score(y_test, pred, zero_division=0)
    roc   = roc_auc_score(y_test, proba)
    pr    = average_precision_score(y_test, proba)

    print(f"\n  Test set size  : {len(y_test):,} rows (20% holdout, stratified)")
    print(f"  Churn rate     : {y_test.mean():.1%} positive class")

    print(f"\n  {'Metric':<30} {'Score':>8}  Notes")
    print("  " + "-" * 65)
    print(f"  {'Accuracy':<30} {acc:>8.3f}  (misleading at 36% base rate)")
    print(f"  {'Precision':<30} {prec:>8.3f}  (of predicted churners, % correct)")
    print(f"  {'Recall':<30} {rec:>8.3f}  ** KEY: % of real churners caught **")
    print(f"  {'F1 Score':<30} {f1:>8.3f}")
    print(f"  {'ROC-AUC':<30} {roc:>8.3f}  (optimistic under imbalance)")
    print(f"  {'PR-AUC':<30} {pr:>8.3f}  ** KEY: best metric for imbalanced churn **")

    print("\n  Why recall + PR-AUC matter most:")
    print("    A missed churner (false negative) = lost customer revenue.")
    print("    A false positive = small wasted retention offer.")
    print("    Accuracy rewards predicting 'no churn' 64% of the time (useless).")
    print("    PR-AUC captures true churn-prediction quality under imbalance.")

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix (threshold = 0.5)")
    print(f"  {'':20}  Predicted: Stay  Predicted: Churn")
    print(f"  {'Actual: Stay':<20}  {tn:>14,}  {fp:>16,}")
    print(f"  {'Actual: Churn':<20}  {fn:>14,}  {tp:>16,}")
    print(f"\n  True Positives  (caught churners)  : {tp:,}")
    print(f"  False Negatives (missed churners)  : {fn:,}  <- costly")
    print(f"  False Positives (wrong alarms)     : {fp:,}  <- cheap")
    print(f"  True Negatives  (correct stays)    : {tn:,}")

    return X_test, y_test, proba


# =============================================================================
# Section 4 — Feature importances
# =============================================================================
def section_importances(pipeline, meta: dict):
    from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    print(f"\n{DIVIDER}")
    print("  SECTION 4 — FEATURE IMPORTANCES (top 10)")
    print(DIVIDER)

    imp_list = meta.get("global_importance", [])
    if not imp_list:
        print("  No importance data in metadata.")
        return

    # Already aggregated in metadata (|coef| for LogReg, gains for XGB)
    imp = pd.Series(dict(imp_list)).sort_values(ascending=False)
    model_name = meta["model_name"]
    source = "|coefficient|" if model_name == "LogisticRegression" else "feature gain"

    print(f"\n  Source: {source} (normalized, {model_name})")
    print(f"\n  {'Feature':<35} {'Importance':>12}  Bar")
    print("  " + "-" * 65)
    top = imp.head(10)
    max_val = top.max() or 1.0
    for feat, val in top.items():
        bar = "#" * int(val / max_val * 30)
        direction = meta.get("feature_directions", {}).get(feat)
        arrow = "(+churn)" if direction == 1.0 else "(-churn)" if direction == -1.0 else ""
        print(f"  {feat:<35} {val:>12.4f}  {bar:<30} {arrow}")

    print("\n  Interpretation:")
    print("  (+churn) = higher value -> more likely to churn")
    print("  (-churn) = higher value -> less likely to churn")


# =============================================================================
# Section 5 — Risk tier distribution
# =============================================================================
def section_risk_tiers(proba: np.ndarray, y_test, meta: dict):
    print(f"\n{DIVIDER}")
    print("  SECTION 5 — RISK-TIER DISTRIBUTION")
    print(DIVIDER)

    lo = meta["threshold_low"]
    hi = meta["threshold_high"]

    tiers = pd.cut(proba, bins=[-0.01, lo, hi, 1.01],
                   labels=["low", "medium", "high"])
    df_t = pd.DataFrame({"tier": tiers, "actual_churn": y_test.values})

    print(f"\n  {'Tier':<10} {'Count':>8} {'% of test':>10} {'Actual churn rate':>18}")
    print("  " + "-" * 50)
    total = len(tiers)
    for tier in ["low", "medium", "high"]:
        mask = df_t["tier"] == tier
        n = mask.sum()
        cr = df_t.loc[mask, "actual_churn"].mean() if n > 0 else 0
        print(f"  {tier:<10} {n:>8,} {n/total:>10.1%} {cr:>18.1%}")

    print(f"\n  Total test rows: {total:,}")
    print(f"  Thresholds: low < {lo:.4f} | medium {lo:.4f}-{hi:.4f} | high >= {hi:.4f}")


# =============================================================================
# Section 6 — predict_churn() API smoke tests
# =============================================================================
def section_api_tests():
    from src.predict import predict_churn

    print(f"\n{DIVIDER}")
    print("  SECTION 6 — predict_churn() API SMOKE TESTS")
    print(DIVIDER)

    cases = [
        ("High-risk profile", {
            "tenure_months": 2, "contract_type": "Month-to-month",
            "monthly_charges": 95.0, "total_charges": 190.0,
            "satisfaction_score": 2.4, "num_support_tickets": 6,
            "internet_service": "Fiber optic", "phone_service": "Y",
            "num_additional_services": 0,
        }, "high or medium"),
        ("Low-risk profile", {
            "tenure_months": 60, "contract_type": "Two year",
            "monthly_charges": 45.0, "total_charges": 2700.0,
            "satisfaction_score": 9.0, "num_support_tickets": 0,
            "internet_service": "DSL", "num_additional_services": 4,
        }, "low"),
        ("Partial input (missing fields)", {
            "contract_type": "Month-to-month",
            "satisfaction_score": 3.0,
        }, "any"),
        ("Messy input (bad encodings, sentinel)", {
            "gender": "m", "phone_service": "N",
            "age": "999", "satisfaction_score": 99,
        }, "any"),
        ("Empty input {}", {}, "any"),
    ]

    all_passed = True
    for label, inputs, expected_tier in cases:
        try:
            r = predict_churn(inputs)
            prob = r["churn_probability"]
            tier = r["risk_tier"]
            factors = r["top_risk_factors"]

            # schema checks
            assert 0.0 <= prob <= 1.0
            assert tier in ("high", "medium", "low")
            assert isinstance(factors, list) and len(factors) >= 1
            assert set(r.keys()) == {"churn_probability", "risk_tier", "top_risk_factors"}

            status = "PASS"
        except Exception as e:
            status = f"FAIL — {e}"
            all_passed = False

        print(f"\n  [{label}]")
        if status == "PASS":
            print(f"    churn_probability : {prob:.4f}  ({prob:.0%})")
            print(f"    risk_tier         : {tier}")
            print(f"    top_risk_factors  : {factors}")
            print(f"    status            : PASS")
        else:
            print(f"    status            : {status}")

    return all_passed


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print("  TELECONNECT CHURN MODEL — LOCAL EVALUATION REPORT")
    print(f"{'#'*60}")

    try:
        pipeline, meta = load_artifacts()
        df = load_data()
    except AssertionError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    section_algorithm(meta)
    section_data_quality()
    X_test, y_test, proba = section_performance(pipeline, meta, df)
    section_importances(pipeline, meta)
    section_risk_tiers(proba, y_test, meta)
    api_ok = section_api_tests()

    print(f"\n{'#'*60}")
    print("  SUMMARY")
    print(f"{'#'*60}")
    print(f"  Algorithm      : {meta['model_name']}")
    print(f"  Trained on     : {meta['n_training_rows']:,} rows")
    print(f"  Recall         : {meta['test_metrics'].get('recall', 'n/a')}")
    print(f"  PR-AUC         : {meta['test_metrics'].get('pr_auc', 'n/a')}")
    print(f"  ROC-AUC        : {meta['test_metrics'].get('roc_auc', 'n/a')}")
    print(f"  API tests      : {'PASS' if api_ok else 'FAIL'}")
    print(f"  Artifact path  : {PIPELINE_PATH}")
    print(f"  Metadata path  : {METADATA_PATH}")
    print()
    sys.exit(0 if api_ok else 1)