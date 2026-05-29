"""
Generate docs/model_comparison.md — a full metrics comparison of all trained models.

Trains LogisticRegression and XGBoost with the same pipeline and split used in
the notebook, then writes a detailed markdown report with every metric, the
confusion matrix, and feature importances for both.

Run:
    python tests/generate_model_comparison.py
Output:
    docs/model_comparison.md
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_cleaning import clean_dataframe
from src.features import add_engineered_features, ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

OUT = ROOT / "docs" / "model_comparison.md"
OUT.parent.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
def build_pipeline(clf) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", preprocess), ("clf", clf)])


def evaluate(name: str, pipeline, X_test, y_test, thresholds=(0.3, 0.4, 0.5)) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    rows = {}
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows[t] = {
            "threshold": t,
            "accuracy":  round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, pred, zero_division=0), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
    return {
        "name": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc":  round(average_precision_score(y_test, proba), 4),
        "by_threshold": rows,
        "proba": proba,
    }


def get_importances(pipeline, name: str) -> list[tuple[str, float]]:
    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]
    ohe_cols = list(prep.named_transformers_["cat"]["onehot"]
                    .get_feature_names_out(CATEGORICAL_FEATURES))
    feat_names = NUMERIC_FEATURES + ohe_cols

    if hasattr(clf, "feature_importances_"):
        raw = clf.feature_importances_
    else:
        raw = np.abs(clf.coef_).ravel()

    # aggregate one-hot back to original categorical column
    agg: dict[str, float] = {}
    for fname, val in zip(feat_names, raw):
        base = fname
        for cc in CATEGORICAL_FEATURES:
            if fname.startswith(cc + "_"):
                base = cc
                break
        agg[base] = agg.get(base, 0.0) + float(val)

    total = sum(agg.values()) or 1.0
    return sorted(((k, v / total) for k, v in agg.items()), key=lambda t: t[1], reverse=True)


# --------------------------------------------------------------------------- #
def main():
    print("Loading and cleaning data...")
    raw = pd.read_csv(ROOT / "data" / "test_datafile.csv")
    df, quality_report = clean_dataframe(raw)
    df = add_engineered_features(df)
    X = df[ALL_FEATURES]
    y = df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = [
        ("Logistic Regression", build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced")
        )),
        ("XGBoost", build_pipeline(
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=pos_weight,
                eval_metric="logloss", random_state=42, n_jobs=4,
            )
        )),
    ]

    results = []
    importances = {}
    for name, pipeline in models:
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        results.append(evaluate(name, pipeline, X_test, y_test))
        importances[name] = get_importances(pipeline, name)
        print(f"  ROC-AUC={results[-1]['roc_auc']}  PR-AUC={results[-1]['pr_auc']}")

    # ---- write markdown --------------------------------------------------- #
    lines = []
    def w(*args): lines.append(" ".join(str(a) for a in args))

    w(f"# TeleConnect Churn Model — Full Model Comparison Report")
    w(f"")
    w(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    w(f"Dataset: `data/test_datafile.csv` ({len(raw):,} raw rows → {len(df):,} after cleaning)")
    w(f"Test split: 20% holdout, stratified by churn label (seed=42)")
    w(f"Base churn rate: **{y.mean():.1%}**")
    w(f"")
    w(f"---")
    w(f"")

    # -- Overview table
    w(f"## 1. Overview — Threshold-Independent Metrics")
    w(f"")
    w(f"These metrics summarise ranking quality across ALL thresholds.")
    w(f"")
    w(f"| Model | ROC-AUC | PR-AUC | Notes |")
    w(f"|---|---|---|---|")
    for r in results:
        best = "✅ **Best**" if r == max(results, key=lambda x: x["pr_auc"]) else ""
        w(f"| {r['name']} | {r['roc_auc']} | {r['pr_auc']} | {best} |")
    w(f"")
    w(f"> **Why PR-AUC is the primary metric:** at a 36.5% base rate, ROC-AUC is")
    w(f"> optimistic because the large true-negative pool inflates it.")
    w(f"> PR-AUC focuses on the positive (churn) class — the one we actually care about.")
    w(f"")
    w(f"---")
    w(f"")

    # -- Per-threshold metrics for each model
    w(f"## 2. Per-Threshold Metrics")
    w(f"")
    w(f"> Threshold 0.5 is the default. In production, the operating point should")
    w(f"> be tuned to the economics of your retention offers vs. saved-customer value.")
    w(f"")
    for r in results:
        w(f"### {r['name']}")
        w(f"")
        w(f"| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |")
        w(f"|---|---|---|---|---|---|---|---|---|")
        for t, m in r["by_threshold"].items():
            w(f"| {t} | {m['accuracy']} | {m['precision']} | {m['recall']} "
              f"| {m['f1']} | {m['tp']:,} | {m['fp']:,} | {m['fn']:,} | {m['tn']:,} |")
        w(f"")
        w(f"**Metric definitions:**")
        w(f"- **Recall** = TP / (TP + FN) — % of actual churners caught. *Most important for retention.*")
        w(f"- **Precision** = TP / (TP + FP) — % of flagged customers who truly churn.")
        w(f"- **FN (False Negatives)** = missed churners → lost revenue, the costly mistake.")
        w(f"- **FP (False Positives)** = wasted retention offer → cheap mistake.")
        w(f"")

    # -- Confusion matrix at 0.5
    w(f"---")
    w(f"")
    w(f"## 3. Confusion Matrices (threshold = 0.5)")
    w(f"")
    for r in results:
        m = r["by_threshold"][0.5]
        w(f"### {r['name']}")
        w(f"")
        w(f"```")
        w(f"                    Predicted: Stay   Predicted: Churn")
        w(f"  Actual: Stay      {m['tn']:>14,}   {m['fp']:>15,}")
        w(f"  Actual: Churn     {m['fn']:>14,}   {m['tp']:>15,}")
        w(f"```")
        total = m['tp'] + m['fp'] + m['fn'] + m['tn']
        w(f"")
        w(f"| | Count | % of test |")
        w(f"|---|---|---|")
        w(f"| True Positives (caught churners) | {m['tp']:,} | {m['tp']/total:.1%} |")
        w(f"| False Negatives (missed churners) ⚠️ | {m['fn']:,} | {m['fn']/total:.1%} |")
        w(f"| False Positives (wrong alarms) | {m['fp']:,} | {m['fp']/total:.1%} |")
        w(f"| True Negatives (correct stays) | {m['tn']:,} | {m['tn']/total:.1%} |")
        w(f"")

    # -- Feature importances
    w(f"---")
    w(f"")
    w(f"## 4. Feature Importances (Top 15)")
    w(f"")
    w(f"Logistic Regression: absolute standardised coefficients (|coef|).")
    w(f"XGBoost: feature gain (total improvement in loss from splits on that feature).")
    w(f"")
    all_feats = sorted(set(k for imps in importances.values() for k, _ in imps))
    imp_dicts = {name: dict(imps) for name, imps in importances.items()}

    header = "| Feature | " + " | ".join(r["name"] for r in results) + " | Notes |"
    sep    = "|---|" + "---|" * len(results) + "---|"
    w(header); w(sep)

    feat_set = list(imp_dicts[results[0]["name"]].keys())[:15]
    for feat in feat_set:
        cols = " | ".join(
            f"{imp_dicts[r['name']].get(feat, 0):.4f}" for r in results
        )
        note = ""
        if feat == "contract_type":    note = "Strongest driver — month-to-month vs locked-in"
        elif feat == "tenure_bucket":  note = "Early churn hazard is non-monotonic"
        elif feat == "satisfaction_score": note = "Direct satisfaction signal"
        elif feat == "tenure_months":  note = "Longer tenure = stickier"
        elif feat == "num_support_tickets": note = "Friction proxy"
        w(f"| `{feat}` | {cols} | {note} |")
    w(f"")

    # -- Model comparison narrative
    w(f"---")
    w(f"")
    w(f"## 5. Which Model to Use and Why")
    w(f"")
    best = max(results, key=lambda x: x["pr_auc"])
    other = [r for r in results if r != best][0]
    w(f"**Winner by PR-AUC: {best['name']}** (PR-AUC {best['pr_auc']} vs {other['pr_auc']})")
    w(f"")
    w(f"| Dimension | Logistic Regression | XGBoost |")
    w(f"|---|---|---|")
    w(f"| Interpretability | High — coefficients explain each prediction | Lower — requires SHAP |")
    w(f"| Captures interactions | No — linear in log-odds | Yes — tree splits |")
    w(f"| Calibration | Good out-of-box | Needs Platt/isotonic scaling |")
    w(f"| Training speed | Fast | Moderate |")
    w(f"| Overfit risk | Low | Medium (needs tuning) |")
    w(f"| Best for | Baseline, stakeholder trust, small data | Larger data, complex patterns |")
    w(f"")
    w(f"**On this dataset**, the churn signal is largely additive")
    w(f"(contract type + tenure + satisfaction dominate), which explains why")
    w(f"Logistic Regression is competitive — there are limited interaction effects")
    w(f"that XGBoost can exploit. With a larger dataset or richer feature engineering,")
    w(f"XGBoost would likely pull ahead.")
    w(f"")

    # -- Imbalance handling
    w(f"---")
    w(f"")
    w(f"## 6. Class Imbalance Handling")
    w(f"")
    w(f"Base churn rate: **{y.mean():.1%}** (36.5% positive, 63.5% negative)")
    w(f"")
    w(f"| Approach | Applied to | How |")
    w(f"|---|---|---|")
    w(f"| `class_weight='balanced'` | Logistic Regression | Up-weights minority (churn) in the loss |")
    w(f"| `scale_pos_weight` = {pos_weight:.2f} | XGBoost | Multiplies positive-class gradient |")
    w(f"| SMOTE (not applied) | — | Considered but rejected — see note |")
    w(f"")
    w(f"> **Why not SMOTE?** SMOTE synthesises new samples in a mixed continuous+one-hot")
    w(f"> feature space, which can create statistically implausible customers (e.g. a")
    w(f"> 'synthetic' customer with half a contract type). Reweighting is simpler, keeps")
    w(f"> the original feature distribution intact, and avoids data leakage if applied")
    w(f"> inside a cross-validation fold. With more data, SMOTE inside a pipeline CV")
    w(f"> would be worth revisiting.")
    w(f"")

    # -- Data quality summary
    w(f"---")
    w(f"")
    w(f"## 7. Data Quality Summary")
    w(f"")
    w(f"| Column | Issue | Rows affected | Strategy |")
    w(f"|---|---|---|---|")
    for _, row in quality_report.iterrows():
        w(f"| `{row['column']}` | {row['issue']} | {row['rows_affected']:,} | {row['strategy']} |")
    w(f"")

    # -- What to do next
    w(f"---")
    w(f"")
    w(f"## 8. Limitations & What to Do Next")
    w(f"")
    w(f"| Limitation | Suggested fix |")
    w(f"|---|---|")
    w(f"| Threshold fixed at 0.5 | Tune to retention economics (offer cost vs. CLV saved) |")
    w(f"| No probability calibration | Apply Platt scaling / isotonic regression |")
    w(f"| No temporal validation | Build a time-based train/test split; add recency features |")
    w(f"| No hyperparameter search | `GridSearchCV` / `Optuna` on XGBoost |")
    w(f"| `top_risk_factors` = approximation | Replace with full SHAP TreeExplainer |")
    w(f"| No drift monitoring | Track feature distributions + prediction scores in prod |")
    w(f"")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(f"  Models compared : {', '.join(r['name'] for r in results)}")
    print(f"  Sections        : 8")
    print(f"  Best by PR-AUC  : {best['name']} ({best['pr_auc']})")


if __name__ == "__main__":
    main()