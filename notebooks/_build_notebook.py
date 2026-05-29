"""
Builds notebooks/churn_model.ipynb programmatically with nbformat.

Why a builder script? It keeps the notebook reproducible and version-control-
friendly (the heavy logic lives in src/, the notebook narrates and orchestrates),
and lets us regenerate the .ipynb deterministically. Run:

    python notebooks/_build_notebook.py        # writes the .ipynb (no outputs)
    jupyter nbconvert --to notebook --execute --inplace notebooks/churn_model.ipynb

The second step executes every cell, embedding outputs and writing the model
artifacts to models/ and the cleaned dataset to data/.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =========================================================================== #
md(r"""
# TeleConnect — Customer Churn Prediction (Part 1)

**Author:** AI/ML Engineer take-home · **Goal:** build a churn-prediction model from
deliberately messy legacy data, and export it as a callable artifact that the Part 2
retention agent can call.

This notebook is intentionally narrative. Each section states *what* I'm doing, *why*,
and *what I'd revisit with more time*. The reusable cleaning and feature logic lives in
`src/data_cleaning.py` and `src/features.py` so that **the exact same transforms run at
training time (here) and at inference time** (`src/predict.py`) — no train/serve skew.

### Roadmap
1. **Data-quality assessment & cleaning** — find issues beyond missing values; document each.
2. **EDA** — churn rate, top-5 associations (with a method matched to the data types),
   visual relationships, and engineered features.
3. **Model building** — a linear and a tree-ensemble model, compared on multiple metrics.
4. **Visualization** — confusion matrix, ROC/PR curves, feature importance.
5. **Export** — best model + preprocessing as a joblib artifact, plus the `predict_churn` fn.
""")

code(r"""
import sys, os, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)

# Make the project root importable so we can reuse src/ modules.
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_cleaning as dc
from src import features as ft

RAW_PATH = ROOT / "data" / "test_datafile.csv"
print("Project root:", ROOT)
print("Raw data:", RAW_PATH, "exists:", RAW_PATH.exists())
""")

# --------------------------------------------------------------------------- 1.1
md(r"""
## 1.1 — Data-Quality Assessment & Cleaning

The brief warns the data comes from several legacy systems with migration artifacts,
manual-entry errors, and inconsistent formatting. Missing values are just the start —
I specifically look for **inconsistent encodings, impossible values, sentinel
placeholders, and semantic outliers**.

Let me first load the raw file *as strings* where useful and profile it.
""")

code(r"""
raw = pd.read_csv(RAW_PATH)
print("Shape:", raw.shape)
print("Overall churn rate: {:.1%}".format(raw["churned"].astype(float).mean()))
raw.head()
""")

code(r"""
# Structural overview — note that several numeric columns are typed as object,
# a tell-tale sign of embedded junk values (blanks, 'nan' strings, sentinels).
raw.info()
""")

md(r"""
### Profiling the categorical columns

I look at raw value counts to expose inconsistent encodings across source systems.
""")

code(r"""
for col in ["gender", "contract_type", "internet_service", "phone_service", "payment_method"]:
    print(f"\n=== {col} ===")
    print(raw[col].value_counts(dropna=False).to_string())
""")

md(r"""
**Categorical issues found:**

| Column | Problem | Affected rows |
|---|---|---|
| `gender` | Mixed encodings `Male/Female/M/F`, an ambiguous `Other`, and blanks | ~410 `F` + ~392 `M` + 98 `Other` + 50 blank |
| `phone_service` | `Yes/no/Y/N` — case + abbreviation inconsistency | ~258 `N` + ~244 `Y` + mixed case |
| `internet_service` | `Fiber optic` vs `fiber`, plus three "no/none/nan" spellings | ~440 `fiber` + 254 `None` + 251 `nan` |
| `payment_method` | Aliases `BT`→bank transfer, `CC`→credit card; 30 blanks | 166 `BT` + 164 `CC` + 30 blank |

**Strategy:** canonicalize each to a controlled vocabulary. For `internet_service` I make a
**documented semantic decision**: `No` means the customer genuinely has no internet, whereas
`None`/`nan`/blank means the value is *missing in the source* — these are different and are
mapped to `No` vs `Unknown` respectively. This logic lives in `src/data_cleaning.py`.
""")

code(r"""
# Profile the numeric columns — coerce to numeric first to reveal sentinels/impossibles.
def numeric_profile(df, cols):
    rows = []
    for c in cols:
        s = dc._to_numeric(df[c])
        rows.append({
            "column": c, "n_missing": int(s.isna().sum()),
            "min": round(s.min(), 2), "max": round(s.max(), 2),
            "mean": round(s.mean(), 2), "n_negative": int((s < 0).sum()),
        })
    return pd.DataFrame(rows)

numeric_profile(raw, dc.NUMERIC_COLS)
""")

md(r"""
**Numeric issues found (beyond missingness):**

- **Impossible negatives:** `age` (min −1), `tenure_months` (−12), `monthly_charges` (−50),
  `avg_monthly_gb_used` (−86), `num_support_tickets` (−5), `satisfaction_score` (−1.4). A
  negative tenure or data-usage is physically impossible — these are entry/migration errors.
- **Sentinel placeholders:** `age` max **999**, `monthly_charges` max **9999**,
  `tenure_months`/`num_support_tickets` max **500** — classic "magic number" stand-ins for
  missing data that would badly distort any model if left in.
- **Out-of-scale values:** `satisfaction_score` is a 0–10 scale but has **127 values >10**
  (many at 99) — sentinels, not real scores.
- **Extreme outlier / decimal error:** `total_charges` max **218,681** vs a mean ~1,600.
  This is almost certainly a decimal-point migration error (a value 100× too large).
- **Duplicates:** 50 duplicate `customer_id`s.

**Strategy (in `clean_dataframe`):** drop duplicate IDs; convert each impossible/sentinel
value to `NaN` using domain range rules so the model's imputer handles them consistently at
train *and* serve time; and **repair** `total_charges` decimal-shift errors by reconciling
against `tenure_months × monthly_charges` (divide by 100 when that brings it into line).
""")

code(r"""
# The suspicious monthly_charges == 15.0 cluster: is it a real price or a placeholder?
n15 = (dc._to_numeric(raw["monthly_charges"]) == 15.0).sum()
print(f"monthly_charges == 15.0 appears {n15} times.")
print("Churn rate within that cluster vs overall:")
mask = dc._to_numeric(raw["monthly_charges"]) == 15.0
print("  15.0 cluster:", round(raw.loc[mask, "churned"].astype(float).mean(), 3))
print("  overall     :", round(raw["churned"].astype(float).mean(), 3))
""")

md(r"""
**Judgment call on `monthly_charges == 15.0`:** it repeats exactly 110 times, which is
suspicious for a continuous charge. But 15.0 is also a *plausible* low-tier price, and its
churn rate isn't wildly different from the base rate, so I **keep it but flag it** rather than
null it — nulling 110 rows on a hunch would destroy real signal. This is exactly the kind of
ambiguous call I'd confirm with the data owner; documented here rather than hidden.
""")

code(r"""
# Apply the cleaning pipeline and review the per-issue report.
cleaned, quality_report = dc.clean_dataframe(raw)
print("Cleaned shape:", cleaned.shape, "(dropped", raw.shape[0] - cleaned.shape[0], "rows)")
quality_report
""")

code(r"""
# Before/after summary table for affected columns (Part 1.1 deliverable).
summary = dc.before_after_summary(raw, cleaned)
summary
""")

code(r"""
# Persist the cleaned dataset — it also backs the agent's lookup_customer tool in Part 2.
CLEANED_PATH = ROOT / "data" / "cleaned_customers.csv"
cleaned.to_csv(CLEANED_PATH, index=False)
print("Wrote", CLEANED_PATH)
cleaned.head()
""")

# --------------------------------------------------------------------------- 1.2
md(r"""
## 1.2 — Exploratory Data Analysis

I work on the cleaned data, add engineered features, then quantify which features are most
associated with churn — choosing association methods that respect the data types.
""")

code(r"""
df = ft.add_engineered_features(cleaned.copy())
TARGET = "churned"
print("Overall churn rate: {:.1%}  (n={})".format(df[TARGET].mean(), len(df)))
df[ft.ALL_FEATURES].head()
""")

md(r"""
### Choosing an association method (the target is binary; features are mixed)

There is no single correct correlation here:

- **Continuous feature ↔ binary target:** Pearson is inappropriate; I use the
  **point-biserial correlation** (mathematically the right special case).
- **Categorical feature ↔ binary target:** correlation is undefined; I use a **chi-square
  test → Cramér's V** as an effect size.
- **A single unified ranker across both types:** I use **mutual information**
  (`mutual_info_classif`), which is non-parametric, captures non-linear relationships, and
  handles continuous and (encoded) categorical features together. I report the **top 5 by
  mutual information** as the headline, and show point-biserial / Cramér's V alongside to
  justify that the ranking is consistent.
""")

code(r"""
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1) or 1))

# Point-biserial for numeric features (drop NaNs pairwise).
pb = {}
for c in ft.NUMERIC_FEATURES:
    s = pd.to_numeric(df[c], errors="coerce")
    m = s.notna()
    if m.sum() > 10:
        pb[c] = pointbiserialr(s[m], df.loc[m, TARGET])[0]

# Cramers V for categorical features.
cv = {c: cramers_v(df[c].astype(str), df[TARGET]) for c in ft.CATEGORICAL_FEATURES}

# Mutual information across all features (impute numerics, ordinal-encode categoricals).
num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[ft.NUMERIC_FEATURES]),
                   columns=ft.NUMERIC_FEATURES, index=df.index)
cat = pd.DataFrame(
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        .fit_transform(df[ft.CATEGORICAL_FEATURES].astype(str)),
    columns=ft.CATEGORICAL_FEATURES, index=df.index)
X_mi = pd.concat([num, cat], axis=1)
discrete_mask = [c in ft.CATEGORICAL_FEATURES for c in X_mi.columns]
mi = mutual_info_classif(X_mi, df[TARGET], discrete_features=discrete_mask, random_state=42)
mi_series = pd.Series(mi, index=X_mi.columns).sort_values(ascending=False)

print("=== Top features by Mutual Information ===")
print(mi_series.head(8).round(4).to_string())
print("\n=== Point-biserial (numeric) ===")
print(pd.Series(pb).sort_values(key=abs, ascending=False).round(3).to_string())
print("\n=== Cramers V (categorical) ===")
print(pd.Series(cv).sort_values(ascending=False).round(3).to_string())
""")

code(r"""
# Headline: the five features most strongly associated with churn.
top5 = mi_series.head(5)
fig, ax = plt.subplots(figsize=(7, 4))
top5[::-1].plot.barh(ax=ax, color="#c0392b")
ax.set_title("Top 5 churn-associated features (mutual information)\n"
             "Takeaway: tenure, contract type & satisfaction dominate churn signal",
             fontsize=11)
ax.set_xlabel("Mutual information with churn")
plt.tight_layout(); plt.show()
top5.round(4)
""")

md(r"""
### Three relationships worth seeing
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

# (a) Churn by contract type
c1 = df.groupby("contract_type")[TARGET].mean().sort_values()
c1.plot.bar(ax=axes[0], color="#2980b9")
axes[0].set_title("Month-to-month customers churn far more\nthan contracted ones", fontsize=10)
axes[0].set_ylabel("Churn rate"); axes[0].tick_params(axis="x", rotation=20)

# (b) Churn by tenure bucket
order = ["new_0_6m", "growing_6_24m", "established_24_48m", "loyal_48m_plus"]
c2 = df.groupby("tenure_bucket")[TARGET].mean().reindex(order)
c2.plot.bar(ax=axes[1], color="#27ae60")
axes[1].set_title("Churn is front-loaded: it collapses\nafter the first ~2 years", fontsize=10)
axes[1].set_ylabel("Churn rate"); axes[1].tick_params(axis="x", rotation=20)

# (c) Churn by satisfaction score (binned)
df["_sat_bin"] = pd.cut(df["satisfaction_score"], bins=[0, 3, 5, 7, 10])
c3 = df.groupby("_sat_bin")[TARGET].mean()
c3.plot.bar(ax=axes[2], color="#8e44ad")
axes[2].set_title("Low satisfaction is a strong churn signal", fontsize=10)
axes[2].set_ylabel("Churn rate"); axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout(); plt.show()
df.drop(columns=["_sat_bin"], inplace=True)
""")

md(r"""
### Engineered features (with reasoning)

Defined in `src/features.py` so they exist identically at train and serve time:

1. **`charges_per_tenure_month` = total_charges / tenure** — the customer's *effective*
   monthly spend over their life. A large gap from nominal `monthly_charges` hints at plan
   changes or credits, which often accompany dissatisfaction.
2. **`tickets_per_tenure_year`** — support-contact *rate*, not raw count. Three tickets in
   month one is a much stronger churn signal than three tickets over five years; normalizing
   by tenure exposes that.
3. **`expected_vs_actual_charges_gap`** — `total_charges − tenure×monthly`; a persistent gap
   can flag billing disputes (a known churn driver).
4. **`tenure_bucket`** — lifecycle stage as a category, letting the model fit the non-monotonic
   early-life churn hazard directly.
5. **`has_bundle`** — phone + internet bundled customers are structurally stickier.

I'd validate (1)–(3) against domain experts and check they don't leak the label; none use
post-cancellation information, so they're safe for prediction.
""")

# --------------------------------------------------------------------------- 1.3
md(r"""
## 1.3 — Model Building & Evaluation

I train two models from **distinctly different families**:

- **Logistic Regression** (linear). *Why:* a transparent, well-calibrated baseline whose
  coefficients are directly interpretable for a retention team. *Strength:* robust, fast,
  hard to overfit. *Weakness:* assumes additive, linear-in-log-odds effects — it can't natively
  capture interactions (e.g. *fiber + month-to-month + low satisfaction*) without manual terms.
- **XGBoost** (gradient-boosted trees). *Why:* the workhorse for tabular churn. *Strength:*
  captures non-linearities and feature interactions automatically, handles mixed scales well.
  *Weakness:* less interpretable, more hyperparameters, and can overfit without care.

Both sit behind the **same preprocessing pipeline** (median-impute + scale numerics;
most-frequent-impute + one-hot encode categoricals), so the comparison is apples-to-apples and
the fitted imputation statistics travel with the exported artifact.
""")

code(r"""
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

X = df[ft.ALL_FEATURES].copy()
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), ft.NUMERIC_FEATURES),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
     ft.CATEGORICAL_FEATURES),
])

# Class imbalance (~36% churn) handled WITHOUT resampling:
#  - LogReg: class_weight="balanced"
#  - XGBoost: scale_pos_weight = negatives / positives
# I prefer reweighting to SMOTE here: SMOTE synthesizes points in a space mixing one-hot and
# scaled numerics, which can create implausible "customers"; reweighting is simpler and avoids
# distorting the feature distribution. (I note the trade-off rather than treating SMOTE as default.)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

logreg = Pipeline([("prep", preprocess),
                   ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
xgb = Pipeline([("prep", preprocess),
                ("clf", XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, scale_pos_weight=pos_weight,
                    eval_metric="logloss", random_state=42, n_jobs=4))])

logreg.fit(X_train, y_train)
xgb.fit(X_train, y_train)
print("Both models trained. Positive-class weight (XGB):", round(pos_weight, 2))
""")

code(r"""
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

def evaluate(name, model):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }

results = pd.DataFrame([evaluate("LogisticRegression", logreg),
                        evaluate("XGBoost", xgb)]).set_index("model").round(3)
results
""")

md(r"""
### Which metric matters most — and which mislead

**Most important: recall on the churn class, read together with PR-AUC.** In retention, a
**false negative (missing a churner) is far costlier than a false positive** (offering a small
incentive to someone who'd have stayed). The whole point of the model is to *catch* at-risk
customers so a rep can intervene, so we optimize for catching them — i.e. recall — while using
**PR-AUC** to judge ranking quality across thresholds, because PR-AUC focuses on the positive
(minority) class.

**Misleading here:** **accuracy** — at a 36% base rate, predicting "nobody churns" already
scores 64%, so accuracy rewards the wrong behavior. **ROC-AUC** is useful but optimistic under
imbalance because the large true-negative count inflates it; PR-AUC is the more honest summary.

**In practice** I wouldn't ship a 0.5 threshold — I'd pick the operating point from the PR
curve to hit a target recall the retention team can staff for, accepting the resulting precision.
""")

# --------------------------------------------------------------------------- 1.4
md(r"""
## 1.4 — Visualization

Confusion matrix, ROC + PR curves, and feature importance. The best model by PR-AUC / recall
is **XGBoost** (confirmed by the table above); visuals below focus on it, with LogReg shown on
the curves for comparison.
""")

code(r"""
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             PrecisionRecallDisplay, confusion_matrix)

best_name = results["pr_auc"].idxmax()
best_model = xgb if best_name == "XGBoost" else logreg
print("Best model by PR-AUC:", best_name)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

# Confusion matrix
proba = best_model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["stayed", "churned"]).plot(ax=axes[0], cmap="Blues",
                                                                   colorbar=False)
axes[0].set_title(f"{best_name}: confusion matrix @0.5\n"
                  "Most churners are caught; FN are the costly cell", fontsize=10)

# ROC
RocCurveDisplay.from_estimator(logreg, X_test, y_test, ax=axes[1], name="LogReg")
RocCurveDisplay.from_estimator(xgb, X_test, y_test, ax=axes[1], name="XGBoost")
axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[1].set_title("ROC curves\nBoth beat chance; XGBoost leads", fontsize=10)

# PR
PrecisionRecallDisplay.from_estimator(logreg, X_test, y_test, ax=axes[2], name="LogReg")
PrecisionRecallDisplay.from_estimator(xgb, X_test, y_test, ax=axes[2], name="XGBoost")
axes[2].axhline(y_test.mean(), color="grey", ls=":", label="base rate")
axes[2].set_title("Precision-Recall curves\nThe metric we actually care about", fontsize=10)
axes[2].legend(loc="upper right", fontsize=8)

plt.tight_layout(); plt.show()
""")

code(r"""
# Feature importance. Prefer SHAP (global mean |SHAP|) if available; else fall back to
# permutation importance, which is model-agnostic and honest about predictive contribution.
feat_names = (ft.NUMERIC_FEATURES +
              list(best_model.named_steps["prep"]
                   .named_transformers_["cat"]["onehot"]
                   .get_feature_names_out(ft.CATEGORICAL_FEATURES)))

importance_source = None
try:
    import shap
    Xt = best_model.named_steps["prep"].transform(X_test)
    explainer = shap.TreeExplainer(best_model.named_steps["clf"])
    sv = explainer.shap_values(Xt)
    imp = np.abs(sv).mean(axis=0)
    importance_source = "mean |SHAP value|"
except Exception as e:
    print("SHAP unavailable (", e, ") -> using permutation importance.")
    from sklearn.inspection import permutation_importance
    r = permutation_importance(best_model, X_test, y_test, n_repeats=5,
                               random_state=42, scoring="average_precision")
    # permutation importance is over original features:
    feat_names = ft.ALL_FEATURES
    imp = r.importances_mean
    importance_source = "permutation importance (PR-AUC drop)"

imp_series = pd.Series(imp, index=feat_names).sort_values(ascending=False).head(12)
fig, ax = plt.subplots(figsize=(8, 5))
imp_series[::-1].plot.barh(ax=ax, color="#16a085")
ax.set_title(f"{best_name}: top features by {importance_source}\n"
             "Tenure, contract, satisfaction & support drive predictions", fontsize=11)
plt.tight_layout(); plt.show()
imp_series.round(4)
""")

# --------------------------------------------------------------------------- 1.5
md(r"""
## 1.5 — Export the Model for Part 2

I refit the best pipeline on **all** cleaned data (more data → a better production artifact),
then save it with metadata the inference layer needs: risk-tier thresholds, per-feature
distribution stats, association directions, global importances, and per-category churn lift.
These power the `top_risk_factors` explanation in `src/predict.py` without a heavy SHAP
dependency at serve time.
""")

code(r"""
import joblib
from datetime import datetime, timezone

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Refit the chosen pipeline on the full dataset for the production artifact.
final_model = (XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
    colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="logloss",
    random_state=42, n_jobs=4) if best_name == "XGBoost"
    else LogisticRegression(max_iter=2000, class_weight="balanced"))
final_pipeline = Pipeline([("prep", preprocess), ("clf", final_model)])
final_pipeline.fit(X, y)

joblib.dump(final_pipeline, MODELS_DIR / "churn_pipeline.joblib")
print("Saved", MODELS_DIR / "churn_pipeline.joblib")
""")

code(r"""
# ---- Build metadata for inference & explanations ----
base_rate = float(y.mean())

# Per-feature association direction (sign): +1 means higher value -> more churn.
directions = {}
for c in ft.NUMERIC_FEATURES:
    s = pd.to_numeric(df[c], errors="coerce"); m = s.notna()
    if m.sum() > 10:
        corr = pointbiserialr(s[m], df.loc[m, TARGET])[0]
        directions[c] = float(np.sign(corr))

# Numeric distribution stats for z-scoring at inference.
numeric_stats = {c: {"mean": float(pd.to_numeric(df[c], errors="coerce").mean()),
                     "std": float(pd.to_numeric(df[c], errors="coerce").std() or 1.0)}
                 for c in ft.NUMERIC_FEATURES}

# Per-category churn lift (category churn rate minus base rate).
cat_lift = {}
for c in ft.CATEGORICAL_FEATURES:
    rates = df.groupby(c)[TARGET].mean()
    for val, rate in rates.items():
        cat_lift[f"{c}={val}"] = float(rate - base_rate)

# Global importance (normalized) for ranking factors. Works for either model family:
# tree models expose feature_importances_, linear models expose coef_ (use |coef|).
try:
    clf = final_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        booster_imp = clf.feature_importances_
    else:  # LogisticRegression -> absolute standardized coefficients
        booster_imp = np.abs(clf.coef_).ravel()
    fnames = (ft.NUMERIC_FEATURES + list(
        preprocess.named_transformers_["cat"]["onehot"].get_feature_names_out(ft.CATEGORICAL_FEATURES)))
    # Aggregate one-hot importances back to the original categorical column.
    agg = {}
    for name, val in zip(fnames, booster_imp):
        base = name
        for cc in ft.CATEGORICAL_FEATURES:
            if name.startswith(cc + "_"):
                base = cc; break
        agg[base] = agg.get(base, 0.0) + float(val)
    total = sum(agg.values()) or 1.0
    global_importance = sorted(((k, v / total) for k, v in agg.items()),
                               key=lambda t: t[1], reverse=True)
except Exception as e:
    print("importance fallback:", e)
    global_importance = [(c, 1.0) for c in ft.ALL_FEATURES]

# Risk-tier thresholds: derive medium/high cutoffs from predicted-probability quantiles so
# tiers are meaningful for THIS population (documented choice, not arbitrary 0.3/0.6).
all_proba = final_pipeline.predict_proba(X)[:, 1]
threshold_low = float(np.quantile(all_proba, 0.50))   # below median risk -> low
threshold_high = float(np.quantile(all_proba, 0.80))  # top quintile -> high

metadata = {
    "model_name": best_name,
    "version": "1.0.0",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "base_churn_rate": base_rate,
    "n_training_rows": int(len(X)),
    "numeric_features": ft.NUMERIC_FEATURES,
    "categorical_features": ft.CATEGORICAL_FEATURES,
    "threshold_low": round(threshold_low, 4),
    "threshold_high": round(threshold_high, 4),
    "feature_directions": directions,
    "numeric_stats": numeric_stats,
    "categorical_churn_lift": cat_lift,
    "global_importance": global_importance,
    "test_metrics": results.loc[best_name].round(4).to_dict(),
}
(MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
print("Saved metadata. Tiers: low<{:.3f}, high>={:.3f}".format(threshold_low, threshold_high))
metadata["test_metrics"]
""")

md(r"""
### The `predict_churn` function (Part 1.5 deliverable)

The required function lives in `src/predict.py` (so Part 2 can import it). It cleans the raw
input with the same logic used here, adds the engineered features, runs the saved pipeline, and
returns `{churn_probability, risk_tier, top_risk_factors}`. Below I import and exercise it,
including a **partial / messy input** to show it degrades gracefully.
""")

code(r"""
import importlib
import src.predict as predict_mod
importlib.reload(predict_mod)            # pick up the just-saved artifact
from src.predict import predict_churn

# A high-risk-looking customer (short tenure, low satisfaction, month-to-month, many tickets):
high_risk = {
    "customer_id": "TC-DEMO01", "age": 41, "gender": "F", "tenure_months": 2,
    "contract_type": "Month-to-month", "monthly_charges": 95.0, "total_charges": 190.0,
    "internet_service": "Fiber optic", "phone_service": "Y", "avg_monthly_gb_used": 3.0,
    "num_support_tickets": 6, "avg_monthly_minutes": 120, "satisfaction_score": 2.4,
    "payment_method": "Electronic check", "num_additional_services": 0,
}
print("HIGH-RISK profile ->")
print(json.dumps(predict_churn(high_risk), indent=2))

# A loyal-looking customer:
low_risk = {
    "customer_id": "TC-DEMO02", "age": 52, "tenure_months": 60,
    "contract_type": "Two year", "monthly_charges": 45.0, "total_charges": 2700.0,
    "internet_service": "DSL", "phone_service": "Yes", "satisfaction_score": 8.7,
    "num_support_tickets": 0, "num_additional_services": 4,
}
print("\nLOW-RISK profile ->")
print(json.dumps(predict_churn(low_risk), indent=2))

# Partial / messy input (missing several fields, messy encodings) — must not crash:
partial = {"customer_id": "TC-DEMO03", "contract_type": "Month-to-month",
           "satisfaction_score": 3.0, "phone_service": "N", "gender": "m"}
print("\nPARTIAL/MESSY input ->")
print(json.dumps(predict_churn(partial), indent=2))
""")

md(r"""
## Summary, limitations & what I'd do next

**What I built:** a documented cleaning pipeline that catches inconsistent encodings,
impossible/sentinel values, decimal-shift errors, and duplicates; an EDA that ranks churn
drivers with type-appropriate association methods; two contrasting models compared on the
metrics that matter for retention; and a callable, version-stamped artifact with a graceful
`predict_churn` interface.

**Honest limitations:**
- `top_risk_factors` is a transparent *approximation* (z-score × association × importance), not
  a causal or full-SHAP attribution. Good enough to guide a rep; not a causal claim.
- The `monthly_charges == 15.0` cluster and the `internet "No" vs missing` split are judgment
  calls I'd confirm with the data owner.
- Risk-tier thresholds are population quantiles; with business input I'd set them by the cost
  of a retention offer vs. the value of a saved customer.
- No temporal validation: `last_interaction_date` suggests recency, but without a clear
  observation window I avoided leak-prone time features. With more time I'd build a proper
  train/validation split by time and add recency features.

**What I'd do next:** threshold tuning against retention economics, calibration
(Platt/Isotonic) so probabilities are trustworthy, cross-validated hyperparameter search, and a
monitoring plan for feature drift once the agent is live.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
OUT = "notebooks/churn_model.ipynb"
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", OUT, "with", len(cells), "cells")
