"""
Reusable data-cleaning logic for the TeleConnect churn dataset.

This module is the single source of truth for *how raw values are normalized*.
It is imported by:
  - the Part 1 notebook (to clean the full training dataset), and
  - src/predict.py (to clean a single inbound customer record at inference time),

so the exact same value-level transforms run during training and serving — no
train/serve skew.

Design split (deliberate):
  * THIS module only *normalizes encodings* and *nulls-out impossible / sentinel
    values* (turning them into NaN). It does NOT impute.
  * Imputation + scaling + one-hot encoding live in the scikit-learn Pipeline
    built in the notebook, so the fitted imputation statistics travel with the
    model artifact. clean_record() therefore produces NaNs that the saved
    pipeline fills using statistics learned on the training data.

Every cleaning decision here is a judgment call on deliberately messy, legacy
data; the rationale is documented inline and narrated further in the notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column groups                                                               #
# --------------------------------------------------------------------------- #
NUMERIC_COLS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_monthly_gb_used",
    "num_support_tickets",
    "avg_monthly_minutes",
    "satisfaction_score",
    "num_additional_services",
]
CATEGORICAL_COLS = [
    "gender",
    "contract_type",
    "internet_service",
    "phone_service",
    "payment_method",
]
ID_COL = "customer_id"
TARGET_COL = "churned"

# --------------------------------------------------------------------------- #
# Plausible domain ranges (inclusive). Values outside -> NaN (impute later).  #
# These bounds encode real-world telecom domain knowledge, not data peeking.  #
# --------------------------------------------------------------------------- #
RANGE_RULES = {
    "age": (18, 100),               # adult account holders; 999 / -1 are sentinels
    "tenure_months": (0, 120),      # 0–10 yrs; 500 is a sentinel, negatives impossible
    "monthly_charges": (0, 250),    # negative & 9999 are impossible/sentinel
    "total_charges": (0, 20000),    # see repair_total_charges for decimal-shift fix
    "avg_monthly_gb_used": (0, 1000),
    "num_support_tickets": (0, 60),  # 500 is a sentinel
    "avg_monthly_minutes": (0, 2000),
    "satisfaction_score": (0, 10),  # documented assumption: 0–10 scale; 99 is a sentinel
    "num_additional_services": (0, 10),
}

# Tokens that mean "missing" across the various legacy source systems.
NULL_TOKENS = {"", "nan", "none", "null", "n/a", "na", "unknown", "?"}


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to float, turning blanks / 'nan' strings into NaN."""
    s = series.astype(str).str.strip().str.lower()
    s = s.where(~s.isin(NULL_TOKENS), other=np.nan)
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------------- #
# Categorical normalizers — pure per-value functions (work on df or scalar)   #
# --------------------------------------------------------------------------- #
def normalize_gender(value) -> str:
    v = str(value).strip().lower()
    if v in NULL_TOKENS or v == "other":
        return "Unknown"
    if v in {"m", "male"}:
        return "Male"
    if v in {"f", "female"}:
        return "Female"
    return "Unknown"


def normalize_phone_service(value) -> str:
    """Return 'Yes' / 'No' / 'Unknown'."""
    v = str(value).strip().lower()
    if v in NULL_TOKENS:
        return "Unknown"
    if v in {"y", "yes", "true", "1"}:
        return "Yes"
    if v in {"n", "no", "false", "0"}:
        return "No"
    return "Unknown"


def normalize_internet_service(value) -> str:
    """
    Canonicalize to {'Fiber optic', 'DSL', 'No', 'Unknown'}.

    Documented assumption: 'No' = customer genuinely has no internet service;
    'None'/'nan'/blank = the field is *missing* in the source system. These are
    semantically different and are mapped separately ('No' vs 'Unknown').
    """
    v = str(value).strip().lower()
    if v in {"fiber optic", "fiber", "fibre", "fiber-optic"}:
        return "Fiber optic"
    if v == "dsl":
        return "DSL"
    if v == "no":
        return "No"
    if v in NULL_TOKENS:  # '', 'nan', 'none', ...
        return "Unknown"
    return "Unknown"


def normalize_payment_method(value) -> str:
    v = str(value).strip().lower()
    if v in NULL_TOKENS:
        return "Unknown"
    if v in {"bank transfer", "bt", "banktransfer"}:
        return "Bank transfer"
    if v in {"credit card", "cc", "creditcard"}:
        return "Credit card"
    if v in {"electronic check", "e-check", "echeck"}:
        return "Electronic check"
    if v in {"mailed check", "mail check", "check"}:
        return "Mailed check"
    return "Unknown"


_NORMALIZERS = {
    "gender": normalize_gender,
    "phone_service": normalize_phone_service,
    "internet_service": normalize_internet_service,
    "payment_method": normalize_payment_method,
}


def normalize_contract_type(value) -> str:
    """contract_type is already clean in the source; normalize defensively."""
    v = str(value).strip().lower()
    mapping = {
        "month-to-month": "Month-to-month",
        "monthly": "Month-to-month",
        "one year": "One year",
        "two year": "Two year",
    }
    return mapping.get(v, str(value).strip())


_NORMALIZERS["contract_type"] = normalize_contract_type


# --------------------------------------------------------------------------- #
# total_charges repair                                                        #
# --------------------------------------------------------------------------- #
def repair_total_charges(total: float, tenure: float, monthly: float) -> float:
    """
    Fix decimal-shift errors in total_charges (e.g. 218,681 vs an expected ~2,186).

    Heuristic: the *expected* lifetime spend is ~ tenure_months * monthly_charges.
    If the recorded total is more than ~4x that expectation but dividing it by 100
    brings it within a sane band of the expectation, we treat it as a decimal-point
    migration error and repair it. Otherwise we leave it for range-rule nulling.
    Returns NaN if inputs are unusable.
    """
    if pd.isna(total):
        return np.nan
    if pd.isna(tenure) or pd.isna(monthly) or tenure <= 0 or monthly <= 0:
        return total
    expected = tenure * monthly
    if expected <= 0:
        return total
    if total > 4 * expected and abs((total / 100) - expected) < abs(total - expected):
        return total / 100.0
    return total


# --------------------------------------------------------------------------- #
# Single-record cleaning (inference path)                                     #
# --------------------------------------------------------------------------- #
def clean_record(record: dict) -> dict:
    """
    Normalize a single raw customer dict for inference.

    Tolerant of missing keys and messy values. Produces normalized categoricals
    and float numerics with NaN where values are missing/impossible — the saved
    sklearn pipeline then imputes. Mirrors clean_dataframe() exactly.
    """
    out: dict = {}

    # Categoricals
    for col, fn in _NORMALIZERS.items():
        out[col] = fn(record.get(col)) if record.get(col) is not None else (
            "Unknown" if col != "contract_type" else "Month-to-month"
        )

    # Numerics: coerce + apply range rules
    for col in NUMERIC_COLS:
        raw = record.get(col, None)
        val = _scalar_to_numeric(raw)
        lo, hi = RANGE_RULES.get(col, (-np.inf, np.inf))
        if not pd.isna(val) and not (lo <= val <= hi):
            val = np.nan
        out[col] = val

    # total_charges decimal-shift repair (after numeric coercion, before final clamp)
    repaired = repair_total_charges(
        out.get("total_charges", np.nan),
        out.get("tenure_months", np.nan),
        out.get("monthly_charges", np.nan),
    )
    lo, hi = RANGE_RULES["total_charges"]
    out["total_charges"] = repaired if (pd.isna(repaired) or lo <= repaired <= hi) else np.nan

    if ID_COL in record:
        out[ID_COL] = record[ID_COL]
    return out


def _scalar_to_numeric(raw) -> float:
    if raw is None:
        return np.nan
    s = str(raw).strip().lower()
    if s in NULL_TOKENS:
        return np.nan
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


# --------------------------------------------------------------------------- #
# Full-dataframe cleaning (training path) + quality report                    #
# --------------------------------------------------------------------------- #
def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the full dataset. Returns (cleaned_df, quality_report_df).

    The quality_report_df has one row per (column, issue) with the number of
    affected rows and the strategy applied — this drives the Part 1.1 summary.
    """
    df = df.copy()
    report_rows: list[dict] = []

    def log(column, issue, n, strategy):
        report_rows.append(
            {"column": column, "issue": issue, "rows_affected": int(n), "strategy": strategy}
        )

    # --- 1. Duplicate rows / duplicate customer IDs ------------------------ #
    n_exact = df.duplicated().sum()
    if n_exact:
        df = df.drop_duplicates()
        log(ID_COL, "exact duplicate rows", n_exact, "dropped")
    n_id_dup = df.duplicated(subset=[ID_COL]).sum()
    if n_id_dup:
        df = df.drop_duplicates(subset=[ID_COL], keep="first")
        log(ID_COL, "duplicate customer_id (non-identical)", n_id_dup, "kept first occurrence")

    # --- 2. Categorical normalization -------------------------------------- #
    for col, fn in _NORMALIZERS.items():
        if col not in df.columns:
            continue
        before = df[col].astype(str)
        df[col] = df[col].map(fn)
        changed = (before.str.strip() != df[col].astype(str)).sum()
        n_unknown = (df[col] == "Unknown").sum() if col != "contract_type" else 0
        log(
            col,
            f"inconsistent encodings ({before.nunique()} distinct raw values)",
            changed,
            f"canonicalized to {sorted(df[col].unique().tolist())[:6]}"
            + (f"; {n_unknown} -> 'Unknown'" if n_unknown else ""),
        )

    # --- 3. Numeric coercion + range / sentinel nulling -------------------- #
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        coerced = _to_numeric(df[col])
        n_missing_before = coerced.isna().sum()

        lo, hi = RANGE_RULES.get(col, (-np.inf, np.inf))
        out_of_range = coerced.notna() & ~coerced.between(lo, hi)
        n_oor = int(out_of_range.sum())
        coerced = coerced.mask(out_of_range, np.nan)

        df[col] = coerced
        if n_oor:
            log(
                col,
                f"impossible / sentinel values outside [{lo}, {hi}]",
                n_oor,
                "set to NaN (imputed by pipeline)",
            )
        if n_missing_before:
            log(col, "missing / blank / 'nan' tokens", n_missing_before,
                "set to NaN (imputed by pipeline)")

    # --- 4. total_charges decimal-shift repair ----------------------------- #
    if {"total_charges", "tenure_months", "monthly_charges"}.issubset(df.columns):
        repaired = df.apply(
            lambda r: repair_total_charges(
                r["total_charges"], r["tenure_months"], r["monthly_charges"]
            ),
            axis=1,
        )
        n_repaired = int((repaired != df["total_charges"]).fillna(False).sum())
        df["total_charges"] = repaired
        # re-apply the range clamp after repair
        lo, hi = RANGE_RULES["total_charges"]
        still_oor = df["total_charges"].notna() & ~df["total_charges"].between(lo, hi)
        df["total_charges"] = df["total_charges"].mask(still_oor, np.nan)
        if n_repaired:
            log("total_charges", "decimal-shift error vs tenure*monthly", n_repaired,
                "divided by 100 to reconcile")

    # --- 5. Target sanity --------------------------------------------------- #
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        n_bad_target = df[TARGET_COL].isna().sum()
        if n_bad_target:
            df = df.dropna(subset=[TARGET_COL])
            log(TARGET_COL, "missing/invalid label", n_bad_target, "row dropped (cannot train)")
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    report = pd.DataFrame(report_rows, columns=["column", "issue", "rows_affected", "strategy"])
    return df.reset_index(drop=True), report


def before_after_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build a before/after statistics table for affected columns (Part 1.1)."""
    rows = []
    for col in NUMERIC_COLS:
        if col not in raw.columns:
            continue
        rb = _to_numeric(raw[col])
        ca = cleaned[col] if col in cleaned.columns else pd.Series(dtype=float)
        rows.append({
            "column": col,
            "missing_before": int(rb.isna().sum()),
            "missing_after": int(ca.isna().sum()),
            "min_before": round(float(rb.min()), 2) if rb.notna().any() else None,
            "min_after": round(float(ca.min()), 2) if ca.notna().any() else None,
            "max_before": round(float(rb.max()), 2) if rb.notna().any() else None,
            "max_after": round(float(ca.max()), 2) if ca.notna().any() else None,
            "mean_before": round(float(rb.mean()), 2) if rb.notna().any() else None,
            "mean_after": round(float(ca.mean()), 2) if ca.notna().any() else None,
        })
    for col in CATEGORICAL_COLS:
        if col not in raw.columns:
            continue
        rows.append({
            "column": col,
            "missing_before": int(raw[col].astype(str).str.strip().isin(NULL_TOKENS).sum()),
            "missing_after": int((cleaned[col] == "Unknown").sum()) if col in cleaned else None,
            "min_before": f"{raw[col].nunique()} distinct",
            "min_after": f"{cleaned[col].nunique()} distinct" if col in cleaned else None,
            "max_before": None, "max_after": None, "mean_before": None, "mean_after": None,
        })
    return pd.DataFrame(rows)
