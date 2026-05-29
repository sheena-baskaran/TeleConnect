"""
Engineered features for the churn model.

Imported by both the training notebook and src/predict.py so the feature set is
identical at train and serve time. Each feature has a one-line rationale; the
notebook narrates the reasoning in full (Part 1.2).

All functions operate on a DataFrame (a single inference record becomes a 1-row
DataFrame), keeping the transform path identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Final feature lists AFTER engineering (consumed by the sklearn ColumnTransformer).
NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_monthly_gb_used",
    "num_support_tickets",
    "avg_monthly_minutes",
    "satisfaction_score",
    "num_additional_services",
    # engineered numerics:
    "charges_per_tenure_month",
    "tickets_per_tenure_year",
    "expected_vs_actual_charges_gap",
]
CATEGORICAL_FEATURES = [
    "gender",
    "contract_type",
    "internet_service",
    "phone_service",
    "payment_method",
    # engineered categorical:
    "tenure_bucket",
    "has_bundle",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append engineered features. Safe on cleaned data containing NaNs."""
    df = df.copy()
    tenure = df.get("tenure_months")
    monthly = df.get("monthly_charges")
    total = df.get("total_charges")
    tickets = df.get("num_support_tickets")

    # Avoid divide-by-zero: a brand-new customer has ~1 month of exposure.
    safe_tenure = tenure.clip(lower=1) if tenure is not None else 1

    # 1) Effective monthly spend over the customer's life. A large gap vs the
    #    nominal monthly_charges hints at plan changes, credits, or billing churn risk.
    df["charges_per_tenure_month"] = (total / safe_tenure) if total is not None else np.nan

    # 2) Support friction *rate* (per year), not raw count — a new customer with
    #    3 tickets is far more at-risk than a 5-year customer with 3 tickets.
    df["tickets_per_tenure_year"] = (
        (tickets / safe_tenure * 12) if tickets is not None else np.nan
    )

    # 3) Reconciliation gap: how far recorded total is from tenure*monthly.
    #    A persistent gap can indicate billing disputes (a churn driver).
    if total is not None and monthly is not None:
        expected = (tenure if tenure is not None else 0) * monthly
        df["expected_vs_actual_charges_gap"] = total - expected
    else:
        df["expected_vs_actual_charges_gap"] = np.nan

    # 4) Tenure lifecycle stage — churn is heavily front-loaded; treat as categorical
    #    so the model can fit a non-monotonic early-life hazard.
    df["tenure_bucket"] = pd.cut(
        tenure if tenure is not None else pd.Series([np.nan] * len(df)),
        bins=[-0.1, 6, 24, 48, np.inf],
        labels=["new_0_6m", "growing_6_24m", "established_24_48m", "loyal_48m_plus"],
    ).astype("object").fillna("unknown")

    # 5) Bundled customers (phone + internet) are structurally stickier.
    has_phone = df.get("phone_service").eq("Yes") if "phone_service" in df else False
    has_net = df.get("internet_service").isin(["DSL", "Fiber optic"]) if "internet_service" in df else False
    df["has_bundle"] = np.where(has_phone & has_net, "bundled", "unbundled")

    return df
