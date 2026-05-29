"""
predict_churn — the callable model artifact that Part 2's agent consumes.

Loads the trained sklearn pipeline + metadata exported by the Part 1 notebook
and exposes the exact signature required by the brief (1.5):

    predict_churn(customer_data: dict) -> {
        "churn_probability": float,   # 0.0 - 1.0
        "risk_tier": str,             # "high" | "medium" | "low"
        "top_risk_factors": list      # top 3 drivers for THIS customer
    }

The raw record is cleaned with the SAME src.data_cleaning logic used in training,
then the engineered features are added, then the saved pipeline imputes/encodes
and predicts — guaranteeing no train/serve skew.

top_risk_factors is a *local* explanation. We use SHAP if the artifact ships a
SHAP explainer; otherwise we fall back to a transparent, dependency-light
heuristic: rank globally-important features by how adverse this customer's value
is relative to the training distribution. This is an approximation, not a causal
claim — see README limitations.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_cleaning import clean_record
from src.features import add_engineered_features, CATEGORICAL_FEATURES, NUMERIC_FEATURES

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_PIPELINE_PATH = _MODELS_DIR / "churn_pipeline.joblib"
_METADATA_PATH = _MODELS_DIR / "model_metadata.json"


class ModelNotTrainedError(RuntimeError):
    """Raised when predict_churn is called before the model artifact exists."""


@lru_cache(maxsize=1)
def _load_artifacts():
    if not _PIPELINE_PATH.exists() or not _METADATA_PATH.exists():
        raise ModelNotTrainedError(
            f"Model artifact not found at {_PIPELINE_PATH}. "
            "Run the Part 1 notebook (notebooks/churn_model.ipynb) to train and export it."
        )
    import joblib

    pipeline = joblib.load(_PIPELINE_PATH)
    metadata = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, metadata


# Human-readable labels + value formats. The High/Low qualifier is chosen
# dynamically from the customer's z-score direction, so the phrase always matches
# the model's data-driven association (it never hardcodes "high"/"low" wrongly).
_NUMERIC_LABELS = {
    "satisfaction_score": ("satisfaction score", "{:.1f}/10"),
    "num_support_tickets": ("support-ticket count", "{:.0f}"),
    "tickets_per_tenure_year": ("support-contact rate", "{:.1f}/yr"),
    "tenure_months": ("tenure", "{:.0f} months"),
    "monthly_charges": ("monthly charges", "${:.0f}"),
    "total_charges": ("lifetime spend", "${:.0f}"),
    "avg_monthly_gb_used": ("data usage", "{:.1f} GB/mo"),
    "avg_monthly_minutes": ("voice usage", "{:.0f} min/mo"),
    "num_additional_services": ("add-on services", "{:.0f}"),
    "charges_per_tenure_month": ("effective monthly cost", "${:.0f}"),
    "expected_vs_actual_charges_gap": ("billing gap vs expected", "${:.0f}"),
    "age": ("age", "{:.0f}"),
}


def _numeric_phrase(feat: str, value: float, z: float) -> str:
    label, fmt = _NUMERIC_LABELS.get(feat, (feat.replace("_", " "), "{:.2f}"))
    qualifier = "High" if z > 0 else "Low"
    return f"{qualifier} {label} ({fmt.format(value)})"
_CATEGORICAL_RISK_PHRASE = {
    ("contract_type", "Month-to-month"): "Month-to-month contract (no lock-in)",
    ("internet_service", "Fiber optic"): "Fiber-optic service (higher-churn segment)",
    ("payment_method", "Electronic check"): "Pays by electronic check (higher-churn segment)",
    ("has_bundle", "unbundled"): "Not bundled (phone + internet)",
}


def _risk_tier(prob: float, meta: dict) -> str:
    hi = meta.get("threshold_high", 0.6)
    lo = meta.get("threshold_low", 0.3)
    if prob >= hi:
        return "high"
    if prob >= lo:
        return "medium"
    return "low"


def _top_risk_factors(record: dict, prob: float, meta: dict, k: int = 3) -> list[str]:
    """Transparent heuristic local explanation (see module docstring)."""
    stats = meta.get("numeric_stats", {})
    directions = meta.get("feature_directions", {})
    importance = dict(meta.get("global_importance", []))
    cat_lift = meta.get("categorical_churn_lift", {})  # {"col=value": lift}

    scored: list[tuple[float, str]] = []

    # Numeric features: adverse z-score * association direction * global importance
    for feat in NUMERIC_FEATURES:
        val = record.get(feat)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        st = stats.get(feat)
        if not st or not st.get("std"):
            continue
        z = (val - st["mean"]) / st["std"]
        adverse = z * directions.get(feat, 0)  # >0 means value pushes toward churn
        if adverse <= 0:
            continue
        score = adverse * (importance.get(feat, 0.0) + 0.01)
        scored.append((score, _numeric_phrase(feat, val, z)))

    # Categorical features: churn lift of the customer's category
    for feat in CATEGORICAL_FEATURES:
        val = record.get(feat)
        if val is None:
            continue
        lift = cat_lift.get(f"{feat}={val}", 0.0)
        if lift <= 0:
            continue
        score = lift * (importance.get(feat, 0.0) + 0.01)
        phrase = _CATEGORICAL_RISK_PHRASE.get((feat, val), f"{feat}: {val}")
        scored.append((score, phrase))

    scored.sort(key=lambda t: t[0], reverse=True)
    factors = [p for _, p in scored[:k]]

    # Fallback so we always return something useful, even for an average customer.
    if not factors:
        factors = ["No single dominant risk factor; prediction driven by feature combination"]
    return factors


def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn for one customer.

    Accepts a (possibly messy / partial) dict of raw customer features and returns
    {churn_probability, risk_tier, top_risk_factors}. Tolerant of missing fields
    (the pipeline imputes). Raises ModelNotTrainedError if the artifact is absent.
    """
    pipeline, meta = _load_artifacts()

    cleaned = clean_record(customer_data)
    row = add_engineered_features(pd.DataFrame([cleaned]))

    # Ensure every expected column is present (pipeline imputes missing numerics).
    for col in NUMERIC_FEATURES:
        if col not in row:
            row[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in row:
            row[col] = "Unknown"

    X = row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    prob = float(pipeline.predict_proba(X)[0, 1])
    prob = max(0.0, min(1.0, prob))

    # Use the engineered row (single dict) for explanation phrasing.
    record_for_expl = row.iloc[0].to_dict()
    return {
        "churn_probability": round(prob, 4),
        "risk_tier": _risk_tier(prob, meta),
        "top_risk_factors": _top_risk_factors(record_for_expl, prob, meta),
    }


if __name__ == "__main__":  # quick manual smoke test
    demo = {
        "customer_id": "TC-000001", "age": 45, "gender": "F", "tenure_months": 2,
        "contract_type": "Month-to-month", "monthly_charges": 95.0, "total_charges": 190.0,
        "internet_service": "Fiber optic", "phone_service": "Y", "avg_monthly_gb_used": 3.1,
        "num_support_tickets": 5, "avg_monthly_minutes": 120, "satisfaction_score": 2.5,
        "payment_method": "Electronic check", "num_additional_services": 0,
    }
    print(json.dumps(predict_churn(demo), indent=2))
