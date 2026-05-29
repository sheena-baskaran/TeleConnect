"""
Mock data backends for the agent's tools.

- Customer profiles are backed by the real (cleaned) dataset, so lookup_customer
  returns genuine rows the churn model was trained to reason about.
- The retention-offer catalog is a sensible design of our own (the brief asks for
  this), structured so offers are filterable by risk tier and contract type.

Everything here is in-memory / file-backed and deterministic — suitable for a
demo and for reproducible evals.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_CLEANED = _DATA_DIR / "cleaned_customers.csv"
_RAW = _DATA_DIR / "test_datafile.csv"


@lru_cache(maxsize=1)
def _customers() -> pd.DataFrame:
    """
    Load customer profiles. Prefer the cleaned dataset produced by the notebook;
    if it doesn't exist yet, clean the raw file on the fly so the agent still works.
    """
    if _CLEANED.exists():
        df = pd.read_csv(_CLEANED)
    else:
        from src.data_cleaning import clean_dataframe

        df, _ = clean_dataframe(pd.read_csv(_RAW))
    df["customer_id"] = df["customer_id"].astype(str)
    return df.set_index("customer_id", drop=False)


def lookup_customer(customer_id: str) -> dict:
    """
    Retrieve a customer profile by ID.

    Returns a profile dict on success, or {"error": ...} if the ID is unknown —
    the agent is expected to handle the error path (e.g. ask the rep to re-check).
    """
    df = _customers()
    cid = str(customer_id).strip().upper()
    if cid not in df.index:
        return {
            "found": False,
            "error": f"No customer found with id '{customer_id}'.",
            "hint": "IDs look like 'TC-001234'. Ask the rep to confirm the ID.",
        }
    row = df.loc[cid]
    if isinstance(row, pd.DataFrame):  # paranoia: duplicate index
        row = row.iloc[0]

    def val(col, default=None):
        v = row.get(col, default)
        if isinstance(v, (np.floating,)):
            v = float(v)
        if isinstance(v, (np.integer,)):
            v = int(v)
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    return {
        "found": True,
        "customer_id": cid,
        "demographics": {"age": val("age"), "gender": val("gender")},
        "contract": {
            "contract_type": val("contract_type"),
            "payment_method": val("payment_method"),
        },
        "tenure_months": val("tenure_months"),
        "charges": {
            "monthly_charges": val("monthly_charges"),
            "total_charges": val("total_charges"),
        },
        "services": {
            "internet_service": val("internet_service"),
            "phone_service": val("phone_service"),
            "num_additional_services": val("num_additional_services"),
            "avg_monthly_gb_used": val("avg_monthly_gb_used"),
            "avg_monthly_minutes": val("avg_monthly_minutes"),
        },
        "satisfaction_score": val("satisfaction_score"),
        # raw feature dict the predict_churn tool can consume directly:
        "_features": {c: val(c) for c in df.columns if c not in ("customer_id", "churned")},
    }


# --------------------------------------------------------------------------- #
# Retention offer catalog (designed by us; the brief invites our own design).  #
# Offers are tagged with eligible risk tiers and contract types so the tool    #
# can filter. Costs are illustrative monthly margins for ROI framing.          #
# --------------------------------------------------------------------------- #
OFFER_CATALOG = [
    {
        "offer_id": "RET-LOYALTY-20",
        "name": "20% Loyalty Discount (12 mo)",
        "description": "20% off the monthly bill for 12 months in exchange for a 1-year commitment.",
        "eligible_risk_tiers": ["high", "medium"],
        "eligible_contracts": ["Month-to-month", "One year"],
        "est_monthly_cost": 14.0,
        "best_for": "Price-sensitive customers on month-to-month plans.",
    },
    {
        "offer_id": "RET-CONTRACT-SWITCH",
        "name": "Free Upgrade + 2-Year Lock-In",
        "description": "Free speed/tier upgrade if the customer moves to a 2-year contract.",
        "eligible_risk_tiers": ["high", "medium"],
        "eligible_contracts": ["Month-to-month", "One year"],
        "est_monthly_cost": 10.0,
        "best_for": "Customers with no lock-in who value service quality.",
    },
    {
        "offer_id": "RET-SERVICE-RECOVERY",
        "name": "Service Recovery Credit + Priority Support",
        "description": "One-time bill credit plus 90 days of priority support routing.",
        "eligible_risk_tiers": ["high"],
        "eligible_contracts": ["Month-to-month", "One year", "Two year"],
        "est_monthly_cost": 25.0,
        "best_for": "Customers with high support-ticket counts or low satisfaction.",
    },
    {
        "offer_id": "RET-BUNDLE-ADDON",
        "name": "Free Add-On Service (6 mo)",
        "description": "Six months of a complimentary add-on (e.g. streaming or security).",
        "eligible_risk_tiers": ["medium", "low"],
        "eligible_contracts": ["Month-to-month", "One year", "Two year"],
        "est_monthly_cost": 6.0,
        "best_for": "Engaged customers with few add-on services; increases stickiness.",
    },
    {
        "offer_id": "RET-DATA-BOOST",
        "name": "Data / Minutes Boost",
        "description": "Doubled data allowance or bonus minutes at no extra cost for 6 months.",
        "eligible_risk_tiers": ["medium", "low"],
        "eligible_contracts": ["Month-to-month", "One year", "Two year"],
        "est_monthly_cost": 8.0,
        "best_for": "Heavy or growing usage customers.",
    },
    {
        "offer_id": "RET-GOODWILL",
        "name": "Goodwill Check-In (no discount)",
        "description": "Proactive check-in call and account review; no financial concession.",
        "eligible_risk_tiers": ["low"],
        "eligible_contracts": ["Month-to-month", "One year", "Two year"],
        "est_monthly_cost": 0.0,
        "best_for": "Low-risk customers — preserve margin, reinforce relationship.",
    },
]


def get_retention_offers(risk_tier: str, contract_type: str | None = None) -> dict:
    """Return offers eligible for the given risk tier (optionally also filtered by contract)."""
    tier = (risk_tier or "").strip().lower()
    offers = [o for o in OFFER_CATALOG if tier in o["eligible_risk_tiers"]]
    if contract_type:
        offers = [o for o in offers if contract_type in o["eligible_contracts"]]
    return {
        "risk_tier": tier,
        "contract_type": contract_type,
        "count": len(offers),
        "offers": offers,
    }
