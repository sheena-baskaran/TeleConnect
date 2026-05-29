"""
Test 1 — predict_churn (the real trained model, no API key needed).

Run:  python -m pytest tests/test_predict.py -v
  or: python tests/test_predict.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_churn


# ---- helpers ----------------------------------------------------------------
def assert_valid_output(result: dict, label: str = ""):
    assert isinstance(result, dict), f"{label}: not a dict"
    assert "churn_probability" in result, f"{label}: missing churn_probability"
    assert "risk_tier" in result, f"{label}: missing risk_tier"
    assert "top_risk_factors" in result, f"{label}: missing top_risk_factors"
    assert 0.0 <= result["churn_probability"] <= 1.0, f"{label}: probability out of range"
    assert result["risk_tier"] in ("high", "medium", "low"), f"{label}: bad risk_tier"
    assert isinstance(result["top_risk_factors"], list), f"{label}: factors not a list"
    assert len(result["top_risk_factors"]) >= 1, f"{label}: empty factors"


# ---- test cases -------------------------------------------------------------
def test_high_risk_customer():
    """Short tenure, low satisfaction, month-to-month -> should be high risk."""
    result = predict_churn({
        "customer_id": "TC-TEST-HIGH",
        "age": 41, "gender": "F",
        "tenure_months": 2,
        "contract_type": "Month-to-month",
        "monthly_charges": 95.0,
        "total_charges": 190.0,
        "internet_service": "Fiber optic",
        "phone_service": "Y",
        "avg_monthly_gb_used": 3.0,
        "num_support_tickets": 6,
        "avg_monthly_minutes": 120,
        "satisfaction_score": 2.4,
        "payment_method": "Electronic check",
        "num_additional_services": 0,
    })
    assert_valid_output(result, "high-risk")
    assert result["churn_probability"] > 0.5, "Expected high churn probability"
    assert result["risk_tier"] in ("high", "medium")
    print(f"  high-risk -> {result['risk_tier']} ({result['churn_probability']:.0%})")
    print(f"  factors: {result['top_risk_factors']}")


def test_low_risk_customer():
    """Long tenure, two-year contract, high satisfaction -> should be low risk."""
    result = predict_churn({
        "customer_id": "TC-TEST-LOW",
        "tenure_months": 60,
        "contract_type": "Two year",
        "monthly_charges": 45.0,
        "total_charges": 2700.0,
        "internet_service": "DSL",
        "phone_service": "Yes",
        "satisfaction_score": 8.7,
        "num_support_tickets": 0,
        "num_additional_services": 4,
    })
    assert_valid_output(result, "low-risk")
    assert result["churn_probability"] < 0.5, "Expected low churn probability"
    print(f"  low-risk -> {result['risk_tier']} ({result['churn_probability']:.0%})")


def test_partial_messy_input():
    """Only a few messy fields — must not crash; pipeline imputes the rest."""
    result = predict_churn({
        "contract_type": "Month-to-month",
        "satisfaction_score": 3.0,
        "phone_service": "N",
        "gender": "m",          # messy encoding
        "age": "999",           # sentinel — should be cleaned to NaN
    })
    assert_valid_output(result, "partial-messy")
    print(f"  partial/messy -> {result['risk_tier']} ({result['churn_probability']:.0%})")


def test_empty_input():
    """Completely empty dict — must not crash."""
    result = predict_churn({})
    assert_valid_output(result, "empty")
    print(f"  empty -> {result['risk_tier']} ({result['churn_probability']:.0%})")


def test_output_schema():
    """Return value must exactly match the Part 1.5 required schema."""
    result = predict_churn({"tenure_months": 12, "contract_type": "One year"})
    assert set(result.keys()) == {"churn_probability", "risk_tier", "top_risk_factors"}
    assert isinstance(result["churn_probability"], float)
    assert isinstance(result["risk_tier"], str)
    assert isinstance(result["top_risk_factors"], list)
    print("  schema OK")


# ---- runner -----------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_high_risk_customer,
        test_low_risk_customer,
        test_partial_messy_input,
        test_empty_input,
        test_output_schema,
    ]
    passed = 0
    for t in tests:
        try:
            print(f"\n[{t.__name__}]")
            t()
            print("  PASS OK")
            passed += 1
        except Exception as e:
            print(f"  FAIL FAIL — {e}")
    print(f"\n{'='*40}")
    print(f"predict_churn: {passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
