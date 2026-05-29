"""
Test 2 — retention agent tool-chain (mock mode, no API key needed).

Run:  python -m pytest tests/test_agent.py -v
  or: python tests/test_agent.py

All tests run against the deterministic MockClient (FORCE_MOCK_LLM=1) so they
are fast, free, and reproducible with no network calls.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["FORCE_MOCK_LLM"] = "1"

from src.agent.orchestrator import run_agent


# ---- test cases -------------------------------------------------------------
def test_full_chain_high_risk():
    """High-risk customer -> agent must chain lookup -> predict -> get_offers."""
    r = run_agent("Customer TC-001096 is thinking of leaving, help!")
    tools = r.tool_names_in_order()
    assert "lookup_customer" in tools, "Must call lookup_customer"
    assert "predict_churn" in tools, "Must call predict_churn"
    assert "get_retention_offers" in tools, "Must call get_retention_offers"
    assert tools.index("lookup_customer") < tools.index("predict_churn"), \
        "lookup must come before predict"
    assert r.final_text, "Must produce a final response"
    print(f"  tools: {tools}")
    print(f"  response preview: {r.final_text[:100]}...")


def test_ambiguous_no_id():
    """No customer ID -> agent must ask for it, must NOT call any tools."""
    r = run_agent("I have a high-risk customer on the phone, what should I offer?")
    tools = r.tool_names_in_order()
    assert "lookup_customer" not in tools, "Must NOT look up without an ID"
    assert "predict_churn" not in tools, "Must NOT predict without an ID"
    assert any(w in r.final_text.lower() for w in ["id", "customer id", "tc-"]), \
        "Response must ask for a customer ID"
    print(f"  tools: {tools} (correctly empty)")
    print(f"  response: {r.final_text[:100]}...")


def test_escalation_legal_threat():
    """Legal threat -> agent must escalate, not attempt a retention offer."""
    r = run_agent("Customer TC-003356 says they will get a lawyer and sue us.")
    tools = r.tool_names_in_order()
    assert "escalate_to_supervisor" in tools, "Must escalate on legal threat"
    assert "get_retention_offers" not in tools, "Must NOT offer discounts on legal threat"
    print(f"  tools: {tools}")


def test_out_of_scope():
    """Unrelated request -> agent must decline and call no tools."""
    r = run_agent("What is the weather like in London today?")
    tools = r.tool_names_in_order()
    assert tools == [], f"Must call NO tools for out-of-scope, got: {tools}"
    assert any(w in r.final_text.lower() for w in ["outside", "scope", "retention", "help"]), \
        "Must politely redirect"
    print(f"  tools: {tools} (correctly empty)")


def test_unknown_customer_id():
    """Unknown ID -> agent looks up, gets not-found, must NOT predict churn."""
    r = run_agent("Look up customer TC-999999 and tell me their churn risk.")
    tools = r.tool_names_in_order()
    assert "lookup_customer" in tools, "Must attempt the lookup"
    assert "predict_churn" not in tools, "Must NOT predict for an unknown customer"
    assert "not found" in r.final_text.lower(), "Must tell rep the customer wasn't found"
    print(f"  tools: {tools}")
    print(f"  response: {r.final_text[:100]}...")


def test_final_response_not_empty():
    """Agent must always produce a non-empty final response."""
    r = run_agent("Customer TC-004460 — is she a retention risk?")
    assert r.final_text.strip(), "Final response must not be empty"
    print(f"  response length: {len(r.final_text)} chars")


def test_low_risk_customer():
    """Low-risk customer -> agent chains correctly and recommends margin-preserving action."""
    r = run_agent("Should I give customer TC-004460 a big retention deal?")
    tools = r.tool_names_in_order()
    assert "lookup_customer" in tools
    assert "predict_churn" in tools
    assert "low" in r.final_text.lower(), "Should mention low risk"
    print(f"  tools: {tools}")
    print(f"  response preview: {r.final_text[:100]}...")


# ---- runner -----------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_full_chain_high_risk,
        test_ambiguous_no_id,
        test_escalation_legal_threat,
        test_out_of_scope,
        test_unknown_customer_id,
        test_final_response_not_empty,
        test_low_risk_customer,
    ]
    passed = 0
    for t in tests:
        try:
            print(f"\n[{t.__name__}]")
            t()
            print("  PASS OK")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL FAIL — {e}")
        except Exception as e:
            print(f"  ERROR FAIL — {type(e).__name__}: {e}")
    print(f"\n{'='*40}")
    print(f"agent: {passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
