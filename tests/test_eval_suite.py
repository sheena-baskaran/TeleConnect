"""
Test 3 — evaluation pipeline smoke test (no API key needed).

Verifies that the eval suite runs end-to-end and produces a valid scorecard.

Run:  python -m pytest tests/test_eval_suite.py -v
  or: python tests/test_eval_suite.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["FORCE_MOCK_LLM"] = "1"

from eval.test_suite import TEST_CASES, by_category
from eval.run_eval import run
from eval.metrics import (
    tool_selection_accuracy,
    parameter_extraction_accuracy,
    response_completeness,
)


def test_test_suite_has_required_categories():
    """Brief requires: single-tool, multi-step, ambiguous, out-of-scope, escalation,
    model-disagreement, adversarial. All must be present."""
    required = {
        "single_tool_happy_path",
        "multi_step_chaining",
        "ambiguous_input",
        "out_of_scope",
        "escalation_trigger",
        "model_disagreement",
        "adversarial_edge",
    }
    present = set(by_category().keys())
    missing = required - present
    assert not missing, f"Missing required categories: {missing}"
    assert len(TEST_CASES) >= 12, f"Brief requires ≥12 cases, got {len(TEST_CASES)}"
    print(f"  {len(TEST_CASES)} cases across {len(present)} categories OK")


def test_metrics_run_without_error():
    """Automated metrics must be computable for every test case."""
    from src.agent.orchestrator import run_agent
    case = next(c for c in TEST_CASES if c.category == "multi_step_chaining")
    r = run_agent(case.user_input)
    ts = tool_selection_accuracy(case, r.tool_names_in_order(), r.tool_calls)
    assert "score" in ts and 0.0 <= ts["score"] <= 1.0
    print(f"  tool_selection score: {ts['score']} OK")


def test_scorecard_produced():
    """Running the full eval (no judge) must produce a valid scorecard."""
    agg = run(use_judge=False)
    assert "overall_pass_rate" in agg
    assert 0.0 <= agg["overall_pass_rate"] <= 1.0
    assert agg["n_cases"] == len(TEST_CASES)
    assert "tool_selection_accuracy" in agg["automated_metrics"]
    assert "parameter_extraction_accuracy" in agg["automated_metrics"]
    assert "response_completeness" in agg["automated_metrics"]
    # scorecard file written
    sc = ROOT / "eval" / "results" / "scorecard.md"
    assert sc.exists(), "scorecard.md must be written"
    print(f"  pass rate: {agg['overall_pass_rate']:.0%} | cases: {agg['n_cases']} OK")
    print(f"  automated metrics: {agg['automated_metrics']} OK")


# ---- runner -----------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_test_suite_has_required_categories,
        test_metrics_run_without_error,
        test_scorecard_produced,
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
    print(f"eval suite: {passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
