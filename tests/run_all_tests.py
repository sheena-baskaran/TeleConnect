"""
Run all tests in one command — no pytest needed.

Usage:
    python tests/run_all_tests.py

Runs:
  Test 1 — predict_churn  (real model, no API key)
  Test 2 — agent tool chain  (mock mode)
  Test 3 — eval suite smoke  (mock mode)

Exit code 0 = all passed.  Exit code 1 = one or more failed.
"""

import sys
import os
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["FORCE_MOCK_LLM"] = "1"
os.environ["PYTHONPATH"] = str(ROOT)

MODULES = [
    ("Test 1 — predict_churn",    "tests.test_predict"),
    ("Test 2 — agent tool chain", "tests.test_agent"),
    ("Test 3 — eval suite",       "tests.test_eval_suite"),
]


def run_module(label: str, module_name: str) -> bool:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    mod = importlib.import_module(module_name)
    tests = [v for k, v in vars(mod).items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            print(f"\n  [{t.__name__}]")
            t()
            print("    PASS OK")
            passed += 1
        except AssertionError as e:
            print(f"    FAIL FAIL — {e}")
        except Exception as e:
            print(f"    ERROR FAIL — {type(e).__name__}: {e}")
    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    results = []
    for label, module in MODULES:
        results.append(run_module(label, module))

    print(f"\n{'='*55}")
    print(f"  OVERALL: {sum(results)}/{len(results)} test modules passed")
    print(f"{'='*55}")
    sys.exit(0 if all(results) else 1)
