"""
Automated, deterministic evaluation metrics (Part 2.2b).

These three metrics need NO LLM — they inspect the agent's tool-call trace and
final text directly, so they're cheap, reproducible, and run in CI. The
LLM-as-judge (judge.py) complements them on the subjective dimensions.

Why these three:
  1. tool_selection_accuracy  — the agent is only as good as its orchestration;
     calling the wrong tool (or a forbidden one) is the most consequential failure.
  2. parameter_extraction_accuracy — correct tool, wrong arguments (e.g. a mangled
     customer_id) silently produces wrong answers; this catches it.
  3. response_completeness — did the final, rep-facing answer actually contain the
     elements a rep needs to act? Tool calls mean nothing if the synthesis is empty.

We also REPORT latency and token cost (not pass/fail) for cost-awareness.
"""

from __future__ import annotations

from eval.test_suite import TestCase


def _norm(s: str) -> str:
    return str(s).strip().lower()


def tool_selection_accuracy(case: TestCase, actual_tools: list[str],
                            actual_calls: list) -> dict:
    """
    Score in [0,1] combining:
      - no forbidden tool fired (hard requirement),
      - overlap (Jaccard) between expected and actual tool *sets*,
      - whether the expected tools appear in the correct relative order.
    For pure must_not_call cases (no expected tools), score is 1.0 iff clean.
    """
    expected = [e.name for e in case.expected_tools]
    forbidden_called = [t for t in case.must_not_call if t in actual_tools]
    no_forbidden = len(forbidden_called) == 0

    if not expected:
        return {"score": 1.0 if no_forbidden else 0.0,
                "no_forbidden": no_forbidden,
                "forbidden_called": forbidden_called,
                "ordered_correct": no_forbidden, "set_jaccard": None}

    exp_set, act_set = set(expected), set(actual_tools)
    jaccard = len(exp_set & act_set) / len(exp_set | act_set) if (exp_set | act_set) else 1.0

    # Ordered subsequence check: do the expected tools appear in order within actual?
    ordered_correct = _is_subsequence(expected, actual_tools)

    score = 0.6 * jaccard + 0.4 * (1.0 if ordered_correct else 0.0)
    if not no_forbidden:
        score *= 0.5  # heavy penalty for calling something explicitly forbidden
    return {"score": round(score, 3), "no_forbidden": no_forbidden,
            "forbidden_called": forbidden_called, "ordered_correct": ordered_correct,
            "set_jaccard": round(jaccard, 3)}


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    it = iter(actual)
    return all(tool in it for tool in expected)


def parameter_extraction_accuracy(case: TestCase, actual_calls: list) -> dict | None:
    """
    Fraction of expected key-params that were passed correctly. Returns None when
    no expected call declares params (so it's excluded from the aggregate).
    actual_calls: list of ToolCallRecord (has .name, .input).
    """
    checks, correct, details = 0, 0, []
    for exp in case.expected_tools:
        if not exp.params:
            continue
        match = next((c for c in actual_calls if c.name == exp.name), None)
        for key, want in exp.params.items():
            checks += 1
            got = None
            if match is not None:
                got = match.input.get(key)
                # predict_churn nests features under customer_data; accept either.
                if got is None and isinstance(match.input.get("customer_data"), dict):
                    got = match.input["customer_data"].get(key)
            ok = got is not None and _norm(got) == _norm(want)
            correct += int(ok)
            details.append({"tool": exp.name, "param": key, "want": want,
                            "got": got, "ok": ok})
    if checks == 0:
        return None
    return {"score": round(correct / checks, 3), "checks": checks,
            "correct": correct, "details": details}


def response_completeness(case: TestCase, final_text: str) -> dict | None:
    """
    Fraction of required substrings present in the final response (case-insensitive).
    Returns None when the case declares no required content.
    """
    required = case.response_must_contain
    if not required:
        return None
    text = _norm(final_text)
    hits = [r for r in required if _norm(r) in text]
    return {"score": round(len(hits) / len(required), 3),
            "found": hits, "missing": [r for r in required if _norm(r) not in text]}


def case_passed(tool_sel: dict, param: dict | None, completeness: dict | None) -> bool:
    """A case passes if tool selection is essentially correct AND (where applicable)
    parameters and completeness clear their bars."""
    if tool_sel["score"] < 0.99:
        return False
    if param is not None and param["score"] < 0.99:
        return False
    if completeness is not None and completeness["score"] < 0.5:
        return False
    return True
