"""
Run the full evaluation suite and emit a scorecard (Parts 2.2 + 2.4).

Usage:
    python -m eval.run_eval                 # agent + judge (mock if no funded key)
    python -m eval.run_eval --no-judge      # automated metrics only (fast, free)

Outputs:
    eval/results/scorecard.md    — human-readable aggregate scorecard
    eval/results/scorecard.json  — machine-readable (per-case + aggregates) for CI

The scorecard reports: overall pass rate, per-category pass rates, average automated
metric scores, average LLM-judge score by dimension, and cost (latency/tokens).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from eval import metrics as M
from eval.judge import RUBRIC, judge_response
from eval.test_suite import TEST_CASES
from src.agent.orchestrator import run_agent
from src.llm_client import using_mock

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _trace(result) -> list[dict]:
    return [{"order": c.order, "name": c.name, "input": c.input, "output": c.output}
            for c in result.tool_calls]


def run(use_judge: bool = True) -> dict:
    per_case = []
    for case in TEST_CASES:
        result = run_agent(case.user_input)
        tools = result.tool_names_in_order()

        tool_sel = M.tool_selection_accuracy(case, tools, result.tool_calls)
        param = M.parameter_extraction_accuracy(case, result.tool_calls)
        completeness = M.response_completeness(case, result.final_text)
        passed = M.case_passed(tool_sel, param, completeness)

        judged = None
        if use_judge:
            try:
                judged = judge_response(case, result.final_text, _trace(result))
            except Exception as e:
                judged = {"error": str(e), "dimensions": {}, "mean_score": None}

        per_case.append({
            "id": case.id, "category": case.category, "user_input": case.user_input,
            "tools_called": tools, "final_text": result.final_text,
            "tool_selection": tool_sel, "parameter_extraction": param,
            "completeness": completeness, "passed": passed, "judge": judged,
            "latency_ms": round(result.total_latency_ms, 1),
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "is_mock_agent": result.is_mock,
        })

    return _aggregate(per_case, use_judge)


def _aggregate(per_case: list[dict], use_judge: bool) -> dict:
    n = len(per_case)
    overall_pass = mean(1.0 if c["passed"] else 0.0 for c in per_case)

    # Per-category pass rates.
    cat = defaultdict(list)
    for c in per_case:
        cat[c["category"]].append(c["passed"])
    per_category = {k: {"n": len(v), "pass_rate": round(mean(1.0 if x else 0.0 for x in v), 3)}
                    for k, v in cat.items()}

    # Average automated-metric scores (skip None).
    def avg_metric(key):
        vals = [c[key]["score"] for c in per_case if c[key] is not None]
        return round(mean(vals), 3) if vals else None

    automated = {
        "tool_selection_accuracy": avg_metric("tool_selection"),
        "parameter_extraction_accuracy": avg_metric("parameter_extraction"),
        "response_completeness": avg_metric("completeness"),
    }

    # Average judge score by dimension.
    judge_dims = {}
    if use_judge:
        for dim in RUBRIC:
            vals = [c["judge"]["dimensions"].get(dim, {}).get("score")
                    for c in per_case if c["judge"] and c["judge"].get("dimensions")]
            vals = [v for v in vals if isinstance(v, (int, float))]
            judge_dims[dim] = round(mean(vals), 2) if vals else None

    cost = {
        "avg_latency_ms": round(mean(c["latency_ms"] for c in per_case), 1),
        "total_input_tokens": sum(c["input_tokens"] for c in per_case),
        "total_output_tokens": sum(c["output_tokens"] for c in per_case),
    }

    return {
        "n_cases": n,
        "overall_pass_rate": round(overall_pass, 3),
        "per_category": per_category,
        "automated_metrics": automated,
        "judge_dimensions": judge_dims,
        "cost": cost,
        "mock_mode": using_mock(),
        "per_case": per_case,
    }


def write_scorecard(agg: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "scorecard.json").write_text(json.dumps(agg, indent=2, default=str),
                                                encoding="utf-8")

    lines = ["# Retention Agent — Evaluation Scorecard", ""]
    if agg["mock_mode"]:
        lines += ["> **MOCK MODE** — no funded API key detected. The agent and judge are the "
                  "deterministic mock. Automated metrics still grade the (mock) agent's tool "
                  "use; judge scores are placeholders. Set a funded `ANTHROPIC_API_KEY` "
                  "(and unset `FORCE_MOCK_LLM`) for real LLM agent + judge results.", ""]
    lines += [f"- **Cases:** {agg['n_cases']}",
              f"- **Overall pass rate:** {agg['overall_pass_rate']:.0%}",
              f"- **Avg latency/case:** {agg['cost']['avg_latency_ms']} ms",
              f"- **Tokens (in/out):** {agg['cost']['total_input_tokens']} / "
              f"{agg['cost']['total_output_tokens']}", ""]

    lines += ["## Per-category pass rate", "", "| Category | Cases | Pass rate |", "|---|---|---|"]
    for cat, v in sorted(agg["per_category"].items()):
        lines.append(f"| {cat} | {v['n']} | {v['pass_rate']:.0%} |")

    lines += ["", "## Automated metrics (deterministic)", "", "| Metric | Score |", "|---|---|"]
    for k, v in agg["automated_metrics"].items():
        lines.append(f"| {k} | {v if v is not None else 'n/a'} |")

    lines += ["", "## LLM-judge average by dimension (1-5, anchored)", "",
              "| Dimension | Avg score |", "|---|---|"]
    for k, v in agg["judge_dimensions"].items():
        lines.append(f"| {k} | {v if v is not None else 'n/a'} |")

    lines += ["", "## Per-case results", "",
              "| Case | Category | Tools called | Pass | Judge mean |", "|---|---|---|---|---|"]
    for c in agg["per_case"]:
        jm = c["judge"]["mean_score"] if c["judge"] else None
        tools = ", ".join(c["tools_called"]) or "(none)"
        lines.append(f"| {c['id']} | {c['category']} | {tools} | "
                     f"{'PASS' if c['passed'] else 'FAIL'} | {jm if jm is not None else '-'} |")

    (RESULTS_DIR / "scorecard.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", RESULTS_DIR / "scorecard.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-judge")
    args = ap.parse_args()

    agg = run(use_judge=not args.no_judge)
    write_scorecard(agg)
    print(f"\nOverall pass rate: {agg['overall_pass_rate']:.0%}  ({agg['n_cases']} cases)")
    print("Automated:", agg["automated_metrics"])
    if not args.no_judge:
        print("Judge dims:", agg["judge_dimensions"])
    print("Mock mode:", agg["mock_mode"])


if __name__ == "__main__":
    main()
