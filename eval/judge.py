"""
LLM-as-judge evaluator (Part 2.2c).

Uses a SEPARATE model (JUDGE_MODEL, deliberately different from the agent model)
to score the agent's output on four dimensions, each with explicit 1/3/5 ANCHORS
so scores are calibrated rather than an unanchored "rate 1-10". The judge returns
structured rubric scores + justifications, not a binary pass/fail.

Judge reliability (the meta-question, addressed honestly):
  * Positivity bias — LLM judges tend to be lenient. We counter this by (a) anchoring
    every score with a concrete description of what a 1/3/5 looks like, and (b)
    explicitly defining hallucination so "5 = no hallucination" is the *default-safe*
    score and low scores require evidence.
  * Prompt sensitivity — scores can shift with wording, so the rubric is fixed and
    versioned; we keep temperature=0 for determinism.
  * Independence — judging with a different model than the agent avoids a model
    grading its own style.
  * Calibration — the deterministic metrics (metrics.py) act as a cross-check: where
    tool-selection accuracy is 0 but the judge gives a high tool-use score, that's a
    calibration red flag worth a human spot-check. We recommend periodically labeling
    a sample by hand and measuring judge-vs-human agreement (e.g. Cohen's kappa).
"""

from __future__ import annotations

import json
import re

from src.llm_client import get_client

RUBRIC_VERSION = "1.0"

# Each dimension defines what a 1, 3, and 5 look like. The judge must anchor to these.
RUBRIC = {
    "factual_correctness": {
        "question": "Are the claims in the response consistent with the tool outputs "
                    "(customer profile, churn score, offers)?",
        "anchors": {
            1: "Contains claims that contradict the tool data (wrong risk tier, wrong "
               "numbers, wrong customer).",
            3: "Mostly accurate but with a minor mismatch or an unsupported aside.",
            5: "Every factual claim is directly supported by the tool outputs.",
        },
    },
    "tool_use_appropriateness": {
        "question": "Did the agent use the right tools, in a sensible order, and avoid "
                    "tools it shouldn't have used?",
        "anchors": {
            1: "Wrong tools, skipped a required step, or used a tool it should not have "
               "(e.g. predicted on an unknown customer).",
            3: "Right tools but a questionable order, a redundant call, or a missed "
               "secondary step (e.g. didn't log).",
            5: "Exactly the tools the situation called for, in the right order, with no "
               "inappropriate calls.",
        },
    },
    "actionability": {
        "question": "Could a busy retention rep act on this response immediately?",
        "anchors": {
            1: "A raw data dump or vague non-answer; the rep wouldn't know what to do.",
            3: "Useful but generic; missing a specific offer or a clear next step.",
            5: "A specific recommendation (offer + why) plus a concrete talking point the "
               "rep can use on the call.",
        },
    },
    "hallucination": {
        "question": "Did the agent invent facts not present in the tool outputs? "
                    "(5 = no hallucination; lower = more invention.)",
        "anchors": {
            1: "Fabricates customer details, offers, or numbers that no tool returned.",
            3: "One small unsupported detail or over-confident inference.",
            5: "No fabrication; everything traces to a tool result or is appropriately hedged.",
        },
    },
}

_JUDGE_SYSTEM = (
    "You are a rigorous QA evaluator for a customer-retention AI agent. You grade the "
    "agent's response against a fixed rubric. Be calibrated and slightly skeptical: do "
    "not award a 5 unless the response clearly meets the 5-anchor. Reward grounded, "
    "actionable answers; penalize fabrication and wrong tool use. Respond with ONLY a "
    "JSON object, no prose around it."
)


def _build_prompt(case, agent_final_text: str, tool_trace: list[dict]) -> str:
    rubric_text = []
    for dim, spec in RUBRIC.items():
        rubric_text.append(
            f"- {dim}: {spec['question']}\n"
            f"    1 = {spec['anchors'][1]}\n"
            f"    3 = {spec['anchors'][3]}\n"
            f"    5 = {spec['anchors'][5]}"
        )
    rubric_block = "\n".join(rubric_text)

    trace_block = json.dumps(tool_trace, indent=2, default=str)[:4000]
    criteria = "\n".join(f"- {c}" for c in case.quality_criteria) or "- (none specified)"

    return f"""\
Evaluate the agent's handling of this retention scenario.

## Rep's message
{case.user_input}

## Scenario category
{case.category}

## What a good response should do
{criteria}

## Tools the agent actually called (name, input, output)
{trace_block}

## Agent's final response to the rep
\"\"\"{agent_final_text}\"\"\"

## Scoring rubric (anchor each score to these definitions; scores are 1-5 integers)
{rubric_block}

Return ONLY this JSON shape:
{{
  "factual_correctness": {{"score": <1-5>, "justification": "<one sentence>"}},
  "tool_use_appropriateness": {{"score": <1-5>, "justification": "<one sentence>"}},
  "actionability": {{"score": <1-5>, "justification": "<one sentence>"}},
  "hallucination": {{"score": <1-5>, "justification": "<one sentence>"}},
  "overall_comment": "<one or two sentences>"
}}"""


def _extract_json(text: str) -> dict:
    """Robustly pull the first JSON object out of the model's reply."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError(f"Judge did not return parseable JSON: {text[:200]!r}")


def judge_response(case, agent_final_text: str, tool_trace: list[dict],
                   client=None) -> dict:
    """
    Score one agent response. Returns:
      {dimensions: {dim: {score, justification}}, overall_comment, mean_score,
       rubric_version, judge_model, is_mock}
    """
    client = client or get_client(role="judge")
    prompt = _build_prompt(case, agent_final_text, tool_trace)
    resp = client.respond(_JUDGE_SYSTEM, [{"role": "user", "content": prompt}],
                          tools=None, max_tokens=700, temperature=0.0)
    parsed = _extract_json(resp.text)

    dims = {}
    scores = []
    for dim in RUBRIC:
        entry = parsed.get(dim, {})
        score = entry.get("score")
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = None
        dims[dim] = {"score": score, "justification": entry.get("justification", "")}
        if score is not None:
            scores.append(score)

    return {
        "dimensions": dims,
        "overall_comment": parsed.get("overall_comment", ""),
        "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
        "rubric_version": RUBRIC_VERSION,
        "judge_model": getattr(client, "model", "unknown"),
        "is_mock": getattr(client, "is_mock", False),
        "judge_latency_ms": round(resp.latency_ms, 1),
    }
