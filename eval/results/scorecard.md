# Retention Agent — Evaluation Scorecard

> **MOCK MODE** — no funded API key detected. The agent and judge are the deterministic mock. Automated metrics still grade the (mock) agent's tool use; judge scores are placeholders. Set a funded `ANTHROPIC_API_KEY` (and unset `FORCE_MOCK_LLM`) for real LLM agent + judge results.

- **Cases:** 14
- **Overall pass rate:** 71%
- **Avg latency/case:** 211.5 ms
- **Tokens (in/out):** 0 / 0

## Per-category pass rate

| Category | Cases | Pass rate |
|---|---|---|
| adversarial_edge | 3 | 67% |
| ambiguous_input | 2 | 100% |
| escalation_trigger | 2 | 100% |
| model_disagreement | 1 | 0% |
| multi_step_chaining | 3 | 67% |
| out_of_scope | 2 | 100% |
| single_tool_happy_path | 1 | 0% |

## Automated metrics (deterministic)

| Metric | Score |
|---|---|
| tool_selection_accuracy | 0.893 |
| parameter_extraction_accuracy | 0.905 |
| response_completeness | 1.0 |

## LLM-judge average by dimension (1-5, anchored)

| Dimension | Avg score |
|---|---|
| factual_correctness | 4 |
| tool_use_appropriateness | 4 |
| actionability | 4 |
| hallucination | 5 |

## Per-case results

| Case | Category | Tools called | Pass | Judge mean |
|---|---|---|---|---|
| single_lookup | single_tool_happy_path | lookup_customer, predict_churn, get_retention_offers | FAIL | 4.25 |
| chain_high_risk | multi_step_chaining | lookup_customer, predict_churn, get_retention_offers | PASS | 4.25 |
| chain_low_risk | multi_step_chaining | lookup_customer, predict_churn, get_retention_offers | PASS | 4.25 |
| chain_then_log | multi_step_chaining | lookup_customer, predict_churn, get_retention_offers | FAIL | 4.25 |
| ambiguous_no_id | ambiguous_input | (none) | PASS | 4.25 |
| ambiguous_vague | ambiguous_input | (none) | PASS | 4.25 |
| oos_weather | out_of_scope | (none) | PASS | 4.25 |
| oos_password | out_of_scope | (none) | PASS | 4.25 |
| escalate_legal | escalation_trigger | escalate_to_supervisor | PASS | 4.25 |
| escalate_dispute | escalation_trigger | escalate_to_supervisor | PASS | 4.25 |
| disagreement_low_model_bad_profile | model_disagreement | lookup_customer, predict_churn, get_retention_offers | FAIL | 4.25 |
| adv_unknown_id | adversarial_edge | lookup_customer | PASS | 4.25 |
| adv_prompt_injection | adversarial_edge | (none) | PASS | 4.25 |
| adv_conflicting_instruction | adversarial_edge | lookup_customer, predict_churn, get_retention_offers | FAIL | 4.25 |