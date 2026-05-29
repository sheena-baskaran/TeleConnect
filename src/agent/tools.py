"""
Agent tools: schemas + implementations + a single TOOL_REGISTRY.

Design goal (explicit eval criterion): adding a SIXTH tool requires only adding
one entry to TOOL_REGISTRY below — one schema dict + one implementation function.
The orchestrator iterates the registry; nothing else changes.

Each tool returns a JSON-serializable dict. Tool descriptions are written for the
LLM: they state when to use the tool, what the inputs mean, and what comes back,
because description quality directly drives tool-selection accuracy.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.agent import mock_db

_INTERACTION_LOG = Path(__file__).resolve().parents[2] / "data" / "interaction_log.jsonl"


# --------------------------------------------------------------------------- #
# Implementations                                                             #
# --------------------------------------------------------------------------- #
def _impl_predict_churn(customer_data: dict | None = None, **kwargs) -> dict:
    """Run the trained Part 1 model. Falls back to a clearly-labeled mock if the
    artifact hasn't been trained yet (per the brief: 'mock it — but say so')."""
    data = customer_data or kwargs or {}
    try:
        from src.predict import predict_churn, ModelNotTrainedError

        try:
            return predict_churn(data)
        except ModelNotTrainedError as e:
            return {
                "churn_probability": 0.5,
                "risk_tier": "medium",
                "top_risk_factors": ["MODEL NOT TRAINED — mock output"],
                "_warning": f"Mock prediction: {e}",
            }
    except Exception as e:  # never crash the agent loop on a tool error
        return {"error": f"predict_churn failed: {e}"}


def _impl_lookup_customer(customer_id: str = "", **kwargs) -> dict:
    return mock_db.lookup_customer(customer_id)


def _impl_get_retention_offers(risk_tier: str = "", contract_type: str | None = None,
                               **kwargs) -> dict:
    return mock_db.get_retention_offers(risk_tier, contract_type)


def _impl_log_interaction(customer_id: str = "", rep_id: str = "unknown",
                          churn_risk_tier: str | None = None,
                          churn_probability: float | None = None,
                          offers_presented: list | None = None,
                          outcome: str = "pending", notes: str = "", **kwargs) -> dict:
    """Append a production-realistic interaction record (JSONL)."""
    record = {
        "interaction_id": f"INT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "customer_id": customer_id,
        "rep_id": rep_id,
        "churn_risk_tier": churn_risk_tier,
        "churn_probability": churn_probability,
        "offers_presented": offers_presented or [],
        "outcome": outcome,  # accepted | declined | escalated | pending | callback
        "notes": notes,
        "schema_version": "1.0",
    }
    try:
        _INTERACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _INTERACTION_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return {"logged": True, "interaction_id": record["interaction_id"], "record": record}
    except Exception as e:
        return {"logged": False, "error": str(e), "record": record}


def _impl_escalate_to_supervisor(customer_id: str = "", reason: str = "",
                                 context_summary: str = "", urgency: str = "normal",
                                 **kwargs) -> dict:
    """Create a supervisor escalation ticket with a context handoff summary."""
    ticket = {
        "escalation_id": f"ESC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "customer_id": customer_id,
        "reason": reason,
        "context_summary": context_summary,
        "urgency": urgency,  # normal | high | critical
        "status": "queued_for_supervisor",
    }
    return {"escalated": True, "ticket": ticket}


# --------------------------------------------------------------------------- #
# Schemas (Anthropic tool format) + registry                                  #
# --------------------------------------------------------------------------- #
TOOL_REGISTRY: dict[str, dict] = {
    "lookup_customer": {
        "impl": _impl_lookup_customer,
        "schema": {
            "name": "lookup_customer",
            "description": (
                "Retrieve a customer's profile by their account ID. Use this FIRST whenever "
                "you have a customer ID, before assessing churn risk or recommending offers. "
                "Returns demographics, contract type, tenure, monthly/total charges, services, "
                "and satisfaction score. If the ID is unknown it returns an error you should "
                "relay to the rep so they can re-check the ID."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Account ID in the form 'TC-001234'.",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    "predict_churn": {
        "impl": _impl_predict_churn,
        "schema": {
            "name": "predict_churn",
            "description": (
                "Run the churn-prediction model on a customer's features. Use this AFTER "
                "lookup_customer — pass the profile's feature fields as 'customer_data'. "
                "Returns churn_probability (0-1), risk_tier ('high'/'medium'/'low'), and "
                "top_risk_factors (the 3 features driving this customer's risk). Use the "
                "risk_tier to choose retention offers."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_data": {
                        "type": "object",
                        "description": (
                            "Customer feature dictionary (e.g. tenure_months, monthly_charges, "
                            "contract_type, satisfaction_score, ...). Pass the '_features' object "
                            "returned by lookup_customer, or individual known fields."
                        ),
                    }
                },
                "required": ["customer_data"],
            },
        },
    },
    "get_retention_offers": {
        "impl": _impl_get_retention_offers,
        "schema": {
            "name": "get_retention_offers",
            "description": (
                "Return retention offers the customer is eligible for, filtered by their churn "
                "risk_tier and (optionally) contract_type. Use this AFTER predict_churn so you "
                "can match offer aggressiveness to risk. Higher-risk tiers unlock stronger "
                "concessions; low risk returns margin-preserving options."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "risk_tier": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "The churn risk tier from predict_churn.",
                    },
                    "contract_type": {
                        "type": "string",
                        "description": "Optional: 'Month-to-month', 'One year', or 'Two year'.",
                    },
                },
                "required": ["risk_tier"],
            },
        },
    },
    "log_interaction": {
        "impl": _impl_log_interaction,
        "schema": {
            "name": "log_interaction",
            "description": (
                "Record the outcome of a retention conversation for analytics and audit. "
                "Call this once you've made a recommendation and (where possible) know the "
                "outcome. Captures which offers were presented and the result."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "rep_id": {"type": "string", "description": "ID of the retention rep, if known."},
                    "churn_risk_tier": {"type": "string"},
                    "churn_probability": {"type": "number"},
                    "offers_presented": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Offer IDs or names presented to the customer.",
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["accepted", "declined", "escalated", "pending", "callback"],
                    },
                    "notes": {"type": "string"},
                },
                "required": ["customer_id", "outcome"],
            },
        },
    },
    "escalate_to_supervisor": {
        "impl": _impl_escalate_to_supervisor,
        "schema": {
            "name": "escalate_to_supervisor",
            "description": (
                "Transfer the case to a human supervisor with a context summary. Use this when "
                "the situation is OUTSIDE the agent's remit: the customer threatens legal or "
                "regulatory action, raises a complex billing dispute you cannot resolve, is "
                "highly distressed, or asks for something none of the other tools can handle. "
                "Do NOT use it for routine retention — handle those yourself."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Why this needs a human."},
                    "context_summary": {
                        "type": "string",
                        "description": "Concise handoff summary so the supervisor has context.",
                    },
                    "urgency": {"type": "string", "enum": ["normal", "high", "critical"]},
                },
                "required": ["reason", "context_summary"],
            },
        },
    },
}


def get_tool_schemas() -> list[dict]:
    """Anthropic-format tool list for the messages API."""
    return [t["schema"] for t in TOOL_REGISTRY.values()]


def execute_tool(name: str, tool_input: dict) -> dict:
    """Dispatch a tool call by name. Unknown tools return an error (never raise)."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return entry["impl"](**(tool_input or {}))
    except TypeError as e:
        # Bad / missing params — surface to the agent so it can retry.
        return {"error": f"Invalid arguments for '{name}': {e}"}
    except Exception as e:
        return {"error": f"Tool '{name}' raised: {e}"}
