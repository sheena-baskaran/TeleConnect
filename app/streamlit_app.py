"""
TeleConnect Retention Assistant — live demo (Part 2.3).

A retention rep types a natural-language message; the agent looks up the customer,
runs the churn model, retrieves offers, and returns a recommendation. The UI makes
the orchestration VISIBLE: every tool call is shown in order with its inputs and
outputs — per the brief, the tool chain is part of the deliverable, not hidden.

Deploy: set ANTHROPIC_API_KEY in the host's secrets for a live LLM agent. Without a
key the app runs the deterministic mock agent (clearly labeled) so the demo still works.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

# Make the project root importable (so `src` resolves on Streamlit Cloud).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bridge Streamlit secrets -> env so src.llm_client (which reads os.getenv) picks them up.
for _key in ("ANTHROPIC_API_KEY", "AGENT_MODEL", "JUDGE_MODEL", "FORCE_MOCK_LLM"):
    try:
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
    except Exception:
        pass

from src.agent.orchestrator import RetentionAgent  # noqa: E402
from src.llm_client import using_mock  # noqa: E402

st.set_page_config(page_title="TeleConnect Retention Assistant", page_icon="📞",
                   layout="wide")

# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📞 Retention Assistant")
    st.caption("AI copilot for TeleConnect retention reps")

    if using_mock():
        st.warning("**Mock mode** — no funded API key detected. Running the deterministic "
                   "mock agent. Set `ANTHROPIC_API_KEY` in secrets for the live LLM agent.")
    else:
        st.success(f"**Live mode** — model: `{os.getenv('AGENT_MODEL', 'claude-sonnet-4-6')}`")

    st.markdown("### Try an example")
    examples = {
        "High-risk customer": "Customer TC-001096 called in saying they might cancel. "
                              "What should I do?",
        "Low-risk check": "Should I proactively give customer TC-004460 a retention deal?",
        "No ID (ambiguous)": "I've got a high-risk customer on the phone, what should I offer?",
        "Legal threat (escalate)": "Customer TC-003356 is furious and says they're getting a "
                                   "lawyer to sue us over their bill.",
        "Model disagreement": "The model says TC-002360 is low risk, but they sound really "
                              "unhappy on the call. Should I trust the score?",
        "Unknown ID": "Look up customer TC-999999 and tell me their churn risk.",
    }
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["pending_input"] = text

    st.markdown("---")
    st.markdown("**Tools available to the agent**")
    st.markdown("- `lookup_customer`\n- `predict_churn`\n- `get_retention_offers`\n"
                "- `log_interaction`\n- `escalate_to_supervisor`")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Tool-call rendering                                                         #
# --------------------------------------------------------------------------- #
_TOOL_ICON = {
    "lookup_customer": "🔍", "predict_churn": "📈", "get_retention_offers": "🎁",
    "log_interaction": "📝", "escalate_to_supervisor": "⚠️",
}


def render_tool_calls(tool_calls: list[dict]):
    if not tool_calls:
        st.caption("_No tools were called for this message._")
        return
    st.markdown(f"**🔧 Tool chain ({len(tool_calls)} calls)** — shown in execution order:")
    for tc in tool_calls:
        icon = _TOOL_ICON.get(tc["name"], "🔧")
        with st.expander(f"{icon} **{tc['order']}. {tc['name']}**  ·  {tc['latency_ms']:.0f} ms"):
            st.markdown("**Input**")
            st.json(tc["input"])
            st.markdown("**Output**")
            st.json(tc["output"])


# --------------------------------------------------------------------------- #
# Conversation state + rendering                                              #
# --------------------------------------------------------------------------- #
st.title("TeleConnect Retention Assistant")
st.caption("Type what's happening with the customer. The assistant chains its tools and "
           "shows its work below each answer.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {role, content, tool_calls?, meta?}

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("tool_calls") is not None:
            render_tool_calls(msg["tool_calls"])
            meta = msg.get("meta", {})
            st.caption(f"⏱ {meta.get('latency_ms', 0):.0f} ms · "
                       f"tokens in/out {meta.get('in_tok', 0)}/{meta.get('out_tok', 0)} · "
                       f"model `{meta.get('model', '?')}`")


def handle(user_text: str):
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("Working through the tools…"):
            agent = RetentionAgent()
            result = agent.run(user_text)
            tool_calls = [{"order": c.order, "name": c.name, "input": c.input,
                           "output": c.output, "latency_ms": c.latency_ms}
                          for c in result.tool_calls]
        st.markdown(result.final_text)
        render_tool_calls(tool_calls)
        st.caption(f"⏱ {result.total_latency_ms:.0f} ms · "
                   f"tokens in/out {result.input_tokens}/{result.output_tokens} · "
                   f"model `{result.model}`")
    st.session_state["messages"].append({
        "role": "assistant", "content": result.final_text, "tool_calls": tool_calls,
        "meta": {"latency_ms": result.total_latency_ms, "in_tok": result.input_tokens,
                 "out_tok": result.output_tokens, "model": result.model},
    })


# Handle example-button input or chat input.
pending = st.session_state.pop("pending_input", None)
typed = st.chat_input("e.g. 'Customer TC-001096 is threatening to cancel — what should I do?'")
if pending:
    handle(pending)
elif typed:
    handle(typed)
