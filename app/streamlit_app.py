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

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background and text */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ecc71;
        --warning-color: #e74c3c;
        --info-color: #3498db;
    }

    /* Header styling */
    h1 {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    h2 {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }

    /* Sidebar improvements */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Chat messages */
    .stChatMessage {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f0f2f6;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Success/warning boxes */
    .stSuccess {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
    }

    .stWarning {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
    }

    /* Caption styling */
    .stCaption {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Main content area */
    .main {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📞 Retention Assistant")
    st.caption("AI copilot for TeleConnect retention reps")

    if using_mock():
        st.warning("**Mock mode** — no LLM provider configured. Running the deterministic "
                   "mock agent. Set `ANTHROPIC_API_KEY` in secrets to use Anthropic Claude.")
    else:
        provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        if provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
            st.success(f"**Live mode** — local Ollama · model: `{model}`")
        else:
            st.success(f"**Live mode** — Anthropic Claude (Haiku) · model: "
                       f"`{os.getenv('AGENT_MODEL', 'claude-haiku-4-5-20251001')}`")

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

    # Header with tool chain info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### 🔧 Tool Orchestration")
    with col2:
        st.metric("Calls", len(tool_calls))
    with col3:
        total_latency = sum(tc["latency_ms"] for tc in tool_calls)
        st.metric("Total time", f"{total_latency:.0f}ms")

    # Tool chain visualization
    for idx, tc in enumerate(tool_calls):
        icon = _TOOL_ICON.get(tc["name"], "🔧")
        latency_color = "🟢" if tc["latency_ms"] < 100 else "🟡" if tc["latency_ms"] < 500 else "🔴"

        with st.expander(
            f"{icon} **{tc['order']}. {tc['name']}** {latency_color} {tc['latency_ms']:.0f}ms",
            expanded=(idx == 0)
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📥 Input**")
                st.json(tc["input"], expanded=False)
            with col2:
                st.markdown("**📤 Output**")
                st.json(tc["output"], expanded=False)

            # Add a visual divider
            st.markdown("---")


# --------------------------------------------------------------------------- #
# Conversation state + rendering                                              #
# --------------------------------------------------------------------------- #
# Header with branding
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">📞 TeleConnect</h1>
    <h3 style="color: #7f8c8d; font-weight: 400; margin-top: 0;">AI-Powered Retention Assistant</h3>
    <p style="color: #95a5a6; font-size: 1.1rem; margin-top: 1rem;">
        Real-time churn prediction and personalized retention strategies for your customers
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
💡 **How to use:** Describe what's happening with your customer. The agent will:
1. 🔍 Look up their profile
2. 📈 Predict churn risk
3. 🎁 Suggest retention offers
4. 📝 Log the interaction

*All tool calls are visible below so you can see exactly how the decision was made.*
""")

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
