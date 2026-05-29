"""
Thin LLM provider abstraction.

The rest of the codebase speaks one normalized format (Anthropic-style content
blocks). Swapping providers means adding one client class here — the orchestrator,
tools, and app do not change.

Two clients ship:
  * AnthropicClient — real Claude tool-calling (used when ANTHROPIC_API_KEY is set).
  * MockClient      — a deterministic, rule-based stand-in so the agent, evals, and
                      demo run end-to-end with NO API key. It performs the correct
                      tool chain (lookup -> predict -> offers -> synthesize), detects
                      escalation triggers, and handles ambiguity/out-of-scope. It is
                      clearly labeled "mock mode" in the UI per the brief's guidance.

Normalized response = a list of blocks, each:
  {"type": "text", "text": str}
  {"type": "tool_use", "id": str, "name": str, "input": dict}
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass


@dataclass
class LLMResponse:
    blocks: list[dict]
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    is_mock: bool = False

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.blocks if b["type"] == "text")

    @property
    def tool_calls(self) -> list[dict]:
        return [b for b in self.blocks if b["type"] == "tool_use"]


def _force_mock() -> bool:
    return os.getenv("FORCE_MOCK_LLM", "").lower() in {"1", "true", "yes"}


def _provider() -> str:
    """Resolve the active provider. FORCE_MOCK_LLM always wins."""
    if _force_mock():
        return "mock"
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


def get_client(role: str = "agent"):
    """
    Factory. role='agent' uses AGENT_MODEL, role='judge' uses JUDGE_MODEL.

    Provider is chosen by LLM_PROVIDER (env), else inferred:
      - "ollama"            -> OllamaClient (local, free, no API key; needs Ollama running)
      - "anthropic"         -> AnthropicClient (needs a funded ANTHROPIC_API_KEY)
      - "mock" / fallback   -> deterministic MockClient
    FORCE_MOCK_LLM=1 overrides everything (handy for offline/CI runs).
    """
    provider = _provider()
    if provider == "ollama":
        return OllamaClient(role=role)
    if provider == "anthropic":
        return AnthropicClient(role=role)
    return MockClient(role=role)


def using_mock() -> bool:
    return _provider() == "mock"


# --------------------------------------------------------------------------- #
# Real Anthropic client                                                       #
# --------------------------------------------------------------------------- #
class AnthropicClient:
    def __init__(self, role: str = "agent"):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        if role == "judge":
            self.model = os.getenv("JUDGE_MODEL", "claude-opus-4-8")
        else:
            self.model = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
        self.is_mock = False

    def respond(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                max_tokens: int = 1500, temperature: float = 0.0) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs = dict(model=self.model, system=system, messages=messages,
                      max_tokens=max_tokens, temperature=temperature)
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000

        blocks = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return LLMResponse(
            blocks=blocks, stop_reason=resp.stop_reason,
            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
            latency_ms=latency, model=self.model, is_mock=False,
        )


# --------------------------------------------------------------------------- #
# Local Ollama client (open-source LLMs, no API key, no extra pip deps)       #
# --------------------------------------------------------------------------- #
# Talks to Ollama's native /api/chat endpoint via the standard library only.
# Ollama is OpenAI-style under the hood and supports tool calling for models
# like qwen2.5 and llama3.1. To use:
#     1) install Ollama (https://ollama.com) and `ollama pull qwen2.5:7b-instruct`
#     2) set  LLM_PROVIDER=ollama  (optionally OLLAMA_MODEL / OLLAMA_BASE_URL)
# The SAME class works against any remote Ollama (e.g. an ngrok tunnel) — just
# point OLLAMA_BASE_URL at the public URL.
class OllamaClient:
    def __init__(self, role: str = "agent"):
        import urllib.request  # stdlib — declared here so import cost is local
        self._urllib = urllib.request
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        default = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        if role == "judge":
            self.model = os.getenv("OLLAMA_JUDGE_MODEL", default)
        else:
            self.model = default
        self.is_mock = False

    # ---- format translation: our Anthropic-style blocks <-> Ollama/OpenAI ---- #
    @staticmethod
    def _to_chat_messages(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if m["role"] == "user":
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                    continue
                # list of blocks: tool_result(s) and/or text
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_result":
                        out.append({"role": "tool", "content": str(b.get("content", ""))})
                    elif b.get("type") == "text":
                        out.append({"role": "user", "content": b["text"]})
            elif m["role"] == "assistant":
                if isinstance(content, str):
                    out.append({"role": "assistant", "content": content})
                    continue
                text = "".join(b["text"] for b in content
                               if isinstance(b, dict) and b.get("type") == "text")
                tool_calls = [{
                    "type": "function",
                    "function": {"name": b["name"], "arguments": b.get("input", {})},
                } for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
                msg = {"role": "assistant", "content": text}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
        return out

    @staticmethod
    def _to_ollama_tools(tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t.get("description", ""),
                              "parameters": t.get("input_schema", {"type": "object"})}}
                for t in tools]

    def respond(self, system: str, messages: list[dict], tools: list[dict] | None = None,
                max_tokens: int = 1500, temperature: float = 0.0) -> LLMResponse:
        t0 = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": self._to_chat_messages(system, messages),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        ot = self._to_ollama_tools(tools)
        if ot:
            payload["tools"] = ot

        req = self._urllib.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._urllib.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = (time.perf_counter() - t0) * 1000

        message = data.get("message", {}) or {}
        blocks: list[dict] = []
        text = message.get("content") or ""
        if text:
            blocks.append({"type": "text", "text": text})

        tool_calls = message.get("tool_calls") or []
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            blocks.append({"type": "tool_use", "id": f"ollama_{i}",
                           "name": fn.get("name", ""), "input": args})

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(
            blocks=blocks, stop_reason=stop_reason,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=latency, model=self.model, is_mock=False,
        )


# --------------------------------------------------------------------------- #
# Deterministic mock client                                                   #
# --------------------------------------------------------------------------- #
_ID_RE = re.compile(r"\bTC[-\s]?(\d{4,6})\b", re.IGNORECASE)
_ESCALATION_KEYWORDS = [
    "lawyer", "legal action", "sue", "lawsuit", "attorney", "court",
    "regulator", "ombudsman", "discrimination", "media", "press", "complaint",
]
_OUT_OF_SCOPE_KEYWORDS = ["weather", "password", "joke", "stock price", "recipe"]

# Match on whole words/phrases, not substrings — otherwise "media" matches
# "immediately", "sue" matches "issue", "court" matches "discount", etc.
def _kw_hit(text: str, keywords: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


class MockClient:
    """Rule-based agent good enough to demo and to baseline the eval harness."""

    def __init__(self, role: str = "agent"):
        self.model = "mock-judge" if role == "judge" else "mock-agent"
        self.role = role
        self.is_mock = True

    # -- judge path: return a deterministic structured rubric ---------------- #
    def respond(self, system: str, messages: list[dict], tools=None,
                max_tokens: int = 1500, temperature: float = 0.0) -> LLMResponse:
        t0 = time.perf_counter()
        if self.role == "judge":
            blocks = [{"type": "text", "text": self._mock_judge(messages)}]
            return LLMResponse(blocks, "end_turn", 0, 0,
                               (time.perf_counter() - t0) * 1000, self.model, True)
        blocks, stop = self._mock_agent(messages)
        return LLMResponse(blocks, stop, 0, 0,
                           (time.perf_counter() - t0) * 1000, self.model, True)

    # ----------------------------------------------------------------------- #
    def _mock_agent(self, messages: list[dict]):
        first_user = self._first_user_text(messages)
        called = self._tools_called(messages)
        last_result = self._last_tool_result(messages)
        lower = first_user.lower()

        # 1) Escalation triggers take precedence.
        if _kw_hit(lower, _ESCALATION_KEYWORDS):
            if "escalate_to_supervisor" not in called:
                return ([{
                    "type": "tool_use", "id": "mock_esc", "name": "escalate_to_supervisor",
                    "input": {
                        "customer_id": self._extract_id(first_user) or "UNKNOWN",
                        "reason": "Customer raised a legal/regulatory threat — outside agent scope.",
                        "context_summary": first_user[:300],
                    },
                }], "tool_use")
            return ([{"type": "text", "text":
                      "I've escalated this to a human supervisor with a summary of the "
                      "situation, since it involves a legal/regulatory matter the agent "
                      "should not handle alone. A supervisor will take over the case."}], "end_turn")

        # 2) Out-of-scope.
        if _kw_hit(lower, _OUT_OF_SCOPE_KEYWORDS):
            return ([{"type": "text", "text":
                      "That request is outside what I can help with — I'm focused on "
                      "customer retention (looking up customers, assessing churn risk, and "
                      "recommending offers). Is there an at-risk customer I can help with?"}],
                    "end_turn")

        cid = self._extract_id(first_user)

        # 3) Ambiguity: no customer ID provided.
        if not cid and "lookup_customer" not in called:
            return ([{"type": "text", "text":
                      "Happy to help with the at-risk customer. Could you give me the "
                      "customer ID (it looks like 'TC-001234')? With that I can pull up "
                      "their profile, run a churn-risk assessment, and suggest retention offers."}],
                    "end_turn")

        # 4) Tool chain: lookup -> predict -> offers -> synthesize.
        if "lookup_customer" not in called:
            return ([{"type": "tool_use", "id": "mock_lk", "name": "lookup_customer",
                      "input": {"customer_id": cid}}], "tool_use")

        # Safety: if the lookup failed (unknown ID), STOP — do not predict on a
        # non-existent customer (that would fabricate a churn score).
        if (isinstance(last_result, dict) and last_result.get("found") is False
                and "predict_churn" not in called):
            return ([{"type": "text", "text":
                      f"I couldn't find a customer with ID {cid or 'that ID'} — the lookup "
                      "returned 'not found'. Could you double-check the ID with the customer? "
                      "It should look like 'TC-001234'. I don't want to assess risk for the "
                      "wrong account."}], "end_turn")

        if "predict_churn" not in called:
            features = {}
            if isinstance(last_result, dict):
                features = last_result.get("_features", {}) or {}
                if "customer_id" not in features and cid:
                    features["customer_id"] = cid
            return ([{"type": "tool_use", "id": "mock_pr", "name": "predict_churn",
                      "input": {"customer_data": features}}], "tool_use")

        if "get_retention_offers" not in called:
            tier, contract = self._risk_from_history(messages)
            return ([{"type": "tool_use", "id": "mock_of", "name": "get_retention_offers",
                      "input": {"risk_tier": tier, "contract_type": contract}}], "tool_use")

        # 5) Final synthesis.
        return ([{"type": "text", "text": self._synthesize(messages, cid)}], "end_turn")

    # ---- helpers ---------------------------------------------------------- #
    @staticmethod
    def _extract_id(text: str):
        m = _ID_RE.search(text or "")
        return f"TC-{int(m.group(1)):06d}" if m else None

    @staticmethod
    def _first_user_text(messages):
        for m in messages:
            if m["role"] == "user":
                c = m["content"]
                if isinstance(c, str):
                    return c
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        return b["text"]
        return ""

    @staticmethod
    def _tools_called(messages):
        names = set()
        for m in messages:
            if m["role"] == "assistant" and isinstance(m["content"], list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        names.add(b["name"])
        return names

    @staticmethod
    def _last_tool_result(messages):
        for m in reversed(messages):
            if m["role"] == "user" and isinstance(m["content"], list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        content = b["content"]
                        if isinstance(content, list):
                            content = "".join(x.get("text", "") for x in content
                                               if isinstance(x, dict))
                        try:
                            return json.loads(content)
                        except Exception:
                            return content
        return None

    def _risk_from_history(self, messages):
        tier, contract = "medium", None
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        content = b["content"]
                        if isinstance(content, list):
                            content = "".join(x.get("text", "") for x in content
                                               if isinstance(x, dict))
                        try:
                            data = json.loads(content)
                        except Exception:
                            continue
                        if isinstance(data, dict):
                            if "risk_tier" in data:
                                tier = data["risk_tier"]
                            if data.get("contract", {}).get("contract_type"):
                                contract = data["contract"]["contract_type"]
        return tier, contract

    def _synthesize(self, messages, cid):
        prob, tier, factors, offers = None, None, [], []
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        content = b["content"]
                        if isinstance(content, list):
                            content = "".join(x.get("text", "") for x in content
                                               if isinstance(x, dict))
                        try:
                            data = json.loads(content)
                        except Exception:
                            continue
                        if isinstance(data, dict):
                            if "churn_probability" in data:
                                prob = data["churn_probability"]
                                tier = data.get("risk_tier")
                                factors = data.get("top_risk_factors", [])
                            if "offers" in data:
                                offers = data["offers"]
        lines = [f"**Retention summary for {cid or 'the customer'}**", ""]
        if prob is not None:
            lines.append(f"- **Churn risk:** {tier.upper()} ({prob:.0%} probability)")
        if factors:
            lines.append(f"- **Top risk factors:** {', '.join(factors)}")
        if offers:
            top = offers[0]
            lines.append(f"- **Recommended offer:** {top['name']} — {top['description']}")
            if len(offers) > 1:
                alt = ", ".join(o["name"] for o in offers[1:3])
                lines.append(f"- **Alternatives:** {alt}")
        lines += ["", ("**Suggested approach:** Lead with empathy, acknowledge the customer's "
                       "concerns, and present the recommended offer framed around the value "
                       "they'll keep. Confirm acceptance and log the outcome.")]
        return "\n".join(lines)

    def _mock_judge(self, messages):
        """Deterministic, plausible rubric so the eval pipeline is exercisable offline."""
        return json.dumps({
            "factual_correctness": {"score": 4,
                "justification": "(mock judge) No API key set; returning a neutral-positive "
                                 "placeholder score. Set ANTHROPIC_API_KEY for real grading."},
            "tool_use_appropriateness": {"score": 4, "justification": "(mock) placeholder"},
            "actionability": {"score": 4, "justification": "(mock) placeholder"},
            "hallucination": {"score": 5, "justification": "(mock) placeholder — assumes grounded"},
            "overall_comment": "Mock judge output. Scores are placeholders, not real evaluations.",
        })
