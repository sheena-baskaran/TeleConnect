"""
Agent orchestration: a provider-agnostic tool-calling loop.

The loop is generic — it iterates the TOOL_REGISTRY and runs until the model stops
requesting tools. Adding a sixth tool requires no change here. It records a full
trace of every tool call (name, order, input, output, latency) so the UI and the
eval harness can inspect the agent's reasoning — the brief requires the demo to
make tool calls visible.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import execute_tool, get_tool_schemas
from src.llm_client import get_client


@dataclass
class ToolCallRecord:
    order: int
    name: str
    input: dict
    output: dict
    latency_ms: float


@dataclass
class AgentResult:
    final_text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0.0
    model: str = ""
    is_mock: bool = False
    stopped_reason: str = ""

    def tool_names_in_order(self) -> list[str]:
        return [t.name for t in self.tool_calls]


class RetentionAgent:
    def __init__(self, client=None, system_prompt: str = SYSTEM_PROMPT, max_turns: int = 8):
        self.client = client or get_client(role="agent")
        self.system_prompt = system_prompt
        self.tools = get_tool_schemas()
        self.max_turns = max_turns

    def run(self, user_message: str, history: list[dict] | None = None) -> AgentResult:
        """Run one rep request to completion, returning the final answer + full trace."""
        messages: list[dict] = list(history or [])
        messages.append({"role": "user", "content": user_message})

        result = AgentResult(final_text="", model=getattr(self.client, "model", ""),
                             is_mock=getattr(self.client, "is_mock", False))
        order = 0
        t_start = time.perf_counter()

        for _ in range(self.max_turns):
            resp = self.client.respond(self.system_prompt, messages, tools=self.tools)
            result.input_tokens += resp.input_tokens
            result.output_tokens += resp.output_tokens

            # Persist the assistant turn verbatim (text + any tool_use blocks).
            messages.append({"role": "assistant", "content": resp.blocks})

            tool_calls = resp.tool_calls
            if not tool_calls:
                result.final_text = resp.text
                result.stopped_reason = resp.stop_reason
                break

            # Execute every requested tool and feed results back.
            tool_results = []
            for call in tool_calls:
                order += 1
                t0 = time.perf_counter()
                output = execute_tool(call["name"], call.get("input", {}))
                latency = (time.perf_counter() - t0) * 1000
                result.tool_calls.append(
                    ToolCallRecord(order, call["name"], call.get("input", {}), output, latency)
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": json.dumps(output, default=str),
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            result.final_text = (result.final_text
                                 or "I wasn't able to complete this within the step limit. "
                                    "Please try rephrasing or provide the customer ID.")
            result.stopped_reason = "max_turns"

        result.messages = messages
        result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        return result


def run_agent(user_message: str, history: list[dict] | None = None) -> AgentResult:
    """Convenience wrapper used by the app and eval harness."""
    return RetentionAgent().run(user_message, history)
