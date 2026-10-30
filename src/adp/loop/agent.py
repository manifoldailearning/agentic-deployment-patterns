"""Pattern 02: the bounded agent loop.

The model decides the next step; the application owns the
loop. Every run is capped by steps, wall-clock time, tokens
and cost, and stops early on repeated identical tool calls or
too many consecutive tool errors. A run always ends with an
explicit stop reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from adp.core.clock import Clock, SystemClock
from adp.core.llm import LLM, Message, ToolCall
from adp.tools.registry import ToolRegistry, ToolResult


@dataclass(frozen=True)
class Budget:
    max_steps: int = 8
    max_seconds: float = 60.0
    max_tokens: int = 20_000
    max_cost_usd: float = 0.50
    max_tool_errors: int = 3
    max_repeats: int = 2


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens."""

    input_per_m: float = 3.0
    output_per_m: float = 15.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in * self.input_per_m
            + tokens_out * self.output_per_m
        ) / 1_000_000


@dataclass
class LoopResult:
    status: str  # "completed" | "stopped"
    stop_reason: str
    answer: str | None
    steps: int
    tokens: int
    cost_usd: float
    tool_results: list[ToolResult] = field(default_factory=list)


class AgentLoop:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        budget: Budget | None = None,
        pricing: Pricing | None = None,
        clock: Clock | None = None,
        system: str = "You are a careful support agent.",
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.budget = budget or Budget()
        self.pricing = pricing or Pricing()
        self.clock = clock or SystemClock()
        self.system = system

    def run(self, task: str) -> LoopResult:
        b = self.budget
        msgs = [
            Message("system", self.system),
            Message("user", task),
        ]
        start = self.clock.now()
        tokens = 0
        cost = 0.0
        errors = 0
        seen: dict[str, int] = {}
        results: list[ToolResult] = []

        def stop(reason: str, steps: int) -> LoopResult:
            return LoopResult(
                "stopped",
                reason,
                None,
                steps,
                tokens,
                cost,
                results,
            )

        for step in range(1, b.max_steps + 1):
            if self.clock.now() - start > b.max_seconds:
                return stop("time_budget", step - 1)
            resp = self.llm.complete(msgs, self.tools.schemas())
            u = resp.usage
            tokens += u.total
            cost += self.pricing.cost(
                u.input_tokens, u.output_tokens
            )
            if tokens > b.max_tokens:
                return stop("token_budget", step)
            if cost > b.max_cost_usd:
                return stop("cost_budget", step)
            if not resp.tool_calls:
                return LoopResult(
                    "completed",
                    "answer",
                    resp.text,
                    step,
                    tokens,
                    cost,
                    results,
                )
            msgs.append(
                Message(
                    "assistant",
                    resp.text,
                    tool_calls=resp.tool_calls,
                )
            )
            for tc in resp.tool_calls:
                key = _fingerprint(tc)
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > b.max_repeats:
                    return stop("repeated_call", step)
                res = self.tools.call(tc)
                results.append(res)
                errors = 0 if res.ok else errors + 1
                msgs.append(
                    Message(
                        "tool",
                        res.to_message(),
                        tool_call_id=tc.id,
                    )
                )
                if errors >= b.max_tool_errors:
                    return stop("tool_errors", step)
        return stop("step_budget", b.max_steps)


def _fingerprint(tc: ToolCall) -> str:
    return tc.name + json.dumps(tc.args, sort_keys=True)
