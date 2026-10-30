"""Pattern 14: a tool gateway between agents and systems.

Agents never hold credentials or call backends directly. Every
call goes through one gateway that enforces, in order:
per-agent allowlists, argument size limits and per-agent rate
limits, then executes and writes an audit record with secrets
redacted. One choke point means one place to change policy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from adp.core.clock import Clock, SystemClock
from adp.core.llm import ToolCall
from adp.tools.registry import ToolRegistry, ToolResult

SECRET = re.compile(
    r"(?i)(api[_-]?key|token|password|secret)"
    r"(\"?\s*[:=]\s*\"?)([^\s\",}]+)")


def redact_secrets(text: str) -> str:
    return SECRET.sub(lambda m: m.group(1) + m.group(2) + "***",
                      text)


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int,
                 clock: Clock) -> None:
        self.rate = rate_per_sec
        self.capacity = burst
        self.tokens = float(burst)
        self.clock = clock
        self.last = clock.now()

    def take(self) -> bool:
        now = self.clock.now()
        self.tokens = min(self.capacity,
                          self.tokens + (now - self.last)
                          * self.rate)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


@dataclass
class AgentPolicy:
    allowed_tools: frozenset[str]
    rate_per_sec: float = 2.0
    burst: int = 5
    max_arg_bytes: int = 4096


@dataclass
class AuditRecord:
    agent: str
    tool: str
    decision: str  # allowed | denied | rate_limited | too_large
    args: str
    ok: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class ToolGateway:
    def __init__(self, tools: ToolRegistry,
                 policies: dict[str, AgentPolicy],
                 clock: Clock | None = None) -> None:
        self.tools = tools
        self.policies = policies
        self.clock = clock or SystemClock()
        self.audit: list[AuditRecord] = []
        self._buckets: dict[str, TokenBucket] = {}

    def schemas_for(self, agent: str) -> list[dict[str, Any]]:
        allowed = self.policies[agent].allowed_tools
        return [s for s in self.tools.schemas()
                if s["name"] in allowed]

    def call(self, agent: str, tc: ToolCall) -> ToolResult:
        args = json.dumps(tc.args, sort_keys=True)
        safe = redact_secrets(args)
        pol = self.policies.get(agent)

        def deny(decision: str, msg: str) -> ToolResult:
            self.audit.append(
                AuditRecord(agent, tc.name, decision, safe))
            return ToolResult(tc.id, tc.name, False, error=msg)

        if pol is None:
            return deny("denied", f"unknown agent '{agent}'")
        if tc.name not in pol.allowed_tools:
            return deny("denied",
                        f"'{tc.name}' not allowed for {agent}")
        if len(args.encode()) > pol.max_arg_bytes:
            return deny("too_large", "arguments too large")
        bucket = self._buckets.setdefault(
            agent, TokenBucket(pol.rate_per_sec, pol.burst,
                               self.clock))
        if not bucket.take():
            return deny("rate_limited",
                        "rate limit exceeded; retry later")
        res = self.tools.call(tc)
        self.audit.append(AuditRecord(agent, tc.name, "allowed",
                                      safe, res.ok))
        return res
