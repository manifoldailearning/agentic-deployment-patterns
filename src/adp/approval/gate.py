"""Pattern 05: human-in-the-loop approval gate.

The agent may *propose* any action, but write actions that
match a policy rule are parked as approval requests instead of
executing. A named human approves or rejects; only then does
the gate run the tool. Every decision is written to an audit
log, and stale requests expire instead of running late.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from adp.core.clock import Clock, SystemClock
from adp.core.llm import ToolCall
from adp.tools.registry import ToolRegistry, ToolResult

Rule = Callable[[ToolCall], str | None]  # returns a reason


def writes_need_approval(reg: ToolRegistry) -> Rule:
    def rule(tc: ToolCall) -> str | None:
        t = reg.get(tc.name)
        if t is not None and t.side_effect == "write":
            return f"{tc.name} changes customer data"
        return None

    return rule


def amount_over(limit: float, key: str = "amount") -> Rule:
    def rule(tc: ToolCall) -> str | None:
        v = tc.args.get(key)
        if isinstance(v, int | float) and v > limit:
            return f"{key} {v} exceeds {limit}"
        return None

    return rule


@dataclass
class ApprovalRequest:
    id: str
    call: ToolCall
    reasons: list[str]
    created_at: float
    status: str = "pending"  # approved|rejected|expired
    decided_by: str | None = None
    result: ToolResult | None = None


@dataclass
class AuditEvent:
    request_id: str
    event: str
    actor: str
    detail: dict[str, Any] = field(default_factory=dict)


class ApprovalGate:
    def __init__(
        self,
        tools: ToolRegistry,
        rules: list[Rule],
        ttl_seconds: float = 3600,
        clock: Clock | None = None,
    ) -> None:
        self.tools = tools
        self.rules = rules
        self.ttl = ttl_seconds
        self.clock = clock or SystemClock()
        self.requests: dict[str, ApprovalRequest] = {}
        self.audit: list[AuditEvent] = []
        self._ids = itertools.count(1)

    def submit(
        self, tc: ToolCall, actor: str = "agent"
    ) -> ToolResult:
        reasons = [
            r
            for rule in self.rules
            if (r := rule(tc)) is not None
        ]
        if not reasons:
            res = self.tools.call(tc)
            self._log("auto", "executed", actor, ok=res.ok)
            return res
        req = ApprovalRequest(
            f"apr-{next(self._ids)}",
            tc,
            reasons,
            self.clock.now(),
        )
        self.requests[req.id] = req
        self._log(req.id, "requested", actor, reasons=reasons)
        return ToolResult(
            tc.id,
            tc.name,
            False,
            output={"approval_id": req.id},
            error="pending_approval: " + "; ".join(reasons),
        )

    def pending(self) -> list[ApprovalRequest]:
        self._expire()
        return [
            r
            for r in self.requests.values()
            if r.status == "pending"
        ]

    def approve(self, req_id: str, approver: str) -> ToolResult:
        req = self._open(req_id, approver)
        req.status, req.decided_by = "approved", approver
        req.result = self.tools.call(req.call)
        self._log(req_id, "approved", approver, ok=req.result.ok)
        return req.result

    def reject(
        self, req_id: str, approver: str, why: str
    ) -> None:
        req = self._open(req_id, approver)
        req.status, req.decided_by = "rejected", approver
        self._log(req_id, "rejected", approver, why=why)

    def _open(self, req_id: str, actor: str) -> ApprovalRequest:
        self._expire()
        req = self.requests.get(req_id)
        if req is None:
            raise KeyError(req_id)
        if req.status != "pending":
            raise ValueError(f"{req_id} is {req.status}")
        if actor == "agent":
            raise PermissionError("agents cannot self-approve")
        return req

    def _expire(self) -> None:
        now = self.clock.now()
        for r in self.requests.values():
            if (
                r.status == "pending"
                and now - r.created_at > self.ttl
            ):
                r.status = "expired"
                self._log(r.id, "expired", "system")

    def _log(
        self, rid: str, event: str, actor: str, **detail: Any
    ) -> None:
        self.audit.append(AuditEvent(rid, event, actor, detail))
