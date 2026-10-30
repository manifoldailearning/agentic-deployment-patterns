"""Pattern 14: one gateway enforces policy for every agent."""

from support_tools import build_registry

from adp.core import FakeClock, ToolCall
from adp.gateway import AgentPolicy, ToolGateway

gw = ToolGateway(
    build_registry(),
    {
        "triage-agent": AgentPolicy(
            frozenset({"lookup_order"}), rate_per_sec=1, burst=2
        ),
        "refund-agent": AgentPolicy(
            frozenset({"lookup_order", "issue_refund"})
        ),
    },
    clock=FakeClock(),
)
look = ToolCall("lookup_order", {"order_id": "ORD-100001"})
refund = ToolCall(
    "issue_refund",
    {
        "order_id": "ORD-100001",
        "amount": 10.0,
        "reason": "Late delivery",
        "api_key": "sk-live-123",
    },
)

print([s["name"] for s in gw.schemas_for("triage-agent")])
for agent, tc in [
    ("triage-agent", refund),
    ("triage-agent", look),
    ("triage-agent", look),
    ("triage-agent", look),
]:
    print(agent, tc.name, "->", gw.call(agent, tc).to_message())
for rec in gw.audit:
    print(rec.decision, rec.tool, rec.args[:60])
