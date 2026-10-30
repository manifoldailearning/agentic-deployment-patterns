"""Pattern 05: the agent proposes, a human approves."""

from support_tools import build_registry

from adp.approval import ApprovalGate, writes_need_approval
from adp.core import ToolCall

reg = build_registry()
gate = ApprovalGate(reg, [writes_need_approval(reg)])

# Reads run immediately.
print(
    gate.submit(
        ToolCall("lookup_order", {"order_id": "ORD-100002"})
    ).to_message()
)

# Writes are parked until a human decides.
res = gate.submit(
    ToolCall(
        "issue_refund",
        {
            "order_id": "ORD-100002",
            "amount": 310.0,
            "reason": "Damaged on arrival",
        },
    )
)
print(res.to_message())

req_id = gate.pending()[0].id
print(gate.approve(req_id, approver="lead.maya").to_message())
for e in gate.audit:
    print(e.request_id, e.event, e.actor)
