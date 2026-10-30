import pytest
from support_tools import build_registry

from adp.approval import ApprovalGate, amount_over, writes_need_approval
from adp.core import FakeClock, ToolCall

REFUND = ToolCall("issue_refund", {"order_id": "ORD-100002",
                                   "amount": 310.0,
                                   "reason": "Damaged item"})


def gate(**kw):
    reg = build_registry()
    return ApprovalGate(reg, [writes_need_approval(reg)], **kw)


def test_reads_execute_without_approval():
    g = gate()
    r = g.submit(ToolCall("lookup_order", {"order_id": "ORD-100001"}))
    assert r.ok and g.pending() == []


def test_writes_are_parked_not_executed():
    g = gate()
    r = g.submit(REFUND)
    assert not r.ok and r.error.startswith("pending_approval")
    assert len(g.pending()) == 1


def test_approval_executes_and_audits():
    g = gate()
    g.submit(REFUND)
    rid = g.pending()[0].id
    res = g.approve(rid, "lead.maya")
    assert res.ok and res.output["refunded"] == 310.0
    assert [e.event for e in g.audit] == ["requested", "approved"]
    assert g.audit[-1].actor == "lead.maya"


def test_reject_and_double_decision():
    g = gate()
    g.submit(REFUND)
    rid = g.pending()[0].id
    g.reject(rid, "lead.maya", "duplicate refund")
    with pytest.raises(ValueError):
        g.approve(rid, "lead.maya")


def test_agent_cannot_self_approve():
    g = gate()
    g.submit(REFUND)
    with pytest.raises(PermissionError):
        g.approve(g.pending()[0].id, "agent")


def test_requests_expire():
    clock = FakeClock()
    g = gate(ttl_seconds=60, clock=clock)
    g.submit(REFUND)
    rid = g.pending()[0].id
    clock.advance(61)
    assert g.pending() == []
    with pytest.raises(ValueError, match="expired"):
        g.approve(rid, "lead.maya")


def test_amount_rule_combines_with_other_rules():
    reg = build_registry()
    g = ApprovalGate(reg, [amount_over(100)])
    small = ToolCall("issue_refund", {"order_id": "ORD-100001",
                                      "amount": 20.0,
                                      "reason": "Late delivery"})
    assert g.submit(small).ok
    assert not g.submit(REFUND).ok
