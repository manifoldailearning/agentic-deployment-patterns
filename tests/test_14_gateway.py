from support_tools import build_registry

from adp.core import FakeClock, ToolCall
from adp.gateway import AgentPolicy, ToolGateway, redact_secrets

LOOK = ToolCall("lookup_order", {"order_id": "ORD-100001"})


def gw(clock=None, **pol):
    return ToolGateway(build_registry(), {
        "a": AgentPolicy(frozenset({"lookup_order"}), **pol)},
        clock=clock or FakeClock())


def test_allowlist_and_schema_filtering():
    g = gw()
    assert [s["name"] for s in g.schemas_for("a")] == [
        "lookup_order"]
    r = g.call("a", ToolCall("issue_refund", {}))
    assert not r.ok and "not allowed" in r.error
    assert g.audit[-1].decision == "denied"


def test_unknown_agent_denied():
    r = gw().call("ghost", LOOK)
    assert not r.ok and "unknown agent" in r.error


def test_rate_limit_refills_over_time():
    clock = FakeClock()
    g = gw(clock, rate_per_sec=1, burst=2)
    assert g.call("a", LOOK).ok and g.call("a", LOOK).ok
    r = g.call("a", LOOK)
    assert not r.ok and g.audit[-1].decision == "rate_limited"
    clock.advance(1)
    assert g.call("a", LOOK).ok


def test_argument_size_limit():
    big = ToolCall("lookup_order", {"order_id": "x" * 500})
    g = gw(max_arg_bytes=100)
    r = g.call("a", big)
    assert not r.ok and g.audit[-1].decision == "too_large"


def test_secrets_never_reach_audit_log():
    s = redact_secrets('{"api_key": "sk-1", "password": "p@ss"}')
    assert "sk-1" not in s and "p@ss" not in s
    g = gw()
    g.call("a", ToolCall("lookup_order",
                         {"order_id": "ORD-100001",
                          "token": "abc123"}))
    assert "abc123" not in g.audit[-1].args


def test_allowed_calls_are_audited_with_outcome():
    g = gw()
    g.call("a", LOOK)
    rec = g.audit[-1]
    assert (rec.decision, rec.ok) == ("allowed", True)
