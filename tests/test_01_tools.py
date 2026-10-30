from support_tools import build_registry

from adp.core import ToolCall


def test_valid_call_runs_tool():
    reg = build_registry()
    r = reg.call(ToolCall("lookup_order", {"order_id": "ORD-100001"}))
    assert r.ok and r.output["status"] == "shipped"


def test_invalid_args_become_structured_error():
    reg = build_registry()
    r = reg.call(ToolCall("lookup_order", {"order_id": "123"}))
    assert not r.ok
    assert r.error.startswith("invalid arguments: order_id")


def test_unknown_tool_lists_known_tools():
    reg = build_registry()
    r = reg.call(ToolCall("drop_db", {}))
    assert not r.ok and "lookup_order" in r.error


def test_tool_exception_is_returned_not_raised():
    reg = build_registry()
    r = reg.call(ToolCall("lookup_order", {"order_id": "ORD-999999"}))
    assert not r.ok and r.error.startswith("tool failed: KeyError")


def test_schema_and_side_effects_are_declared():
    reg = build_registry()
    names = [s["name"] for s in reg.schemas()]
    assert names == ["lookup_order", "issue_refund"]
    assert reg.get("issue_refund").side_effect == "write"
    schema = reg.schemas()[1]["input_schema"]
    assert schema["properties"]["amount"]["maximum"] == 5000


def test_duplicate_tool_names_rejected():
    reg = build_registry()
    import pytest
    from pydantic import BaseModel

    from adp.tools import Tool

    class A(BaseModel):
        pass

    with pytest.raises(ValueError):
        reg.register(Tool("lookup_order", "x", A, lambda a: 1))
