"""Pattern 01: validate every model-proposed tool call."""

from support_tools import build_registry

from adp.core import ToolCall

reg = build_registry()

calls = [
    ToolCall("lookup_order", {"order_id": "ORD-100001"}),
    ToolCall("lookup_order", {"order_id": "12345"}),
    ToolCall("delete_order", {"order_id": "ORD-100001"}),
    ToolCall("lookup_order", {"order_id": "ORD-999999"}),
]
for tc in calls:
    print(tc.name, "->", reg.call(tc).to_message())
