"""Shared example domain: a small customer-support backend."""

from __future__ import annotations

from pydantic import BaseModel, Field

from adp.tools import ToolRegistry

ORDERS = {
    "ORD-100001": {"status": "shipped", "total": 42.50},
    "ORD-100002": {"status": "processing", "total": 310.00},
}


class OrderLookup(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d{6}$")


class RefundRequest(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d{6}$")
    amount: float = Field(gt=0, le=5000)
    reason: str = Field(min_length=5, max_length=200)


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(
        "lookup_order", "Get order status and total", OrderLookup
    )
    def lookup_order(args: OrderLookup) -> dict:
        if args.order_id not in ORDERS:
            raise KeyError(f"no order {args.order_id}")
        return ORDERS[args.order_id]

    @reg.tool(
        "issue_refund",
        "Refund part of an order",
        RefundRequest,
        side_effect="write",
    )
    def issue_refund(args: RefundRequest) -> dict:
        return {"refunded": args.amount, "order": args.order_id}

    return reg
