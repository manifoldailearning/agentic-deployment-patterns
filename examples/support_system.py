"""Reference support assistant built from the patterns.

Router (03) + tools (01) + permission-aware RAG (09) + guards
(08). The model is a deterministic FakeLLM so evaluations are
repeatable; swap in a real LLM adapter in production.
"""

from __future__ import annotations

import json
import re

from kb import DOCS
from support_tools import build_registry

from adp.core import FakeLLM, ToolCall
from adp.guardrails import (
    Guard,
    block_injection,
    forbid_terms,
    redact_pii,
)
from adp.rag import PermissionAwareRetriever, User, answer
from adp.router import Router

AGENT = User("agent", frozenset({"agents"}))
KEYWORDS = [
    ("refund", "refund"),
    ("order", "order_status"),
    ("how", "how_to"),
    ("reset", "how_to"),
]


def classifier(messages, tools):
    text = messages[-1].content.lower()
    for word, intent in KEYWORDS:
        if word in text:
            return json.dumps(
                {"intent": intent, "confidence": 0.9}
            )
    return json.dumps({"intent": "how_to", "confidence": 0.3})


def writer(messages, tools):
    first = re.search(
        r"\[doc:([\w-]+)\] ([^\n]+)", messages[-1].content
    )
    return f"{first.group(2)} [doc:{first.group(1)}]"


def build_system():
    tools = build_registry()
    retriever = PermissionAwareRetriever(DOCS)
    rag_llm = FakeLLM(responder=writer)
    in_guard = Guard([block_injection, redact_pii])
    out_guard = Guard([forbid_terms("override code")])

    def order_status(text):
        m = re.search(r"ORD-\d{6}", text)
        if not m:
            return "Please share your order number (ORD-xxxxxx)."
        r = tools.call(
            ToolCall("lookup_order", {"order_id": m.group(0)})
        )
        return r.to_message()

    handlers = {
        "order_status": order_status,
        "refund": lambda t: "A lead will review this refund.",
        "how_to": lambda t: answer(rag_llm, retriever, AGENT, t)[
            0
        ],
    }
    router = Router(
        FakeLLM(responder=classifier),
        handlers,
        lambda t: "Routed to a human agent.",
    )

    def system(text: str) -> str:
        g = in_guard.run(text)
        if not g.allowed:
            return "route=blocked | I can't help with that."
        r = router.route(g.text)
        route = r.route.split(":")[0]
        out = out_guard.run(str(r.output))
        body = out.text if out.allowed else "[withheld]"
        return f"route={route} | {body}"

    return system
