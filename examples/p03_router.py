"""Pattern 03: the model classifies, deterministic code routes."""

from adp.core import FakeLLM
from adp.router import Router

handlers = {
    "order_status": lambda t: "status workflow",
    "refund": lambda t: "refund workflow (needs approval)",
    "how_to": lambda t: "knowledge-base workflow",
}
human_queue: list[str] = []


def to_human(text: str) -> str:
    human_queue.append(text)
    return "queued for a human"


llm = FakeLLM([
    '{"intent": "order_status", "confidence": 0.93}',
    '{"intent": "refund", "confidence": 0.41}',
    "Sure! I think this is a refund.",
    '{"intent": "refund", "confidence": 0.88}',
])
router = Router(llm, handlers, to_human)
for text in ["Where is my order?", "money back??",
             "I want a refund for ORD-100002"]:
    r = router.route(text)
    print(f"{text!r} -> {r.route} ({r.attempts} attempt/s)")
print("human queue:", human_queue)
