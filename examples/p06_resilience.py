"""Pattern 06: retry transient errors, make writes repeatable."""

import random

from adp.core import FakeClock
from adp.resilience import (
    CircuitBreaker,
    CircuitOpen,
    IdempotencyStore,
    RetryPolicy,
    TransientError,
    retry,
)

clock = FakeClock()
attempts = {"n": 0}


def flaky_model_call():
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise TransientError("429 rate limited")
    return "draft reply"


out = retry(
    flaky_model_call, RetryPolicy(), clock, random.Random(7)
)
print(
    out,
    "after",
    attempts["n"],
    "attempts; waited",
    [round(s, 2) for s in clock.sleeps],
)

store = IdempotencyStore()
refund = {"order_id": "ORD-100002", "amount": 310.0}
first = store.run(
    "refund-ORD-100002-1",
    refund,
    lambda p: {"refund_id": "rf-1", **p},
)
again = store.run(
    "refund-ORD-100002-1",
    refund,
    lambda p: {"refund_id": "rf-2", **p},
)
print("retried write returns", again["refund_id"])

breaker = CircuitBreaker(
    failure_threshold=2, reset_after=30, clock=clock
)
for _ in range(3):
    try:
        breaker.call(lambda: 1 / 0)
    except CircuitOpen as e:
        print("fast fail:", e)
    except ZeroDivisionError:
        print("dependency error; breaker", breaker.state)
