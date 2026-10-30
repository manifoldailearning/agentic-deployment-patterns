import random

import pytest

from adp.core import FakeClock
from adp.resilience import (
    CircuitBreaker,
    CircuitOpen,
    IdempotencyConflict,
    IdempotencyStore,
    PermanentError,
    RetryPolicy,
    TransientError,
    retry,
)


def flaky(fail_times, exc=TransientError):
    n = {"calls": 0}

    def fn():
        n["calls"] += 1
        if n["calls"] <= fail_times:
            raise exc("x")
        return "ok"

    return fn, n


def test_retry_recovers_with_backoff():
    clock = FakeClock()
    fn, n = flaky(2)
    p = RetryPolicy(base_delay=1, jitter=0)
    assert retry(fn, p, clock) == "ok"
    assert n["calls"] == 3 and clock.sleeps == [1, 2]


def test_retry_gives_up_after_max_attempts():
    fn, n = flaky(10)
    with pytest.raises(TransientError):
        retry(fn, RetryPolicy(max_attempts=3), FakeClock())
    assert n["calls"] == 3


def test_permanent_errors_are_not_retried():
    fn, n = flaky(1, PermanentError)
    with pytest.raises(PermanentError):
        retry(fn, RetryPolicy(), FakeClock())
    assert n["calls"] == 1


def test_backoff_is_capped_and_jittered():
    p = RetryPolicy(base_delay=1, max_delay=4, jitter=0.1)
    rng = random.Random(1)
    d = [p.delay(a, rng) for a in range(1, 7)]
    assert all(0.9 <= x <= 4.4 for x in d) and d[-1] > 3.5


def test_idempotency_returns_first_result():
    s = IdempotencyStore()
    calls = []
    f = lambda p: calls.append(p) or len(calls)  # noqa: E731
    assert s.run("k1", {"a": 1}, f) == 1
    assert s.run("k1", {"a": 1}, f) == 1
    assert len(calls) == 1


def test_idempotency_conflict_on_changed_payload():
    s = IdempotencyStore()
    s.run("k1", {"a": 1}, lambda p: 1)
    with pytest.raises(IdempotencyConflict):
        s.run("k1", {"a": 2}, lambda p: 2)


def test_breaker_opens_fails_fast_and_recovers():
    clock = FakeClock()
    b = CircuitBreaker(
        failure_threshold=2, reset_after=10, clock=clock
    )
    for _ in range(2):
        with pytest.raises(ZeroDivisionError):
            b.call(lambda: 1 / 0)
    assert b.state == "open"
    with pytest.raises(CircuitOpen):
        b.call(lambda: "never called")
    clock.advance(10)
    assert b.state == "half_open"
    assert b.call(lambda: "ok") == "ok"
    assert b.state == "closed"


def test_failed_probe_reopens_breaker():
    clock = FakeClock()
    b = CircuitBreaker(
        failure_threshold=1, reset_after=5, clock=clock
    )
    with pytest.raises(ZeroDivisionError):
        b.call(lambda: 1 / 0)
    clock.advance(5)
    with pytest.raises(ZeroDivisionError):
        b.call(lambda: 1 / 0)
    assert b.state == "open"
