"""Pattern 06: retries, idempotency keys, circuit breakers.

* ``retry`` retries only transient errors, with capped
  exponential backoff and jitter.
* ``IdempotencyStore`` makes a write safe to repeat: the same
  key returns the first result; the same key with a different
  payload is refused.
* ``CircuitBreaker`` stops calling a failing dependency and
  probes it again after a cool-down.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from adp.core.clock import Clock, SystemClock

T = TypeVar("T")


class TransientError(Exception):
    """Safe to retry: timeouts, 429s, 503s."""


class PermanentError(Exception):
    """Never retry: bad input, auth, policy refusal."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.1  # +/- fraction of the delay
    retry_on: tuple[type[BaseException], ...] = (
        TransientError, TimeoutError, ConnectionError)

    def delay(self, attempt: int, rng: random.Random) -> float:
        d = min(self.max_delay,
                self.base_delay * 2 ** (attempt - 1))
        return d * (1 + rng.uniform(-self.jitter, self.jitter))


def retry(fn: Callable[[], T], policy: RetryPolicy | None = None,
          clock: Clock | None = None,
          rng: random.Random | None = None) -> T:
    p = policy or RetryPolicy()
    clock = clock or SystemClock()
    rng = rng or random.Random()
    for attempt in range(1, p.max_attempts + 1):
        try:
            return fn()
        except p.retry_on:
            if attempt == p.max_attempts:
                raise
            clock.sleep(p.delay(attempt, rng))
    raise AssertionError("unreachable")


class IdempotencyConflict(Exception):
    pass


class IdempotencyStore:
    def __init__(self) -> None:
        self._done: dict[str, tuple[str, Any]] = {}

    @staticmethod
    def fingerprint(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def run(self, key: str, payload: dict[str, Any],
            fn: Callable[[dict[str, Any]], T]) -> T:
        fp = self.fingerprint(payload)
        if key in self._done:
            seen_fp, result = self._done[key]
            if seen_fp != fp:
                raise IdempotencyConflict(
                    f"key {key} reused with a different payload")
            return result
        result = fn(payload)
        self._done[key] = (fp, result)
        return result


class CircuitOpen(Exception):
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5,
                 reset_after: float = 30.0,
                 clock: Clock | None = None) -> None:
        self.threshold = failure_threshold
        self.reset_after = reset_after
        self.clock = clock or SystemClock()
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock.now() - self.opened_at >= self.reset_after:
            return "half_open"
        return "open"

    def call(self, fn: Callable[[], T]) -> T:
        state = self.state
        if state == "open":
            raise CircuitOpen("dependency unavailable; failing fast")
        try:
            result = fn()
        except Exception:
            self.failures += 1
            if state == "half_open" or \
                    self.failures >= self.threshold:
                self.opened_at = self.clock.now()
            raise
        self.failures = 0
        self.opened_at = None
        return result
