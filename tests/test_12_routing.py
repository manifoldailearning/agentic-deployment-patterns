import pytest

from adp.core import FakeClock, FakeLLM, Message
from adp.resilience import CircuitBreaker
from adp.routing import (
    AllProvidersFailed,
    CachedLLM,
    FallbackLLM,
    ModelRouter,
    length_and_keyword_policy,
)

M = [Message("user", "hello")]


def echo(name):
    return FakeLLM(responder=lambda m, t: name, model=name)


def test_cache_hits_identical_requests_until_ttl():
    clock = FakeClock()
    inner = echo("x")
    c = CachedLLM(inner, "x", ttl_seconds=10, clock=clock)
    c.complete(M)
    c.complete(M)
    assert (c.hits, c.misses, len(inner.calls)) == (1, 1, 1)
    c.complete([Message("user", "different")])
    clock.advance(11)
    c.complete(M)
    assert c.misses == 3


def test_cache_key_includes_tools():
    c = CachedLLM(echo("x"), "x", clock=FakeClock())
    c.complete(M)
    c.complete(M, tools=[{"name": "t"}])
    assert c.misses == 2


def test_router_policy():
    r = ModelRouter(
        {"small": echo("s"), "large": echo("l")},
        length_and_keyword_policy(max_chars=20),
    )
    assert r.complete(M).text == "s"
    assert (
        r.complete([Message("user", "legal question")]).text
        == "l"
    )
    assert r.complete([Message("user", "x" * 21)]).text == "l"
    assert r.routed == {"small": 1, "large": 2}


class Down:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None):
        self.calls += 1
        raise TimeoutError("down")


def test_fallback_and_breaker_skip():
    clock = FakeClock()
    down = Down()
    fb = FallbackLLM(
        [("a", down), ("b", echo("b"))],
        {
            "a": CircuitBreaker(2, 30, clock),
            "b": CircuitBreaker(2, 30, clock),
        },
    )
    for _ in range(4):
        assert fb.complete(M).text == "b"
    assert down.calls == 2  # breaker stopped further calls
    clock.advance(30)
    fb.complete(M)
    assert down.calls == 3  # half-open probe


def test_all_providers_failed():
    fb = FallbackLLM([("a", Down()), ("b", Down())])
    with pytest.raises(AllProvidersFailed, match="a: Timeout"):
        fb.complete(M)
