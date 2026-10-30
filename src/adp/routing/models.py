"""Pattern 12: response caching and model routing/fallback.

* ``CachedLLM`` returns a stored response for an identical
  request (same model, messages and tools) within a TTL.
  Only cache deterministic, non-personal prompts.
* ``ModelRouter`` sends easy requests to a cheaper model and
  hard ones to a stronger model, using a policy you can test.
* ``FallbackLLM`` tries providers in order, skipping any whose
  circuit breaker is open.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict

from adp.core.clock import Clock, SystemClock
from adp.core.llm import LLM, LLMResponse, Message
from adp.resilience.policies import CircuitBreaker, CircuitOpen


def request_key(
    model: str,
    messages: Sequence[Message],
    tools: Sequence[dict] | None,
) -> str:
    blob = json.dumps(
        {
            "model": model,
            "messages": [asdict(m) for m in messages],
            "tools": list(tools or []),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedLLM:
    def __init__(
        self,
        inner: LLM,
        model: str,
        ttl_seconds: float = 3600,
        clock: Clock | None = None,
    ) -> None:
        self.inner = inner
        self.model = model
        self.ttl = ttl_seconds
        self.clock = clock or SystemClock()
        self._store: dict[str, tuple[float, LLMResponse]] = {}
        self.hits = 0
        self.misses = 0

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
    ) -> LLMResponse:
        key = request_key(self.model, messages, tools)
        hit = self._store.get(key)
        if hit and self.clock.now() - hit[0] < self.ttl:
            self.hits += 1
            return hit[1]
        self.misses += 1
        resp = self.inner.complete(messages, tools)
        self._store[key] = (self.clock.now(), resp)
        return resp


Policy = Callable[[Sequence[Message]], str]


def length_and_keyword_policy(
    max_chars: int = 600,
    hard_words: tuple[str, ...] = (
        "refund",
        "legal",
        "contract",
        "escalate",
    ),
) -> Policy:
    def choose(messages: Sequence[Message]) -> str:
        text = messages[-1].content.lower() if messages else ""
        if len(text) > max_chars or any(
            w in text for w in hard_words
        ):
            return "large"
        return "small"

    return choose


class ModelRouter:
    def __init__(
        self, models: dict[str, LLM], policy: Policy
    ) -> None:
        self.models = models
        self.policy = policy
        self.routed: dict[str, int] = {k: 0 for k in models}

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
    ) -> LLMResponse:
        tier = self.policy(messages)
        self.routed[tier] += 1
        return self.models[tier].complete(messages, tools)


class AllProvidersFailed(Exception):
    pass


class FallbackLLM:
    def __init__(
        self,
        providers: list[tuple[str, LLM]],
        breakers: dict[str, CircuitBreaker] | None = None,
    ) -> None:
        self.providers = providers
        self.breakers = breakers or {
            name: CircuitBreaker() for name, _ in providers
        }
        self.last_used: str | None = None

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
    ) -> LLMResponse:
        errors = []
        for name, llm in self.providers:
            breaker = self.breakers[name]
            try:
                resp = breaker.call(
                    lambda llm=llm: llm.complete(messages, tools)
                )
            except CircuitOpen:
                errors.append(f"{name}: circuit open")
                continue
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}")
                continue
            self.last_used = name
            return resp
        raise AllProvidersFailed("; ".join(errors))
