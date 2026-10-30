"""Pattern 11: tracing, metrics and cost accounting.

One trace per request, one span per step (model call, tool
call, retrieval, guard). Spans carry the attributes needed to
debug a bad answer later: model and prompt version, tokens,
cost, tool name, outcome. The span model mirrors
OpenTelemetry, so an exporter can forward it unchanged.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from adp.core.llm import LLM, LLMResponse, Message
from adp.loop.agent import Pricing


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start: float
    end: float | None = None
    status: str = "ok"
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def ms(self) -> float:
        return round(
            ((self.end or self.start) - self.start) * 1000, 2
        )


_current: ContextVar[Span | None] = ContextVar(
    "span", default=None
)


class Tracer:
    def __init__(self, service: str = "agent") -> None:
        self.service = service
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = _current.get()
        s = Span(
            trace_id=parent.trace_id
            if parent
            else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            name=name,
            start=time.perf_counter(),
            attrs={"service": self.service, **attrs},
        )
        token = _current.set(s)
        try:
            yield s
        except Exception as exc:
            s.status = "error"
            s.attrs["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            s.end = time.perf_counter()
            _current.reset(token)
            self.spans.append(s)

    def export_jsonl(self, path: str) -> None:
        with open(path, "w") as fh:
            for s in self.spans:
                fh.write(
                    json.dumps(asdict(s) | {"ms": s.ms}) + "\n"
                )

    def tree(self, trace_id: str) -> list[str]:
        spans = [s for s in self.spans if s.trace_id == trace_id]
        kids: dict[str | None, list[Span]] = defaultdict(list)
        for s in sorted(spans, key=lambda s: s.start):
            kids[s.parent_id].append(s)
        lines: list[str] = []

        def walk(pid: str | None, depth: int) -> None:
            for s in kids[pid]:
                lines.append(
                    "  " * depth + f"{s.name} [{s.status}]"
                )
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return lines


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.samples: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0) -> None:
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        self.samples[name].append(value)

    def percentile(self, name: str, p: float) -> float:
        xs = sorted(self.samples[name])
        if not xs:
            return 0.0
        rank = math.ceil(p / 100 * len(xs))  # nearest rank
        return xs[max(0, rank - 1)]


class TracedLLM:
    """Wrap any LLM so every call emits a span and metrics."""

    def __init__(
        self,
        inner: LLM,
        tracer: Tracer,
        metrics: Metrics,
        prompt_version: str,
        pricing: Pricing | None = None,
    ) -> None:
        self.inner = inner
        self.tracer = tracer
        self.metrics = metrics
        self.prompt_version = prompt_version
        self.pricing = pricing or Pricing()

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict] | None = None,
    ) -> LLMResponse:
        with self.tracer.span(
            "llm.complete", prompt_version=self.prompt_version
        ) as s:
            try:
                r = self.inner.complete(messages, tools)
            except Exception:
                self.metrics.inc("llm.errors")
                raise
            cost = self.pricing.cost(
                r.usage.input_tokens, r.usage.output_tokens
            )
            s.attrs.update(
                model=r.model,
                tokens_in=r.usage.input_tokens,
                tokens_out=r.usage.output_tokens,
                cost_usd=round(cost, 6),
                tool_calls=len(r.tool_calls),
            )
        self.metrics.inc("llm.calls")
        self.metrics.inc("llm.cost_usd", cost)
        self.metrics.observe("llm.latency_ms", s.ms)
        return r
