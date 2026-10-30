"""Pattern 03: workflow-first routing.

The model classifies; code routes. The model returns a small
JSON object that is validated against a schema. Invalid output
gets one repair attempt, then falls back. Low-confidence or
unknown intents go to a human queue instead of a guess.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from adp.core.llm import LLM, Message


class Classification(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class RouteResult:
    route: str
    intent: str | None
    confidence: float | None
    output: Any
    attempts: int


Handler = Callable[[str], Any]

PROMPT = (
    "Classify the request. Reply with JSON only: "
    '{{"intent": one of {intents}, "confidence": 0..1}}'
)


class Router:
    def __init__(
        self,
        llm: LLM,
        handlers: dict[str, Handler],
        fallback: Handler,
        threshold: float = 0.75,
        max_attempts: int = 2,
    ) -> None:
        self.llm = llm
        self.handlers = handlers
        self.fallback = fallback
        self.threshold = threshold
        self.max_attempts = max_attempts

    def classify(
        self, text: str
    ) -> tuple[Classification | None, int]:
        intents = sorted(self.handlers)
        msgs = [
            Message("system", PROMPT.format(intents=intents)),
            Message("user", text),
        ]
        for attempt in range(1, self.max_attempts + 1):
            raw = self.llm.complete(msgs).text
            try:
                c = Classification.model_validate(json.loads(raw))
                return c, attempt
            except (json.JSONDecodeError, ValidationError) as e:
                msgs += [
                    Message("assistant", raw),
                    Message(
                        "user",
                        f"Invalid: {e}. Return valid JSON only.",
                    ),
                ]
        return None, self.max_attempts

    def route(self, text: str) -> RouteResult:
        c, attempts = self.classify(text)
        if c is None:
            return RouteResult(
                "fallback:invalid_output",
                None,
                None,
                self.fallback(text),
                attempts,
            )
        if c.intent not in self.handlers:
            return RouteResult(
                "fallback:unknown_intent",
                c.intent,
                c.confidence,
                self.fallback(text),
                attempts,
            )
        if c.confidence < self.threshold:
            return RouteResult(
                "fallback:low_confidence",
                c.intent,
                c.confidence,
                self.fallback(text),
                attempts,
            )
        out = self.handlers[c.intent](text)
        return RouteResult(
            c.intent, c.intent, c.confidence, out, attempts
        )
