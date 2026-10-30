"""Model interface used by every pattern.

Patterns depend on the small ``LLM`` protocol, never on a vendor
SDK. ``FakeLLM`` returns scripted responses so that every test
and example runs offline and deterministically.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = "call-1"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    model: str = "fake-1"


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class LLM(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


class ScriptExhausted(RuntimeError):
    """FakeLLM was called more times than it was scripted."""


Responder = Callable[
    [Sequence[Message], Sequence[dict[str, Any]] | None],
    "LLMResponse | str",
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: about four characters per token."""
    return max(1, len(text) // 4)


class FakeLLM:
    """Deterministic stand-in for a real model.

    Give it either a ``script`` (responses returned in order)
    or a ``responder`` function that inspects the messages.
    """

    def __init__(
        self,
        script: Sequence[LLMResponse | str] | None = None,
        responder: Responder | None = None,
        model: str = "fake-1",
    ) -> None:
        self._script = list(script or [])
        self._responder = responder
        self.model = model
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self._responder is not None:
            out = self._responder(messages, tools)
        elif self._script:
            out = self._script.pop(0)
        else:
            raise ScriptExhausted("no scripted response left")
        if isinstance(out, str):
            prompt = " ".join(m.content for m in messages)
            out = LLMResponse(
                text=out,
                usage=Usage(
                    estimate_tokens(prompt), estimate_tokens(out)
                ),
                model=self.model,
            )
        return out


def say(text: str, tokens_in: int = 50, tokens_out: int = 20):
    """Build a plain text response (test helper)."""
    return LLMResponse(text=text, usage=Usage(tokens_in, tokens_out))


def call(
    name: str,
    call_id: str = "call-1",
    tokens_in: int = 50,
    tokens_out: int = 20,
    **args: Any,
) -> LLMResponse:
    """Build a response that requests one tool call."""
    return LLMResponse(
        tool_calls=(ToolCall(name, dict(args), call_id),),
        usage=Usage(tokens_in, tokens_out),
    )
