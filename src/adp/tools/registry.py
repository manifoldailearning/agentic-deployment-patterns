"""Pattern 01: typed tool contracts.

Every tool declares a Pydantic model for its arguments. The
registry validates each model-proposed call before running
it, and returns structured errors the model can recover from
instead of raising into the agent loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from adp.core.llm import ToolCall

SideEffect = Literal["read", "write"]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any], Any]
    side_effect: SideEffect = "read"

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: Any = None
    error: str | None = None

    def to_message(self) -> str:
        if self.ok:
            return f"OK: {self.output}"
        return f"ERROR: {self.error}"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        side_effect: SideEffect = "read",
    ):
        """Decorator form of :meth:`register`."""

        def wrap(fn: Callable[[Any], Any]) -> Callable:
            self.register(
                Tool(
                    name, description, args_model, fn, side_effect
                )
            )
            return fn

        return wrap

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def call(self, tc: ToolCall) -> ToolResult:
        tool = self._tools.get(tc.name)
        if tool is None:
            known = ", ".join(self.names())
            return ToolResult(
                tc.id,
                tc.name,
                False,
                error=f"unknown tool '{tc.name}'; known: {known}",
            )
        try:
            args = tool.args_model.model_validate(tc.args)
        except ValidationError as exc:
            return ToolResult(
                tc.id,
                tc.name,
                False,
                error="invalid arguments: " + _summarize(exc),
            )
        try:
            out = tool.fn(args)
        except Exception as exc:  # tool failures become data
            return ToolResult(
                tc.id,
                tc.name,
                False,
                error=f"tool failed: {type(exc).__name__}: {exc}",
            )
        return ToolResult(tc.id, tc.name, True, output=out)


def _summarize(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "args"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
