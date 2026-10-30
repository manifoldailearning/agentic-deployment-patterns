"""Pattern 08: guardrails around the model.

Input checks run before the model sees anything; output
checks run before anything leaves the system. Each check
returns a verdict, and the policy engine combines them:
``block`` beats ``redact`` beats ``allow``. Deterministic
checks are cheap, testable and explainable, so they come
first; model-based judges can be added as more checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

ACTIONS = ("allow", "redact", "block")


@dataclass(frozen=True)
class Verdict:
    check: str
    action: str  # allow | redact | block
    reason: str = ""
    text: str | None = None  # rewritten text for "redact"


Check = Callable[[str], Verdict]

PII_PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE": re.compile(r"\+?\d[\d -]{8,}\d"),
}


def redact_pii(text: str) -> Verdict:
    out, found = text, []
    for label, pat in PII_PATTERNS.items():
        out, n = pat.subn(f"[{label}]", out)
        if n:
            found.append(label)
    if found:
        return Verdict(
            "pii", "redact", "found " + ", ".join(found), out
        )
    return Verdict("pii", "allow")


INJECTION = re.compile(
    r"ignore (all|any|previous|prior) (instructions|rules)"
    r"|you are now|system prompt|developer mode",
    re.IGNORECASE,
)


def block_injection(text: str) -> Verdict:
    m = INJECTION.search(text)
    if m:
        return Verdict(
            "injection", "block", f"matched '{m.group(0)}'"
        )
    return Verdict("injection", "allow")


def max_length(limit: int) -> Check:
    def check(text: str) -> Verdict:
        if len(text) > limit:
            return Verdict(
                "length", "block", f"{len(text)} > {limit} chars"
            )
        return Verdict("length", "allow")

    return check


def forbid_terms(*terms: str) -> Check:
    pat = re.compile(
        "|".join(map(re.escape, terms)), re.IGNORECASE
    )

    def check(text: str) -> Verdict:
        m = pat.search(text)
        if m:
            return Verdict(
                "forbidden", "block", f"contains '{m.group(0)}'"
            )
        return Verdict("forbidden", "allow")

    return check


def require_citation(text: str) -> Verdict:
    if re.search(r"\[doc:[\w-]+\]", text):
        return Verdict("citation", "allow")
    return Verdict("citation", "block", "answer cites no source")


@dataclass
class GuardResult:
    allowed: bool
    text: str
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [
            f"{v.check}: {v.reason}"
            for v in self.verdicts
            if v.action != "allow"
        ]


class Guard:
    def __init__(self, checks: list[Check]) -> None:
        self.checks = checks

    def run(self, text: str) -> GuardResult:
        verdicts: list[Verdict] = []
        current = text
        for check in self.checks:
            v = check(current)
            verdicts.append(v)
            if v.action == "block":
                return GuardResult(False, "", verdicts)
            if v.action == "redact" and v.text is not None:
                current = v.text
        return GuardResult(True, current, verdicts)


@dataclass
class GuardedLLMCall:
    """Wrap any text-in, text-out function with guards."""

    input_guard: Guard
    output_guard: Guard
    refusal: str = "I can't help with that request."

    def __call__(
        self, fn: Callable[[str], str], user_text: str
    ) -> tuple[str, list[str]]:
        gi = self.input_guard.run(user_text)
        if not gi.allowed:
            return self.refusal, gi.reasons
        answer = fn(gi.text)
        go = self.output_guard.run(answer)
        if not go.allowed:
            return self.refusal, gi.reasons + go.reasons
        return go.text, gi.reasons + go.reasons
