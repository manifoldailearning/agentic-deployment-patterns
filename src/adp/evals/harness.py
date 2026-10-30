"""Pattern 10: evaluation harness and release gate.

Cases are data (JSONL), graders are small functions, and the
gate is a policy: overall pass rate, per-category pass rates,
critical cases that must all pass, and no regression against
the last accepted baseline. CI runs the gate on every change.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    input: str
    category: str
    expect: dict[str, Any]
    critical: bool = False


Grader = Callable[[str, dict[str, Any]], tuple[bool, str]]


def grade_contains(out: str, exp: dict) -> tuple[bool, str]:
    need = exp.get("contains", [])
    missing = [s for s in need if s.lower() not in out.lower()]
    return not missing, f"missing {missing}" if missing else ""


def grade_forbidden(out: str, exp: dict) -> tuple[bool, str]:
    bad = [s for s in exp.get("forbidden", [])
           if s.lower() in out.lower()]
    return not bad, f"forbidden {bad}" if bad else ""


def grade_citation(out: str, exp: dict) -> tuple[bool, str]:
    if not exp.get("cite"):
        return True, ""
    ok = re.search(r"\[doc:[\w-]+\]", out) is not None
    return ok, "" if ok else "no citation"


def grade_route(out: str, exp: dict) -> tuple[bool, str]:
    want = exp.get("route")
    if want is None:
        return True, ""
    return out.startswith(f"route={want}"), f"want {want}"


DEFAULT_GRADERS: list[Grader] = [grade_contains, grade_forbidden,
                                 grade_citation, grade_route]


@dataclass
class CaseResult:
    id: str
    category: str
    critical: bool
    passed: bool
    reasons: list[str]
    output: str


@dataclass
class Report:
    results: list[CaseResult]
    pass_rate: float
    by_category: dict[str, float]
    critical_failures: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_cases(path: str | Path) -> list[Case]:
    lines = Path(path).read_text().splitlines()
    return [Case(**json.loads(x)) for x in lines if x.strip()]


def evaluate(system: Callable[[str], str], cases: list[Case],
             graders: list[Grader] | None = None) -> Report:
    graders = graders or DEFAULT_GRADERS
    results = []
    for c in cases:
        try:
            out = system(c.input)
        except Exception as exc:
            out = f"EXCEPTION {type(exc).__name__}: {exc}"
        reasons = [why for g in graders
                   for ok, why in [g(out, c.expect)] if not ok]
        results.append(CaseResult(c.id, c.category, c.critical,
                                  not reasons, reasons, out))
    cats: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        cats[r.category].append(r.passed)
    return Report(
        results,
        _rate([r.passed for r in results]),
        {k: _rate(v) for k, v in sorted(cats.items())},
        [r.id for r in results if r.critical and not r.passed],
    )


def _rate(xs: list[bool]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


@dataclass(frozen=True)
class GatePolicy:
    min_pass_rate: float = 0.90
    min_category_rate: float = 0.80
    max_regression: float = 0.02  # vs baseline pass rate


@dataclass
class GateDecision:
    passed: bool
    failures: list[str]


def gate(report: Report, policy: GatePolicy,
         baseline: float | None = None) -> GateDecision:
    f = []
    if report.critical_failures:
        f.append(f"critical cases failed: "
                 f"{report.critical_failures}")
    if report.pass_rate < policy.min_pass_rate:
        f.append(f"pass rate {report.pass_rate:.2%} < "
                 f"{policy.min_pass_rate:.0%}")
    for cat, rate in report.by_category.items():
        if rate < policy.min_category_rate:
            f.append(f"category {cat} {rate:.2%} < "
                     f"{policy.min_category_rate:.0%}")
    if baseline is not None and \
            report.pass_rate < baseline - policy.max_regression:
        f.append(f"regression: {report.pass_rate:.2%} vs "
                 f"baseline {baseline:.2%}")
    return GateDecision(not f, f)
