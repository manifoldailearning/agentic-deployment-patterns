"""CLI: python -m adp.evals CASES SYSTEM [--baseline FILE]

SYSTEM is "module:function" returning a callable str -> str.
Exits non-zero when the release gate fails.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from adp.evals.harness import (
    GatePolicy,
    evaluate,
    gate,
    load_cases,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adp.evals")
    ap.add_argument("cases")
    ap.add_argument("system")
    ap.add_argument("--baseline")
    ap.add_argument("--report")
    ap.add_argument("--min-pass", type=float, default=0.90)
    a = ap.parse_args(argv)

    mod, fn = a.system.split(":")
    system = getattr(importlib.import_module(mod), fn)()
    report = evaluate(system, load_cases(a.cases))
    base = None
    if a.baseline and Path(a.baseline).exists():
        base = json.loads(Path(a.baseline).read_text())[
            "pass_rate"
        ]
    decision = gate(
        report, GatePolicy(min_pass_rate=a.min_pass), base
    )
    if a.report:
        Path(a.report).write_text(report.to_json())
    print(
        f"pass rate {report.pass_rate:.2%} "
        f"by category {report.by_category}"
    )
    for r in report.results:
        if not r.passed:
            print(f"  FAIL {r.id}: {r.reasons}")
    print("GATE", "PASSED" if decision.passed else "FAILED")
    for f in decision.failures:
        print("  -", f)
    return 0 if decision.passed else 1


if __name__ == "__main__":
    sys.exit(main())
