from pathlib import Path

from adp.evals import Case, GatePolicy, evaluate, gate, load_cases
from adp.evals.__main__ import main

ROOT = Path(__file__).resolve().parent.parent
CASES = [
    Case("a", "q1", "x", {"contains": ["yes"]}),
    Case("b", "q2", "x", {"forbidden": ["secret"]}, critical=True),
    Case("c", "q3", "y", {"cite": True}),
]


def test_graders_and_report():
    rep = evaluate(lambda q: {"q1": "yes", "q2": "secret!",
                              "q3": "fact [doc:a]"}[q], CASES)
    assert [r.passed for r in rep.results] == [True, False, True]
    assert rep.critical_failures == ["b"]
    assert rep.by_category == {"x": 0.5, "y": 1.0}


def test_exceptions_are_failures_not_crashes():
    def boom(q):
        raise RuntimeError("model down")
    rep = evaluate(boom, CASES[:1])
    assert not rep.results[0].passed
    assert rep.results[0].output.startswith("EXCEPTION")


def test_gate_rules():
    good = evaluate(lambda q: "yes [doc:a]", CASES)
    assert gate(good, GatePolicy()).passed
    d = gate(good, GatePolicy(), baseline=1.0)
    assert d.passed
    bad = evaluate(lambda q: "secret", CASES)
    d = gate(bad, GatePolicy(), baseline=1.0)
    text = " ".join(d.failures)
    assert not d.passed
    for word in ("critical", "pass rate", "category",
                 "regression"):
        assert word in text


def test_reference_system_passes_gate():
    cases = load_cases(ROOT / "evals" / "cases.jsonl")
    assert len(cases) >= 12
    code = main([str(ROOT / "evals" / "cases.jsonl"),
                 "support_system:build_system",
                 "--baseline", str(ROOT / "evals" /
                                   "baseline.json")])
    assert code == 0


def test_cli_fails_gate_for_broken_system(tmp_path, monkeypatch):
    mod = tmp_path / "broken_sys.py"
    mod.write_text("def build():\n    return lambda q: 'nope'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    code = main([str(ROOT / "evals" / "cases.jsonl"),
                 "broken_sys:build"])
    assert code == 1
