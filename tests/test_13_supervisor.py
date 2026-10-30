import json

from adp.core import FakeLLM
from adp.multiagent import Supervisor, Worker

calls = []
W = [Worker("a", "A", lambda t: calls.append(t) or f"A:{t}", 2),
     Worker("b", "B", lambda t: f"B:{t}", 1)]


def sup(plan, **kw):
    raw = plan if isinstance(plan, str) else json.dumps(plan)
    return Supervisor(FakeLLM([raw]), W, **kw)


def st(worker, task):
    return {"worker": worker, "task": task}


def test_valid_plan_runs_in_order_and_combines():
    r = sup({"subtasks": [st("a", "one"), st("b", "two")]}).run(
        "q", combine=lambda o: "|".join(o.values()))
    assert r.status == "completed"
    assert r.answer == "A:one|B:two" and r.units_used == 3


def test_unknown_worker_rejects_whole_plan_before_running():
    calls.clear()
    r = sup({"subtasks": [st("a", "ok task"),
                          st("root", "bad")]}).run("q", str)
    assert r.status == "rejected_plan" and calls == []
    assert "unknown worker 'root'" in r.problems


def test_plan_size_and_duplicates_are_limited():
    many = {"subtasks": [st("b", f"task {i}") for i in range(5)]}
    r = sup(many, max_subtasks=4).run("q", str)
    assert r.status == "rejected_plan"
    dup = {"subtasks": [st("b", "same"), st("b", "same")]}
    assert "duplicate subtasks" in sup(dup).run("q", str).problems


def test_unparseable_plan_is_rejected():
    r = sup("I'll just handle it myself!").run("q", str)
    assert r.status == "rejected_plan"
    assert r.problems[0].startswith("unparseable plan")


def test_shared_budget_stops_the_run():
    plan = {"subtasks": [st("a", "task 1"), st("a", "task 2"),
                         st("b", "task 3")]}
    r = sup(plan, budget_units=3).run("q", str)
    assert r.status == "budget_exceeded"
    assert list(r.outputs) == ["1:a"] and r.units_used == 2
