from adp.durable import (
    DurableRunner,
    SQLiteCheckpointStore,
    Workflow,
)


def counting_wf(counts, fail_at=None):
    def mk(name):
        def step(s):
            counts[name] = counts.get(name, 0) + 1
            if fail_at and fail_at.get(name):
                raise RuntimeError("boom")
            s.setdefault("trail", []).append(name)
            return s

        return name, step

    return Workflow("w", [mk("a"), mk("b"), mk("c")])


def test_runs_all_steps_and_completes():
    counts = {}
    r = DurableRunner(SQLiteCheckpointStore())
    cp = r.run(counting_wf(counts), "r1", {})
    assert cp.status == "completed"
    assert cp.state["trail"] == ["a", "b", "c"]


def test_failure_is_checkpointed_and_resume_skips_done(tmp_path):
    db = str(tmp_path / "cp.db")
    counts, fail = {}, {"b": True}
    wf = counting_wf(counts, fail)
    cp = DurableRunner(SQLiteCheckpointStore(db)).run(
        wf, "r1", {}
    )
    assert cp.status == "failed" and cp.step_index == 1
    assert cp.error.startswith("b: RuntimeError")
    fail["b"] = False
    # new store object = new process, same database file
    cp = DurableRunner(SQLiteCheckpointStore(db)).run(
        wf, "r1", {}
    )
    assert cp.status == "completed"
    assert counts == {"a": 1, "b": 2, "c": 1}
    assert cp.state["trail"] == ["a", "b", "c"]


def test_completed_run_is_not_rerun():
    counts = {}
    r = DurableRunner(SQLiteCheckpointStore())
    wf = counting_wf(counts)
    r.run(wf, "r1", {})
    r.run(wf, "r1", {})
    assert counts == {"a": 1, "b": 1, "c": 1}


def test_runs_are_isolated_by_id():
    counts = {}
    r = DurableRunner(SQLiteCheckpointStore())
    wf = counting_wf(counts)
    r.run(wf, "r1", {})
    r.run(wf, "r2", {})
    assert counts["a"] == 2
