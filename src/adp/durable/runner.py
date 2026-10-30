"""Pattern 04: durable execution with checkpoints.

A workflow is a list of named steps. After each step the run
state is written to a checkpoint store. If the process dies,
re-running the same ``run_id`` resumes after the last
completed step, so finished work (and its side effects) is
never repeated.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

State = dict[str, Any]
Step = Callable[[State], State]


@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    step_index: int  # number of completed steps
    status: str  # "running" | "completed" | "failed"
    state: State
    error: str | None = None


class SQLiteCheckpointStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints ("
            " run_id TEXT PRIMARY KEY, step_index INTEGER,"
            " status TEXT, state TEXT, error TEXT)"
        )

    def load(self, run_id: str) -> Checkpoint | None:
        row = self.db.execute(
            "SELECT step_index, status, state, error"
            " FROM checkpoints WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        idx, status, state, error = row
        return Checkpoint(run_id, idx, status,
                          json.loads(state), error)

    def save(self, cp: Checkpoint) -> None:
        with self.db:  # one transaction per checkpoint
            self.db.execute(
                "INSERT OR REPLACE INTO checkpoints"
                " VALUES (?, ?, ?, ?, ?)",
                (cp.run_id, cp.step_index, cp.status,
                 json.dumps(cp.state), cp.error),
            )


class Workflow:
    def __init__(self, name: str,
                 steps: list[tuple[str, Step]]) -> None:
        self.name = name
        self.steps = steps


class DurableRunner:
    def __init__(self, store: SQLiteCheckpointStore) -> None:
        self.store = store

    def run(self, wf: Workflow, run_id: str,
            initial: State) -> Checkpoint:
        cp = self.store.load(run_id)
        if cp is None:
            cp = Checkpoint(run_id, 0, "running", dict(initial))
            self.store.save(cp)
        if cp.status == "completed":
            return cp
        state = dict(cp.state)
        for i in range(cp.step_index, len(wf.steps)):
            name, step = wf.steps[i]
            try:
                state = step(dict(state))
            except Exception as exc:
                failed = Checkpoint(
                    run_id, i, "failed", state,
                    f"{name}: {type(exc).__name__}: {exc}")
                self.store.save(failed)
                return failed
            self.store.save(
                Checkpoint(run_id, i + 1, "running", state))
        done = Checkpoint(run_id, len(wf.steps), "completed",
                          state)
        self.store.save(done)
        return done
