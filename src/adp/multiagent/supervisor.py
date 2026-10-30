"""Pattern 13: a bounded supervisor over specialist workers.

The supervisor model writes a plan: a short list of subtasks,
each assigned to a named worker. Code validates the plan
(known workers, size limit, no duplicates) before anything
runs. Workers cannot call each other; only the supervisor
delegates, and the whole run shares one budget.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from adp.core.llm import LLM, Message


class Subtask(BaseModel):
    worker: str
    task: str = Field(min_length=3, max_length=500)


class Plan(BaseModel):
    subtasks: list[Subtask] = Field(min_length=1)


@dataclass(frozen=True)
class Worker:
    name: str
    description: str
    run: Callable[[str], str]
    cost_units: int = 1


@dataclass
class SupervisorResult:
    status: str  # completed | rejected_plan | budget_exceeded
    answer: str | None
    outputs: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    units_used: int = 0


PLAN_PROMPT = (
    "Split the request into at most {n} subtasks. Use only "
    "these workers: {workers}. Reply with JSON: "
    '{{"subtasks": [{{"worker": "...", "task": "..."}}]}}'
)


class Supervisor:
    def __init__(self, llm: LLM, workers: list[Worker],
                 max_subtasks: int = 4,
                 budget_units: int = 6) -> None:
        self.llm = llm
        self.workers = {w.name: w for w in workers}
        self.max_subtasks = max_subtasks
        self.budget_units = budget_units

    def plan(self, request: str) -> tuple[Plan | None,
                                           list[str]]:
        prompt = PLAN_PROMPT.format(
            n=self.max_subtasks,
            workers=", ".join(sorted(self.workers)))
        raw = self.llm.complete([Message("system", prompt),
                                 Message("user", request)]).text
        try:
            plan = Plan.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            return None, [f"unparseable plan: {exc}"[:200]]
        problems = []
        if len(plan.subtasks) > self.max_subtasks:
            problems.append(f"{len(plan.subtasks)} subtasks > "
                            f"{self.max_subtasks}")
        for st in plan.subtasks:
            if st.worker not in self.workers:
                problems.append(f"unknown worker '{st.worker}'")
        pairs = [(s.worker, s.task) for s in plan.subtasks]
        if len(set(pairs)) != len(pairs):
            problems.append("duplicate subtasks")
        return (None, problems) if problems else (plan, [])

    def run(self, request: str,
            combine: Callable[[dict[str, str]], str]
            ) -> SupervisorResult:
        plan, problems = self.plan(request)
        if plan is None:
            return SupervisorResult("rejected_plan", None,
                                    problems=problems)
        outputs: dict[str, str] = {}
        used = 0
        for i, st in enumerate(plan.subtasks, 1):
            w = self.workers[st.worker]
            if used + w.cost_units > self.budget_units:
                return SupervisorResult(
                    "budget_exceeded", None, outputs,
                    [f"stopped before subtask {i}"], used)
            used += w.cost_units
            outputs[f"{i}:{w.name}"] = w.run(st.task)
        return SupervisorResult("completed", combine(outputs),
                                outputs, [], used)
