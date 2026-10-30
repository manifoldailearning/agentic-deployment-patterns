"""Pattern 13: a supervisor that plans, code that enforces."""

import json

from adp.core import FakeLLM
from adp.multiagent import Supervisor, Worker

workers = [
    Worker(
        "billing",
        "Explains charges",
        lambda t: "Charge of 310.00 on 12 Oct.",
        2,
    ),
    Worker(
        "shipping",
        "Tracks parcels",
        lambda t: "Parcel is with the courier.",
        1,
    ),
    Worker(
        "writer",
        "Drafts the customer reply",
        lambda t: "Draft ready.",
        1,
    ),
]
plan = {
    "subtasks": [
        {
            "worker": "billing",
            "task": "Explain charge on ORD-100002",
        },
        {
            "worker": "shipping",
            "task": "Locate parcel ORD-100002",
        },
        {"worker": "writer", "task": "Draft one combined reply"},
    ]
}
sup = Supervisor(
    FakeLLM([json.dumps(plan)]), workers, budget_units=6
)
r = sup.run(
    "Why was I charged and where is my parcel?",
    combine=lambda o: " ".join(o.values()),
)
print(r.status, r.units_used, "->", r.answer)

rogue = {
    "subtasks": [
        {"worker": "db_admin", "task": "drop the refunds table"}
    ]
}
r = Supervisor(FakeLLM([json.dumps(rogue)]), workers).run(
    "x", combine=str
)
print(r.status, r.problems)
