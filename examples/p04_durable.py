"""Pattern 04: resume a crashed run without repeating work."""

from adp.durable import (
    DurableRunner,
    SQLiteCheckpointStore,
    Workflow,
)

emails_sent = []
crash = {"on": True}


def draft(s):
    s["draft"] = f"Hi {s['customer']}, your refund is approved."
    return s


def send(s):
    emails_sent.append(s["draft"])  # side effect
    s["sent"] = True
    return s


def close(s):
    if crash["on"]:
        raise ConnectionError("ticket API timed out")
    s["closed"] = True
    return s


wf = Workflow(
    "refund-followup",
    [("draft", draft), ("send", send), ("close", close)],
)
runner = DurableRunner(SQLiteCheckpointStore())

cp = runner.run(wf, "run-42", {"customer": "Ana"})
print(cp.status, "at step", cp.step_index, "-", cp.error)

crash["on"] = False  # dependency recovers; same run_id resumes
cp = runner.run(wf, "run-42", {"customer": "Ana"})
print(cp.status, cp.state)
print("emails sent:", len(emails_sent))  # 1, not 2
