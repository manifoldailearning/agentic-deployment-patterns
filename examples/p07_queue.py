"""Pattern 07: bounded queue, fixed workers, fast refusal."""

import asyncio

from adp.queue import JobQueue, QueueFull


async def agent_run(payload):
    await asyncio.sleep(payload.get("seconds", 0.01))
    return f"answered {payload['ticket']}"


async def main():
    q = JobQueue(
        agent_run, workers=2, max_queue=3, job_timeout=0.2
    )
    accepted, refused = [], 0
    for i in range(6):  # burst of 6 against capacity 3
        try:
            accepted.append(q.submit({"ticket": f"T{i}"}))
        except QueueFull:
            refused += 1
    await q.start()
    await q.drain()
    print("accepted", len(accepted), "refused", refused)
    for j in accepted:
        print(j.id, j.status, j.result)
    await q.start()
    slow = q.submit({"ticket": "T9", "seconds": 1})
    await q.drain()
    print(slow.id, slow.status, slow.error)


asyncio.run(main())
