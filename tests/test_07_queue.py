import asyncio

import pytest

from adp.queue import JobQueue, QueueFull


async def echo(p):
    await asyncio.sleep(p.get("s", 0))
    if p.get("fail"):
        raise ValueError("bad ticket")
    return p["v"]


def test_processes_jobs_with_worker_pool():
    async def go():
        q = JobQueue(echo, workers=3, max_queue=10)
        jobs = [q.submit({"v": i}) for i in range(5)]
        await q.start()
        await q.drain()
        return jobs

    jobs = asyncio.run(go())
    assert [j.result for j in jobs] == [0, 1, 2, 3, 4]
    assert {j.status for j in jobs} == {"succeeded"}


def test_backpressure_refuses_when_full():
    async def go():
        q = JobQueue(echo, workers=1, max_queue=2)
        q.submit({"v": 1})
        q.submit({"v": 2})
        with pytest.raises(QueueFull):
            q.submit({"v": 3})
        assert q.depth() == 2

    asyncio.run(go())


def test_timeouts_and_failures_are_isolated():
    async def go():
        q = JobQueue(
            echo, workers=2, max_queue=5, job_timeout=0.05
        )
        slow = q.submit({"v": 1, "s": 1})
        bad = q.submit({"v": 2, "fail": True})
        good = q.submit({"v": 3})
        await q.start()
        await q.drain()
        return slow, bad, good

    slow, bad, good = asyncio.run(go())
    assert slow.status == "timed_out"
    assert bad.status == "failed" and "bad ticket" in bad.error
    assert good.status == "succeeded"


def test_concurrency_never_exceeds_worker_count():
    live = {"now": 0, "peak": 0}

    async def track(p):
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        await asyncio.sleep(0.01)
        live["now"] -= 1

    async def go():
        q = JobQueue(track, workers=2, max_queue=20)
        for _ in range(10):
            q.submit({})
        await q.start()
        await q.drain()

    asyncio.run(go())
    assert live["peak"] == 2
