"""Pattern 07: async job queue with backpressure.

Agent runs are slow and bursty, so requests are accepted onto
a bounded queue and processed by a fixed pool of workers.
When the queue is full, new work is refused immediately (the
HTTP layer maps this to 429 + Retry-After) instead of piling
up until everything times out. Each job has its own timeout.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

JobFn = Callable[[dict[str, Any]], Awaitable[Any]]


class QueueFull(Exception):
    """Refuse fast: the caller should retry later."""


@dataclass
class Job:
    id: str
    payload: dict[str, Any]
    status: str = "queued"  # running|succeeded|failed|timed_out
    result: Any = None
    error: str | None = None


class JobQueue:
    def __init__(
        self,
        handler: JobFn,
        workers: int = 4,
        max_queue: int = 100,
        job_timeout: float = 30.0,
    ) -> None:
        self.handler = handler
        self.n_workers = workers
        self.job_timeout = job_timeout
        self.q: asyncio.Queue[Job] = asyncio.Queue(max_queue)
        self.jobs: dict[str, Job] = {}
        self._ids = itertools.count(1)
        self._tasks: list[asyncio.Task] = []

    def submit(self, payload: dict[str, Any]) -> Job:
        if self.q.full():
            raise QueueFull(f"{self.q.qsize()} jobs waiting")
        job = Job(f"job-{next(self._ids)}", payload)
        self.q.put_nowait(job)
        self.jobs[job.id] = job
        return job

    def depth(self) -> int:
        return self.q.qsize()

    async def start(self) -> None:
        for i in range(self.n_workers):
            self._tasks.append(
                asyncio.create_task(self._worker(i))
            )

    async def drain(self) -> None:
        """Wait for queued work, then stop the workers."""
        await self.q.join()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _worker(self, n: int) -> None:
        while True:
            job = await self.q.get()
            job.status = "running"
            try:
                job.result = await asyncio.wait_for(
                    self.handler(job.payload), self.job_timeout
                )
                job.status = "succeeded"
            except TimeoutError:
                job.status = "timed_out"
                job.error = f"exceeded {self.job_timeout}s"
            except Exception as exc:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.q.task_done()
