"""Pattern 15: serving an agent as a production service.

* configuration from the environment, validated at start-up
* liveness (/healthz) separate from readiness (/readyz)
* synchronous answers with a hard timeout (/v1/ask)
* slow work through a bounded queue (/v1/jobs) that returns
  429 + Retry-After when full
* a request id on every response, JSON logs, basic metrics
* release management (pattern 16) decides which variant serves
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from adp.observability.tracing import Metrics
from adp.queue.worker import JobQueue, QueueFull
from adp.release.manager import ReleaseManager, Variant

System = Callable[[str], str]
log = logging.getLogger("adp.serving")


@dataclass(frozen=True)
class Settings:
    system: str = "support_system:build_system"
    prompt_version: str = "support-v1"
    model: str = "fake-1"
    request_timeout: float = 10.0
    workers: int = 4
    max_queue: int = 100
    job_timeout: float = 60.0

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        s = cls(
            system=e("ADP_SYSTEM", cls.system),
            prompt_version=e("ADP_PROMPT_VERSION",
                             cls.prompt_version),
            model=e("ADP_MODEL", cls.model),
            request_timeout=float(e("ADP_REQUEST_TIMEOUT", 10)),
            workers=int(e("ADP_WORKERS", 4)),
            max_queue=int(e("ADP_MAX_QUEUE", 100)),
            job_timeout=float(e("ADP_JOB_TIMEOUT", 60)),
        )
        if s.workers < 1 or s.max_queue < 1:
            raise ValueError("workers and max_queue must be >= 1")
        return s


def load_system(spec: str) -> System:
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)()


class AskIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class AskOut(BaseModel):
    answer: str
    variant: str
    request_id: str


def create_app(settings: Settings | None = None,
               system: System | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    metrics = Metrics()
    state: dict = {"ready": False}

    async def handle(payload: dict) -> str:
        return await asyncio.to_thread(state["system"],
                                       payload["text"])

    queue = JobQueue(handle, cfg.workers, cfg.max_queue,
                     cfg.job_timeout)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["system"] = system or load_system(cfg.system)
        state["release"] = ReleaseManager(
            Variant(cfg.prompt_version, cfg.model))
        await queue.start()
        state["ready"] = True
        yield
        state["ready"] = False
        await queue.drain()

    app = FastAPI(title="agent-service", lifespan=lifespan)
    app.state.queue = queue
    app.state.metrics = metrics
    app.state.state = state

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or \
            uuid.uuid4().hex
        request.state.rid = rid
        t0 = time.perf_counter()
        resp = await call_next(request)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        resp.headers["x-request-id"] = rid
        metrics.inc(f"http.{resp.status_code}")
        metrics.observe("http.latency_ms", ms)
        log.info(json.dumps({"rid": rid,
                             "path": request.url.path,
                             "status": resp.status_code,
                             "ms": ms}))
        return resp

    @app.get("/healthz")
    async def healthz():
        return {"status": "alive"}

    @app.get("/readyz")
    async def readyz():
        if not state["ready"]:
            raise HTTPException(503, "not ready")
        return {"status": "ready",
                "queue_depth": queue.depth()}

    @app.post("/v1/ask", response_model=AskOut)
    async def ask(body: AskIn, request: Request):
        rid = request.state.rid
        rm: ReleaseManager = state["release"]
        v = rm.choose(rid)
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(state["system"], body.text),
                cfg.request_timeout)
        except TimeoutError:
            metrics.inc("ask.timeouts")
            raise HTTPException(504, "agent timed out") from None
        return AskOut(answer=answer, variant=v.id,
                      request_id=rid)

    @app.post("/v1/jobs", status_code=202)
    async def submit(body: AskIn):
        try:
            job = queue.submit({"text": body.text})
        except QueueFull:
            metrics.inc("jobs.refused")
            return JSONResponse(
                {"detail": "busy, retry later"}, status_code=429,
                headers={"Retry-After": "5"})
        return {"job_id": job.id, "status": job.status}

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str):
        job = queue.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return {"job_id": job.id, "status": job.status,
                "result": job.result, "error": job.error}

    @app.get("/metrics")
    async def get_metrics():
        return {"counters": dict(metrics.counters),
                "latency_p95_ms":
                metrics.percentile("http.latency_ms", 95)}

    return app
