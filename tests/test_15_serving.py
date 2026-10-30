import threading
import time

from fastapi.testclient import TestClient

from adp.serving import Settings, create_app


def client(system, **kw):
    app = create_app(Settings(**kw), system=system)
    return TestClient(app)


def test_health_ready_and_ask():
    with client(lambda t: f"echo {t}") as c:
        assert c.get("/healthz").json() == {"status": "alive"}
        assert c.get("/readyz").json()["status"] == "ready"
        r = c.post(
            "/v1/ask",
            json={"text": "hi"},
            headers={"x-request-id": "rid-1"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "echo hi"
        assert body["request_id"] == "rid-1"
        assert r.headers["x-request-id"] == "rid-1"


def test_input_validation():
    with client(lambda t: t) as c:
        assert (
            c.post("/v1/ask", json={"text": ""}).status_code
            == 422
        )
        assert (
            c.post(
                "/v1/ask", json={"text": "x" * 5000}
            ).status_code
            == 422
        )


def test_sync_timeout_returns_504():
    def slow(t):
        time.sleep(0.5)
        return "late"

    with client(slow, request_timeout=0.05) as c:
        assert (
            c.post("/v1/ask", json={"text": "x"}).status_code
            == 504
        )


def test_jobs_accept_complete_and_backpressure():
    gate = threading.Event()

    def blocked(t):
        gate.wait(2)
        return t.upper()

    with client(blocked, workers=1, max_queue=1) as c:
        ids = []
        codes = []
        for i in range(4):
            r = c.post("/v1/jobs", json={"text": f"j{i}"})
            codes.append(r.status_code)
            if r.status_code == 202:
                ids.append(r.json()["job_id"])
            time.sleep(0.05)
        assert 429 in codes
        refused = c.post("/v1/jobs", json={"text": "late"})
        assert refused.headers.get("retry-after") == "5"
        gate.set()
        deadline = time.time() + 3
        while time.time() < deadline:
            st = c.get(f"/v1/jobs/{ids[0]}").json()
            if st["status"] == "succeeded":
                break
            time.sleep(0.02)
        assert st["result"] == "J0"
        assert c.get("/v1/jobs/nope").status_code == 404
        m = c.get("/metrics").json()["counters"]
        assert m["jobs.refused"] >= 1


def test_reference_system_loads_from_settings():
    with TestClient(create_app(Settings())) as c:
        r = c.post(
            "/v1/ask", json={"text": "Where is order ORD-100001?"}
        )
        assert "shipped" in r.json()["answer"]
