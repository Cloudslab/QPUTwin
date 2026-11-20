"""Tests the asynchronous job queue and worker replicas.

Usage: uv run pytest tests/test_async_jobs.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import asyncio
import math
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from qputwin_core.devices import DEVICES, IONQ_FORTE
from qputwin_core.simulators import TrappedIonSimulator, simulator_for
from qputwin_orchestrator.app import create_app
from qputwin_orchestrator.registry import BackendEntry, BackendRegistry, OrchestratorSettings
from qputwin_worker.app import create_worker_app

from test_simulators import star_ghz

SLOW_S = 0.4


class Router(httpx.AsyncBaseTransport):
    """Route by host name to in-process worker apps, with optional per-host latency."""

    def __init__(self, apps: dict[str, object], delays: dict[str, float]) -> None:
        self.transports = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}
        self.delays = delays

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        t = self.transports.get(request.url.host)
        if t is None:
            raise httpx.ConnectError(f"connection refused by {request.url.host}", request=request)
        if request.url.path.endswith("/execute") and request.url.host in self.delays:
            await asyncio.sleep(self.delays[request.url.host])
        return await t.handle_async_request(request)


def _make_client(**settings) -> TestClient:
    forte = create_worker_app(IONQ_FORTE, TrappedIonSimulator, overrides=False)
    apps = {bid: create_worker_app(spec, type(simulator_for(spec)), overrides=False) for bid, spec in DEVICES.items()}
    apps |= {"slow-forte": forte, "rep-a": forte, "rep-b": forte, "slow-a": forte, "slow-b": forte}
    entries = {bid: BackendEntry(id=bid, url=f"http://{bid}") for bid in DEVICES}
    entries["slow"] = BackendEntry(id="slow", url="http://slow-forte")
    entries["replicated"] = BackendEntry(id="replicated", url="http://rep-a", urls=["http://rep-a", "http://rep-b"])
    entries["failover"] = BackendEntry(id="failover", url="http://dead", urls=["http://dead", "http://rep-a"])
    entries["slowpair"] = BackendEntry(id="slowpair", url="http://slow-a", urls=["http://slow-a", "http://slow-b"])
    reg = BackendRegistry(entries=entries, settings=OrchestratorSettings(**settings))
    delays = {"slow-forte": SLOW_S, "slow-a": SLOW_S, "slow-b": SLOW_S}
    return TestClient(create_app(reg, transport=Router(apps, delays)))


@pytest.fixture(scope="module")
def client():
    with _make_client() as c:
        yield c


def _identities(t: dict) -> None:
    b = t["breakdown"]
    assert math.isclose(t["total_roundtrip_ms"], t["orchestration_overhead_ms"] + b["worker_service_ms"],
                        abs_tol=1e-5)
    parts = (b["ingress_ms"] + b["validation_ms"] + b["routing_ms"] + b["orchestrator_queue_ms"]
             + b["dispatch_rtt_ms"] + b["response_assembly_ms"] + b["unattributed_ms"])
    assert math.isclose(parts, t["total_roundtrip_ms"], abs_tol=1e-5)
    if b["dispatch_rtt_ms"]:
        assert math.isclose(b["dispatch_rtt_ms"], b["network_transit_ms"] + b["worker_service_ms"], abs_tol=1e-5)
    assert all(v >= -1e-6 for v in b.values()), b


def _wait_for_status(c: TestClient, job_id: str, status: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = c.get(f"/v1/jobs/{job_id}").json()
        if body["status"] == status:
            return body
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {status}")


# =================================================================== async
def test_async_submit_poll_and_result(client: TestClient) -> None:
    r = client.post("/v1/jobs", params={"mode": "async"},
                    json={"backend": "ionq_forte", "qasm": star_ghz(6), "shots": 1000, "seed": 3})
    assert r.status_code == 202, r.text
    acc = r.json()
    assert acc["status"] == "QUEUED" and acc["execution_mode"] == "async"
    assert r.headers["location"] == f"/v1/jobs/{acc['job_id']}" == acc["links"]["self"]
    assert r.headers["x-qputwin-job-id"] == acc["job_id"]
    done = client.get(acc["links"]["self"], params={"wait": 10}).json()
    assert done["status"] == "COMPLETED" and done["http_status"] == 200
    assert set(done["result"]["counts"]) == {"000000", "111111"} and done["result"]["seed"] == 3
    t = done["telemetry"]
    assert t["execution_mode"] == "async" and t["schema_version"] == "1.2" and t["engine_invoked"]
    _identities(t)
    sync = client.post("/v1/jobs", json={"backend": "ionq_forte", "qasm": star_ghz(6), "shots": 1000, "seed": 3})
    assert sync.json()["result"]["counts"] == done["result"]["counts"]        # same physics, same seed
    assert sync.json()["telemetry"]["breakdown"]["orchestrator_queue_ms"] == 0.0
    # exactly one dataset record, written when the job finished
    recs = [x for x in client.get("/v1/telemetry", params={"limit": 1000}).json()["records"]
            if x["job_id"] == acc["job_id"]]
    assert len(recs) == 1 and recs[0]["execution_mode"] == "async" and recs[0]["status"] == "COMPLETED"
    assert "submit_response_ms" in recs[0] and "server_timing_total_ms" not in recs[0]


def test_prefer_respond_async_header(client: TestClient) -> None:
    r = client.post("/v1/jobs", headers={"prefer": "respond-async"},
                    json={"backend": "iqm_emerald", "qasm": star_ghz(2), "shots": 10})
    assert r.status_code == 202
    assert client.get(f"/v1/jobs/{r.json()['job_id']}", params={"wait": 10}).json()["status"] == "COMPLETED"


def test_async_validation_errors_are_immediate(client: TestClient) -> None:
    r = client.post("/v1/jobs", params={"mode": "async"},
                    json={"backend": "ionq_forte", "qasm": "OPENQASM 2.0;\nqreg q[2]\n"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "QASMSyntaxError"
    assert r.json()["telemetry"]["execution_mode"] == "async"
    recs = [x for x in client.get("/v1/telemetry", params={"limit": 1000}).json()["records"]
            if x["job_id"] == r.json()["job_id"]]
    assert len(recs) == 1 and recs[0]["status"] == "REJECTED"
    assert client.post("/v1/jobs", params={"mode": "async"},
                       json={"backend": "nope", "qasm": "OPENQASM 2.0;"}).status_code == 404


def test_async_worker_rejection_is_reported_in_the_job(client: TestClient) -> None:
    r = client.post("/v1/jobs", params={"mode": "async"},
                    json={"backend": "rigetti_ankaa_3", "qasm": star_ghz(12), "shots": 10})
    assert r.status_code == 202
    done = client.get(f"/v1/jobs/{r.json()['job_id']}", params={"wait": 10}).json()
    assert done["status"] == "REJECTED" and done["http_status"] == 422
    assert done["error"]["type"] == "TopologicalCompilationError"
    _identities(done["telemetry"])


def test_queueing_cancellation_and_state_errors(client: TestClient) -> None:
    job = {"backend": "slow", "qasm": star_ghz(3), "shots": 10}
    a = client.post("/v1/jobs", params={"mode": "async"}, json=job).json()["job_id"]
    _wait_for_status(client, a, "RUNNING")
    b = client.post("/v1/jobs", params={"mode": "async"}, json=job).json()
    c = client.post("/v1/jobs", params={"mode": "async"}, json=job).json()
    assert b["queue_position"] == 1 and c["queue_position"] == 2           # capacity: 1 replica × 1 slot
    queues = client.get("/v1/jobs", params={"backend": "slow"}).json()["queues"]["slow"]
    assert queues == {"queued": 2, "running": 1, "capacity": 1}
    cancelled = client.delete(f"/v1/jobs/{b['job_id']}")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "CANCELLED"
    assert client.get(f"/v1/jobs/{c['job_id']}").json()["queue_position"] == 1
    assert client.delete(f"/v1/jobs/{a}").status_code == 409                # running jobs cannot be cancelled
    assert client.delete(f"/v1/jobs/{b['job_id']}").json()["error"]["type"] == "JobStateError"
    assert client.get("/v1/jobs/does-not-exist").status_code == 404
    done = client.get(f"/v1/jobs/{c['job_id']}", params={"wait": 10}).json()
    assert done["status"] == "COMPLETED"
    # c waited for a: the wait shows up as orchestrator queue time, not worker queue time
    assert done["telemetry"]["breakdown"]["orchestrator_queue_ms"] > 0.5 * SLOW_S * 1000
    _identities(done["telemetry"])
    recs = {x["job_id"]: x for x in client.get("/v1/telemetry", params={"limit": 1000}).json()["records"]}
    assert recs[b["job_id"]]["status"] == "CANCELLED"
    statuses = {j["job_id"]: j["status"] for j in client.get("/v1/jobs").json()["jobs"]}
    assert statuses[b["job_id"]] == "CANCELLED" and statuses[a] == "COMPLETED"


def test_long_poll_returns_early_and_times_out(client: TestClient) -> None:
    jid = client.post("/v1/jobs", params={"mode": "async"},
                      json={"backend": "slow", "qasm": star_ghz(2), "shots": 5}).json()["job_id"]
    t0 = time.monotonic()
    pending = client.get(f"/v1/jobs/{jid}", params={"wait": 0.05})
    assert pending.json()["status"] in ("QUEUED", "RUNNING") and pending.headers["retry-after"] == "1"
    done = client.get(f"/v1/jobs/{jid}", params={"wait": 10}).json()
    assert done["status"] == "COMPLETED" and time.monotonic() - t0 < 5


def test_queue_limit_and_retention() -> None:
    with _make_client(async_max_queued=1, async_ttl_s=0.0) as c:
        job = {"backend": "slow", "qasm": star_ghz(2), "shots": 5}
        a = c.post("/v1/jobs", params={"mode": "async"}, json=job).json()["job_id"]
        _wait_for_status(c, a, "RUNNING")
        assert c.post("/v1/jobs", params={"mode": "async"}, json=job).status_code == 202
        full = c.post("/v1/jobs", params={"mode": "async"}, json=job)
        assert full.status_code == 503 and full.json()["error"]["type"] == "QueueFullError"
        assert c.get(f"/v1/jobs/{a}", params={"wait": 10}).json()["status"] == "COMPLETED"
        c.get("/v1/jobs")                                     # triggers pruning; TTL 0 expires finished jobs
        assert c.get(f"/v1/jobs/{a}").status_code == 404


# ================================================================ replicas
def test_least_in_flight_balancing_across_replicas(client: TestClient) -> None:
    for i in range(10):
        r = client.post("/v1/jobs", json={"backend": "replicated", "qasm": star_ghz(3), "shots": 10, "seed": i})
        assert r.status_code == 200
        assert r.json()["telemetry"]["environment"]["replica"] in ("http://rep-a", "http://rep-b")
    b = next(x for x in client.get("/v1/backends").json()["backends"] if x["id"] == "replicated")
    assert b["status"] == "online" and sorted(r["dispatched"] for r in b["replicas"]) == [5, 5]


def test_connection_failover_to_a_live_replica(client: TestClient) -> None:
    r = client.post("/v1/jobs", json={"backend": "failover", "qasm": star_ghz(3), "shots": 10})
    assert r.status_code == 200, r.text
    assert r.json()["telemetry"]["environment"]["replica"] == "http://rep-a"
    b = next(x for x in client.get("/v1/backends").json()["backends"] if x["id"] == "failover")
    dead = next(x for x in b["replicas"] if x["url"] == "http://dead")
    assert dead["status"] == "offline" and dead["failures"] >= 1 and b["status"] == "online"


def test_async_capacity_scales_with_replicas(client: TestClient) -> None:
    reg: BackendRegistry = client.app.state.registry          # type: ignore[attr-defined]
    assert reg.capacity("replicated") == 2 and reg.capacity("slow") == 1


def test_dns_discovery_expands_a_host_name() -> None:
    reg = BackendRegistry(entries={"x": BackendEntry(id="x", url="http://localhost:8001")},
                          settings=OrchestratorSettings(discovery="dns"))
    asyncio.run(reg.refresh_replicas(reg.entries["x"]))
    urls = [r.url for r in reg.entries["x"].replicas]
    assert "http://127.0.0.1:8001" in urls
    reg.entries["x"].urls = ["http://no-such-host.invalid:8001"]
    asyncio.run(reg.refresh_replicas(reg.entries["x"], force=True))
    assert [r.url for r in reg.entries["x"].replicas] == urls   # failed resolution keeps the last known set


def test_registry_reads_replica_lists(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "b.toml"
    cfg.write_text('[orchestrator]\ndiscovery = "static"\n[backends.a]\nurls = ["http://a1:1", "http://a2:1/"]\n'
                   '[backends.b]\nurl = "http://b:1"\n')
    monkeypatch.setenv("QPUTWIN_BACKEND_URL_B", "http://b1:1, http://b2:1")
    reg = BackendRegistry.from_config(cfg)
    assert [r.url for r in reg.entries["a"].replicas] == ["http://a1:1", "http://a2:1"]
    assert [r.url for r in reg.entries["b"].replicas] == ["http://b1:1", "http://b2:1"]
