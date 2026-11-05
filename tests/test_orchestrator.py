"""Orchestrator + workers wired in-process (httpx ASGI transport, no Docker).

Verifies routing, error mapping and — most importantly — the additive telemetry
identities that the research dataset relies on.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from qputwin_core.devices import DEVICES
from qputwin_core.simulators import simulator_for
from qputwin_orchestrator.app import create_app
from qputwin_orchestrator.registry import BackendEntry, BackendRegistry, OrchestratorSettings
from qputwin_worker.app import create_worker_app

from test_simulators import chain_ghz, star_ghz


class HostRouter(httpx.AsyncBaseTransport):
    """Route requests to in-process ASGI worker apps by hostname."""

    def __init__(self, apps: dict[str, object]) -> None:
        self.transports = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        t = self.transports.get(request.url.host)
        if t is None:
            raise httpx.ConnectError(f"no route to {request.url.host}", request=request)
        return await t.handle_async_request(request)


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    apps = {bid: create_worker_app(spec, type(simulator_for(spec))) for bid, spec in DEVICES.items()}
    entries = {bid: BackendEntry(id=bid, url=f"http://{bid}") for bid in DEVICES}
    entries["ghost"] = BackendEntry(id="ghost", url="http://ghost")
    path = tmp_path_factory.mktemp("telemetry") / "telemetry.jsonl"
    reg = BackendRegistry(entries=entries, settings=OrchestratorSettings(telemetry_path=str(path)))
    app = create_app(reg, transport=HostRouter(apps))
    with TestClient(app) as c:
        c.telemetry_path = path  # type: ignore[attr-defined]
        yield c


def _check_identities(t: dict) -> None:
    b = t["breakdown"]
    assert math.isclose(t["total_roundtrip_ms"], t["orchestration_overhead_ms"] + b["worker_service_ms"],
                        abs_tol=1e-5)
    assert math.isclose(t["qpu_compute_time_ms"], b["simulation_ms"] + b["sampling_ms"], abs_tol=1e-5)
    parts = (b["ingress_ms"] + b["validation_ms"] + b["routing_ms"] + b["orchestrator_queue_ms"]
             + b["dispatch_rtt_ms"] + b["response_assembly_ms"] + b["unattributed_ms"])
    assert math.isclose(parts, t["total_roundtrip_ms"], abs_tol=1e-5)
    if b["dispatch_rtt_ms"]:
        assert math.isclose(b["dispatch_rtt_ms"], b["network_transit_ms"] + b["worker_service_ms"], abs_tol=1e-5)
    for v in b.values():
        assert v >= -1e-6


def test_meta_endpoints(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    root = client.get("/").json()
    assert root["contact"] == {"name": "Koustav Kumar Mondal", "email": "erkoustavkumar@gmail.com"}
    backends = {b["id"]: b for b in client.get("/v1/backends").json()["backends"]}
    assert backends["ionq_forte"]["status"] == "online" and backends["ghost"]["status"] == "offline"
    assert backends["rigetti_ankaa_3"]["n_physical"] == 84
    dev = client.get("/v1/backends/iqm_emerald").json()
    assert dev["coupling_map"]["kind"] == "grid" and dev["emulator"]["max_sim_qubits"] == 54


def test_completed_job_and_telemetry(client: TestClient) -> None:
    r = client.post("/v1/jobs", json={"backend": "quantinuum_helios", "qasm": star_ghz(10), "shots": 2048,
                                      "seed": 11})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "COMPLETED" and set(body["result"]["counts"]) == {"0" * 10, "1" * 10}
    assert body["result"]["seed"] == 11
    t = body["telemetry"]
    assert t["engine_invoked"] and t["qpu_compute_time_ms"] > 0
    _check_identities(t)
    assert t["qpu_compute_time_ms"] <= t["breakdown"]["worker_service_ms"] <= t["total_roundtrip_ms"]
    assert "total;dur=" in r.headers["server-timing"] and r.headers["x-qputwin-job-id"] == body["job_id"]
    assert body["compilation"]["native_gate_histogram"]["zzmax"] == 9
    assert t["schema_version"] == "1.2" and t["execution_mode"] == "sync"
    assert t["breakdown"]["orchestrator_queue_ms"] == 0.0 and t["breakdown"]["worker_route_ms"] == 0.0
    assert t["environment"]["replica"] == "http://quantinuum_helios"


def test_noise_routing_and_duration_estimate_over_http(client: TestClient) -> None:
    r = client.post("/v1/jobs", json={
        "backend": "iqm_emerald", "qasm": star_ghz(6), "shots": 2000, "seed": 4,
        "options": {"routing": {"method": "sabre"}, "noise": {"model": "device", "scale": 1.0}}})
    assert r.status_code == 200, r.text
    body = r.json()
    comp, res, t = body["compilation"], body["result"], body["telemetry"]
    assert comp["routing"]["swaps_inserted"] > 0 and comp["hardware_time_estimate"]["circuit_duration_us"] > 0
    assert res["noise"]["engine"] == "density_matrix" and 0 < res["noise"]["purity"] < 1
    assert len(res["counts"]) > 2                                   # noise populates non-GHZ outcomes
    assert t["workload"]["engine"] == "density_matrix" and t["workload"]["noisy"] and t["workload"]["routed"]
    assert t["breakdown"]["worker_route_ms"] > 0
    _check_identities(t)
    bad = client.post("/v1/jobs", json={"backend": "iqm_emerald", "qasm": star_ghz(2),
                                        "options": {"noise": {"scale": -1}}})
    assert bad.status_code == 400 and bad.json()["error"]["type"] == "InvalidRequestError"
    bad = client.post("/v1/jobs", json={"backend": "iqm_emerald", "qasm": star_ghz(2),
                                        "options": {"routing": {"method": "sabre"}, "compilation": "strict"}})
    assert bad.status_code == 400


def test_topological_rejection_carries_telemetry(client: TestClient) -> None:
    r = client.post("/v1/jobs", json={"backend": "rigetti_ankaa_3", "qasm": star_ghz(12), "shots": 100})
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "REJECTED" and body["error"]["type"] == "TopologicalCompilationError"
    t = body["telemetry"]
    assert t["engine_invoked"] is False and t["qpu_compute_time_ms"] == 0.0
    assert t["breakdown"]["worker_service_ms"] > 0
    _check_identities(t)


def test_chain_accepted_on_superconducting(client: TestClient) -> None:
    r = client.post("/v1/jobs", json={"backend": "iqm_emerald", "qasm": chain_ghz(9), "shots": 500, "seed": 1})
    assert r.status_code == 200 and set(r.json()["result"]["counts"]) == {"0" * 9, "1" * 9}


@pytest.mark.parametrize("payload,status,etype", [
    ({"backend": "nope", "qasm": "OPENQASM 2.0;"}, 404, "UnknownBackendError"),
    ({"backend": "ionq_forte", "qasm": "OPENQASM 2.0;\nqreg q[2]\n"}, 400, "QASMSyntaxError"),
    ({"backend": "ionq_forte", "qasm": "OPENQASM 2.0;\nqreg q[40];\n"}, 422, "ResourceLimitError"),
    ({"backend": "ionq_forte", "qasm": "x", "shots": 0}, 400, "InvalidRequestError"),
    ({"backend": "ghost", "qasm": "OPENQASM 2.0;"}, 503, "BackendUnavailableError"),
    ({"backend": "ionq_forte", "qasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n',
      "options": {"compilation": "strict"}}, 422, "NativeGateError"),
])
def test_error_mapping(client: TestClient, payload: dict, status: int, etype: str) -> None:
    r = client.post("/v1/jobs", json=payload)
    assert r.status_code == status, r.text
    body = r.json()
    assert body["error"]["type"] == etype
    assert body["telemetry"]["engine_invoked"] is False
    _check_identities(body["telemetry"])


def test_validation_rejects_before_dispatch(client: TestClient) -> None:
    r = client.post("/v1/jobs", json={"backend": "ionq_forte", "qasm": "OPENQASM 2.0;\nqreg q[2]\n"})
    b = r.json()["telemetry"]["breakdown"]
    assert b["dispatch_rtt_ms"] == 0 and b["worker_service_ms"] == 0


def test_telemetry_store_and_dataset(client: TestClient) -> None:
    for _ in range(3):
        client.post("/v1/jobs", json={"backend": "ionq_forte", "qasm": star_ghz(4), "shots": 10})
    recent = client.get("/v1/telemetry", params={"limit": 2, "backend": "ionq_forte"}).json()["records"]
    assert len(recent) == 2 and all(r["backend"] == "ionq_forte" for r in recent)
    assert recent[-1]["server_timing_total_ms"] >= recent[-1]["telemetry"]["total_roundtrip_ms"]
    summary = client.get("/v1/telemetry/summary").json()["summary"]
    assert summary["ionq_forte"]["completed"] >= 3
    assert summary["ionq_forte"]["metrics"]["total_roundtrip_ms"]["median"] > 0
    lines = Path(client.telemetry_path).read_text().splitlines()  # type: ignore[attr-defined]
    assert lines and all("telemetry" in json.loads(line) for line in lines)
