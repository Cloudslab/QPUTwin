"""QPUTwin integration & benchmarking test-suite (runs against the live Docker stack).

    docker compose up -d --build
    uv run pytest test_benchmark.py -v -s

What is verified
----------------
1. A highly entangled 12-qubit star-GHZ circuit — interaction graph K_{1,11}, a hub
   of degree 11 that no square lattice can host — runs on both trapped-ion
   backends with exact GHZ statistics/amplitudes, and its ``telemetry`` object
   satisfies the additive identities the research dataset relies on.
2. The *same* circuit is mathematically rejected by both superconducting
   backends: the reported qubit pair has adjacency-matrix entry 0, as
   independently recomputed here from the device's published coupling map.
3. Control experiment: an equally entangled GHZ built as a nearest-neighbour
   chain is accepted by the superconducting backends — the rejection in (2) is
   topological, not a blanket refusal.
4. A complete-graph state (K_8, every pair interacts) matches its analytic
   amplitudes 2^{-n/2}(-1)^{C(|x|,2)} on trapped-ion hardware.
5. Per-container resource model: qubit caps default to the vendor sizes, and the
   memory guard rejects an infeasible 36-qubit dense statevector with a precise
   ``ResourceLimitError`` instead of crashing the worker.
6. Benchmark: N repetitions per backend; median / p95 of the three headline
   metrics are printed and cross-checked with ``/v1/telemetry/summary``.
7. Asynchronous execution: ``202`` + polling returns the same physics and
   telemetry obeying the same identities (plus ``orchestrator_queue_ms``).
8. Routing vs rejection: the star-GHZ that (2) rejects is SABRE-routed into a
   correct GHZ on both superconducting backends; the cost (SWAPs, latency) is
   printed next to the rejection latency.
9. Noise: the calibration-driven density-matrix engine degrades a Bell pair on
   every backend, consistently with the reported estimated success probability.
10. Gate-duration model: every job carries a hardware wall-clock estimate.
11. Horizontal scaling: when a backend runs several replicas (``.env``
    ``QPUTWIN_<DEVICE>_REPLICAS`` or ``--scale``), concurrent async jobs spread over
    distinct worker containers.
12. Wide noisy circuits run on the trajectory engine; coherent, crosstalk and
    time-correlated noise terms are applied; every result lists the calibration
    values it relied on that are not vendor-published.
13. Operations: ``/ready`` and the Prometheus ``/metrics`` endpoint.

Only ``pytest``, ``httpx`` and the standard library are used, so the suite runs
from a fresh ``uv sync`` (dev group) without the scientific extras.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import cmath
import math
import os
import statistics
import time

import httpx
import pytest

BASE_URL = os.environ.get("QPUTWIN_BASE_URL", "http://localhost:8000")
N_BENCH = int(os.environ.get("QPUTWIN_BENCH_REPS", "20"))
TRAPPED_ION = ("quantinuum_helios", "ionq_forte")
SUPERCONDUCTING = ("rigetti_ankaa_3", "iqm_emerald")
GHZ_N = 12

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------ circuits
def star_ghz(n: int) -> str:
    """|GHZ_n> with every CNOT fanned out from qubit 0 (star interaction graph)."""
    body = "".join(f"cx q[0],q[{k}];\n" for k in range(1, n))
    return (f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\ncreg c[{n}];\n'
            f"h q[0];\n{body}measure q -> c;\n")


def chain_ghz(n: int) -> str:
    """|GHZ_n> built along a 1-D nearest-neighbour chain q0-q1-...-q(n-1)."""
    body = "".join(f"cx q[{k}],q[{k + 1}];\n" for k in range(n - 1))
    return (f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\ncreg c[{n}];\n'
            f"h q[0];\n{body}measure q -> c;\n")


def complete_graph_state(n: int) -> str:
    """Graph state on K_n written in OpenQASM 3 (needs all-to-all connectivity)."""
    cz = "".join(f"cz q[{i}], q[{j}];\n" for i in range(n) for j in range(i + 1, n))
    return f'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[{n}] q;\nbit[{n}] c;\nh q;\n{cz}c = measure q;\n'


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session")
def api() -> httpx.Client:
    client = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(300.0, connect=5.0))
    try:
        health = client.get("/health")
    except httpx.HTTPError as exc:
        pytest.fail(f"QPUTwin orchestrator is not reachable at {BASE_URL} ({exc!r}). "
                    "Start the stack first: `docker compose up -d --build`.", pytrace=False)
    assert health.status_code == 200, health.text
    backends = {b["id"]: b for b in client.get("/v1/backends").json()["backends"]}
    offline = [b for b in (*TRAPPED_ION, *SUPERCONDUCTING) if backends.get(b, {}).get("status") != "online"]
    if offline:
        pytest.fail(f"backends not online: {offline}", pytrace=False)
    yield client
    client.close()


@pytest.fixture(scope="session")
def adjacency(api: httpx.Client) -> dict[str, list[list[int]]]:
    """Adjacency matrices rebuilt *independently* from each SC device's published edge list."""
    out: dict[str, list[list[int]]] = {}
    for b in SUPERCONDUCTING:
        dev = api.get(f"/v1/backends/{b}").json()
        n = dev["n_physical"]
        a = [[0] * n for _ in range(n)]
        for p, q in dev["coupling_map"]["edges"]:
            a[p][q] = a[q][p] = 1
        out[b] = a
    return out


def submit(api: httpx.Client, backend: str, qasm: str, **kw) -> httpx.Response:
    return api.post("/v1/jobs", json={"backend": backend, "qasm": qasm, **kw})


def assert_telemetry(t: dict, *, engine: bool) -> None:
    """Structural and additive-identity checks on one telemetry object."""
    for key in ("orchestration_overhead_ms", "qpu_compute_time_ms", "total_roundtrip_ms"):
        assert key in t, f"telemetry missing {key}"
        assert isinstance(t[key], float | int) and math.isfinite(t[key]), f"{key} not finite"
    b = t["breakdown"]
    assert t["schema_version"] == "1.2" and t["clock"] == "perf_counter_ns"
    assert t["engine_invoked"] is engine
    assert t["total_roundtrip_ms"] > 0 and t["orchestration_overhead_ms"] > 0
    # identity 1: total = orchestration overhead + worker service (by construction)
    assert math.isclose(t["total_roundtrip_ms"], t["orchestration_overhead_ms"] + b["worker_service_ms"],
                        abs_tol=1e-5)
    # identity 2: qpu compute = simulation + sampling
    assert math.isclose(t["qpu_compute_time_ms"], b["simulation_ms"] + b["sampling_ms"], abs_tol=1e-5)
    # identity 3: orchestrator spans partition the round trip
    parts = (b["ingress_ms"] + b["validation_ms"] + b["routing_ms"] + b["orchestrator_queue_ms"]
             + b["dispatch_rtt_ms"] + b["response_assembly_ms"] + b["unattributed_ms"])
    assert math.isclose(parts, t["total_roundtrip_ms"], abs_tol=1e-5)
    # identity 4: dispatch round trip = network transit + worker service
    assert math.isclose(b["dispatch_rtt_ms"], b["network_transit_ms"] + b["worker_service_ms"], abs_tol=1e-5)
    # ordering: engine ⊂ worker ⊂ round trip
    assert 0 <= t["qpu_compute_time_ms"] <= b["worker_service_ms"] <= t["total_roundtrip_ms"]
    assert all(v >= -1e-6 for v in b.values()), b
    if engine:
        assert t["qpu_compute_time_ms"] > 0 and b["simulation_ms"] > 0
    else:
        assert t["qpu_compute_time_ms"] == 0.0


# ============================================== 1. trapped ion: accept + telemetry
@pytest.mark.parametrize("backend", TRAPPED_ION)
def test_entangled_circuit_runs_on_trapped_ion_with_valid_telemetry(api: httpx.Client, backend: str) -> None:
    shots = 8192
    r = submit(api, backend, star_ghz(GHZ_N), shots=shots, seed=2026)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "COMPLETED" and body["backend"] == backend

    # --- physics: GHZ statistics -------------------------------------------
    counts = body["result"]["counts"]
    assert set(counts) == {"0" * GHZ_N, "1" * GHZ_N}, counts
    assert sum(counts.values()) == shots
    k = counts["1" * GHZ_N]
    assert abs(k - shots / 2) < 5 * math.sqrt(shots / 4), "GHZ outcomes not balanced (5σ)"

    # --- compilation: native gates and exactly one entangler per CNOT --------
    comp = body["compilation"]
    entangler = {"quantinuum_helios": "zzmax", "ionq_forte": "ms"}[backend]
    assert comp["native_gate_histogram"][entangler] == GHZ_N - 1
    assert comp["native_two_qubit_count"] == GHZ_N - 1 and comp["max_synthesis_error"] < 1e-12

    # --- telemetry ------------------------------------------------------------
    t = body["telemetry"]
    assert_telemetry(t, engine=True)
    w = t["workload"]
    assert w["simulated_qubits"] == GHZ_N and w["statevector_bytes"] == 16 * 2 ** GHZ_N
    assert w["execution_path"] == "terminal_sampling"
    assert t["environment"]["worker"]["vendor"] == backend
    server_total = float(r.headers["server-timing"].split(",")[0].split("dur=")[1])
    assert server_total >= t["total_roundtrip_ms"], "Server-Timing total must include serialisation"
    assert r.headers["x-qputwin-job-id"] == body["job_id"]
    recent = api.get("/v1/telemetry", params={"limit": 50, "backend": backend}).json()["records"]
    assert any(rec["job_id"] == body["job_id"] for rec in recent), "job missing from telemetry dataset"
    print(f"\n[{backend}] total={t['total_roundtrip_ms']:.3f} ms  "
          f"orchestration={t['orchestration_overhead_ms']:.3f} ms  qpu={t['qpu_compute_time_ms']:.3f} ms")


@pytest.mark.parametrize("backend", TRAPPED_ION)
def test_ghz_statevector_is_exact(api: httpx.Client, backend: str) -> None:
    r = submit(api, backend, star_ghz(GHZ_N), shots=1, seed=1, options={"return_statevector": True})
    assert r.status_code == 200, r.text
    sv = r.json()["result"]["statevector"]
    amps = [complex(re, im) for re, im in sv]
    assert len(amps) == 2 ** GHZ_N
    for idx, a in enumerate(amps):
        expected = 1 / math.sqrt(2) if idx in (0, 2 ** GHZ_N - 1) else 0.0
        assert abs(a - expected) < 1e-9, (idx, a)          # phase-exact, not just |a|


@pytest.mark.parametrize("backend", TRAPPED_ION)
def test_complete_graph_state_amplitudes(api: httpx.Client, backend: str) -> None:
    n = 8
    r = submit(api, backend, complete_graph_state(n), shots=1, seed=3, options={"return_statevector": True})
    assert r.status_code == 200, r.text
    amps = [complex(re, im) for re, im in r.json()["result"]["statevector"]]
    for x, a in enumerate(amps):
        w = bin(x).count("1")
        expected = (-1) ** (w * (w - 1) // 2) / math.sqrt(2 ** n)
        assert abs(a - expected) < 1e-9, (x, a, expected)
    assert r.json()["compilation"]["native_two_qubit_count"] == n * (n - 1) // 2


# ======================================== 2. superconducting: mathematical rejection
@pytest.mark.parametrize("backend", SUPERCONDUCTING)
def test_same_circuit_rejected_by_superconducting_adjacency(api: httpx.Client, backend: str,
                                                            adjacency: dict[str, list[list[int]]]) -> None:
    qasm = star_ghz(GHZ_N)
    r = submit(api, backend, qasm, shots=1024, seed=2026)
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["status"] == "REJECTED"
    err = body["error"]
    assert err["type"] == "TopologicalCompilationError"
    d = err["details"]
    p, q = d["qubits"]
    a = adjacency[backend]

    # The rejection is grounded in the adjacency matrix: A[p][q] == 0 ...
    assert a[p][q] == 0 and d["adjacency_entry"] == 0
    assert q not in d["neighbors"][str(p)]
    # ... and it is the *first* uncoupled CNOT of the program, recomputed independently.
    first_bad = next(k for k in range(1, GHZ_N) if a[0][k] == 0)
    assert (p, q) == (0, first_bad)
    assert d["gate"] == "cx" and d["statement"] == f"cx q[0],q[{first_bad}];"
    assert d["line"] == 5 + first_bad                     # header(2) + qreg + creg + h, then cx lines
    # every star edge the grid *does* host is genuinely nearest-neighbour
    assert all(a[0][k] == 1 for k in range(1, first_bad))

    t = body["telemetry"]
    assert_telemetry(t, engine=False)
    assert t["breakdown"]["worker_service_ms"] > 0, "rejection must come from the emulator itself"
    assert t["breakdown"]["worker_compile_ms"] > 0


# ============================================================ 3. control experiment
@pytest.mark.parametrize("backend", SUPERCONDUCTING)
def test_nearest_neighbour_chain_is_accepted(api: httpx.Client, backend: str) -> None:
    n = 9                                                 # fits in one row of both lattices
    r = submit(api, backend, chain_ghz(n), shots=4096, seed=7)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["result"]["counts"]) == {"0" * n, "1" * n}
    assert body["compilation"]["native_gate_histogram"]["cz"] == n - 1
    assert_telemetry(body["telemetry"], engine=True)


# ============================================================ 4. resource model (D6)
def test_per_container_caps_and_memory_guard(api: httpx.Client) -> None:
    caps = {b["id"]: b["max_sim_qubits"] for b in api.get("/v1/backends").json()["backends"]}
    assert caps == {"quantinuum_helios": 98, "ionq_forte": 36, "rigetti_ankaa_3": 84, "iqm_emerald": 54}
    r = submit(api, "ionq_forte", star_ghz(36), shots=10)   # 36 qubits: ~3 TiB working set
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["type"] == "ResourceLimitError"
    d = err["details"]
    assert d["simulated_qubits"] == 36 and d["required_bytes"] > d["available_bytes"]
    assert 10 <= d["max_feasible_qubits"] < 36
    assert_telemetry(r.json()["telemetry"], engine=False)
    # the worker survived and still serves jobs
    assert submit(api, "ionq_forte", star_ghz(4), shots=8).status_code == 200


def test_strict_native_mode_over_http(api: httpx.Client) -> None:
    natives = ('OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\ngpi2(0) q[0];\nms(0,0,pi/2) q[0],q[1];\n'
               "measure q -> c;\n")
    ok = submit(api, "ionq_forte", natives, shots=100, options={"compilation": "strict"})
    assert ok.status_code == 200 and ok.json()["compilation"]["mode"] == "strict"
    bad = submit(api, "ionq_forte", star_ghz(3), shots=100, options={"compilation": "strict"})
    assert bad.status_code == 422 and bad.json()["error"]["type"] == "NativeGateError"


# ================================================================== 5. benchmark
def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))]


def test_benchmark_latency_decomposition(api: httpx.Client) -> None:
    workloads = {b: star_ghz(GHZ_N) for b in TRAPPED_ION} | {b: chain_ghz(9) for b in SUPERCONDUCTING}
    rows = []
    for backend, qasm in workloads.items():
        submit(api, backend, qasm, shots=1024, seed=0)       # warm-up (connection pool, caches)
        samples: dict[str, list[float]] = {"total": [], "orch": [], "qpu": []}
        for i in range(N_BENCH):
            r = submit(api, backend, qasm, shots=1024, seed=i)
            assert r.status_code == 200, r.text
            t = r.json()["telemetry"]
            assert_telemetry(t, engine=True)
            samples["total"].append(t["total_roundtrip_ms"])
            samples["orch"].append(t["orchestration_overhead_ms"])
            samples["qpu"].append(t["qpu_compute_time_ms"])
        row = {k: (statistics.median(v), _pct(v, 0.95)) for k, v in samples.items()}
        rows.append((backend, row))
        assert row["total"][0] < 5_000, f"{backend}: implausible median latency"

    print(f"\n{'backend':<20}{'total med/p95 (ms)':>24}{'orchestration med/p95':>26}{'qpu med/p95':>22}")
    for backend, row in rows:
        fmt = lambda k: f"{row[k][0]:8.3f} / {row[k][1]:8.3f}"  # noqa: E731
        print(f"{backend:<20}{fmt('total'):>24}{fmt('orch'):>26}{fmt('qpu'):>22}")

    summary = api.get("/v1/telemetry/summary").json()["summary"]
    for backend in workloads:
        s = summary[backend]
        assert s["completed"] >= N_BENCH
        assert s["metrics"]["total_roundtrip_ms"]["median"] > 0


def test_width_scaling_on_helios(api: httpx.Client) -> None:
    """QPU compute time must grow with simulated width (dense statevector ∝ 2^n)."""
    times = {}
    for n in (4, 12, 20):
        r = submit(api, "quantinuum_helios", star_ghz(n), shots=1024, seed=n)
        assert r.status_code == 200, r.text
        times[n] = r.json()["telemetry"]["qpu_compute_time_ms"]
    print(f"\nHelios star-GHZ qpu_compute_time_ms by width: {times}")
    assert times[20] > times[4]


def test_seed_reproducibility_over_http(api: httpx.Client) -> None:
    qasm = complete_graph_state(6)
    a = submit(api, "quantinuum_helios", qasm, shots=500, seed=99).json()["result"]["counts"]
    b = submit(api, "quantinuum_helios", qasm, shots=500, seed=99).json()["result"]["counts"]
    assert a == b
    # unseeded jobs draw and echo a seed, which reproduces the run
    first = submit(api, "ionq_forte", qasm, shots=500).json()["result"]
    again = submit(api, "ionq_forte", qasm, shots=500, seed=first["seed"]).json()["result"]
    assert first["counts"] == again["counts"]


def test_phase_is_reported_consistently_across_vendors(api: httpx.Client) -> None:
    """All four emulators reproduce the same exact statevector for a NN-compatible circuit."""
    qasm = ('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'
            "rx(0.3) q[0];\nry(1.1) q[1];\ncx q[0], q[1];\nrz(0.7) q[1];\nh q[0];\n")
    vecs = {}
    for b in (*TRAPPED_ION, *SUPERCONDUCTING):
        r = submit(api, b, qasm, shots=1, seed=0, options={"return_statevector": True})
        assert r.status_code == 200, r.text
        vecs[b] = [complex(re, im) for re, im in r.json()["result"]["statevector"]]
    ref = vecs["quantinuum_helios"]
    for b, v in vecs.items():
        assert max(abs(x - y) for x, y in zip(v, ref, strict=True)) < 1e-9, b
    assert abs(sum(abs(x) ** 2 for x in ref) - 1) < 1e-12
    assert all(cmath.isfinite(x) for x in ref)


# ======================================================== 7. asynchronous jobs
def poll(api: httpx.Client, job_id: str, timeout_s: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        body = api.get(f"/v1/jobs/{job_id}", params={"wait": 10}).json()
        if body["status"] not in ("QUEUED", "RUNNING"):
            return body
        assert time.monotonic() < deadline, f"job {job_id} still {body['status']}"


@pytest.mark.parametrize("backend", (*TRAPPED_ION, *SUPERCONDUCTING))
def test_async_job_matches_sync_physics_and_telemetry(api: httpx.Client, backend: str) -> None:
    qasm = star_ghz(8) if backend in TRAPPED_ION else chain_ghz(8)
    r = api.post("/v1/jobs", params={"mode": "async"},
                 json={"backend": backend, "qasm": qasm, "shots": 2048, "seed": 5})
    assert r.status_code == 202, r.text
    assert r.headers["location"] == f"/v1/jobs/{r.json()['job_id']}"
    done = poll(api, r.json()["job_id"])
    assert done["status"] == "COMPLETED" and done["http_status"] == 200
    sync = submit(api, backend, qasm, shots=2048, seed=5).json()
    assert done["result"]["counts"] == sync["result"]["counts"]
    t = done["telemetry"]
    assert t["execution_mode"] == "async" and t["breakdown"]["orchestrator_queue_ms"] >= 0
    assert_telemetry(t, engine=True)


def test_async_cancel_and_queue_accounting(api: httpx.Client) -> None:
    heavy = star_ghz(22)                                  # ~0.1-1 s each on one worker
    ids = [api.post("/v1/jobs", params={"mode": "async"},
                    json={"backend": "ionq_forte", "qasm": heavy, "shots": 100, "seed": k}).json()["job_id"]
           for k in range(4)]
    cancelled = api.delete(f"/v1/jobs/{ids[-1]}")
    if cancelled.status_code == 200:                       # still queued: cancellation must succeed
        assert cancelled.json()["status"] == "CANCELLED"
    else:                                                  # already dispatched: illegal transition
        assert cancelled.status_code == 409
    finished = [poll(api, j) for j in ids[:-1]]
    assert all(f["status"] == "COMPLETED" for f in finished)
    waits = [f["telemetry"]["breakdown"]["orchestrator_queue_ms"] for f in finished]
    print(f"\nasync orchestrator_queue_ms for 3 back-to-back 22-qubit jobs: {[round(w, 2) for w in waits]}")


# ==================================================== 8. routing vs rejection
@pytest.mark.parametrize("backend", SUPERCONDUCTING)
def test_routing_versus_rejection(api: httpx.Client, backend: str) -> None:
    qasm = star_ghz(GHZ_N)
    rejected = submit(api, backend, qasm, shots=1024, seed=2026)
    assert rejected.status_code == 422
    routed = submit(api, backend, qasm, shots=1024, seed=2026, options={"routing": {"method": "sabre"}})
    assert routed.status_code == 200, routed.text
    body = routed.json()
    assert set(body["result"]["counts"]) == {"0" * GHZ_N, "1" * GHZ_N}
    st = body["compilation"]["routing"]
    assert st["swaps_inserted"] > 0
    assert body["compilation"]["native_two_qubit_count"] == st["routed_two_qubit_gates"] \
        == GHZ_N - 1 + st["added_cx"] - st["cancelled_two_qubit_gates"]
    t = body["telemetry"]
    assert_telemetry(t, engine=True)
    assert t["breakdown"]["worker_route_ms"] > 0
    print(f"\n[{backend}] reject: {rejected.json()['telemetry']['total_roundtrip_ms']:.3f} ms | "
          f"route: {st['swaps_inserted']} SWAPs (+{st['added_cx']} CX), route {t['breakdown']['worker_route_ms']:.3f} ms, "
          f"total {t['total_roundtrip_ms']:.3f} ms, est. hardware time "
          f"{body['compilation']['hardware_time_estimate']['circuit_duration_us']:.3f} us/shot")


# ===================================================================== 9. noise
@pytest.mark.parametrize("backend", (*TRAPPED_ION, *SUPERCONDUCTING))
def test_calibrated_noise_degrades_a_bell_pair(api: httpx.Client, backend: str) -> None:
    shots = 20_000
    bell = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'
    r = submit(api, backend, bell, shots=shots, seed=11, options={"noise": {"model": "device"}})
    assert r.status_code == 200, r.text
    body = r.json()
    noise, counts = body["result"]["noise"], body["result"]["counts"]
    assert noise["engine"] == "density_matrix" and 0.5 < noise["purity"] < 1.0
    good = (counts.get("00", 0) + counts.get("11", 0)) / shots
    esp = noise["estimated_success_probability"]
    assert 0.8 < good < 1.0 and 0.8 < esp < 1.0
    assert abs(good - esp) < 0.05, (good, esp)          # ESP is a first-order proxy for the parity success
    assert body["telemetry"]["workload"]["engine"] == "density_matrix"
    assert_telemetry(body["telemetry"], engine=True)
    print(f"\n[{backend}] Bell parity success {good:.4f}  ESP {esp:.4f}  purity {noise['purity']:.4f}")


# ====================================================== 10. gate-duration model
def test_hardware_time_estimates(api: httpx.Client) -> None:
    est = {}
    for b in (*TRAPPED_ION, *SUPERCONDUCTING):
        r = submit(api, b, chain_ghz(5), shots=1000, seed=1).json()
        e = r["compilation"]["hardware_time_estimate"]
        assert e["complete"] and e["circuit_duration_us"] > 0
        assert math.isclose(e["estimated_job_duration_s"], 1000 * e["circuit_duration_us"] * 1e-6, rel_tol=1e-9)
        est[b] = e["circuit_duration_us"]
    print(f"\nestimated circuit duration per shot (us): {est}")
    assert min(est[b] for b in TRAPPED_ION) > 100 * max(est[b] for b in SUPERCONDUCTING)


# ================================================== 11. horizontal scaling
def test_replicas_share_the_async_load(api: httpx.Client) -> None:
    backends = {b["id"]: b for b in api.get("/v1/backends").json()["backends"]}
    scaled = [b for b, v in backends.items() if len(v.get("replicas") or []) >= 2]
    if not scaled:
        pytest.skip("no backend runs >= 2 replicas; set QPUTWIN_<DEVICE>_REPLICAS=2 in .env to exercise scaling")
    backend = scaled[0]
    qasm = star_ghz(20) if backend in TRAPPED_ION else chain_ghz(12)
    ids = [api.post("/v1/jobs", params={"mode": "async"},
                    json={"backend": backend, "qasm": qasm, "shots": 64, "seed": k}).json()["job_id"]
           for k in range(8)]
    done = [poll(api, j) for j in ids]
    assert all(d["status"] == "COMPLETED" for d in done)
    workers = {d["telemetry"]["environment"]["worker"]["worker_id"] for d in done}
    assert len(workers) >= 2, f"all jobs ran on one replica: {workers}"
    print(f"\n[{backend}] {len(backends[backend]['replicas'])} replicas; 8 async jobs served by {len(workers)} workers")


# ====================================================== 12. extended noise
def test_wide_noisy_circuit_uses_trajectories(api: httpx.Client) -> None:
    r = submit(api, "quantinuum_helios", chain_ghz(16), shots=256, seed=3, options={"noise": {"model": "device"}})
    assert r.status_code == 200, r.text
    body = r.json()
    noise = body["result"]["noise"]
    assert body["telemetry"]["workload"]["engine"] == "trajectory" and noise["exact_sampling"]
    good = sum(v for k, v in body["result"]["counts"].items() if k in ("0" * 16, "1" * 16)) / 256
    assert 0.8 < good < 1.0
    assert_telemetry(body["telemetry"], engine=True)


def test_extended_noise_terms_and_provenance(api: httpx.Client) -> None:
    opts = {"noise": {"model": "device", "overrotation": 0.02, "crosstalk_zz_rad": 0.05,
                      "quasi_static_khz": 20.0, "leakage_rate": 0.001}}
    r = submit(api, "iqm_emerald", chain_ghz(5), shots=500, seed=4, options=opts)
    assert r.status_code == 200, r.text
    noise = r.json()["result"]["noise"]
    assert {"coherent", "crosstalk", "quasi_static", "leakage"} <= set(noise["features"])
    assert noise["engine"] == "trajectory"
    assert "gates.cz.fidelity" in noise["parameters_used"]["published"]
    assert "gates.cz.duration_ns" in noise["assumed_parameters"]
    bad = submit(api, "iqm_emerald", chain_ghz(3), shots=10, options={**opts, "engine": "density_matrix"})
    assert bad.status_code == 400 and "trajectory" in bad.json()["error"]["message"]


# ========================================================== 13. operations
def test_ready_and_metrics(api: httpx.Client) -> None:
    ready = api.get("/ready")
    assert ready.status_code == 200 and ready.json()["job_store"] == "ok"
    assert len(ready.json()["backends_reachable"]) == 4
    text = api.get("/metrics").text
    assert "# TYPE qputwin_jobs_total counter" in text and "qputwin_job_roundtrip_ms_bucket" in text
    assert 'qputwin_async_capacity{backend="ionq_forte"}' in text
