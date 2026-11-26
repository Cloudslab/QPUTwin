"""Runs the QPUTwin benchmark experiments and saves their results.

Usage: uv run --group bench python benchmarks/run_benchmarks.py all

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")       # same BLAS threading as the worker containers
os.environ.setdefault("QPUTWIN_ENGINE_THREADS", "1")  # in-process experiments are per-core; the live stack uses its quota

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from qputwin_core import __version__  # noqa: E402
from qputwin_core.density import apply_readout_model  # noqa: E402
from qputwin_core.devices import DEVICES  # noqa: E402
from qputwin_core.errors import ResourceLimitError  # noqa: E402
from qputwin_core.noise import NoiseModel, NoiseOptions  # noqa: E402
from qputwin_core.simulators import ExecutionOptions, SimulatorConfig, simulator_for  # noqa: E402
from qputwin_core.timing import SpanTimer  # noqa: E402

import circuits as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
BASE_URL = os.environ.get("QPUTWIN_BASE_URL", "http://localhost:8000")
H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
BACKENDS = tuple(DEVICES)           # all seven twins, in registry order


# ================================================================ circuits
_PATHS: dict[str, list[int]] = {}


def _long_path(cmap: Any, budget: int = 200_000) -> list[int]:
    """A long simple path through the coupling graph, deterministic.

    Row-major lattices with every coupler working use the boustrophedon snake. Otherwise
    (heavy-hex, Willow's diamond grid, or a lattice with inactive couplers) a depth-first
    search from the lowest-numbered qubit, trying the neighbour with the fewest onward
    options first (Warnsdorff's rule), keeps the longest path found within ``budget`` steps.
    """
    if "rows" in cmap.layout:
        rows, cols = cmap.layout["rows"], cmap.layout["cols"]
        snake = [r * cols + (c if r % 2 == 0 else cols - 1 - c) for r in range(rows) for c in range(cols)]
        if all(cmap.coupled(a, b) for a, b in zip(snake, snake[1:], strict=False)):
            return snake
    best: list[int] = []
    steps = 0
    start = cmap.usable_qubits[0]
    path, seen = [start], {start}

    def options(q: int) -> list[int]:
        return sorted((v for v in cmap.neighbors(q) if v not in seen),
                      key=lambda v: (sum(w not in seen for w in cmap.neighbors(v)), v))

    stack = [iter(options(start))]
    while stack and steps < budget:
        steps += 1
        nxt = next(stack[-1], None)
        if nxt is None:
            stack.pop()
            seen.discard(path.pop())
            continue
        path.append(nxt)
        seen.add(nxt)
        if len(path) > len(best):
            best = list(path)
            if len(best) == len(cmap.usable_qubits):
                break
        stack.append(iter(options(nxt)))
    return best


def ghz_path(backend: str, n: int) -> list[int]:
    """Physical qubits of an n-qubit nearest-neighbour path: a line on ions, a long path on lattices."""
    cmap = DEVICES[backend].coupling
    if cmap.all_to_all:
        return list(range(n))
    if backend not in _PATHS:
        _PATHS[backend] = _long_path(cmap)
    path = _PATHS[backend]
    if len(path) < n:
        raise ValueError(f"{backend}: longest path found has {len(path)} qubits, need {n}")
    return path[:n]


def ghz(backend: str, n: int) -> str:
    """GHZ_n along ``ghz_path`` (every CX on a coupled pair), measuring the n path qubits."""
    qs = ghz_path(backend, n)
    size = max(qs) + 1
    body = f"h q[{qs[0]}];\n" + "".join(f"cx q[{a}],q[{b}];\n" for a, b in zip(qs, qs[1:], strict=False))
    meas = "".join(f"measure q[{q}] -> c[{k}];\n" for k, q in enumerate(qs))
    return H2 + f"qreg q[{size}];\ncreg c[{n}];\n" + body + meas


def brickwork(backend: str, n: int, layers: int, seed: int = 0) -> str:
    """``layers`` of random single-qubit rotations and CX on alternating pairs along ``ghz_path``.

    Every CX is on a coupled pair, so no routing is needed on any twin, and the state stays dense
    (all n qubits active) for the whole circuit: a long-lived, memory-bound job.
    """
    qs = ghz_path(backend, n)
    rng = random.Random(seed)
    body = []
    for layer in range(layers):
        body += [f"u3({rng.uniform(0, 3):.4f},{rng.uniform(-3, 3):.4f},{rng.uniform(-3, 3):.4f}) q[{q}];\n"
                 for q in qs]
        body += [f"cx q[{qs[i]}],q[{qs[i + 1]}];\n" for i in range(layer % 2, n - 1, 2)]
    meas = "".join(f"measure q[{q}] -> c[{k}];\n" for k, q in enumerate(qs))
    return H2 + f"qreg q[{max(qs) + 1}];\ncreg c[{n}];\n" + "".join(body) + meas


def star_ghz(n: int) -> str:
    return H2 + f"qreg q[{n}];\ncreg c[{n}];\nh q[0];\n" + "".join(
        f"cx q[0],q[{k}];\n" for k in range(1, n)) + "measure q -> c;\n"


def ghz_success(counts: dict[str, int], n: int) -> float:
    shots = sum(counts.values())
    return (counts.get("0" * n, 0) + counts.get("1" * n, 0)) / shots


# ============================================================= environment
def environment() -> dict[str, Any]:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    env = {"qputwin": __version__, "python": platform.python_version(), "numpy": np.__version__,
           "platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
           "load_average": os.getloadavg(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    try:
        env["docker"] = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                                       capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return env


def save(name: str, data: Any, **meta: Any) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    doc = {"experiment": name, "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
           "environment": environment(), **meta, "data": data}
    (RESULTS / f"{name}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"  saved results/{name}.json")


def timed(sim, qasm: str, opts: ExecutionOptions):
    timer = SpanTimer()
    t0 = time.perf_counter()
    r = sim.execute(qasm, opts, timer)
    return r, timer.ms("simulation") + timer.ms("sampling"), (time.perf_counter() - t0) * 1e3


def api() -> httpx.Client:
    c = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(600.0, connect=5.0))
    c.get("/health").raise_for_status()
    return c


# ============================================================ experiments
def exp_latency(reps: int = 100) -> None:
    """E1 — live latency decomposition per backend (12-qubit GHZ, 1024 shots)."""
    c = api()
    rows = []
    for b in BACKENDS:
        qasm = ghz(b, 12)
        for k in range(5):                                         # warm-up
            c.post("/v1/jobs", json={"backend": b, "qasm": qasm, "shots": 1024, "seed": k})
        for k in range(reps):
            r = c.post("/v1/jobs", json={"backend": b, "qasm": qasm, "shots": 1024, "seed": 1000 + k})
            r.raise_for_status()
            t = r.json()["telemetry"]
            st = float(r.headers["server-timing"].split(",")[0].split("dur=")[1])
            rows.append({"backend": b, "rep": k, "server_timing_total_ms": st,
                         **{k2: t[k2] for k2 in ("total_roundtrip_ms", "orchestration_overhead_ms",
                                                 "qpu_compute_time_ms")}, **t["breakdown"]})
        print(f"  {b}: median total {statistics.median(x['total_roundtrip_ms'] for x in rows if x['backend'] == b):.3f} ms")
    save("latency", rows, workload="12-qubit GHZ along a native path, 1024 shots", reps=reps)


def exp_scaling() -> None:
    """E2 — engine cost vs simulated width (in-process, GHZ chain on Helios)."""
    sim = simulator_for(DEVICES["quantinuum_helios"], SimulatorConfig(max_trajectories=64))
    series = {
        "statevector_c128": (range(4, 25, 2), lambda: ExecutionOptions(shots=1024, seed=1)),
        "statevector_c64": (range(4, 25, 2), lambda: ExecutionOptions(shots=1024, seed=1, precision="complex64")),
        "density_matrix": (range(2, 13), lambda: ExecutionOptions(shots=1024, seed=1, noise={}, engine="density_matrix")),
        "trajectory_T32": (range(4, 19, 2), lambda: ExecutionOptions(shots=1024, seed=1, noise={"trajectories": 32},
                                                                      engine="trajectory")),
    }
    rows = []
    for name, (widths, make) in series.items():
        for n in widths:
            times = []
            for _ in range(3):
                _, qpu, _wall = timed(sim, ghz("quantinuum_helios", n), make())
                times.append(qpu)
            med = statistics.median(times)
            rows.append({"engine": name, "qubits": n, "qpu_ms_median": med, "qpu_ms": times})
            print(f"  {name:18s} n={n:2d}  {med:10.3f} ms")
            if med > 60_000:
                break
    save("scaling", rows, workload="GHZ chain, 1024 shots; noisy engines use the Helios calibration; 3 repetitions")


def exp_routing() -> None:
    """E3 — cost of routing vs native all-to-all execution (in-process)."""
    rows = []
    for b in BACKENDS:
        sim = simulator_for(DEVICES[b])
        for k in range(3, 17):
            opts = {"routing": {"method": "sabre", "seed": 0}} if not DEVICES[b].coupling.all_to_all else {}
            r = sim.execute(star_ghz(k), ExecutionOptions(shots=1, seed=1, **opts))
            comp = r.compilation
            noisy = sim.execute(star_ghz(k), ExecutionOptions(shots=1, seed=1, noise={}, **opts))
            row = {"backend": b, "qubits": k, "native_two_qubit": comp["native_two_qubit_count"],
                   "two_qubit_depth": comp["two_qubit_depth"], "native_depth": comp["native_depth"],
                   "simulated_qubits": comp["simulated_qubits"],
                   "circuit_duration_us": comp["hardware_time_estimate"]["circuit_duration_us"],
                   "esp": noisy.noise["estimated_success_probability"],
                   "swaps": (comp["routing"] or {}).get("swaps_inserted", 0),
                   "cancelled": (comp["routing"] or {}).get("cancelled_two_qubit_gates", 0)}
            if comp["simulated_qubits"] <= 10:
                m = sim.execute(star_ghz(k), ExecutionOptions(shots=20_000, seed=2, noise={},
                                                              engine="density_matrix", **opts))
                row["ghz_success"] = ghz_success(m.counts, k)
            rows.append(row)
        print(f"  {b}: done")
    save("routing", rows, workload="star GHZ (hub = qubit 0); SABRE seed 0 on lattices; noisy success on the "
                                   "density-matrix engine (≤ 10 simulated qubits), 20 000 shots")


def exp_routing_latency(reps: int = 15) -> None:
    """E3b — live latency: rejection vs SABRE routing on IQM Emerald."""
    c = api()
    rows = []
    for k in (4, 6, 8, 10, 12, 14):
        for mode in ("reject", "route"):
            opts = {"routing": {"method": "sabre"}} if mode == "route" else {}
            ts, routes = [], []
            for i in range(reps):
                r = c.post("/v1/jobs", json={"backend": "iqm_emerald", "qasm": star_ghz(k), "shots": 256,
                                             "seed": i, "options": opts})
                t = r.json()["telemetry"]
                ts.append(t["total_roundtrip_ms"])
                routes.append(t["breakdown"]["worker_route_ms"])
            rows.append({"qubits": k, "mode": mode, "status": r.status_code, "total_ms_median": statistics.median(ts),
                         "total_ms": ts, "route_ms_median": statistics.median(routes)})
        print(f"  k={k}: reject {rows[-2]['total_ms_median']:.2f} ms, route {rows[-1]['total_ms_median']:.2f} ms")
    save("routing_latency", rows, workload="star GHZ on IQM Emerald, 256 shots", reps=reps)


def exp_fidelity() -> None:
    """E4 — GHZ success under calibrated noise vs width (density matrix, exact)."""
    rows = []
    for b in BACKENDS:
        sim = simulator_for(DEVICES[b])
        for n in range(2, 11):
            r = sim.execute(ghz(b, n), ExecutionOptions(shots=100_000, seed=3, noise={}, engine="density_matrix"))
            rows.append({"backend": b, "qubits": n, "ghz_success": ghz_success(r.counts, n),
                         "esp": r.noise["estimated_success_probability"], "purity": r.noise["purity"]})
        print(f"  {b}: n=10 success {rows[-1]['ghz_success']:.4f}")
    save("fidelity", rows, workload="GHZ along a native path, calibrated device noise, 100 000 shots")


def exp_hwtime() -> None:
    """E5 — estimated hardware time per shot vs width (gate-duration model)."""
    rows = []
    for b in BACKENDS:
        sim = simulator_for(DEVICES[b], SimulatorConfig(memory_guard=False))
        for n in range(2, 37, 2):
            plan = sim.plan(ghz(b, n), ExecutionOptions(shots=1000), SpanTimer())
            est = sim._time_estimate(plan, 1000)
            rows.append({"backend": b, "qubits": n, "circuit_duration_us": est["circuit_duration_us"],
                         "job_duration_s_1000_shots": est["estimated_job_duration_s"],
                         "assumed_parameters": est["assumed_parameters"]})
    save("hwtime", rows, workload="GHZ along a native path; estimate excludes per-shot overhead (unpublished)")


def exp_budget() -> None:
    """E6 — error budget: GHZ_6 infidelity with one noise channel at a time."""
    off = {"depolarizing": False, "relaxation": False, "idle": False, "readout": False}
    conditions = {
        "gate depolarising": {**off, "depolarizing": True},
        "T1/T2 relaxation": {**off, "relaxation": True, "idle": True},
        "readout": {**off, "readout": True},
        "all channels": {},
    }
    rows = []
    for b in BACKENDS:
        sim = simulator_for(DEVICES[b])
        for cname, noise in conditions.items():
            r = sim.execute(ghz(b, 6), ExecutionOptions(shots=200_000, seed=4, noise=noise, engine="density_matrix"))
            rows.append({"backend": b, "condition": cname, "infidelity": 1 - ghz_success(r.counts, 6)})
    save("budget", rows, workload="GHZ_6 along a native path, density matrix, 200 000 shots")


def exp_convergence(seeds: int = 5) -> None:
    """E7 — trajectory sampling error vs number of trajectories (exact reference from the density matrix)."""
    spec = DEVICES["iqm_emerald"]
    noise = {"scale": 3.0}
    qasm = ghz("iqm_emerald", 4)
    sim = simulator_for(spec, SimulatorConfig(max_trajectories=20_000))
    r = sim.execute(qasm, ExecutionOptions(shots=1, noise=noise, return_density_matrix=True))
    rho = np.array(r.density_matrix)
    p = np.clip((rho[..., 0]).diagonal().copy(), 0, None)
    model = NoiseModel(spec.calibration, NoiseOptions.from_dict(noise), spec.coupling)
    exact = apply_readout_model(p / p.sum(), r.statevector_qubits, model.readout_errors, model.readout_groups)
    qpos = {q: j for j, q in enumerate(r.statevector_qubits)}
    path = ghz_path("iqm_emerald", 4)
    exact_counts = {}
    for idx, pr in enumerate(exact):
        key = "".join(str((idx >> qpos[path[k]]) & 1) for k in reversed(range(4)))
        exact_counts[key] = exact_counts.get(key, 0.0) + float(pr)
    shots = 20_000
    rng = np.random.default_rng(7)
    keys = sorted(exact_counts)
    pv = np.array([exact_counts[k] for k in keys])
    floor = [0.5 * float(np.abs(rng.multinomial(shots, pv) / shots - pv).sum()) for _ in range(200)]
    rows = []
    for t in (10, 32, 100, 316, 1000, 3162, 10_000, 20_000):
        tvs = []
        for s in range(seeds):
            out = sim.execute(qasm, ExecutionOptions(shots=shots, seed=100 + s, noise={**noise, "trajectories": t},
                                                     engine="trajectory"))
            tvs.append(0.5 * sum(abs(out.counts.get(k, 0) / shots - exact_counts.get(k, 0.0))
                                 for k in set(out.counts) | set(exact_counts)))
        rows.append({"trajectories": t, "tv_mean": statistics.fmean(tvs), "tv_std": statistics.stdev(tvs), "tv": tvs})
        print(f"  T={t:6d}: TV {rows[-1]['tv_mean']:.4f}")
    save("convergence", rows, shot_noise_floor_mean=statistics.fmean(floor), shot_noise_floor_p95=float(
        np.percentile(floor, 95)), shots=shots, seeds=seeds,
        workload="GHZ_4 on IQM Emerald, noise scale 3; reference = exact density matrix + readout model")


def _compose(replicas: int, orchestrators: int = 1) -> None:
    # QPUTWIN_ENGINE_THREADS=1 is for this process's in-process experiments only; the containers must keep
    # their default (engine threads = the worker's CPU quota), so it is not passed on to compose
    env = {k: v for k, v in os.environ.items() if k != "QPUTWIN_ENGINE_THREADS"}
    env |= {"QPUTWIN_FORTE_REPLICAS": str(replicas), "QPUTWIN_ORCHESTRATOR_REPLICAS": str(orchestrators),
           "QPUTWIN_ORCHESTRATOR_PORTS": "8000" if orchestrators == 1 else f"8000-{7999 + orchestrators}"}
    profile = ["--profile", "ha"] if orchestrators > 1 else []
    if orchestrators == 1:                                       # leave the default single-instance stack
        subprocess.run(["docker", "compose", "--profile", "ha", "rm", "-sf", "gateway"], cwd=ROOT, env=env,
                       capture_output=True, timeout=120)
    subprocess.run(["docker", "compose", *profile, "up", "-d", "--wait", "--remove-orphans"], cwd=ROOT, env=env,
                   check=True, capture_output=True, timeout=600)
    time.sleep(12)                                               # DNS discovery interval (10 s) + margin


REPLICA_ENV = {"quantinuum_helios": "QPUTWIN_HELIOS_REPLICAS", "ionq_forte": "QPUTWIN_FORTE_REPLICAS",
               "rigetti_ankaa_3": "QPUTWIN_ANKAA3_REPLICAS", "iqm_emerald": "QPUTWIN_EMERALD_REPLICAS",
               "ibm_heron": "QPUTWIN_HERON_REPLICAS", "ibm_nighthawk": "QPUTWIN_NIGHTHAWK_REPLICAS",
               "google_willow": "QPUTWIN_WILLOW_REPLICAS"}


def _compose_env(overrides: dict[str, str]) -> None:
    """Restart the stack with ``overrides`` (replicas, worker CPU / memory limits); 1 orchestrator."""
    env = {k: v for k, v in os.environ.items() if k != "QPUTWIN_ENGINE_THREADS"} | overrides
    subprocess.run(["docker", "compose", "--profile", "ha", "rm", "-sf", "gateway"], cwd=ROOT, env=env,
                   capture_output=True, timeout=120)
    subprocess.run(["docker", "compose", "up", "-d", "--wait", "--remove-orphans", "--force-recreate"], cwd=ROOT,
                   env=env, check=True, capture_output=True, timeout=900)
    time.sleep(12)


def _host_sample() -> dict[str, float]:
    """Host CPU jiffies (busy, total) and memory from /proc (this runs on the docker host)."""
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, val = line.split(":")
            mem[k] = int(val.split()[0]) * 1024
    return {"busy": sum(v) - idle, "total": sum(v), "mem_used": mem["MemTotal"] - mem["MemAvailable"],
            "mem_total": mem["MemTotal"]}


def exp_saturation(duration: float = 600.0, light_rate: float = 150.0, replicas: int = 3, worker_mem: str = "7g",
                   big: int = 28, too_big: int = 30) -> None:
    """E22 — the whole host under mixed load: every twin, all vCPUs and most of the memory at once.

    The stack runs ``replicas`` workers per twin (21 in all), each allowed every CPU of the host and
    ``worker_mem`` of memory. For ``duration`` seconds: a Poisson stream of light synchronous jobs over
    all twins; per twin one heavy asynchronous job in flight at all times (``big``-qubit states that fit
    only in place within the worker's memory, deep 22-qubit QV-style circuits, 11-qubit noisy density
    matrices); and a trickle of ``too_big``-qubit jobs that cannot fit and must be refused cleanly. Records host CPU and
    memory, per-container stats, per-class latency and status, and container OOM kills and restarts.
    """
    overrides = {env: str(replicas) for env in REPLICA_ENV.values()} | {
        "QPUTWIN_WORKER_CPUS": str(os.cpu_count() or 16), "QPUTWIN_WORKER_MEM": worker_mem}
    _compose_env(overrides)
    try:
        c = api()
        # two heavy streams per twin (one per replica): memory-bound big states, and compute-bound jobs
        big_specs: dict[str, list[dict[str, Any]]] = {}
        cpu_specs: dict[str, list[dict[str, Any]]] = {}
        for b in BACKENDS:
            a2a = DEVICES[b].coupling.all_to_all
            big_specs[b] = [{"qasm": brickwork(b, big, 12, seed=k), "shots": 1024} for k in range(3)]
            cpu_specs[b] = [{"qasm": ghz(b, 11), "shots": 1024, "options": {"noise": {}, "engine": "density_matrix"}}]
            if a2a:
                cpu_specs[b].append({"qasm": C.FAMILIES["qv"](22, seed=3), "shots": 1024})
        oversized = {b: star_ghz(too_big) if DEVICES[b].coupling.all_to_all else ghz(b, too_big) for b in BACKENDS}
        light = [_light_job(b) for b in BACKENDS]
        heavy_rows: list[dict[str, Any]] = []
        refused: list[dict[str, Any]] = []
        host: list[dict[str, Any]] = []
        stats: list[dict[str, Any]] = []

        async def heavy_loop(b: str, stop: float, specs: list[dict[str, Any]], stream: str) -> None:
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=900.0) as ac:
                t0, k = time.perf_counter(), 0
                while time.perf_counter() - t0 < stop:
                    spec = specs[k % len(specs)]
                    s0 = time.perf_counter()
                    r = await ac.post("/v1/jobs", params={"mode": "async"},
                                      json={"backend": b, "seed": k, "priority": 1, **spec})
                    jid = r.json().get("job_id")
                    body: dict[str, Any] = {}
                    deadline = time.perf_counter() + 900
                    while jid and time.perf_counter() < deadline:
                        body = (await ac.get(f"/v1/jobs/{jid}", params={"wait": 30})).json()
                        if body.get("status") not in ("QUEUED", "RUNNING"):
                            break
                    wl = (body.get("telemetry") or {}).get("workload") or {}
                    heavy_rows.append({"backend": b, "stream": stream, "kind": k % len(specs), "t": s0 - t0,
                                       "latency_s": time.perf_counter() - s0, "status": body.get("status"),
                                       "error": (body.get("error") or {}).get("type"),
                                       "in_place": wl.get("in_place"), "simulated_qubits": wl.get("simulated_qubits"),
                                       "engine": wl.get("engine")})
                    k += 1

        async def oversize_loop(stop: float) -> None:
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as ac:
                t0, k = time.perf_counter(), 0
                while time.perf_counter() - t0 < stop:
                    b = BACKENDS[k % len(BACKENDS)]
                    s0 = time.perf_counter()
                    r = await ac.post("/v1/jobs", json={"backend": b, "qasm": oversized[b], "shots": 16, "seed": k})
                    body = r.json()
                    refused.append({"backend": b, "status": r.status_code, "latency_s": time.perf_counter() - s0,
                                    "error": (body.get("error") or {}).get("type"),
                                    "details": {k2: v for k2, v in ((body.get("error") or {}).get("details") or {}
                                                                    ).items() if k2 in ("required_bytes",
                                                                                        "available_bytes",
                                                                                        "max_feasible_qubits")}})
                    k += 1
                    await asyncio.sleep(5.0)

        async def sampler(stop: float) -> None:
            t0 = time.perf_counter()
            prev = _host_sample()
            while time.perf_counter() - t0 < stop:
                await asyncio.sleep(1.0)
                cur = _host_sample()
                host.append({"t": time.perf_counter() - t0,
                             "cpu_pct": 100.0 * (cur["busy"] - prev["busy"]) / max(1, cur["total"] - prev["total"]),
                             "mem_used_gib": cur["mem_used"] / 2 ** 30, "mem_total_gib": cur["mem_total"] / 2 ** 30})
                prev = cur
                if len(host) % 9 == 1:
                    stats.append({"t": host[-1]["t"], "containers": await asyncio.to_thread(_docker_stats)})

        async def main() -> list[dict[str, Any]]:
            res, *_ = await asyncio.gather(_open_loop(light, light_rate, duration, seed=7),
                                           *(heavy_loop(b, duration, big_specs[b], "memory") for b in BACKENDS),
                                           *(heavy_loop(b, duration, cpu_specs[b], "compute") for b in BACKENDS),
                                           oversize_loop(duration), sampler(duration + 5))
            return res

        c.close()
        import faulthandler                      # if the client ever stalls, log where (every 5 min after the window)
        faulthandler.dump_traceback_later(duration + 300, repeat=True, file=sys.stderr)

        async def guarded() -> list[dict[str, Any]]:
            return await asyncio.wait_for(main(), timeout=duration + 900)   # hard stop: never hang the campaign

        try:
            light_res = asyncio.run(guarded())
        finally:
            faulthandler.cancel_dump_traceback_later()
        inspect = subprocess.run(["docker", "ps", "-a", "--filter", "name=qputwin", "--format", "{{.Names}}"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
        health = []
        for name in inspect:
            out = subprocess.run(["docker", "inspect", "--format",
                                  "{{.State.OOMKilled}} {{.RestartCount}} {{.State.Status}}", name],
                                 capture_output=True, text=True, timeout=30).stdout.split()
            health.append({"name": name, "oom_killed": out[0] == "true", "restarts": int(out[1]), "state": out[2]})
        ok = [r for r in light_res if r["status"] == 200]
        cpu = [h["cpu_pct"] for h in host]
        mem = [h["mem_used_gib"] for h in host]
        print(f"  light {len(ok)}/{len(light_res)} ok, p50 {np.median([r['total_ms'] for r in ok]):.1f} ms; "
              f"heavy {sum(h['status'] == 'COMPLETED' for h in heavy_rows)}/{len(heavy_rows)} completed "
              f"({sum(bool(h['in_place']) for h in heavy_rows)} in place); refused {len(refused)} "
              f"(statuses {sorted({r['status'] for r in refused})}); host CPU median {np.median(cpu):.0f} % "
              f"max {max(cpu):.0f} %, memory max {max(mem):.1f} GiB; OOM kills "
              f"{sum(h['oom_killed'] for h in health)}, restarts {sum(h['restarts'] for h in health)}", flush=True)
        save("saturation", {"light": [_job_fields_compact(r) for r in light_res], "heavy": heavy_rows,
                            "oversized": refused, "host": host, "containers": stats, "health": health},
             duration_s=duration, light_rate_jobs_s=light_rate, replicas_per_twin=replicas, worker_mem=worker_mem,
             worker_cpus=os.cpu_count(), summary=_summary(light_res, duration) | {
                 "heavy_completed": sum(h["status"] == "COMPLETED" for h in heavy_rows),
                 "heavy_total": len(heavy_rows), "heavy_in_place": sum(bool(h["in_place"]) for h in heavy_rows),
                 "oversized_refused": sum(r["error"] == "ResourceLimitError" for r in refused),
                 "oversized_total": len(refused), "host_cpu_pct_median": float(np.median(cpu)),
                 "host_cpu_pct_max": float(max(cpu)), "host_mem_gib_max": float(max(mem)),
                 "oom_kills": sum(h["oom_killed"] for h in health),
                 "restarts": sum(h["restarts"] for h in health)},
             workload=(f"whole-host saturation: {replicas} worker replicas per twin ({7 * replicas} workers, each "
                       "allowed every CPU and "
                       f"{worker_mem}), 1 orchestrator; Poisson light GHZ_8 jobs at {light_rate:.0f}/s over all twins; "
                       f"per twin two heavy async streams, one job in flight each: {big}-qubit, 12-layer brickwork "
                       "circuits along a native path (memory: a dense state held for seconds), and "
                       "11-qubit noisy density matrices plus 22-qubit QV-style circuits on the all-to-all twins "
                       f"(compute); one {too_big}-qubit job every 5 s that "
                       "must be refused; host /proc samples every 1 s, docker stats every ~9 s"))
    finally:
        _compose_env({})


def _job_fields_compact(r: dict[str, Any]) -> list[Any]:
    return [round(r["t"], 3), round(r["total_ms"], 3) if r.get("total_ms") is not None else None, r["status"]]


def exp_async_scaling() -> None:
    """E8 — async throughput vs worker replicas (IonQ Forte): cache-resident vs DRAM-bound statevectors.

    Each point is repeated ``REPEATS`` times; the row keeps every repetition and reports the median, because a
    single short burst (well under a second of makespan) is too noisy to compare replica counts.
    """
    from concurrent.futures import ThreadPoolExecutor
    workloads = {14: 480, 20: 120}                                 # qubits -> jobs (14q: 256 KiB state, 20q: 16 MiB)
    repeats = 5
    submitters = 16              # concurrent submission (~360 jobs/s, E16) outpaces any replica count tested here
    rows = []

    def ts(s: str) -> float:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

    try:
        for replicas in (1, 2, 3):
            _compose(replicas)
            c = api()
            for width, jobs in workloads.items():
                qasm = star_ghz(width)
                for k in range(3 * replicas):                      # warm-up every replica
                    c.post("/v1/jobs", json={"backend": "ionq_forte", "qasm": qasm, "shots": 64, "seed": k})
                runs = []
                for rep in range(repeats):
                    def submit(k: int) -> str:
                        return c.post("/v1/jobs", params={"mode": "async"},
                                      json={"backend": "ionq_forte", "qasm": qasm, "shots": 64, "seed": 10 + k}
                                      ).json()["job_id"]
                    with ThreadPoolExecutor(submitters) as ex:   # a serial submitter, not the workers, was the limit
                        ids = list(ex.map(submit, range(jobs)))
                    done = []
                    for j in ids:
                        while True:
                            r = c.get(f"/v1/jobs/{j}", params={"wait": 20})
                            b = r.json()
                            if "status" not in b:            # transient (e.g. a replica restarting): retry
                                print(f"  poll {j}: HTTP {r.status_code} {str(b)[:160]}", flush=True)
                                time.sleep(0.5)
                                continue
                            if b["status"] not in ("QUEUED", "RUNNING"):
                                done.append(b)
                                break
                    makespan = max(ts(b["finished_at"]) for b in done) - min(ts(b["submitted_at"]) for b in done)
                    drain = max(ts(b["finished_at"]) for b in done) - min(ts(b["started_at"]) for b in done)
                    runs.append({"makespan_s": makespan, "throughput_jobs_s": jobs / makespan,
                                 "drain_jobs_s": jobs / drain,
                                 "errors": sum(b["status"] != "COMPLETED" for b in done),
                                 "workers": sorted({b["telemetry"]["environment"]["worker"]["worker_id"]
                                                    for b in done}),
                                 "queue_ms": [b["telemetry"]["breakdown"]["orchestrator_queue_ms"] for b in done],
                                 "worker_service_ms": [b["telemetry"]["breakdown"]["worker_service_ms"] for b in done],
                                 "qpu_ms": [b["telemetry"]["qpu_compute_time_ms"] for b in done]})
                thr = [r["throughput_jobs_s"] for r in runs]
                rows.append({"replicas": replicas, "qubits": width, "state_bytes": 16 << width, "jobs": jobs,
                             "repeats": repeats, "throughput_jobs_s": statistics.median(thr),
                             "throughput_runs": thr, "errors": sum(r["errors"] for r in runs),
                             "workers_used": len({w for r in runs for w in r["workers"]}),
                             "worker_service_ms": [v for r in runs for v in r["worker_service_ms"]],
                             "queue_ms": [v for r in runs for v in r["queue_ms"]],
                             "qpu_ms": [v for r in runs for v in r["qpu_ms"]],
                             "runs": [{k: v for k, v in r.items() if k in ("makespan_s", "throughput_jobs_s", "drain_jobs_s",
                                                                          "errors", "workers")} for r in runs]})
                print(f"  replicas={replicas} n={width}: median {statistics.median(thr):.2f} jobs/s "
                      f"(runs {', '.join(f'{v:.1f}' for v in thr)}), median worker "
                      f"{statistics.median(rows[-1]['worker_service_ms']):.0f} ms, {rows[-1]['workers_used']} workers")
    finally:
        _compose(1)
    save("async_scaling", rows, workload=f"async star GHZ jobs on IonQ Forte, 64 shots; 14 qubits (cache-resident "
                                         f"256 KiB state, {workloads[14]} jobs) and 20 qubits (16 MiB, DRAM-bound, "
                                         f"{workloads[20]} jobs); submitted by {submitters} concurrent clients; {repeats} repetitions "
                                         f"per point, median reported")


def exp_noise_sweep() -> None:
    """E9 — GHZ_5 success vs global noise scale."""
    rows = []
    for b in BACKENDS:
        sim = simulator_for(DEVICES[b])
        for s in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
            r = sim.execute(ghz(b, 5), ExecutionOptions(shots=100_000, seed=5, noise={"scale": s},
                                                         engine="density_matrix"))
            rows.append({"backend": b, "scale": s, "ghz_success": ghz_success(r.counts, 5)})
    save("noise_sweep", rows, workload="GHZ_5 along a native path, density matrix, 100 000 shots")


def exp_ramsey() -> None:
    """E10 — Ramsey decay: Markovian T2 (exponential) vs quasi-static detuning (Gaussian)."""
    t2_us = 1.0
    sigma_khz = math.sqrt(2) / (2 * math.pi * t2_us * 1e-3)       # same 1/e time as T2
    off = {"depolarizing": False, "relaxation": False, "idle": False, "readout": False, "quasi_static": False}
    sim = simulator_for(DEVICES["iqm_emerald"], SimulatorConfig(max_trajectories=4000))
    rows = []
    for m in range(0, 31, 2):
        src = H2 + ("qreg q[3]; creg c[1];\nh q[0];\n" + "cz q[1],q[2];\n" * m
                    + "barrier q[0],q[1],q[2];\nh q[0];\nmeasure q[0] -> c[0];\n")
        t_us = max(60.0 * m, 40.0) / 1e3
        mk = sim.execute(src, ExecutionOptions(shots=200_000, seed=6, engine="density_matrix",
                                               noise={**off, "relaxation": True, "idle": True,
                                                      "t1_us": 1e9, "t2_us": t2_us}))
        qs = sim.execute(src, ExecutionOptions(shots=4000, seed=6, engine="trajectory",
                                               noise={**off, "quasi_static": True, "quasi_static_khz": sigma_khz}))
        for model, r in (("markovian", mk), ("quasi_static", qs)):
            p1 = r.counts.get("1", 0) / sum(r.counts.values())
            rows.append({"model": model, "delay_us": t_us, "contrast": 1 - 2 * p1, "shots": sum(r.counts.values())})
    save("ramsey", rows, t2_us=t2_us, sigma_khz=sigma_khz,
         workload="Ramsey on IQM Emerald q0; delay from 2m CZ gates on q1-q2 (60 ns each) then a barrier")


# ===================================================== circuit families
def _route_opts(backend: str) -> dict[str, Any]:
    return {} if DEVICES[backend].coupling.all_to_all else {"routing": {"method": "sabre", "seed": 0}}


def ideal_distribution(qasm: str, ncl: int) -> dict[str, float]:
    """Exact output distribution from the ideal statevector (measured clbit k ← logical qubit k).

    Cross-checked against 200 000 sampled shots: the total-variation gap must be within shot noise.
    """
    sim = simulator_for(DEVICES["quantinuum_helios"], SimulatorConfig(memory_guard=False))
    r = sim.execute(qasm, ExecutionOptions(shots=200_000, seed=11, return_statevector=True))
    amps = np.array(r.statevector)
    p = amps[:, 0] ** 2 + amps[:, 1] ** 2
    pos = {q: j for j, q in enumerate(r.statevector_qubits)}
    probs: dict[str, float] = {}
    for idx in np.nonzero(p > 1e-14)[0]:
        key = "".join(str((int(idx) >> pos[k]) & 1) if k in pos else "0" for k in reversed(range(ncl)))
        probs[key] = probs.get(key, 0.0) + float(p[idx])
    tv = 0.5 * sum(abs(probs.get(k, 0.0) - r.counts.get(k, 0) / 200_000) for k in set(probs) | set(r.counts))
    bound = 0.5 * math.sqrt(2 * len(probs) / (math.pi * 200_000)) * 2 + 0.005
    assert tv < bound, f"ideal distribution check failed: TV {tv:.4f} > {bound:.4f}"
    return probs


def exp_circuits() -> None:
    """E11 — nine circuit families × seven devices: compiled cost, engine time and noisy fidelity."""
    noisy_widths = (3, 4, 5, 6, 7, 8, 10)
    ideal_widths = (4, 8, 12, 16, 20)
    shots = 50_000
    rows, cost = [], []
    only = [f for f in os.environ.get("QPUTWIN_BENCH_FAMILIES", "").split(",") if f]
    if only:                                    # re-run some families, keep the others from the last run
        prev = json.loads((RESULTS / "circuits.json").read_text())
        rows = [r for r in prev["data"] if r["family"] not in only]
        cost = [r for r in prev["cost"] if r["family"] not in only]
    for fam, gen in C.FAMILIES.items():
        if only and fam not in only:
            continue
        for n in noisy_widths:
            if fam == "adder" and n % 2:
                continue
            qasm = gen(n, seed=1)
            ncl = n - 1 if fam == "bv" else n
            ideal = ideal_distribution(qasm, ncl)
            for b in BACKENDS:
                sim = simulator_for(DEVICES[b])
                try:
                    r, qpu, wall = timed(sim, qasm, ExecutionOptions(shots=shots, seed=12, noise={},
                                                                     engine="density_matrix", **_route_opts(b)))
                except ResourceLimitError as exc:       # routing spread the circuit beyond the exact-DM limit
                    rows.append({"family": fam, "backend": b, "qubits": n, "hellinger": None,
                                 "skipped": str(exc), "simulated_qubits": exc.details.get("simulated_qubits")})
                    continue
                comp = r.compilation
                rows.append({"family": fam, "backend": b, "qubits": n,
                             "hellinger": C.hellinger_fidelity(ideal, C.normalise(r.counts)),
                             "tv_distance": 0.5 * sum(abs(ideal.get(k, 0) - r.counts.get(k, 0) / shots)
                                                      for k in set(ideal) | set(r.counts)),
                             "esp": r.noise["estimated_success_probability"],
                             "native_two_qubit": comp["native_two_qubit_count"],
                             "two_qubit_depth": comp["two_qubit_depth"], "native_depth": comp["native_depth"],
                             "swaps": (comp["routing"] or {}).get("swaps_inserted", 0),
                             "circuit_duration_us": comp["hardware_time_estimate"]["circuit_duration_us"],
                             "density_matrix_ms": qpu, "wall_ms": wall, "ideal_support": len(ideal)})
            print(f"  {fam:7s} n={n:2d}  " + "  ".join(
                f"{r['backend'][:9]} " + (f"{r['hellinger']:.3f}" if r["hellinger"] is not None else "skip")
                for r in rows[-len(BACKENDS):]))
        for n in ideal_widths:
            if fam == "adder" and n % 2:
                continue
            qasm = gen(n, seed=1)
            for b in BACKENDS:
                sim = simulator_for(DEVICES[b])
                timer = SpanTimer()
                r = sim.execute(qasm, ExecutionOptions(shots=1024, seed=1, **_route_opts(b)), timer)
                comp = r.compilation
                cost.append({"family": fam, "backend": b, "qubits": n, "gates_in": qasm.count(";") - 4 - n,
                             "native_two_qubit": comp["native_two_qubit_count"],
                             "native_one_qubit": comp.get("native_one_qubit_count"),
                             "simulated_qubits": comp["simulated_qubits"],
                             "two_qubit_depth": comp["two_qubit_depth"], "native_depth": comp["native_depth"],
                             "swaps": (comp["routing"] or {}).get("swaps_inserted", 0),
                             "circuit_duration_us": comp["hardware_time_estimate"]["circuit_duration_us"],
                             **{f"{s}_ms": timer.ms(s) for s in ("parse", "route", "compile", "simulation",
                                                                  "sampling")}})
        print(f"  {fam}: cost table done (n ≤ {ideal_widths[-1]})")
    save("circuits", rows, cost=cost, shots=shots, workload=(
        "nine circuit families (benchmarks/circuits.py, seed 1); noisy = calibrated device noise on the exact "
        "density-matrix engine, 50 000 shots, Hellinger fidelity vs the exact ideal distribution; lattices "
        "routed with SABRE (seed 0); cost table = ideal statevector, 1024 shots"))


def exp_qv() -> None:
    """E12 — quantum-volume-style heavy-output probability vs width (exact noise, 12 circuits per width)."""
    rows = []
    for m in range(2, 9):
        circuits = []
        for s in range(12):
            qasm = C.qv(m, seed=1000 * m + s)
            ideal = ideal_distribution(qasm, m)
            heavy = C.heavy_outputs(ideal)
            circuits.append((s, qasm, heavy, sum(ideal[k] for k in heavy)))
        for b in BACKENDS:
            sim = simulator_for(DEVICES[b])
            hops = []
            for s, qasm, heavy, ideal_hop in circuits:
                r = sim.execute(qasm, ExecutionOptions(shots=20_000, seed=s, noise={}, engine="density_matrix",
                                                       **_route_opts(b)))
                hop = sum(v for k, v in r.counts.items() if k in heavy) / 20_000
                hops.append(hop)
                rows.append({"backend": b, "width": m, "circuit": s, "hop": hop, "ideal_hop": ideal_hop,
                             "native_two_qubit": r.compilation["native_two_qubit_count"]})
            mean = statistics.fmean(hops)
            lower = mean - 2 * statistics.stdev(hops) / math.sqrt(len(hops))
            print(f"  m={m} {b:18s} HOP {mean:.3f} (2σ lower {lower:.3f}) {'PASS' if lower > 2 / 3 else 'fail'}")
    save("qv", rows, workload="QV-style circuits (m layers of random 3-CX two-qubit blocks; generic, not Haar "
                              "SU(4)), 12 circuits per width, 20 000 shots, exact calibrated noise; pass = mean "
                              "HOP − 2σ_mean > 2/3")


# ======================================================== load generation
def _job_fields(r: httpx.Response, t_start: float, t_end: float) -> dict[str, Any]:
    row = {"t": t_start, "t_end": t_end, "client_ms": (t_end - t_start) * 1e3, "status": r.status_code}
    try:
        if r.status_code != 200:
            row["error"] = f"{r.status_code} {(r.json().get('error') or {}).get('type', '')}".strip()
        tel = r.json().get("telemetry") or {}
        bd = tel.get("breakdown") or {}
        row.update(total_ms=tel.get("total_roundtrip_ms"), queue_ms=bd.get("worker_queue_ms"),
                   service_ms=bd.get("worker_service_ms"), qpu_ms=tel.get("qpu_compute_time_ms"),
                   overhead_ms=tel.get("orchestration_overhead_ms"), spans=bd)
    except ValueError:
        pass
    return row


async def _closed_loop(jobs: list[dict[str, Any]], concurrency: int, duration: float, warmup: float = 2.0,
                       base_url: str = BASE_URL, start_at: float | None = None, first: int = 0, stride: int = 0
                       ) -> list[dict[str, Any]]:
    """``concurrency`` virtual users; ``start_at`` (a CLOCK_MONOTONIC instant) aligns several processes."""
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)
    out: list[dict[str, Any]] = []
    stride = stride or concurrency
    async with httpx.AsyncClient(base_url=base_url, timeout=600.0, limits=limits) as c:
        if start_at is not None:
            await asyncio.sleep(max(0.0, start_at - time.perf_counter()))
        t0 = start_at if start_at is not None else time.perf_counter()
        stop = t0 + warmup + duration

        async def user(i: int) -> None:
            k = first + i
            while (ts := time.perf_counter()) < stop:
                job = {**jobs[k % len(jobs)], "seed": k}
                k += stride
                try:
                    r = await c.post("/v1/jobs", json=job)
                except httpx.HTTPError as exc:
                    out.append({"t": ts - t0 - warmup, "t_end": time.perf_counter() - t0 - warmup, "status": -1,
                                "error": type(exc).__name__})
                    continue
                out.append(_job_fields(r, ts - t0 - warmup, time.perf_counter() - t0 - warmup))

        await asyncio.gather(*(user(i) for i in range(concurrency)))
    return out


async def _open_loop(jobs: list[dict[str, Any]], rate: float, duration: float, seed: int = 0
                     ) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    arrivals, t = [], 0.0
    while (t := t + rng.expovariate(rate)) < duration:
        arrivals.append(t)
    out: list[dict[str, Any]] = []
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=256)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=600.0, limits=limits) as c:
        t0 = time.perf_counter()

        async def one(k: int, at: float) -> None:
            await asyncio.sleep(max(0.0, t0 + at - time.perf_counter()))
            ts = time.perf_counter() - t0
            try:
                r = await c.post("/v1/jobs", json={**jobs[k % len(jobs)], "seed": k})
                out.append(_job_fields(r, ts, time.perf_counter() - t0) | {"scheduled": at})
            except httpx.HTTPError as exc:
                out.append({"t": ts, "status": -1, "scheduled": at, "error": type(exc).__name__})

        await asyncio.gather(*(one(k, at) for k, at in enumerate(arrivals)))
    return out


def _closed_loop_proc(args: tuple[Any, ...]) -> tuple[list[dict[str, Any]], float]:
    jobs, conc, duration, warmup, base_url, start_at, first, stride = args
    cpu0 = time.process_time()
    rows = asyncio.run(_closed_loop(jobs, conc, duration, warmup, base_url, start_at, first, stride))
    return rows, time.process_time() - cpu0


def _closed_loop_mp(jobs: list[dict[str, Any]], concurrency: int, duration: float, procs: int,
                    base_url: str = BASE_URL) -> tuple[list[dict[str, Any]], float]:
    """Closed loop split over ``procs`` client processes (so the load generator is never the bottleneck)."""
    from concurrent.futures import ProcessPoolExecutor
    procs = max(1, min(procs, concurrency))
    shares = [concurrency // procs + (1 if k < concurrency % procs else 0) for k in range(procs)]
    start_at = time.perf_counter() + 1.5                       # CLOCK_MONOTONIC is shared by all processes
    firsts = [sum(shares[:k]) for k in range(procs)]
    with ProcessPoolExecutor(procs) as ex:
        parts = list(ex.map(_closed_loop_proc, [(jobs, sh, duration, 2.0, base_url, start_at, f, concurrency)
                                                for sh, f in zip(shares, firsts, strict=True)]))
    return [r for rows, _ in parts for r in rows], sum(cpu for _, cpu in parts)


def _summary(rows: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """Steady-state statistics over the jobs that *completed* inside the measurement window [0, duration]."""
    rows = [r for r in rows if 0.0 <= r.get("t_end", 0.0) <= duration]
    ok = [r for r in rows if r["status"] == 200]
    lat = np.array([r["total_ms"] for r in ok]) if ok else np.array([np.nan])
    cl = np.array([r["client_ms"] for r in ok]) if ok else np.array([np.nan])
    failed = [r for r in rows if r["status"] != 200]
    kinds: dict[str, int] = {}
    for r in failed:                            # HTTP status, or the client-side exception (status -1)
        k = r.get("error") or str(r["status"])
        kinds[k] = kinds.get(k, 0) + 1
    return {"completed": len(ok), "errors": len(rows) - len(ok), "error_kinds": kinds,
            "throughput_jobs_s": len(ok) / duration,
            **{f"p{q}_ms": float(np.percentile(lat, q)) for q in (50, 90, 95, 99)},
            "mean_ms": float(lat.mean()), "client_p50_ms": float(np.percentile(cl, 50)),
            "client_p99_ms": float(np.percentile(cl, 99)), "client_mean_ms": float(cl.mean()),
            "queue_p50_ms": float(np.median([r["queue_ms"] for r in ok])) if ok else None,
            "qpu_p50_ms": float(np.median([r["qpu_ms"] for r in ok])) if ok else None,
            "spans_mean_ms": {k: float(np.mean([r["spans"][k] for r in ok])) for k in (ok[0]["spans"] if ok else {})
                              if isinstance(ok[0]["spans"][k], int | float)}}


def _light_job(backend: str) -> dict[str, Any]:
    return {"backend": backend, "qasm": ghz(backend, 8), "shots": 256}


def _heavy_job() -> dict[str, Any]:
    return {"backend": "ionq_forte", "qasm": C.qv(12, seed=5), "shots": 1024}


def exp_load_closed(duration: float = 15.0) -> None:
    """E13 — closed-loop load: throughput and tail latency vs concurrent clients (sync API)."""
    light, heavy = (1, 2, 4, 8, 16, 32, 64, 128, 256), (1, 2, 4, 8, 16, 32, 64)
    conditions = [("light_emerald", [_light_job("iqm_emerald")], 1, light, duration),
                  ("heavy_forte", [_heavy_job()], 1, heavy, 2 * duration),
                  ("heavy_forte_x4", [_heavy_job()], 4, heavy, 2 * duration),
                  ("mixed_all_backends", [_light_job(b) for b in BACKENDS], 1, light, duration)]
    rows, raw = [], {}
    try:
        for name, jobs, replicas, levels, dur in conditions:
            _compose(replicas)
            api()
            for conc in levels:
                cpu0 = time.process_time()
                res = asyncio.run(_closed_loop(jobs, conc, dur))
                s = _summary(res, dur)
                rows.append({"condition": name, "replicas": replicas, "concurrency": conc, "duration_s": dur, **s,
                             "client_cpu_s": time.process_time() - cpu0})
                raw[f"{name}/{conc}"] = [round(r["total_ms"], 3) for r in res
                                         if r["status"] == 200 and 0 <= r["t_end"] <= dur]
                print(f"  {name:17s} c={conc:3d}: {s['throughput_jobs_s']:7.1f} jobs/s  p50 {s['p50_ms']:8.2f}"
                      f"  p99 {s['p99_ms']:8.2f} ms  errors {s['errors']}")
    finally:
        _compose(1)
    save("load_closed", rows, latencies_ms=raw, warmup_s=2.0, workload=(
        "closed loop, N clients each sending the next job as soon as the last returns; light = GHZ_8 256 shots "
        "(IQM Emerald), heavy = QV-style 12 qubits 1024 shots (IonQ Forte, 1 or 4 worker replicas), mixed = "
        "light jobs round-robin over all QPUs"))


def exp_load_ha(duration: float = 15.0, procs: int = 3) -> None:
    """E13b — orchestrator scale-out: light jobs over all QPUs, direct vs nginx gateway with 1 or 3 orchestrators.

    The load is generated by ``procs`` client processes so that the generator is not the bottleneck.
    """
    gateway = os.environ.get("QPUTWIN_GATEWAY_URL", "http://localhost:8080")
    jobs = [_light_job(b) for b in BACKENDS]
    setups = [("direct_1_orch", 1, BASE_URL, False), ("gateway_1_orch", 1, gateway, True),
              ("gateway_3_orch", 3, gateway, True)]
    rows = []
    try:
        for name, orch, url, via_gateway in setups:
            _compose(1, orchestrators=orch)
            if via_gateway and orch == 1:                          # gateway in front of a single orchestrator
                subprocess.run(["docker", "compose", "--profile", "ha", "up", "-d", "--wait", "gateway"], cwd=ROOT,
                               check=True, capture_output=True, timeout=600)
            httpx.get(f"{url}/health", timeout=10).raise_for_status()
            for conc in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                res, cpu = _closed_loop_mp(jobs, conc, duration, procs, base_url=url)
                s = _summary(res, duration)
                rows.append({"condition": name, "orchestrators": orch, "gateway": via_gateway, "concurrency": conc,
                             "duration_s": duration, "client_procs": min(procs, conc), **s, "client_cpu_s": cpu})
                print(f"  {name:15s} c={conc:3d}: {s['throughput_jobs_s']:7.1f} jobs/s  client p50 "
                      f"{s['client_p50_ms']:8.2f}  p99 {s['client_p99_ms']:8.2f} ms  errors {s['errors']}  "
                      f"client CPU {cpu / (duration + 2) / min(procs, conc):.2f} core/proc")
    finally:
        _compose(1)
    save("load_ha", rows, client_procs=procs, workload=(
        "closed loop from 3 client processes; GHZ_8 256 shots round-robin over all QPUs (1 worker each); "
        "direct to one orchestrator, or through the nginx gateway (compose profile ha) to 1 or 3 orchestrators"))


def exp_load_open(duration: float = 30.0) -> None:
    """E14 — open-loop Poisson load vs utilisation, compared with the M/G/1 (Pollaczek–Khinchine) model."""
    jobs = [_heavy_job()]
    api()
    base = asyncio.run(_closed_loop(jobs, 1, 20.0))
    ok = [r for r in base if r["status"] == 200]
    service = np.array([r["service_ms"] - r["queue_ms"] for r in ok]) / 1e3      # worker occupancy per job
    resp0 = float(np.mean([r["total_ms"] for r in ok]))
    es, es2 = float(service.mean()), float((service ** 2).mean())
    capacity = 1.0 / es
    print(f"  service E[S] = {es * 1e3:.2f} ms (cv {service.std() / es:.3f}); capacity {capacity:.1f} jobs/s")
    rows, raw = [], {}
    for rho in (0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.2):
        lam = rho * capacity
        d = duration if rho < 0.8 else 2 * duration if rho < 1 else 20.0      # long windows near saturation
        res = asyncio.run(_open_loop(jobs, lam, d, seed=int(rho * 100)))
        s = _summary(res, d + 1e9) | {"throughput_jobs_s": sum(r["status"] == 200 for r in res) / d}
        wq = lam * es2 / (2 * (1 - rho)) * 1e3 if rho < 1 else None
        rows.append({"rho_target": rho, "arrival_rate": lam, "arrivals": len(res), **s,
                     "pk_mean_ms": resp0 + wq if wq is not None else None})
        raw[str(rho)] = [[round(r["t"], 4), round(r["total_ms"], 3)] for r in res if r["status"] == 200]
        print(f"  ρ={rho:4.2f}: mean {s['mean_ms']:8.2f} ms (P-K {rows[-1]['pk_mean_ms'] or float('nan'):8.2f})"
              f"  p99 {s['p99_ms']:8.2f}")
    save("load_open", rows, latencies_ms=raw, service_mean_ms=es * 1e3, service_cv=float(service.std() / es),
         service_second_moment_s2=es2, response_c1_ms=resp0, capacity_jobs_s=capacity, duration_s=duration,
         workload="Poisson arrivals of the heavy job (QV-style 12 qubits, 1024 shots, IonQ Forte, 1 replica)")


def _docker_stats() -> list[dict[str, Any]]:
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}"], capture_output=True,
                         text=True, timeout=30).stdout
    stats = []
    for line in out.splitlines():
        d = json.loads(line)
        if "qputwin" not in d["Name"]:
            continue
        mem = d["MemUsage"].split("/")[0].strip()
        unit = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, "B": 1 / 2 ** 20}
        num = next((float(mem[: -len(u)]) * f for u, f in unit.items() if mem.endswith(u)), float("nan"))
        stats.append({"name": d["Name"], "mem_mib": num, "cpu_pct": float(d["CPUPerc"].rstrip("%"))})
    return stats


def exp_soak(duration: float = 300.0, rate: float = 120.0) -> None:
    """E15 — five-minute soak: steady Poisson load over all QPUs, latency and memory over time."""
    jobs = [_light_job(b) for b in BACKENDS]
    api()
    samples: list[dict[str, Any]] = []

    async def sampler(stop: float) -> None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < stop:
            st = await asyncio.to_thread(_docker_stats)
            samples.append({"t": time.perf_counter() - t0, "containers": st})
            await asyncio.sleep(5.0)

    async def main() -> list[dict[str, Any]]:
        res, _ = await asyncio.gather(_open_loop(jobs, rate, duration, seed=42), sampler(duration + 5))
        return res

    res = asyncio.run(main())
    ok = [r for r in res if r["status"] == 200]
    print(f"  {len(ok)} ok / {len(res)} requests, p50 {np.median([r['total_ms'] for r in ok]):.2f} ms, "
          f"p99 {np.percentile([r['total_ms'] for r in ok], 99):.2f} ms")
    save("soak", [[round(r["t"], 3), round(r["total_ms"], 3), r["status"]] if r["status"] == 200
                  else [round(r["t"], 3), None, r["status"]] for r in res],
         stats=samples, rate_jobs_s=rate, duration_s=duration, summary=_summary(res, duration + 1e9) | {
             "throughput_jobs_s": len(ok) / duration},
         workload="Poisson arrivals, GHZ_8 256 shots round-robin over all QPUs, docker stats every ~5 s")


def exp_async_burst(jobs_per_backend: int = 500) -> None:
    """E16 — async burst: 500 queued jobs per QPU, queue depth and drain throughput."""
    c = api()

    async def submit_all() -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        specs = [(b, k) for k in range(jobs_per_backend) for b in BACKENDS]
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=600.0,
                                     limits=httpx.Limits(max_connections=32)) as ac:
            sem = asyncio.Semaphore(16)

            async def one(b: str, k: int) -> None:
                async with sem:
                    t = time.perf_counter()
                    r = await ac.post("/v1/jobs", params={"mode": "async"},
                                      json={"backend": b, "qasm": ghz(b, 10), "shots": 64, "seed": k})
                    r.raise_for_status()
                    out.append((r.json()["job_id"], (time.perf_counter() - t) * 1e3))

            await asyncio.gather(*(one(b, k) for b, k in specs))
        return out

    t0 = time.perf_counter()
    submitted = asyncio.run(submit_all())
    submit_s = time.perf_counter() - t0
    done = []
    for jid, _ in submitted:
        while True:
            b = c.get(f"/v1/jobs/{jid}", params={"wait": 30}).json()
            if b["status"] not in ("QUEUED", "RUNNING"):
                break
        done.append({"backend": b["backend"], "status": b["status"], "submitted_at": b["submitted_at"],
                     "started_at": b.get("started_at"), "finished_at": b.get("finished_at"),
                     "queue_ms": (b.get("telemetry") or {}).get("breakdown", {}).get("orchestrator_queue_ms"),
                     "total_ms": (b.get("telemetry") or {}).get("total_roundtrip_ms")})
    acc = [ms for _, ms in submitted]
    print(f"  accepted {len(submitted)} jobs in {submit_s:.2f} s ({len(submitted) / submit_s:.0f}/s, "
          f"p50 accept {np.median(acc):.2f} ms); {sum(d['status'] == 'COMPLETED' for d in done)} completed")
    save("async_burst", done, accept_ms=acc, submit_wall_s=submit_s, workload=(
        f"{jobs_per_backend} async GHZ_10 (64 shots) jobs per QPU submitted by 16 concurrent clients, "
        "one dispatch slot per worker replica"))


def exp_fairshare(heavy_jobs: int = 300, light_jobs: int = 20) -> None:
    """E17 — scheduler isolation: a light client's queue wait while a heavy client floods the queue."""
    c = api()
    conditions = {"same client (FIFO)": ("heavy", 5), "fair share (priority 5)": ("light", 5),
                  "priority 9": ("light", 9)}
    job = {"backend": "ionq_forte", "qasm": star_ghz(14), "shots": 64}
    rows = []
    for cname, (client, prio) in conditions.items():
        async def flood() -> list[str]:                         # the heavy backlog arrives as one burst
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=600.0) as ac:
                sem = asyncio.Semaphore(16)

                async def one(k: int) -> str:
                    async with sem:
                        r = await ac.post("/v1/jobs", params={"mode": "async"},
                                          headers={"x-qputwin-client": "heavy"},
                                          json={**job, "seed": k, "priority": 5})
                        return r.json()["job_id"]

                return list(await asyncio.gather(*(one(k) for k in range(heavy_jobs))))

        heavy_ids = asyncio.run(flood())
        light_ids = []
        for k in range(light_jobs):
            light_ids.append(c.post("/v1/jobs", params={"mode": "async"}, headers={"x-qputwin-client": client},
                                    json={**job, "seed": 10_000 + k, "priority": prio}).json()["job_id"])
            time.sleep(0.1)
        for kind, ids in (("heavy", heavy_ids), ("light", light_ids)):
            for jid in ids:
                while (b := c.get(f"/v1/jobs/{jid}", params={"wait": 30}).json())["status"] in ("QUEUED", "RUNNING"):
                    pass
                rows.append({"condition": cname, "kind": kind,
                             "queue_ms": b["telemetry"]["breakdown"]["orchestrator_queue_ms"],
                             "submitted_at": b["submitted_at"], "finished_at": b["finished_at"]})
        light = [r["queue_ms"] for r in rows if r["condition"] == cname and r["kind"] == "light"]
        print(f"  {cname:24s} light wait p50 {np.median(light):9.1f} ms  max {max(light):9.1f} ms")
    save("fairshare", rows, workload=f"{heavy_jobs} heavy-client jobs submitted as one concurrent burst, then {light_jobs} light "
                                     "jobs 100 ms apart; star GHZ_14, 64 shots, IonQ Forte, 1 replica")


def exp_shots() -> None:
    """E18 — latency and response size vs shots (live, GHZ_12)."""
    c = api()
    rows = []
    for b in BACKENDS:
        for shots in (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000):
            for k in range(7):
                r = c.post("/v1/jobs", json={"backend": b, "qasm": ghz(b, 12), "shots": shots, "seed": k})
                t = r.json()["telemetry"]
                rows.append({"backend": b, "shots": shots, "total_ms": t["total_roundtrip_ms"],
                             "sampling_ms": t["breakdown"]["sampling_ms"],
                             "simulation_ms": t["breakdown"]["simulation_ms"], "bytes": len(r.content)})
        print(f"  {b}: 1e6 shots median {statistics.median(x['total_ms'] for x in rows[-7:]):.2f} ms")
    save("shots", rows, workload="GHZ_12 along a native path, 7 repetitions per point")


def exp_gate_scaling() -> None:
    """E19 — front-end throughput: parse / route / compile time vs circuit size (live, 10 qubits)."""
    c = api()
    rows = []
    for b in BACKENDS:
        for gates in (100, 300, 1_000, 3_000, 10_000, 30_000):
            qasm = C.random_gates(10, gates, seed=gates)
            for k in range(3):
                r = c.post("/v1/jobs", json={"backend": b, "qasm": qasm, "shots": 1, "seed": k,
                                             "options": _route_opts(b)})
                r.raise_for_status()
                bd = r.json()["telemetry"]["breakdown"]
                rows.append({"backend": b, "gates": gates, "qasm_bytes": len(qasm),
                             **{k2: bd[k2] for k2 in ("validation_ms", "worker_parse_ms", "worker_route_ms",
                                                      "worker_compile_ms", "simulation_ms")},
                             "total_ms": r.json()["telemetry"]["total_roundtrip_ms"],
                             "native_two_qubit": r.json()["compilation"]["native_two_qubit_count"]})
        last = rows[-1]
        print(f"  {b}: 30k gates parse {last['worker_parse_ms']:.0f} ms, route {last['worker_route_ms']:.0f} ms, "
              f"compile {last['worker_compile_ms']:.0f} ms")
    save("gate_scaling", rows, workload="random 10-qubit circuits (70 % 1Q / 30 % CX), 1 shot; SABRE on lattices")


EXPERIMENTS = {"latency": exp_latency, "scaling": exp_scaling, "routing": exp_routing,
               "routing_latency": exp_routing_latency, "fidelity": exp_fidelity, "hwtime": exp_hwtime,
               "budget": exp_budget, "convergence": exp_convergence, "async_scaling": exp_async_scaling,
               "noise_sweep": exp_noise_sweep, "ramsey": exp_ramsey, "circuits": exp_circuits, "qv": exp_qv,
               "load_closed": exp_load_closed, "load_ha": exp_load_ha, "load_open": exp_load_open, "soak": exp_soak,
               "async_burst": exp_async_burst, "fairshare": exp_fairshare, "shots": exp_shots,
               "gate_scaling": exp_gate_scaling, "saturation": exp_saturation}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "all"])
    args = ap.parse_args(argv)
    names = list(EXPERIMENTS) if "all" in args.experiments else args.experiments
    for name in names:
        print(f"[{name}] {EXPERIMENTS[name].__doc__.strip().splitlines()[0]}")
        t0 = time.perf_counter()
        EXPERIMENTS[name]()
        print(f"  ({time.perf_counter() - t0:.1f} s)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
