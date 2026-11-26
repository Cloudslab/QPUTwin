"""Runs the IBM and Google emulation workflows and the matching QPUTwin job.

Usage: from workflows import ibm_job, google_job, qputwin_job

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import re
import threading
import time
import warnings
from functools import cache
from typing import Any

import adapters as A

warnings.filterwarnings("ignore")

VENDOR = {"ibm_heron": "fake_boston", "ibm_nighthawk": "fake_miami", "google_willow": "willow_pink"}


@cache
def _ibm(vendor: str, threads: int):
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
    fake = FakeProviderForBackendV2().backend(vendor)
    return fake, AerSimulator.from_backend(fake, max_parallel_threads=threads)


@cache
def _google(vendor: str, threads: int, exact: bool):
    import cirq
    import cirq_google as cg
    import qsimcirq
    dev = cg.engine.create_device_from_processor_id(vendor)
    model = cg.NoiseModelFromGoogleNoiseProperties(cg.engine.load_device_noise_properties(vendor))
    sim = (cirq.DensityMatrixSimulator(noise=model) if exact
           else qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=threads), noise=model))
    return dev, sim


def _counts_from_bits(rows) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = "".join(str(int(b)) for b in row[::-1])
        counts[key] = counts.get(key, 0) + 1
    return counts


def ibm_job(vendor: str, qasm: str, shots: int, seed: int, threads: int = 1) -> dict[str, Any]:
    from qiskit import qasm2, transpile
    fake, sim = _ibm(vendor, threads)
    t0 = time.perf_counter()
    qc = qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    tqc = transpile(qc, backend=fake, optimization_level=1, seed_transpiler=0)
    t1 = time.perf_counter()
    res = sim.run(tqc, shots=shots, seed_simulator=seed).result()
    t2 = time.perf_counter()
    ops = tqc.count_ops()
    return {"counts": dict(res.get_counts()), "run_s": t2 - t0, "compile_s": t1 - t0, "simulate_s": t2 - t1,
            "two_qubit": int(ops.get("cz", 0) + ops.get("ecr", 0)),
            "method": res.results[0].metadata.get("method"),
            "active_qubits": len({tqc.find_bit(q).index for inst in tqc.data for q in inst.qubits})}


def google_job(vendor: str, qasm: str, shots: int, seed: int, threads: int = 1, exact: bool = False
               ) -> dict[str, Any]:
    import cirq
    from cirq.contrib.qasm_import import circuit_from_qasm
    dev, sim = _google(vendor, threads, exact)
    t0 = time.perf_counter()
    gs = cirq.CZTargetGateset()
    # Cirq has no barrier: optimise each barrier-delimited segment on its own (as Qiskit and QPUTwin
    # do), so the optimiser cannot cancel across it (a mirror circuit would otherwise vanish)
    lines = qasm.splitlines()
    head = [ln for ln in lines if ln.strip().split(" ")[0] in ("OPENQASM", "include", "qreg", "creg")]
    body = [ln for ln in lines if ln not in head]
    segs, cur = [], []
    for ln in body:
        if ln.strip().startswith("barrier"):
            segs.append(cur)
            cur = []
        else:
            cur.append(ln)
    segs.append(cur)
    c = cirq.Circuit()
    for seg in segs:
        if any(ln.strip() for ln in seg):
            c += cirq.optimize_for_target_gateset(circuit_from_qasm("\n".join(head + seg)), gateset=gs)
    routed, _, _ = cirq.RouteCQC(dev.metadata.nx_graph).route_circuit(c)
    if len(segs) > 1:      # with a barrier only the inserted SWAPs are lowered, op by op, without merging
        routed = cirq.map_operations_and_unroll(
            routed, lambda op, i: op if op in gs else gs.decompose_to_target_gateset(op, i))
        routed = cirq.drop_empty_moments(routed)
    else:
        routed = cirq.optimize_for_target_gateset(routed, gateset=gs)
    dev.validate_circuit(routed)
    t1 = time.perf_counter()
    res = sim.run(routed, repetitions=shots)
    t2 = time.perf_counter()
    keys = sorted(res.measurements, key=lambda k: int(re.findall(r"\d+", k)[-1]))
    import numpy as np
    bits = np.concatenate([res.measurements[k] for k in keys], axis=1)
    two_q = sum(1 for op in routed.all_operations() if len(op.qubits) == 2 and not cirq.is_measurement(op))
    return {"counts": _counts_from_bits(bits), "run_s": t2 - t0, "compile_s": t1 - t0, "simulate_s": t2 - t1,
            "two_qubit": two_q, "method": "cirq_density_matrix" if exact else "qsim_noisy_trajectories",
            "active_qubits": len(routed.all_qubits())}


_clients = threading.local()


def _client() -> A.QPUTwin:
    """One HTTP client (connection pool) per thread, as a long-running client would keep."""
    c = getattr(_clients, "c", None)
    if c is None:
        c = _clients.c = A.QPUTwin()
    return c


def qputwin_job(backend: str, qasm: str, shots: int, seed: int, noise_aware: bool = True) -> dict[str, Any]:
    routing = {"method": "sabre", "seed": 0, "noise_aware": noise_aware}
    run = _client().sample(qasm, shots, seed, {"routing": routing, "noise": {}}, backend)
    comp = run.extra.get("compilation", {})
    return {"counts": run.counts, "run_s": run.run_s, "engine_s": run.engine_s,
            "two_qubit": comp.get("native_two_qubit_count"), "method": run.extra.get("engine"),
            "active_qubits": comp.get("simulated_qubits")}


def job(pipeline: str, backend: str, qasm: str, shots: int, seed: int, threads: int = 1) -> dict[str, Any]:
    """One device-emulation job through ``pipeline`` (``vendor``, ``google_dm``, ``qputwin`` or ``qputwin_plain``,
    the latter without the calibration-aware re-layout)."""
    vendor = VENDOR[backend]
    if pipeline.startswith("qputwin"):
        return qputwin_job(backend, qasm, shots, seed, noise_aware=pipeline == "qputwin")
    if vendor.startswith("fake_"):
        return ibm_job(vendor, qasm, shots, seed, threads)
    return google_job(vendor, qasm, shots, seed, threads, exact=pipeline == "google_dm")


def warm(pipeline: str, backend: str, threads: int = 1) -> None:
    """Build the per-device simulator objects (outside any timed region)."""
    vendor = VENDOR[backend]
    if pipeline.startswith("qputwin"):
        return
    if vendor.startswith("fake_"):
        _ibm(vendor, threads)
    else:
        _google(vendor, threads, pipeline == "google_dm")
