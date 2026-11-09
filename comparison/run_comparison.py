"""Comparative benchmark: QPUTwin vs Qiskit Aer, Amazon Braket local, Cirq, qsim and Qulacs.

QPUTwin is the **already running** Docker stack (``QPUTWIN_BASE_URL``, default
http://localhost:8000); this script never starts, stops or reconfigures it.

Experiments (each writes ``results/<name>.json`` with environment metadata):

* ``correctness``  exact ideal output distributions of nine circuit families, every
                   simulator against Qiskit Aer (and QPUTwin), 4–16 qubits
* ``sv_speed``     ideal statevector run time (simulate + 1024 shots) vs width for GHZ,
                   QV-style and QFT-adder circuits; ``COMPARE_THREADS`` = 1 (default), 4 (QPUTwin as
                   the running stack) or 16 (all in-process, widths to 28)
* ``dm_speed``     noisy density-matrix run time vs width (same thread budgets)
* ``noise_xval``   QPUTwin's calibrated noise vs an independent calibration-matched Aer
                   model, run on QPUTwin's identical compiled circuit, all seven devices
* ``vendor_xval``  the IBM and Google twins vs the vendors' own noise models (Aer
                   ``from_backend`` of IBM's snapshots, Cirq's ``willow_pink`` model)
* ``vendor_workflow``  device emulation end to end: each vendor's own local-emulation
                   workflow (Qiskit transpile + Aer from_backend; Cirq routing + qsim) vs QPUTwin
* ``workflow_throughput``  96 emulation jobs on the whole host (``WORKFLOW_BACKEND``), vendor
                   workflows in a process pool vs QPUTwin worker replicas (configured by the driver)

Usage (from this folder)::

    uv run python run_comparison.py correctness sv_speed dm_speed noise_xval vendor_xval
    COMPARE_THREADS=4 uv run python run_comparison.py sv_speed dm_speed

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import os

THREADS = int(os.environ.get("COMPARE_THREADS", "1"))
os.environ["OMP_NUM_THREADS"] = str(THREADS)    # OpenMP simulators (Qulacs reads it at import); must precede imports
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"                        # numpy's BLAS stays single-threaded: QPUTwin threads its own
                                                # blocks, and a threaded BLAS inside them would oversubscribe
# QPUTwin with 1 thread, or with more threads than a stack worker's 4-CPU quota, runs in-process
LOCAL = THREADS == 1 or THREADS > 4

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402

import adapters as A  # noqa: E402
import noise_models as N  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SHOTS = 1024


# ================================================================ helpers
def environment() -> dict[str, Any]:
    cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")), "unknown")
    try:
        stack = httpx.get(f"{A.BASE_URL}/health", timeout=10).json()
        dev = httpx.get(f"{A.BASE_URL}/v1/backends/quantinuum_helios", timeout=10).json()["emulator"]
        qputwin = {"base_url": A.BASE_URL, **stack, "worker_precision": dev["precision"],
                   "worker_engine_threads": "the worker's CPU quota (QPUTWIN_WORKER_CPUS, default 4)",
                   "worker_cpu_quota": "4 CPUs / 12 GiB unless overridden"}
    except httpx.HTTPError:            # e.g. the capacity sweep, which stops the stack to free the host
        qputwin = {"base_url": A.BASE_URL, "stack": "not running (in-process QPUTwin only)"}
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "load_average": os.getloadavg(), "threads": THREADS,
            "versions": A.versions(), "qputwin": qputwin}


def save(name: str, data: Any, **meta: Any) -> None:
    RESULTS.mkdir(exist_ok=True)
    doc = {"experiment": name, "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
           "environment": environment(), **meta, "data": data}
    (RESULTS / f"{name}.json").write_text(json.dumps(doc, indent=1))
    print(f"  saved results/{name}.json", flush=True)


def hellinger(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in set(p) | set(q)) ** 2


def normalise(counts: dict[str, int]) -> dict[str, float]:
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()}


def tvd(p: dict[str, float], q: dict[str, float]) -> float:
    """Total-variation distance between two distributions."""
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


# ============================================================ experiments
def exp_correctness() -> None:
    """Exact ideal distributions: every simulator vs Qiskit Aer, nine families, 4–16 qubits."""
    sims = [A.QPUTwin(), A.Aer(), A.Braket(), A.Cirq(), A.Qsim(), A.Qulacs()]
    rows = []
    for fam, gen in A.C.FAMILIES.items():
        for n in (4, 8, 12, 16):
            if fam == "adder" and n % 2:
                continue
            qasm = gen(n, seed=1)
            flat = A.flatten(qasm)
            probs = {s.name: (s.exact(qasm, flat) if isinstance(s, A.QPUTwin) else s.exact(flat)).probs
                     for s in sims}
            ref, twin = probs["Qiskit Aer"], probs["QPUTwin"]
            for name, p in probs.items():
                keys = set(p) | set(ref)
                rows.append({"family": fam, "qubits": n, "simulator": name, "support": len(ref),
                             "max_abs_dp_vs_aer": max(abs(p.get(k, 0) - ref.get(k, 0)) for k in keys),
                             "tv_vs_aer": 0.5 * sum(abs(p.get(k, 0) - ref.get(k, 0)) for k in keys),
                             "tv_vs_qputwin": 0.5 * sum(abs(p.get(k, 0) - twin.get(k, 0))
                                                        for k in set(p) | set(twin))})
            worst = max(r["max_abs_dp_vs_aer"] for r in rows[-len(sims):] if r["simulator"] != "qsim")
            print(f"  {fam:6s} n={n:2d} support={len(ref):5d}  worst |Δp| (double precision) {worst:.1e}", flush=True)
    save("correctness", rows, reference="Qiskit Aer statevector", workload=(
        "nine circuit families from benchmarks/circuits.py (seed 1); exact probabilities of the measured bits; "
        "QPUTwin via return_statevector on quantinuum_helios (all-to-all, no routing); qsim runs in complex64"))


def _timed(fn) -> tuple[list[float], list[float | None]]:
    """Run ``fn`` once, or three times if it is cheap; return (run times, engine times)."""
    runs = [fn()]
    if runs[0].run_s < 2.0:
        runs += [fn(), fn()]
    return [r.run_s for r in runs], [r.engine_s for r in runs]


def _speed_sweep(name: str, families: tuple[str, ...], widths: tuple[int, ...],
                 runners: list[tuple[str, Any]], cap_s: float) -> list[dict[str, Any]]:
    rows = []
    for fam in families:
        for label, fn in runners:
            for n in widths:
                qasm = A.C.FAMILIES[fam](n, seed=1)
                flat = A.flatten(qasm)
                times, engines = _timed(lambda q=qasm, f=flat: fn(q, f))
                eng = [e for e in engines if e is not None]
                rows.append({"family": fam, "qubits": n, "simulator": label, "threads": THREADS,
                             "gates": len(flat.ops), "run_s": times, "run_s_best": min(times),
                             "engine_s": eng, "engine_s_best": min(eng) if eng else None})
                extra = f" (engine {min(eng):.3f} s)" if eng else ""
                print(f"  {name} {fam:4s} {label:22s} n={n:2d} {min(times):9.3f} s{extra}", flush=True)
                if min(times) > cap_s:
                    break                                         # larger widths would only be slower
    return rows


def exp_sv_speed(cap_s: float = 60.0) -> None:
    """Ideal statevector time (simulate + 1024 shots) vs width, at the given thread budget."""
    aer, qsim, qulacs = A.Aer(threads=THREADS), A.Qsim(threads=THREADS), A.Qulacs()
    runners: list[tuple[str, Any]] = []
    if LOCAL:
        local = A.QPUTwinLocal(threads=THREADS)
        runners += [("QPUTwin (in-process)", lambda q, f: local.sample(q, SHOTS, 1)),
                    ("QPUTwin (complex64)", lambda q, f: local.sample(q, SHOTS, 1, {"precision": "complex64"}))]
    else:
        twin = A.QPUTwin()
        runners += [("QPUTwin", lambda q, f: twin.sample(q, SHOTS, 1)),
                    ("QPUTwin (complex64)", lambda q, f: twin.sample(q, SHOTS, 1, {"precision": "complex64"}))]
    runners += [("Qiskit Aer", lambda q, f: aer.sample(f, SHOTS, 1)), ("qsim", lambda q, f: qsim.sample(f, SHOTS, 1)),
                ("Qulacs", lambda q, f: qulacs.sample(f, SHOTS, 1))]
    if THREADS == 1:
        braket, cirq = A.Braket(), A.Cirq()
        runners += [("Braket local", lambda q, f: braket.sample(f, SHOTS, 1)),
                    ("Cirq", lambda q, f: cirq.sample(f, SHOTS, 1))]
    widths = (6, 10, 14, 16, 18, 20, 22, 24) + ((26, 28) if THREADS > 4 else ())
    rows = _speed_sweep("sv", ("ghz", "qv", "qft"), widths, runners, cap_s)
    save(f"sv_speed_t{THREADS}", rows, shots=SHOTS, cap_s=cap_s, workload=(
        "ideal circuits; in-process simulators get the identical U/CX list, QPUTwin the source QASM (compiled to "
        "Helios natives). run_s = simulate + sample 1024 shots, or the client-measured HTTP round trip for the "
        "QPUTwin stack; engine_s = QPUTwin simulation + sampling from its telemetry. With 4 threads QPUTwin is the "
        "running stack (engine threads = worker CPU quota = 4); with 1 or more than 4 threads it is the same code "
        "in-process, with numpy's BLAS pinned to one thread and every simulator given the same thread count. "
        f"Best of 3 repetitions below 2 s; a series stops after the first width above {cap_s:.0f} s"))


def exp_dm_speed(cap_s: float = 120.0) -> None:
    """Noisy density-matrix time vs width, at the given thread budget."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    lam = (1e-3, 1e-2)                               # Qiskit depolarising λ after every U and CX
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(lam[0], 1), ["u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(lam[1], 2), ["cx"])
    aer = A.Aer(threads=THREADS, method="density_matrix")
    noisy = {"noise": {}, "engine": "density_matrix"}
    runners: list[tuple[str, Any]] = []
    if LOCAL:
        local = A.QPUTwinLocal(threads=THREADS)
        runners.append(("QPUTwin (in-process)", lambda q, f: local.sample(q, SHOTS, 1, noisy)))
    else:
        twin = A.QPUTwin()
        runners.append(("QPUTwin", lambda q, f: twin.sample(q, SHOTS, 1, noisy)))
    runners.append(("Qiskit Aer", lambda q, f: aer.sample(f, SHOTS, 1, noise_model=nm)))
    if THREADS == 1:
        braket, cirq = A.Braket("braket_dm"), A.Cirq(density_matrix=True)
        runners += [("Braket local", lambda q, f: braket.sample(f, SHOTS, 1, noise=lam)),
                    ("Cirq", lambda q, f: cirq.sample(f, SHOTS, 1, noise=lam))]
    widths = (4, 6, 8, 10, 11, 12, 13) + ((14,) if THREADS > 4 else ())
    rows = _speed_sweep("dm", ("ghz", "qv"), widths, runners, cap_s)
    save(f"dm_speed_t{THREADS}", rows, shots=SHOTS, cap_s=cap_s, depolarizing_lambda=lam, workload=(
        "noisy density-matrix simulation; QPUTwin: its full calibrated Helios noise model on native gates (engine "
        "forced to density_matrix); others: depolarising λ = 1e-3 after every U and 1e-2 after every CX of the "
        "identical U/CX list"))


def _local_plan(backend: str, qasm: str, opts: dict[str, Any]):
    """QPUTwin's compiled native circuit for this job, computed in-process with the stack's own code."""
    sys.path.insert(0, str(HERE.parent / "src"))
    from qputwin_core.devices import DEVICES
    from qputwin_core.simulators import ExecutionOptions, simulator_for
    from qputwin_core.timing import SpanTimer
    spec = DEVICES[backend]
    plan = simulator_for(spec).plan(qasm, ExecutionOptions(shots=1, routing=opts.get("routing")), SpanTimer())
    return spec, plan.native


def _checked_native(backend: str, qasm: str, opts: dict[str, Any], http_run: A.Run):
    """The in-process compile, asserted identical to what the running worker executed."""
    spec, native = _local_plan(backend, qasm, opts)
    local: dict[str, int] = {}
    for op in native.ops:
        if op.is_gate:
            local[op.name] = local.get(op.name, 0) + 1
    remote = http_run.extra["compilation"]["native_gate_histogram"]
    assert local == remote, f"{backend}: local compile {local} differs from the worker's {remote}"
    return spec, native


CASES = [(fam, n) for fam in ("ghz", "bv", "mirror", "adder", "qft", "qv") for n in range(3, 9)
         if not (fam == "adder" and n % 2)]


def exp_noise_xval(shots: int = 100_000) -> None:
    """QPUTwin's calibrated noise vs an independent Aer model on the identical compiled circuit, seven devices."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    twin = A.QPUTwin()
    ideal_sim = A.Aer()
    aer = AerSimulator(method="density_matrix", max_parallel_threads=THREADS)
    rows = []
    for backend in N.NATIVE:
        model = N.device_model(backend, A.BASE_URL)
        opts = {"routing": {"method": "sabre", "seed": 0}} if model.coupling is not None else {}
        for fam, n in CASES:
            qasm = A.C.FAMILIES[fam](n, seed=1)
            flat = A.flatten(qasm)
            ideal = ideal_sim.exact(flat).probs
            full = twin.sample(qasm, shots, 7, {**opts, "noise": {}, "engine": "density_matrix"}, backend)
            no_idle = twin.sample(qasm, shots, 7, {**opts, "noise": {"idle": False}, "engine": "density_matrix"},
                                  backend)
            spec, native = _checked_native(backend, qasm, opts, full)
            virtual = set(spec.virtual_gates)
            qc, used = N.native_circuit(native, virtual)
            nm = N.matched_model_for(qc, spec.calibration, used, set(spec.native_gates), virtual)
            t0 = time.perf_counter()
            res = aer.run(qc, shots=shots, seed_simulator=8, noise_model=nm).result()
            aer_s = time.perf_counter() - t0
            aer_counts = dict(res.get_counts())
            tqc = transpile(flat.qiskit, basis_gates=model.basis, coupling_map=model.coupling,
                            optimization_level=1, seed_transpiler=0)
            ops = tqc.count_ops()
            rows.append({
                "backend": backend, "family": fam, "qubits": n, "physical_qubits": used,
                "qputwin_full": hellinger(ideal, normalise(full.counts)),
                "qputwin_no_idle": hellinger(ideal, normalise(no_idle.counts)),
                "aer_matched": hellinger(ideal, normalise(aer_counts)),
                "tvd_no_idle_vs_aer": tvd(normalise(no_idle.counts), normalise(aer_counts)),
                "qputwin_two_qubit": full.extra["compilation"]["native_two_qubit_count"],
                "qiskit_two_qubit": sum(v for k, v in ops.items() if k in ("cz", "rzz", "rxx")),
                "qputwin_one_qubit": full.extra["compilation"]["native_one_qubit_count"],
                "qputwin_engine_s": full.engine_s, "aer_run_s": aer_s,
            })
            r = rows[-1]
            print(f"  {backend:18s} {fam:6s} n={n}  QPUTwin {r['qputwin_full']:.4f} (no idle "
                  f"{r['qputwin_no_idle']:.4f})  Aer-matched {r['aer_matched']:.4f}  TVD {r['tvd_no_idle_vs_aer']:.4f}"
                  f"   2Q {r['qputwin_two_qubit']}/{r['qiskit_two_qubit']}", flush=True)
    save("noise_xval", rows, shots=shots, models={b: {"basis": N.NATIVE[b]["basis"], "map": N.NATIVE[b]["map"]}
                                                  for b in N.NATIVE}, workload=(
        "Hellinger fidelity to the exact ideal distribution (Aer statevector). QPUTwin: its own compiler (SABRE seed 0 "
        "on lattices), calibrated noise on the density-matrix engine, with and without idle-window decoherence. "
        "Aer-matched: QPUTwin's identical compiled circuit (same gates, same physical qubits, asserted equal to the "
        "worker's gate histogram), rebuilt with Qiskit gates, with an independently written per-qubit model "
        "(noise_models.py: relaxation then depolarising solved to each gate's own fidelity, per-qubit T1/T2 and "
        "readout); 100 000 shots each. tvd = total-variation distance between the two noisy distributions. The 2Q "
        "columns compare QPUTwin's compiler with Qiskit transpile (optimization_level=1, seed 0)"))


VENDOR = {"ibm_heron": "fake_boston", "ibm_nighthawk": "fake_miami", "google_willow": "willow_pink"}


def _ibm_physical(native, n_device: int):
    """The compiled circuit on IBM's full register with IBM's own gate names (sx, x, rz, cz)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_device, native.num_clbits)
    for op in native.ops:
        if op.name == "measure":
            qc.measure(op.qubits[0], op.clbits[0])
        elif op.name in ("sx", "x", "cz"):
            getattr(qc, op.name)(*op.qubits)
        elif op.name == "rz":
            qc.rz(op.params[0], op.qubits[0])
        elif op.name != "barrier":
            raise ValueError(op.name)
    return qc


def _cirq_counts(native, shots: int, seed: int, measurement_decay: bool = True) -> dict[str, int]:
    """The compiled circuit on Willow's GridQubits, through Cirq's own noise model of willow_pink.

    ``measurement_decay=False`` sets the measurement duration to 0 in Cirq's thermal model, so qubits do
    not also decay for the 600 ns read-out on top of the measured read-out error (which already contains it).
    """
    import dataclasses

    import cirq
    import cirq_google as cg
    import numpy as np
    dev = cg.engine.create_device_from_processor_id("willow_pink")
    grid = sorted(dev.metadata.qubit_set)
    ops = []
    for op in native.ops:
        qs = [grid[q] for q in op.qubits]
        if op.name == "phased_xz":
            t, p, z = op.params
            ops.append(cirq.PhasedXZGate(x_exponent=t / math.pi, z_exponent=z / math.pi,
                                         axis_phase_exponent=p / math.pi)(*qs))
        elif op.name == "rz":
            ops.append(cirq.Z(*qs) ** (op.params[0] / math.pi))
        elif op.name == "cz":
            ops.append(cirq.CZ(*qs))
        elif op.name == "measure":
            ops.append(cirq.measure(*qs, key=f"c{op.clbits[0]}"))
        elif op.name != "barrier":
            raise ValueError(op.name)
    circuit = cirq.Circuit(ops)
    dev.validate_circuit(circuit)
    props = cg.engine.load_device_noise_properties("willow_pink")
    if not measurement_decay:
        times = dict(props.gate_times_ns)
        times[cirq.MeasurementGate] = 0.0
        props = dataclasses.replace(props, gate_times_ns=times)
    model = cg.NoiseModelFromGoogleNoiseProperties(props)
    res = cirq.DensityMatrixSimulator(noise=model, seed=seed).run(circuit, repetitions=shots)
    ncl = native.num_clbits
    bits = np.stack([res.measurements[f"c{k}"][:, 0] for k in range(ncl)], axis=1)
    counts: dict[str, int] = {}
    for row in bits:
        key = "".join(str(int(b)) for b in row[::-1])
        counts[key] = counts.get(key, 0) + 1
    return counts


def exp_vendor_xval(shots: int = 100_000) -> None:
    """The IBM and Google twins vs the vendors' own noise models of the same snapshot, on identical circuits."""
    import warnings
    warnings.filterwarnings("ignore")
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
    twin = A.QPUTwin()
    ideal_sim = A.Aer()
    aer = AerSimulator(method="density_matrix", max_parallel_threads=THREADS)
    provider = FakeProviderForBackendV2()
    rows = []
    for backend, vendor in VENDOR.items():
        fake = provider.backend(vendor) if vendor.startswith("fake_") else None
        vendor_nm = NoiseModel.from_backend(fake) if fake else None
        median = N.device_model(backend, A.BASE_URL)            # a twin built from device medians only
        opts = {"routing": {"method": "sabre", "seed": 0}}
        for fam, n in CASES:
            qasm = A.C.FAMILIES[fam](n, seed=1)
            ideal = ideal_sim.exact(A.flatten(qasm)).probs
            full = twin.sample(qasm, shots, 7, {**opts, "noise": {}, "engine": "density_matrix"}, backend)
            no_idle = twin.sample(qasm, shots, 7, {**opts, "noise": {"idle": False}, "engine": "density_matrix"},
                                  backend)
            spec, native = _checked_native(backend, qasm, opts, full)
            t0 = time.perf_counter()
            if fake is not None:
                phys = _ibm_physical(native, fake.num_qubits)
                vendor_counts = dict(aer.run(phys, shots=shots, seed_simulator=7, noise_model=vendor_nm)
                                     .result().get_counts())
                vendor_s = time.perf_counter() - t0
                floor_counts = dict(aer.run(phys, shots=shots, seed_simulator=9, noise_model=vendor_nm)
                                    .result().get_counts())
                vendor_alt = None
            else:
                vendor_counts = _cirq_counts(native, shots, 7)
                vendor_s = time.perf_counter() - t0
                floor_counts = _cirq_counts(native, shots, 9)
                vendor_alt = _cirq_counts(native, shots, 11, measurement_decay=False)
            # baseline: the same circuit with one median value per gate type, as a launch-blog twin would have
            virtual = set(spec.virtual_gates)
            qc, _used = N.native_circuit(native, virtual)
            med_counts = dict(aer.run(qc, shots=shots, seed_simulator=8,
                                      noise_model=_median_model(median, qc, virtual)).result().get_counts())
            pv, pf, pn, pm, p0 = (normalise(c) for c in (vendor_counts, full.counts, no_idle.counts, med_counts,
                                                          floor_counts))
            row = {"backend": backend, "vendor_model": vendor, "family": fam, "qubits": n,
                   "physical_qubits": sorted({q for op in native.ops for q in op.qubits}),
                   "vendor": hellinger(ideal, pv), "qputwin_full": hellinger(ideal, pf),
                   "qputwin_no_idle": hellinger(ideal, pn), "median_model": hellinger(ideal, pm),
                   "tvd_no_idle_vs_vendor": tvd(pn, pv), "tvd_full_vs_vendor": tvd(pf, pv),
                   "tvd_median_vs_vendor": tvd(pm, pv), "tvd_sampling_floor": tvd(p0, pv),
                   "vendor_run_s": vendor_s, "native_two_qubit": full.extra["compilation"]["native_two_qubit_count"]}
            if vendor_alt is not None:
                pa = normalise(vendor_alt)
                row |= {"vendor_no_measurement_decay": hellinger(ideal, pa), "tvd_full_vs_vendor_no_meas": tvd(pf, pa)}
            rows.append(row)
            r = rows[-1]
            alt = (f"  [no meas. decay {r['vendor_no_measurement_decay']:.4f}, TVD full {r['tvd_full_vs_vendor_no_meas']:.4f}]"
                   if vendor_alt is not None else "")
            print(f"  {backend:14s} {fam:6s} n={n}  vendor {r['vendor']:.4f}  QPUTwin no-idle {r['qputwin_no_idle']:.4f} "
                  f"(TVD {r['tvd_no_idle_vs_vendor']:.4f})  full {r['qputwin_full']:.4f} (TVD {r['tvd_full_vs_vendor']:.4f})"
                  f"  median-only {r['median_model']:.4f} (TVD {r['tvd_median_vs_vendor']:.4f})  floor "
                  f"{r['tvd_sampling_floor']:.4f}{alt}", flush=True)
    save("vendor_xval", rows, shots=shots, vendor_models={
        "ibm_heron": "Qiskit Aer NoiseModel.from_backend(FakeBoston) — IBM's ibm_boston snapshot (qiskit-ibm-runtime)",
        "ibm_nighthawk": "Qiskit Aer NoiseModel.from_backend(FakeMiami) — IBM's ibm_miami snapshot",
        "google_willow": "Cirq NoiseModelFromGoogleNoiseProperties(load_device_noise_properties('willow_pink')); "
                         "also run with the measurement duration set to 0 in its thermal model"},
        workload=(
        "QPUTwin's compiled circuit (SABRE seed 0; asserted equal to the worker's) run unchanged, on the same physical "
        "qubits, through the vendor's own noise model built from the same published snapshot. QPUTwin full = with "
        "idle-window decoherence from its ASAP schedule; no-idle = without (neither vendor model idles an "
        "unscheduled circuit). median-only = an Aer model with one median value per gate type and per-device T1/T2/"
        "readout, i.e. a twin without per-qubit data. Hellinger fidelity to the exact ideal distribution; tvd = "
        "total-variation distance to the vendor model's distribution; tvd_sampling_floor = the vendor model against "
        "itself with another seed (the distance two independent 100 000-shot samples of one distribution show). "
        "100 000 shots each"))


def _median_model(model: N.DeviceModel, qc, virtual: set[str]):
    """Device-median errors attached to the labelled native unitaries of ``qc`` (no per-qubit data)."""
    from qiskit_aer.noise import NoiseModel, ReadoutError
    cal = model.calibration
    t1, t2 = cal["coherence"]["t1_us"], cal["coherence"]["t2_us"]
    nm = NoiseModel(basis_gates=["unitary", "rz", "measure"])
    labels = {inst.operation.label for inst in qc.data if inst.operation.name == "unitary"}
    for label in labels - virtual:
        g = cal["gates"][label]
        nq = 2 if label in N.TWO_QUBIT_NATIVES else 1
        err, _ = N.gate_error(g["fidelity"], g["duration_ns"] or 0.0, t1, t2, nq)
        nm.add_all_qubit_quantum_error(err, [label])
    p01, p10 = cal["readout"]["p01"], cal["readout"]["p10"]
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))
    return nm


WORKFLOW_FAMS = ("ghz", "bv", "qaoa", "qft", "qv", "mirror")


def exp_vendor_workflow(shots: int = 1024, reps: int = 3, cap_s: float = 120.0) -> None:
    """Device emulation end to end: each vendor's own local-emulation workflow vs the QPUTwin twin.

    The same logical program goes through (a) IBM's workflow (``transpile`` to the fake backend, then
    ``AerSimulator.from_backend``), Google's workflow (Cirq ``RouteCQC`` + ``CZTargetGateset``, then qsim
    or Cirq's density matrix with Google's noise model) and (b) one QPUTwin job. Every tool uses its own
    default engine choice and the same thread budget. Reported: time per job (best of ``reps``), the
    native two-qubit count, the simulated register and the predicted Hellinger fidelity to the ideal output.
    """
    import workflows as W
    ideal_sim = A.Aer()
    widths = (4, 6, 8, 10, 12, 14)
    rows = []
    for backend in W.VENDOR:
        pipes = ["qputwin", "qputwin_plain", "vendor"] + (["google_dm"] if backend == "google_willow" else [])
        for p in pipes:
            W.warm(p, backend, THREADS)
        for fam in WORKFLOW_FAMS:
            for p in pipes:
                for n in widths:
                    qasm = A.C.FAMILIES[fam](n, seed=1)
                    ideal = ideal_sim.exact(A.flatten(qasm)).probs
                    runs = []
                    try:
                        for k in range(reps):
                            runs.append(W.job(p, backend, qasm, shots, seed=11 + k, threads=THREADS))
                            if runs[-1]["run_s"] > cap_s:
                                break
                    except Exception as exc:                 # a time-out or failure ends this series, recorded
                        rows.append({"backend": backend, "pipeline": p, "family": fam, "qubits": n,
                                     "threads": THREADS, "status": "failed", "error": str(exc)[:300]})
                        print(f"  {backend:14s} {p:9s} {fam:6s} n={n:2d} FAILED {str(exc)[:120]}", flush=True)
                        break
                    best = min(runs, key=lambda r: r["run_s"])
                    row = {"backend": backend, "pipeline": p, "family": fam, "qubits": n, "threads": THREADS,
                           "status": "ok", "run_s": [r["run_s"] for r in runs], "run_s_best": best["run_s"],
                           "compile_s": best.get("compile_s"), "simulate_s": best.get("simulate_s"),
                           "engine_s": best.get("engine_s"), "method": best["method"],
                           "two_qubit": best["two_qubit"], "active_qubits": best["active_qubits"],
                           "hellinger": hellinger(ideal, normalise(runs[0]["counts"]))}
                    rows.append(row)
                    print(f"  {backend:14s} {p:9s} {fam:6s} n={n:2d} {best['run_s']:8.3f} s  2q {row['two_qubit']}"
                          f"  F_H {row['hellinger']:.3f}  [{row['method']}]", flush=True)
                    save(f"vendor_workflow_t{THREADS}", rows, shots=shots, reps=reps, cap_s=cap_s, workload=(
                        "device emulation end to end from the same OpenQASM source: IBM = qiskit.transpile("
                        "backend=FakeBoston/FakeMiami, optimization_level=1, seed 0) + AerSimulator.from_backend "
                        "(Aer's automatic method); Google = Cirq RouteCQC on willow_pink + CZTargetGateset + qsim "
                        "(QSimSimulator with NoiseModelFromGoogleNoiseProperties) or Cirq DensityMatrixSimulator "
                        "(google_dm); QPUTwin = one HTTP job (SABRE seed 0, calibrated noise, engine auto). "
                        "Per-device simulator objects are built outside the timed region. hellinger = predicted "
                        "Hellinger fidelity to the exact ideal output"))
                    if best["run_s"] > cap_s or (p == "google_dm" and n >= 6):
                        break                       # Cirq's exact density matrix: minutes per job from 8 qubits


def _throughput_jobs(backend: str) -> list[tuple[str, str, int, int]]:
    return [(fam, A.C.FAMILIES[fam](n, seed=1), n, seed) for fam in WORKFLOW_FAMS for n in (4, 6, 8, 10)
            for seed in range(4)]


def _pool_init(pipeline: str, backend: str) -> None:
    import workflows as W
    W.warm(pipeline, backend, 1)


def _pool_job(args: tuple[str, str, str, int, int]) -> tuple[float, int]:
    import resource

    import workflows as W
    pipeline, backend, qasm, n, seed = args
    t0 = time.perf_counter()
    W.job(pipeline, backend, qasm, SHOTS, seed, threads=1)
    return time.perf_counter() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def exp_workflow_throughput() -> None:
    """A fixed batch of 96 device-emulation jobs on the whole host (WORKFLOW_BACKEND, WORKFLOW_PARALLEL).

    Vendor workflows: WORKFLOW_PARALLEL processes, one thread each. QPUTwin: as many concurrent clients
    against the running stack, which the driver has configured with that many single-CPU worker replicas
    of the device (recorded below from /v1/backends).
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    backend = os.environ.get("WORKFLOW_BACKEND", "ibm_heron")
    par = int(os.environ.get("WORKFLOW_PARALLEL", str(os.cpu_count() or 1)))
    jobs = _throughput_jobs(backend)
    info = httpx.get(f"{A.BASE_URL}/v1/backends/{backend}", timeout=10).json()
    rows = []
    for pipeline in ("qputwin", "vendor"):
        t0 = time.perf_counter()
        if pipeline == "qputwin":
            import workflows as W

            def one(j: tuple[str, str, int, int]) -> tuple[float, int]:
                s = time.perf_counter()
                W.qputwin_job(backend, j[1], SHOTS, j[3])
                return time.perf_counter() - s, 0

            with ThreadPoolExecutor(par) as ex:
                list(ex.map(one, jobs[:par]))                     # warm-up: connections, worker caches
                t0 = time.perf_counter()
                lat = list(ex.map(one, jobs))
        else:
            with ProcessPoolExecutor(par, initializer=_pool_init, initargs=(pipeline, backend)) as ex:
                list(ex.map(_pool_job, [(pipeline, backend, j[1], j[2], j[3]) for j in jobs[:par]]))  # warm-up
                t0 = time.perf_counter()
                lat = list(ex.map(_pool_job, [(pipeline, backend, j[1], j[2], j[3]) for j in jobs]))
        wall = time.perf_counter() - t0
        times = sorted(t for t, _ in lat)
        rows.append({"backend": backend, "pipeline": pipeline, "parallel": par, "jobs": len(jobs),
                     "makespan_s": wall, "jobs_per_s": len(jobs) / wall, "job_s_median": times[len(times) // 2],
                     "job_s_p95": times[int(0.95 * (len(times) - 1))],
                     "peak_rss_bytes_max_process": max(m for _, m in lat) if pipeline != "qputwin" else None})
        print(f"  {backend} {pipeline:8s} x{par}: {len(jobs)} jobs in {wall:.2f} s = {len(jobs) / wall:.2f} jobs/s "
              f"(median job {times[len(times) // 2]:.3f} s)", flush=True)
    prev = HERE / "results" / "workflow_throughput.json"
    old = json.loads(prev.read_text())["data"] if prev.exists() else []
    old = [r for r in old if r["backend"] != backend]
    save("workflow_throughput", old + rows, stack_backend=info.get("replicas") or info.get("capacity"), workload=(
        "96 device-emulation jobs (six families x widths 4, 6, 8, 10 x four seeds, 1024 shots); vendor workflows "
        "in a process pool (one thread per process, per-device objects built in each process before timing); "
        "QPUTwin via concurrent HTTP clients against the stack configured with one single-CPU worker replica per "
        "client for this device"))


EXPERIMENTS = {"correctness": exp_correctness, "sv_speed": exp_sv_speed, "dm_speed": exp_dm_speed,
               "noise_xval": exp_noise_xval, "vendor_xval": exp_vendor_xval, "vendor_workflow": exp_vendor_workflow,
               "workflow_throughput": exp_workflow_throughput}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "all"])
    args = ap.parse_args()
    httpx.get(f"{A.BASE_URL}/health", timeout=5).raise_for_status()       # the stack must already be running
    for name in (list(EXPERIMENTS) if "all" in args.experiments else args.experiments):
        print(f"[{name}] threads={THREADS}", flush=True)
        t0 = time.perf_counter()
        EXPERIMENTS[name]()
        print(f"  ({time.perf_counter() - t0:.1f} s)", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
