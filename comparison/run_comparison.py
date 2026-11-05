"""Comparative benchmark: QPUTwin vs Qiskit Aer, Amazon Braket local, Cirq, qsim and Qulacs.

QPUTwin is the **already running** Docker stack (``QPUTWIN_BASE_URL``, default
http://localhost:8000); this script never starts, stops or reconfigures it.

Experiments (each writes ``results/<name>.json`` with environment metadata):

* ``correctness``  exact ideal output distributions of nine circuit families, every
                   simulator against Qiskit Aer (and QPUTwin), 4–16 qubits
* ``sv_speed``     ideal statevector run time (simulate + 1024 shots) vs width for GHZ,
                   QV-style and QFT-adder circuits; ``COMPARE_THREADS`` = 1 (default) or 4
* ``dm_speed``     noisy density-matrix run time vs width (same thread budgets)
* ``noise_xval``   QPUTwin's calibrated noise vs an independent calibration-matched Aer
                   model on the same four devices (fidelity and compiled 2Q counts)

Usage (from this folder)::

    uv run python run_comparison.py correctness sv_speed dm_speed noise_xval
    COMPARE_THREADS=4 uv run python run_comparison.py sv_speed dm_speed

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import os

THREADS = int(os.environ.get("COMPARE_THREADS", "1"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS)               # must precede numpy / qulacs imports

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
    stack = httpx.get(f"{A.BASE_URL}/health", timeout=10).json()
    dev = httpx.get(f"{A.BASE_URL}/v1/backends/quantinuum_helios", timeout=10).json()["emulator"]
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "load_average": os.getloadavg(), "threads": THREADS,
            "versions": A.versions(), "qputwin": {"base_url": A.BASE_URL, **stack,
                                                  "worker_precision": dev["precision"],
                                                  "worker_threads": "1 (OMP_NUM_THREADS=1 in the container)",
                                                  "worker_cpu_quota": "4 CPUs / 12 GiB"}}


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
    if THREADS == 1:
        local = A.QPUTwinLocal(threads=1)
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
    rows = _speed_sweep("sv", ("ghz", "qv", "qft"), (6, 10, 14, 16, 18, 20, 22, 24), runners, cap_s)
    save(f"sv_speed_t{THREADS}", rows, shots=SHOTS, cap_s=cap_s, workload=(
        "ideal circuits; in-process simulators get the identical U/CX list, QPUTwin the source QASM (compiled to "
        "Helios natives). run_s = simulate + sample 1024 shots, or the client-measured HTTP round trip for the "
        "QPUTwin stack; engine_s = QPUTwin simulation + sampling from its telemetry. With 4 threads QPUTwin is the "
        "running stack (engine threads = worker CPU quota = 4); with 1 thread it is the same code in-process. "
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
    if THREADS == 1:
        local = A.QPUTwinLocal(threads=1)
        runners.append(("QPUTwin (in-process)", lambda q, f: local.sample(q, SHOTS, 1, noisy)))
    else:
        twin = A.QPUTwin()
        runners.append(("QPUTwin", lambda q, f: twin.sample(q, SHOTS, 1, noisy)))
    runners.append(("Qiskit Aer", lambda q, f: aer.sample(f, SHOTS, 1, noise_model=nm)))
    if THREADS == 1:
        braket, cirq = A.Braket("braket_dm"), A.Cirq(density_matrix=True)
        runners += [("Braket local", lambda q, f: braket.sample(f, SHOTS, 1, noise=lam)),
                    ("Cirq", lambda q, f: cirq.sample(f, SHOTS, 1, noise=lam))]
    rows = _speed_sweep("dm", ("ghz", "qv"), (4, 6, 8, 10, 11, 12, 13), runners, cap_s)
    save(f"dm_speed_t{THREADS}", rows, shots=SHOTS, cap_s=cap_s, depolarizing_lambda=lam, workload=(
        "noisy density-matrix simulation; QPUTwin: its full calibrated Helios noise model on native gates (engine "
        "forced to density_matrix); others: depolarising λ = 1e-3 after every U and 1e-2 after every CX of the "
        "identical U/CX list"))


def exp_noise_xval(shots: int = 100_000) -> None:
    """QPUTwin's calibrated noise vs an independent calibration-matched Aer model, four devices."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    twin = A.QPUTwin()
    ideal_sim = A.Aer()
    aer = AerSimulator(method="density_matrix", max_parallel_threads=THREADS)
    rows = []
    for backend in N.NATIVE:
        model = N.device_model(backend, A.BASE_URL)
        lattice = model.coupling is not None
        opts = {"routing": {"method": "sabre", "seed": 0}} if lattice else {}
        for fam in ("ghz", "bv", "mirror", "adder", "qft", "qv"):
            for n in range(3, 9):
                if fam == "adder" and n % 2:
                    continue
                qasm = A.C.FAMILIES[fam](n, seed=1)
                flat = A.flatten(qasm)
                ideal = ideal_sim.exact(flat).probs
                full = twin.sample(qasm, shots, 7, {**opts, "noise": {}, "engine": "density_matrix"}, backend)
                no_idle = twin.sample(qasm, shots, 7, {**opts, "noise": {"idle": False},
                                                       "engine": "density_matrix"}, backend)
                tqc = transpile(flat.qiskit, basis_gates=model.basis, coupling_map=model.coupling,
                                optimization_level=1, seed_transpiler=0)
                ops = tqc.count_ops()
                t0 = time.perf_counter()
                res = aer.run(tqc, shots=shots, seed_simulator=7, noise_model=model.noise).result()
                aer_s = time.perf_counter() - t0
                aer_counts = dict(res.get_counts())
                rows.append({
                    "backend": backend, "family": fam, "qubits": n,
                    "qputwin_full": hellinger(ideal, normalise(full.counts)),
                    "qputwin_no_idle": hellinger(ideal, normalise(no_idle.counts)),
                    "aer_matched": hellinger(ideal, normalise(aer_counts)),
                    "qputwin_two_qubit": full.extra["compilation"]["native_two_qubit_count"],
                    "qiskit_two_qubit": sum(v for k, v in ops.items() if k in ("cz", "rzz", "rxx")),
                    "qiskit_one_qubit_noisy": sum(v for k, v in ops.items() if k in ("r", "rx")),
                    "qputwin_one_qubit": full.extra["compilation"]["native_one_qubit_count"],
                    "qputwin_engine_s": full.engine_s, "aer_run_s": aer_s,
                })
                r = rows[-1]
                print(f"  {backend:18s} {fam:6s} n={n}  QPUTwin {r['qputwin_full']:.4f} (no idle "
                      f"{r['qputwin_no_idle']:.4f})  Aer-matched {r['aer_matched']:.4f}   2Q "
                      f"{r['qputwin_two_qubit']}/{r['qiskit_two_qubit']}", flush=True)
    save("noise_xval", rows, shots=shots, models={b: {"basis": N.NATIVE[b]["basis"], "map": N.NATIVE[b]["map"]}
                                                  for b in N.NATIVE}, workload=(
        "Hellinger fidelity to the exact ideal distribution (Aer statevector). QPUTwin: its own compiler (SABRE on "
        "lattices), calibrated noise on the density-matrix engine, with and without idle-window decoherence. "
        "Aer: Qiskit transpile (optimization_level=1, seed 0) to the same native basis and coupling map, with the "
        "noise model in noise_models.py built from the identical calibration; 100 000 shots each"))


EXPERIMENTS = {"correctness": exp_correctness, "sv_speed": exp_sv_speed, "dm_speed": exp_dm_speed,
               "noise_xval": exp_noise_xval}


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
