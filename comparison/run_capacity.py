"""Capacity stress test: the widest circuits each simulator runs on the whole host.

Every point runs in a **fresh subprocess** with all host threads, so its peak resident memory
(``ru_maxrss``) is its own and a crash or out-of-memory kill does not stop the sweep. A series
stops at the first width that fails (out of memory, refused, crashed) or exceeds the time cap.

Simulators (ideal statevector, 1024 shots): QPUTwin complex128 and complex64 (in-process, the
worker's own memory policy: two buffers when they fit, otherwise in place, otherwise refused),
qsim (complex64), Qiskit Aer complex128 and complex64, Qulacs (complex128). Noisy density matrix:
QPUTwin (calibrated Helios model) and Aer (depolarising λ = 1e-3 / 1e-2 per U / CX).

Usage (from this folder; ``capacity`` = statevector + density matrix)::

    uv run python run_capacity.py            # writes results/capacity.json
    uv run python run_capacity.py --point qputwin_c128 qv 28 sv     # one point, prints JSON

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
THREADS = int(os.environ.get("CAPACITY_THREADS", str(os.cpu_count() or 1)))
CAP_S = float(os.environ.get("CAPACITY_CAP_S", "1200"))
SV_SIMS = ("qputwin_c128", "qputwin_c64", "qsim", "aer_c128", "aer_c64", "qulacs")
SV_WIDTHS = (26, 28, 29, 30, 31, 32, 33)
DM_SIMS = ("qputwin_dm", "aer_dm")
DM_WIDTHS = (12, 13, 14, 15, 16)
FAMILIES = ("ghz", "qv")


def _point(sim: str, fam: str, n: int, kind: str) -> dict:
    """Run one point in this (fresh) process and return its measurements."""
    import resource

    import adapters as A
    qasm = A.C.FAMILIES[fam](n, seed=1)
    flat = A.flatten(qasm)
    extra: dict = {}
    if sim.startswith("qputwin"):
        local = A.QPUTwinLocal(threads=THREADS, memory_guard=True)
        opts = {"precision": "complex64"} if sim == "qputwin_c64" else {}
        if kind == "dm":
            opts = {"noise": {}, "engine": "density_matrix"}
        t0 = time.perf_counter()
        r = local.sim.execute(qasm, local._opts(shots=1024, seed=1, **opts))
        run_s = time.perf_counter() - t0
        extra = {"in_place": r.workload.get("in_place"), "estimated_peak_bytes": r.workload.get("estimated_peak_bytes"),
                 "memory_budget_bytes": r.workload.get("memory_budget_bytes")}
    elif sim == "qsim":
        run_s = A.Qsim(threads=THREADS).sample(flat, 1024, 1).run_s
    elif sim.startswith("aer"):
        if kind == "dm":
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1), ["u"])
            nm.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ["cx"])
            run_s = A.Aer(threads=THREADS, method="density_matrix").sample(flat, 1024, 1, noise_model=nm).run_s
        else:
            prec = "single" if sim == "aer_c64" else "double"
            run_s = A.Aer(threads=THREADS, precision=prec).sample(flat, 1024, 1).run_s
    elif sim == "qulacs":
        run_s = A.Qulacs().sample(flat, 1024, 1).run_s
    else:
        raise ValueError(sim)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {"run_s": run_s, "peak_rss_bytes": peak, **extra}


def _spawn(sim: str, fam: str, n: int, kind: str) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, __file__, "--point", sim, fam, str(n), kind], cwd=HERE, env=env,
                           capture_output=True, text=True, timeout=CAP_S)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode == 0 and lines:
        return {"status": "ok", "wall_s": wall, **json.loads(lines[-1])}
    err = (p.stderr.strip().splitlines() or [""])[-1][:300]
    status = "killed" if p.returncode in (-9, 137) else ("refused" if "ResourceLimitError" in p.stderr
                                                            else "out_of_memory" if "MemoryError" in p.stderr
                                                            or "bad_alloc" in p.stderr else "failed")
    return {"status": status, "wall_s": wall, "returncode": p.returncode, "error": err}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--point":
        sim, fam, n, kind = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
        print(json.dumps(_point(sim, fam, n, kind)), flush=True)
        return
    sys.path.insert(0, str(HERE))
    from run_comparison import save
    rows = []
    for kind, sims, widths in (("sv", SV_SIMS, SV_WIDTHS), ("dm", DM_SIMS, DM_WIDTHS)):
        for fam in FAMILIES:
            for sim in sims:
                for n in widths:
                    r = {"kind": kind, "family": fam, "simulator": sim, "qubits": n, "threads": THREADS,
                         **_spawn(sim, fam, n, kind)}
                    rows.append(r)
                    peak = r.get("peak_rss_bytes")
                    print(f"  {kind} {fam:3s} {sim:13s} n={n:2d} {r['status']:13s} "
                          f"{r.get('run_s', r['wall_s']):9.2f} s  peak {peak / 2**30 if peak else 0:6.2f} GiB"
                          f"{'  in place' if r.get('in_place') else ''}{'  ' + r.get('error', '') if r['status'] != 'ok' else ''}",
                          flush=True)
                    save("capacity", rows, threads=THREADS, cap_s=CAP_S, workload=(
                        "widest ideal statevector (1024 shots) and noisy density matrix per simulator on the whole "
                        "host; one fresh subprocess per point (peak RSS = its own); a series stops at the first "
                        "failure or time-out"))
                    if r["status"] != "ok":
                        break


if __name__ == "__main__":
    main()
