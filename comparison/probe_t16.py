"""Scratch probe: deep ideal circuits at 16 threads, QPUTwin in-process vs qsim, Aer and Qulacs.

Run from comparison/ with OMP_NUM_THREADS=16 so Qulacs uses all cores:
    OMP_NUM_THREADS=16 uv run python probe_t16.py 24 26 28
"""
import os
import sys
import time

import adapters as A

T = 16
widths = [int(a) for a in sys.argv[1:]] or [24, 26]
fams = os.environ.get("PROBE_FAMS", "qv,qft").split(",")
only = os.environ.get("PROBE_SIMS")
local = A.QPUTwinLocal(threads=T)
sims = {
    "qputwin_c128": lambda q, f: local.sample(q, 1024, 1),
    "qputwin_c64": lambda q, f: local.sample(q, 1024, 1, {"precision": "complex64"}),
    "qsim": (lambda s: lambda q, f: s.sample(f, 1024, 1))(A.Qsim(threads=T)),
    "aer": (lambda s: lambda q, f: s.sample(f, 1024, 1))(A.Aer(threads=T)),
    "qulacs": (lambda s: lambda q, f: s.sample(f, 1024, 1))(A.Qulacs()),
}
if only:
    sims = {k: v for k, v in sims.items() if k in only.split(",")}
for fam in fams:
    for n in widths:
        qasm = A.C.FAMILIES[fam](n, seed=1)
        flat = A.flatten(qasm)
        for name, fn in sims.items():
            best, eng = 1e9, None
            for _ in range(2):
                t0 = time.perf_counter()
                r = fn(qasm, flat)
                best = min(best, time.perf_counter() - t0)
                eng = r.engine_s if r.engine_s is not None else eng
            extra = f" engine {eng:.3f}" if eng else ""
            print(f"{fam} n={n} {name:13s} {best:8.3f} s{extra}", flush=True)
