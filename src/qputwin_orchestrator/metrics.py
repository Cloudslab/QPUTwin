"""Prometheus metrics of the orchestrator.

Usage: from qputwin_orchestrator.metrics import Metrics

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from typing import Any

__all__ = ["Metrics", "Sample"]

BUCKETS_MS = (0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
              30000.0, 60000.0, 300000.0, math.inf)
_HISTOGRAMS = {
    "qputwin_job_roundtrip_ms": ("total_roundtrip_ms", "Server-side job round trip (ms)"),
    "qputwin_job_orchestration_overhead_ms": ("orchestration_overhead_ms", "Orchestration overhead (ms)"),
    "qputwin_job_qpu_compute_ms": ("qpu_compute_time_ms", "Time inside the numpy engine (ms)"),
}

Sample = tuple[dict[str, str], float]


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(d: dict[str, str]) -> str:
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in d.items()) + "}" if d else ""


def _num(x: float) -> str:
    if math.isinf(x):
        return "+Inf"
    return repr(float(x)) if not float(x).is_integer() else str(int(x))


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jobs: dict[tuple[str, str, str], int] = {}
        self.hist: dict[str, dict[str, list[float]]] = {name: {} for name in _HISTOGRAMS}

    def observe(self, record: dict[str, Any]) -> None:
        backend = str(record.get("backend"))
        key = (backend, str(record.get("status")), str(record.get("execution_mode", "sync")))
        t = record.get("telemetry") or {}
        with self._lock:
            self.jobs[key] = self.jobs.get(key, 0) + 1
            if record.get("status") != "COMPLETED":
                return
            for name, (field, _) in _HISTOGRAMS.items():
                v = t.get(field)
                if v is None:
                    continue
                h = self.hist[name].setdefault(backend, [0.0] * (len(BUCKETS_MS) + 2))   # buckets, sum, count
                for i, le in enumerate(BUCKETS_MS):
                    if float(v) <= le:
                        h[i] += 1
                h[-2] += float(v)
                h[-1] += 1

    def render(self, gauges: Iterable[tuple[str, str, str, list[Sample]]] = ()) -> str:
        out: list[str] = []
        with self._lock:
            out += ["# HELP qputwin_jobs_total Finished jobs.", "# TYPE qputwin_jobs_total counter"]
            for (b, st, mode), n in sorted(self.jobs.items()):
                out.append(f"qputwin_jobs_total{_labels({'backend': b, 'status': st, 'mode': mode})} {n}")
            for name, (_, help_) in _HISTOGRAMS.items():
                out += [f"# HELP {name} {help_}", f"# TYPE {name} histogram"]
                for b, h in sorted(self.hist[name].items()):
                    for i, le in enumerate(BUCKETS_MS):
                        out.append(f"{name}_bucket{_labels({'backend': b, 'le': _num(le)})} {_num(h[i])}")
                    out.append(f"{name}_sum{_labels({'backend': b})} {_num(h[-2])}")
                    out.append(f"{name}_count{_labels({'backend': b})} {_num(h[-1])}")
        for name, kind, help_, samples in gauges:
            out += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}"]
            out += [f"{name}{_labels(lbl)} {_num(v)}" for lbl, v in samples]
        return "\n".join(out) + "\n"
