"""Benchmark telemetry: per-request span recorder, additive decomposition, dataset sink.

Span model (docs/guide/telemetry.md)::

    total_roundtrip ─┬─ ingress            (ASGI entry → handler entry: body read + request validation)
                     ├─ validation         (QASM parse/flatten + resource checks)
                     ├─ routing            (registry lookup + payload serialisation)
                     ├─ orchestrator_queue (async jobs only: enqueue → dispatch; 0 for sync jobs)
                     ├─ dispatch_rtt ──────┬─ network_transit  (= dispatch_rtt − worker_service)
                     │                     └─ worker_service ─┬─ worker_queue
                     │                                        ├─ worker_parse
                     │                                        ├─ worker_route   (SABRE, when requested)
                     │                                        ├─ worker_compile
                     │                                        ├─ qpu_compute ─┬─ simulation
                     │                                        │               └─ sampling
                     │                                        └─ worker_unattributed
                     ├─ response_assembly
                     └─ unattributed

Headline metrics::

    qpu_compute_time_ms       = simulation_ms + sampling_ms          (inside the numpy engine)
    orchestration_overhead_ms = total_roundtrip_ms − worker_service_ms
    ⇒ total_roundtrip_ms     ≡ orchestration_overhead_ms + worker_service_ms   (by construction)

All durations come from ``time.perf_counter_ns`` inside a single process; the
worker reports *durations*, never timestamps, so container clock skew cannot
enter the measurement.

For an asynchronous job (``202`` + polling) the same recorder spans submission to
completion: ``total_roundtrip_ms`` then runs from the submit request's arrival
to the moment the result is ready, and the time spent waiting in the
orchestrator's queue is reported as ``orchestrator_queue_ms``. The latency of the
``202`` response itself is recorded in the dataset as ``submit_response_ms``.

Schema history (purely additive): 1.0 → 1.1 added ``orchestrator_queue_ms``,
``worker_route_ms``, ``execution_mode`` and ``environment.replica``; 1.2 added
``queue_clock`` (``perf_counter`` when the accepting instance executed the job,
``realtime`` when another instance did or after a restart) and ``attempt``.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qputwin_core import __version__
from qputwin_core.timing import ns_to_ms

SCHEMA_VERSION = "1.2"
HEADLINE = ("total_roundtrip_ms", "orchestration_overhead_ms", "qpu_compute_time_ms")

__all__ = ["TelemetryRecorder", "TelemetryStore", "SCHEMA_VERSION", "HEADLINE"]


def _r(x: float) -> float:
    return round(x, 6)


class TelemetryRecorder:
    """Created by the middleware for every HTTP request; filled in by the route handler."""

    def __init__(self, t0_ns: int) -> None:
        self.t0_ns = t0_ns
        self.handler_entry_ns: int | None = None
        self.end_ns: int | None = None
        self.spans_ns: dict[str, int] = {}
        self.worker_timing: dict[str, float] = {}
        self.job_id: str | None = None
        self.backend: str | None = None
        self.status: str | None = None
        self.http_status: int | None = None
        self.error_type: str | None = None
        self.engine_invoked = False
        self.workload: dict[str, Any] = {}
        self.worker_env: dict[str, Any] = {}
        self.telemetry: dict[str, Any] | None = None
        self.response_start_ns: int | None = None
        self.response_end_ns: int | None = None
        self.execution_mode = "sync"
        self.deferred_record = False           # queued async job: the job manager records it on completion
        self.replica: str | None = None
        self.queue_clock: str | None = None    # async only: perf_counter | realtime
        self.attempt = 1
        self.timestamp_utc = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    # ----------------------------------------------------------------- spans
    def enter_handler(self) -> None:
        self.handler_entry_ns = time.perf_counter_ns()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t = time.perf_counter_ns()
        try:
            yield
        finally:
            self.spans_ns[name] = self.spans_ns.get(name, 0) + time.perf_counter_ns() - t

    def ms(self, name: str) -> float:
        return ns_to_ms(self.spans_ns.get(name, 0))

    # -------------------------------------------------------------- finalise
    def finalize(self) -> dict[str, Any]:
        """Stop the clock and build the ``telemetry`` object returned to the client."""
        self.end_ns = time.perf_counter_ns()
        total = ns_to_ms(self.end_ns - self.t0_ns)
        ingress = ns_to_ms((self.handler_entry_ns or self.end_ns) - self.t0_ns)
        validation, routing = self.ms("validation"), self.ms("routing")
        queue = self.ms("queue")
        dispatch, assembly = self.ms("dispatch"), self.ms("response_assembly")
        w = self.worker_timing
        worker_service = float(w.get("service_ms", 0.0))
        simulation = float(w.get("simulation_ms", 0.0))
        sampling = float(w.get("sampling_ms", 0.0))
        qpu = simulation + sampling
        breakdown = {
            "ingress_ms": ingress,
            "validation_ms": validation,
            "routing_ms": routing,
            "orchestrator_queue_ms": queue,
            "dispatch_rtt_ms": dispatch,
            "network_transit_ms": max(0.0, dispatch - worker_service) if dispatch else 0.0,
            "worker_service_ms": worker_service,
            "worker_queue_ms": float(w.get("queue_ms", 0.0)),
            "worker_parse_ms": float(w.get("parse_ms", 0.0)),
            "worker_route_ms": float(w.get("route_ms", 0.0)),
            "worker_compile_ms": float(w.get("compile_ms", 0.0)),
            "simulation_ms": simulation,
            "sampling_ms": sampling,
            "worker_unattributed_ms": float(w.get("unattributed_ms", 0.0)),
            "response_assembly_ms": assembly,
            "unattributed_ms": total - (ingress + validation + routing + queue + dispatch + assembly),
        }
        self.telemetry = {
            "schema_version": SCHEMA_VERSION,
            "orchestration_overhead_ms": _r(total - worker_service),
            "qpu_compute_time_ms": _r(qpu),
            "total_roundtrip_ms": _r(total),
            "engine_invoked": self.engine_invoked,
            "execution_mode": self.execution_mode,
            "queue_clock": self.queue_clock,
            "attempt": self.attempt,
            "breakdown": {k: _r(v) for k, v in breakdown.items()},
            "workload": self.workload,
            "environment": {
                "orchestrator_version": __version__,
                "python": platform.python_version(),
                "replica": self.replica,
                "worker": self.worker_env,
            },
            "clock": "perf_counter_ns",
            "timestamp_utc": self.timestamp_utc,
        }
        return self.telemetry

    def server_timing(self) -> str:
        """W3C Server-Timing header value (``total`` measured *after* JSON serialisation)."""
        now = self.response_start_ns or time.perf_counter_ns()
        parts = [f"total;dur={ns_to_ms(now - self.t0_ns):.3f}"]
        t = self.telemetry
        if t is not None:
            b = t["breakdown"]
            parts += [f"validation;dur={b['validation_ms']:.3f}", f"routing;dur={b['routing_ms']:.3f}",
                      f"dispatch;dur={b['dispatch_rtt_ms']:.3f}", f"worker;dur={b['worker_service_ms']:.3f}",
                      f"qpu;dur={t['qpu_compute_time_ms']:.3f}",
                      f"orchestration;dur={t['orchestration_overhead_ms']:.3f}"]
        return ", ".join(parts)

    def dataset_record(self) -> dict[str, Any] | None:
        if self.telemetry is None:
            return None
        rec = {
            "job_id": self.job_id, "backend": self.backend, "status": self.status,
            "http_status": self.http_status, "error_type": self.error_type,
            "execution_mode": self.execution_mode,
            "timestamp_utc": self.timestamp_utc, "telemetry": self.telemetry,
        }
        if self.deferred_record:
            # the HTTP response belongs to the submission (202), not to the result
            if self.response_start_ns is not None:
                rec["submit_response_ms"] = _r(ns_to_ms(self.response_start_ns - self.t0_ns))
            return rec
        if self.response_start_ns is not None:
            rec["server_timing_total_ms"] = _r(ns_to_ms(self.response_start_ns - self.t0_ns))
        if self.response_end_ns is not None:
            rec["response_complete_ms"] = _r(ns_to_ms(self.response_end_ns - self.t0_ns))
        return rec


class TelemetryStore:
    """In-memory ring buffer + optional append-only JSONL dataset (one record per job)."""

    def __init__(self, maxlen: int = 10_000, path: str | Path | None = None) -> None:
        self.records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.observers: list[Any] = []                 # callables fed every record (e.g. Prometheus metrics)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> None:
        line = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            self.records.append(record)
            if self.path is not None:
                # one O_APPEND write per record: lines from several orchestrator processes never interleave
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
                    os.write(fd, line)
                finally:
                    os.close(fd)
        for observe in self.observers:
            observe(record)

    def recent(self, limit: int = 100, backend: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = [r for r in self.records if backend is None or r.get("backend") == backend]
        return items[-limit:]

    def summary(self) -> dict[str, Any]:
        """Per-backend descriptive statistics of the headline metrics (completed jobs)."""
        with self._lock:
            items = list(self.records)
        out: dict[str, Any] = {}
        for backend in sorted({str(r.get("backend")) for r in items}):
            rows = [r for r in items if str(r.get("backend")) == backend]
            ok = [r for r in rows if r.get("status") == "COMPLETED"]
            stats: dict[str, Any] = {}
            for metric in HEADLINE:
                xs = sorted(float(r["telemetry"][metric]) for r in ok)
                stats[metric] = _describe(xs)
            out[backend] = {"jobs": len(rows), "completed": len(ok),
                            "rejected": sum(1 for r in rows if r.get("status") == "REJECTED"),
                            "failed": sum(1 for r in rows if r.get("status") == "FAILED"),
                            "cancelled": sum(1 for r in rows if r.get("status") == "CANCELLED"),
                            "async": sum(1 for r in rows if r.get("execution_mode") == "async"),
                            "metrics": stats}
        return out


def _describe(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "median": None, "p95": None, "max": None}
    p95 = xs[min(len(xs) - 1, max(0, math.ceil(0.95 * len(xs)) - 1))]
    return {"n": len(xs), "mean": _r(statistics.fmean(xs)),
            "stdev": _r(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
            "min": _r(xs[0]), "median": _r(statistics.median(xs)), "p95": _r(p95), "max": _r(xs[-1])}
