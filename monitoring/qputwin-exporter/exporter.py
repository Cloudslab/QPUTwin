#!/usr/bin/env python3
"""Prometheus exporter for per-job QPUTwin metrics from the telemetry dataset.

Usage: python -u exporter.py  (run by the monitoring docker compose file)

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

log = logging.getLogger("qputwin-exporter")

PATH = os.environ.get("QPUTWIN_TEL_PATH", "/data/telemetry.jsonl")
PORT = int(os.environ.get("QPUTWIN_TEL_PORT", "9108"))
START = os.environ.get("QPUTWIN_TEL_START", "end")
POLL_S = float(os.environ.get("QPUTWIN_TEL_POLL_S", "0.5"))

# The fourteen leaf spans. Together they sum to the job's total round trip.
SPANS = {
    "ingress_ms": "ingress",
    "validation_ms": "validation",
    "routing_ms": "backend_selection",
    "orchestrator_queue_ms": "orchestrator_queue",
    "network_transit_ms": "network_transit",
    "response_assembly_ms": "response_assembly",
    "unattributed_ms": "orchestrator_unattributed",
    "worker_queue_ms": "worker_queue",
    "worker_parse_ms": "worker_parse",
    "worker_route_ms": "worker_route",
    "worker_compile_ms": "worker_compile",
    "simulation_ms": "simulation",
    "sampling_ms": "sampling",
    "worker_unattributed_ms": "worker_unattributed",
}
# The spans sum to the round trip by construction; this only absorbs float rounding.
IDENTITY_TOLERANCE_MS = 0.001

SPAN_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 30000)
ROUNDTRIP_BUCKETS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500,
                     1000, 2500, 5000, 10000, 30000, 60000)
QUBIT_BUCKETS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36)
BYTES_BUCKETS = tuple(4 ** i for i in range(5, 18))   # 1 KiB … 16 GiB


class Histogram:
    __slots__ = ("bounds", "counts", "sum", "n")

    def __init__(self, bounds: tuple[float, ...]) -> None:
        self.bounds = bounds
        self.counts = [0] * len(bounds)
        self.sum = 0.0
        self.n = 0

    def observe(self, value: float) -> None:
        self.sum += value
        self.n += 1
        for i, bound in enumerate(self.bounds):
            if value <= bound:
                self.counts[i] += 1
                break

    def lines(self, name: str, labels: dict[str, str]) -> Iterator[str]:
        running = 0
        for bound, count in zip(self.bounds, self.counts, strict=True):
            running += count
            yield f"{name}_bucket{fmt({**labels, 'le': _le(bound)})} {running}"
        yield f"{name}_bucket{fmt({**labels, 'le': '+Inf'})} {self.n}"
        yield f"{name}_sum{fmt(labels)} {self.sum}"
        yield f"{name}_count{fmt(labels)} {self.n}"


def _le(bound: float) -> str:
    return str(int(bound)) if float(bound).is_integer() else str(bound)


def fmt(labels: dict[str, str]) -> str:
    def esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in labels.items()) + "}" if labels else ""


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[tuple[str, ...], int] = defaultdict(int)
        self.errors: dict[tuple[str, str], int] = defaultdict(int)
        self.shots: dict[str, int] = defaultdict(int)
        self.native_ops: dict[str, int] = defaultdict(int)
        self.native_2q: dict[str, int] = defaultdict(int)
        self.retried: dict[str, int] = defaultdict(int)
        self.violations: dict[str, int] = defaultdict(int)
        self.spans: dict[tuple[str, str], Histogram] = {}
        self.roundtrip: dict[str, Histogram] = {}
        self.qubits: dict[str, Histogram] = {}
        self.peak_bytes: dict[str, Histogram] = {}
        self.records = 0
        self.parse_errors = 0
        self.last_record_ts = 0.0
        self.offset = 0

    def ingest(self, rec: dict[str, Any]) -> None:
        backend = str(rec.get("backend"))
        status = str(rec.get("status"))
        tel = rec.get("telemetry") or {}
        work = tel.get("workload") or {}
        breakdown = tel.get("breakdown") or {}
        mode = str(rec.get("execution_mode") or tel.get("execution_mode") or "sync")
        with self.lock:
            self.records += 1
            self.jobs[(backend, status, mode, str(work.get("engine", "none")),
                       str(work.get("execution_path", "none")), str(bool(work.get("noisy"))).lower(),
                       str(bool(work.get("routed"))).lower())] += 1
            if rec.get("error_type"):
                self.errors[(backend, str(rec["error_type"]))] += 1
            if tel.get("attempt", 1) > 1:
                self.retried[backend] += 1
            stamp = rec.get("timestamp_utc")
            if stamp:
                self.last_record_ts = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
            if status != "COMPLETED":
                return
            self.shots[backend] += int(work.get("shots") or 0)
            self.native_ops[backend] += int(work.get("native_op_count") or 0)
            self.native_2q[backend] += int(work.get("native_two_qubit_count") or 0)
            total = tel.get("total_roundtrip_ms")
            if total is not None:
                self.roundtrip.setdefault(backend, Histogram(ROUNDTRIP_BUCKETS)).observe(float(total))
            if work.get("simulated_qubits") is not None:
                self.qubits.setdefault(backend, Histogram(QUBIT_BUCKETS)).observe(float(work["simulated_qubits"]))
            if work.get("estimated_peak_bytes") is not None:
                self.peak_bytes.setdefault(backend, Histogram(BYTES_BUCKETS)).observe(
                    float(work["estimated_peak_bytes"]))
            leaves = 0.0
            for field, span in SPANS.items():
                value = breakdown.get(field)
                if value is None:
                    continue
                leaves += float(value)
                self.spans.setdefault((backend, span), Histogram(SPAN_BUCKETS)).observe(float(value))
            if total is not None and breakdown and abs(leaves - float(total)) > IDENTITY_TOLERANCE_MS:
                self.violations[backend] += 1

    def render(self) -> str:
        out: list[str] = []

        def head(name: str, kind: str, text: str) -> None:
            out.extend((f"# HELP {name} {text}", f"# TYPE {name} {kind}"))

        def counter(name: str, text: str, rows: Iterable[tuple[dict[str, str], float]]) -> None:
            head(name, "counter", text)
            out.extend(f"{name}{fmt(lbl)} {v}" for lbl, v in rows)

        def histograms(name: str, text: str, rows: Iterable[tuple[dict[str, str], Histogram]]) -> None:
            head(name, "histogram", text)
            for lbl, hist in rows:
                out.extend(hist.lines(name, lbl))

        job_keys = ("backend", "status", "mode", "engine", "execution_path", "noisy", "routed")
        with self.lock:
            counter("qputwin_tel_jobs_total", "Finished jobs, from the telemetry dataset.",
                    ((dict(zip(job_keys, k, strict=True)), v) for k, v in sorted(self.jobs.items())))
            counter("qputwin_tel_errors_total", "Rejected or failed jobs by error type.",
                    (({"backend": b, "error_type": e}, v) for (b, e), v in sorted(self.errors.items())))
            for name, text, data in (
                ("qputwin_tel_shots_total", "Shots of completed jobs.", self.shots),
                ("qputwin_tel_native_ops_total", "Native operations of completed jobs.", self.native_ops),
                ("qputwin_tel_native_two_qubit_ops_total", "Native two-qubit gates of completed jobs.", self.native_2q),
                ("qputwin_tel_retried_jobs_total", "Jobs executed more than once after a lease expired.", self.retried),
                ("qputwin_tel_identity_violations_total",
                 "Completed jobs whose fourteen spans do not sum to the round trip.", self.violations),
            ):
                counter(name, text, (({"backend": b}, v) for b, v in sorted(data.items())))
            histograms("qputwin_tel_span_ms", "Duration of one additive latency span (ms).",
                       (({"backend": b, "span": s}, h) for (b, s), h in sorted(self.spans.items())))
            histograms("qputwin_tel_roundtrip_ms", "Server round trip, in finer buckets than /metrics (ms).",
                       (({"backend": b}, h) for b, h in sorted(self.roundtrip.items())))
            histograms("qputwin_tel_simulated_qubits", "Qubits actually simulated, after routing and compaction.",
                       (({"backend": b}, h) for b, h in sorted(self.qubits.items())))
            histograms("qputwin_tel_estimated_peak_bytes", "Estimated peak engine memory per job (bytes).",
                       (({"backend": b}, h) for b, h in sorted(self.peak_bytes.items())))
            counter("qputwin_tel_records_total", "Telemetry records read.", [({}, self.records)])
            counter("qputwin_tel_parse_errors_total", "Telemetry lines that could not be parsed.",
                    [({}, self.parse_errors)])
            head("qputwin_tel_last_record_timestamp_seconds", "gauge", "Timestamp of the newest record read.")
            out.append(f"qputwin_tel_last_record_timestamp_seconds {self.last_record_ts}")
            head("qputwin_tel_file_offset_bytes", "gauge", "Read position in the telemetry file.")
            out.append(f"qputwin_tel_file_offset_bytes {self.offset}")
        return "\n".join(out) + "\n"


def tail(store: Store) -> None:
    """Follow the telemetry file, across truncation and replacement."""
    handle = None
    inode = None
    position = 0
    partial = b""
    first_open = True
    while True:
        try:
            stat = os.stat(PATH)
        except OSError:
            time.sleep(POLL_S * 4)
            continue
        if handle is None or stat.st_ino != inode or stat.st_size < position:
            if handle is not None:
                handle.close()
            handle = open(PATH, "rb")   # noqa: SIM115 - held open across polls
            inode = stat.st_ino
            position = stat.st_size if first_open and START == "end" else 0
            partial = b""
            first_open = False
            log.info("following %s from byte %d", PATH, position)
        if stat.st_size == position:
            time.sleep(POLL_S)
            continue
        handle.seek(position)
        chunk = handle.read(16 * 1024 * 1024)
        position += len(chunk)
        *lines, partial = (partial + chunk).split(b"\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                store.ingest(json.loads(line))
            except (ValueError, TypeError, KeyError, AttributeError):
                with store.lock:
                    store.parse_errors += 1
        with store.lock:
            store.offset = position


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = Store()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/metrics":
                body, status = store.render().encode(), 200
            elif self.path == "/health":
                body, status = b"ok\n", 200
            else:
                body, status = b"not found\n", 404
            self.send_response(status)
            self.send_header("content-type", "text/plain; version=0.0.4")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    threading.Thread(target=tail, args=(store,), daemon=True).start()
    log.info("serving on :%d", PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
