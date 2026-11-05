"""Benchmarking middleware (pure ASGI).

Implemented as a raw ASGI callable rather than Starlette's ``BaseHTTPMiddleware``,
which adds measurable per-request overhead and buffers response bodies. The
middleware:

1. stamps ``t0 = perf_counter_ns()`` the instant the request enters the
   application and attaches a :class:`TelemetryRecorder` to ``scope["state"]``
   (visible to handlers as ``request.state.qputwin_telemetry``);
2. on ``http.response.start`` injects a W3C ``Server-Timing`` header — whose
   ``total`` is measured *after* JSON serialisation, complementing the in-body
   ``total_roundtrip_ms`` which cannot include the time taken to serialise
   itself — plus ``X-QPUTwin-Job-Id``;
3. after the final body chunk is sent, appends the request's record to the
   :class:`TelemetryStore` (ring buffer + JSONL research dataset). Asynchronous
   jobs are recorded by the job manager when they *finish* instead, so a ``202``
   never produces a record of its own.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import time
from typing import Any

from qputwin_orchestrator.telemetry import TelemetryRecorder, TelemetryStore

ASGIApp = Any
STATE_KEY = "qputwin_telemetry"


class BenchmarkingMiddleware:
    def __init__(self, app: ASGIApp, store: TelemetryStore) -> None:
        self.app = app
        self.store = store

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rec = TelemetryRecorder(time.perf_counter_ns())
        scope.setdefault("state", {})[STATE_KEY] = rec

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                rec.response_start_ns = time.perf_counter_ns()
                headers = list(message.get("headers", []))
                headers.append((b"server-timing", rec.server_timing().encode("latin-1")))
                if rec.job_id:
                    headers.append((b"x-qputwin-job-id", rec.job_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                rec.response_end_ns = time.perf_counter_ns()
                if rec.deferred_record:
                    return
                record = rec.dataset_record()
                if record is not None:
                    self.store.add(record)

        await self.app(scope, receive, send_wrapper)
