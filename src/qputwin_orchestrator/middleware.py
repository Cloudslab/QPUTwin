"""Middleware that measures the time of every request.

Usage: from qputwin_orchestrator.middleware import BenchmarkingMiddleware

Koustav Kumar Mondal <koustav@koustavmondal.com>
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
