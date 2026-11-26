"""FastAPI application of the orchestrator.

Usage: from qputwin_orchestrator.app import create_app

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from qputwin_core import __author__, __email__, __version__
from qputwin_core.errors import (
    AuthenticationError, BackendProtocolError, BackendTimeoutError, BackendUnavailableError, InvalidRequestError,
    QPUTwinError, ResourceLimitError, error_from_dict,
)
from qputwin_core.qasm import load_circuit
from qputwin_orchestrator.jobs import JobManager, make_store
from qputwin_orchestrator.metrics import Metrics
from qputwin_orchestrator.middleware import STATE_KEY, BenchmarkingMiddleware
from qputwin_orchestrator.registry import BackendEntry, BackendRegistry
from qputwin_orchestrator.schemas import (
    JobAccepted, JobErrorResponse, JobRequest, JobResponse, JobStatus,
)
from qputwin_orchestrator.telemetry import TelemetryRecorder, TelemetryStore

log = logging.getLogger("qputwin.orchestrator")

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": JobErrorResponse} for code in (400, 401, 404, 409, 422, 502, 503, 504)
}


def create_app(registry: BackendRegistry | None = None,
               transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """Build the orchestrator. ``transport`` lets tests route worker traffic in-process."""
    registry = registry or BackendRegistry.from_config()
    settings = registry.settings
    store = TelemetryStore(settings.telemetry_buffer, settings.telemetry_path)
    metrics = Metrics()
    store.observers.append(metrics.observe)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_s, connect=settings.connect_timeout_s),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64, keepalive_expiry=30),   # below the workers' 75 s keep-alive
            transport=transport,
        )
        app.state.client = client
        await registry.prefetch(client)
        await app.state.jobs.start()
        try:
            yield
        finally:
            await app.state.jobs.stop()
            await client.aclose()

    app = FastAPI(
        title="QPUTwin Orchestrator",
        version=__version__,
        description="QPUTwin: Calibration-Faithful Digital Twins of Trapped-Ion and Superconducting Quantum "
                    "Processors. "
                    f"{__author__} <{__email__}>.",
        contact={"name": __author__, "email": __email__},
        lifespan=lifespan,
    )
    app.state.registry = registry
    app.state.store = store
    app.add_middleware(BenchmarkingMiddleware, store=store)

    # --------------------------------------------------------------- helpers
    def recorder(request: Request) -> TelemetryRecorder:
        rec = getattr(request.state, STATE_KEY, None)
        if rec is None:  # pragma: no cover - only if the middleware is removed
            import time
            rec = TelemetryRecorder(time.perf_counter_ns())
        return rec

    def error_body(rec: TelemetryRecorder, err: QPUTwinError) -> tuple[int, dict[str, Any]]:
        rec.http_status = err.http_status
        rec.status = "FAILED" if err.http_status >= 500 else "REJECTED"
        rec.error_type = err.type
        telemetry = rec.finalize()
        return err.http_status, {"job_id": rec.job_id, "backend": rec.backend, "status": rec.status,
                                 "error": err.to_dict(), "telemetry": telemetry}

    def error_response(rec: TelemetryRecorder, err: QPUTwinError) -> JSONResponse:
        status, body = error_body(rec, err)
        return JSONResponse(body, status_code=status)

    def authenticate(request: Request) -> tuple[str, int]:
        """``(client id, max priority)``. Without configured API keys every caller is trusted and may
        name itself with ``X-QPUTwin-Client`` (used for fair share)."""
        if not settings.auth_enabled:
            return (request.headers.get("x-qputwin-client") or "anonymous")[:64], 9
        key = request.headers.get("x-api-key", "")
        auth = request.headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        match = None
        for k, ident in settings.api_keys.items():          # constant-time comparison against every key
            if hmac.compare_digest(k.encode(), key.encode()):
                match = ident
        if match is None:
            raise AuthenticationError("missing or invalid API key (send 'Authorization: Bearer <key>' or "
                                      "'X-API-Key: <key>')")
        return match

    def scope(ident: tuple[str, int]) -> str | None:
        """With authentication on, a client sees only its own jobs."""
        return ident[0] if settings.auth_enabled else None

    # ---------------------------------------------------------------- routes
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {"service": "qputwin-orchestrator", "version": __version__,
                "contact": {"name": __author__, "email": __email__},
                "backends": sorted(registry.entries), "docs": "/docs"}

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "orchestrator", "version": __version__}

    @app.get("/ready", tags=["meta"])
    async def ready() -> JSONResponse:
        """Readiness: the job store answers and at least one backend has been reached."""
        checks: dict[str, Any] = {}
        try:
            await jobs._call(jobs.store.ping)
            checks["job_store"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["job_store"] = f"error: {type(exc).__name__}"
        known = [b for b, e in registry.entries.items() if e.device is not None]
        checks["backends_reachable"] = known
        ok = checks["job_store"] == "ok" and bool(known)
        return JSONResponse({"status": "ready" if ok else "not_ready", "instance": jobs.owner, **checks},
                            status_code=200 if ok else 503)

    @app.get("/metrics", tags=["meta"], response_class=PlainTextResponse)
    async def prometheus() -> PlainTextResponse:
        q = await jobs.stats()
        gauges = [
            ("qputwin_build_info", "gauge", "Build information.", [({"version": __version__, "instance": jobs.owner}, 1)]),
            ("qputwin_async_jobs", "gauge", "Asynchronous jobs by state (all instances).",
             [({"backend": b, "state": st}, v[st]) for b, v in q.items() for st in ("queued", "running")]),
            ("qputwin_async_capacity", "gauge", "Dispatch slots per backend.",
             [({"backend": b}, v["capacity"]) for b, v in q.items() if v["capacity"] is not None]),
            ("qputwin_replica_in_flight", "gauge", "In-flight requests per worker replica (this instance).",
             [({"backend": b, "replica": r.url}, r.in_flight) for b, e in registry.entries.items() for r in e.replicas]),
            ("qputwin_replica_dispatched_total", "counter", "Requests sent per worker replica (this instance).",
             [({"backend": b, "replica": r.url}, r.dispatched) for b, e in registry.entries.items() for r in e.replicas]),
            ("qputwin_replica_failures_total", "counter", "Connection failures per worker replica (this instance).",
             [({"backend": b, "replica": r.url}, r.failures) for b, e in registry.entries.items() for r in e.replicas]),
        ]
        return PlainTextResponse(metrics.render(gauges), media_type="text/plain; version=0.0.4")

    @app.get("/v1/backends", tags=["backends"], dependencies=[Depends(authenticate)])
    async def list_backends(request: Request) -> dict[str, Any]:
        client: httpx.AsyncClient = request.app.state.client
        ids = sorted(registry.entries)
        healths = await asyncio.gather(*(registry.health(client, b) for b in ids))
        out = []
        for bid, h in zip(ids, healths, strict=True):
            entry = registry.entries[bid]
            if entry.device is None and h.get("status") == "online":
                try:
                    await registry.device(client, bid)
                except QPUTwinError:
                    pass
            d = entry.device or {}
            out.append({
                "id": bid, "status": h.get("status"), "description": entry.description,
                "vendor": d.get("vendor"), "name": d.get("name"), "technology": d.get("technology"),
                "n_physical": d.get("n_physical"), "connectivity": d.get("connectivity"),
                "native_gates": d.get("native_gates"),
                "max_sim_qubits": (d.get("emulator") or {}).get("max_sim_qubits"),
                "replicas": h.get("replicas"),
                "disabled_qubits": (d.get("coupling_map") or {}).get("disabled_qubits"),
                "calibration": ((d.get("calibration") or {}).get("label")),
            })
        return {"backends": out}

    @app.get("/v1/backends/{backend_id}", tags=["backends"], dependencies=[Depends(authenticate)])
    async def get_backend(backend_id: str, request: Request, refresh: bool = False) -> JSONResponse:
        try:
            return JSONResponse(await registry.device(request.app.state.client, backend_id, refresh=refresh))
        except QPUTwinError as err:
            return JSONResponse({"error": err.to_dict()}, status_code=err.http_status)

    # ------------------------------------------------------ job pipeline
    async def validate(job: JobRequest, rec: TelemetryRecorder, client: httpx.AsyncClient
                       ) -> tuple[BackendEntry, bytes, dict[str, str]]:
        """Validation + routing spans: everything that happens before a job is dispatched."""
        with rec.span("validation"):
            entry = registry.get(job.backend)
            size = len(job.qasm.encode("utf-8"))
            if size > settings.max_qasm_bytes:
                raise InvalidRequestError(f"QASM payload is {size} bytes; limit is {settings.max_qasm_bytes}",
                                          details={"bytes": size, "max_qasm_bytes": settings.max_qasm_bytes})
            if job.shots > settings.max_shots:
                raise InvalidRequestError(f"shots must be <= {settings.max_shots}")
            device = await registry.device(client, job.backend)
            if size > settings.validate_in_threadpool_bytes:
                circuit = await run_in_threadpool(load_circuit, job.qasm)
            else:
                circuit = load_circuit(job.qasm)
            n_phys = int(device["n_physical"])
            if circuit.num_qubits > n_phys:
                raise ResourceLimitError(
                    f"circuit declares {circuit.num_qubits} qubits but {job.backend} has {n_phys} physical qubits",
                    details={"declared_qubits": circuit.num_qubits, "physical_qubits": n_phys,
                             "device": job.backend})
        with rec.span("routing"):
            seed = job.seed if job.seed is not None else secrets.randbits(63)
            payload = json.dumps({
                "qasm": job.qasm, "shots": job.shots, "seed": seed,
                "options": job.options.model_dump(), "job_id": rec.job_id,
            }, separators=(",", ":")).encode("utf-8")
            headers = {"content-type": "application/json", "x-qputwin-job-id": rec.job_id or ""}
        return entry, payload, headers

    async def post_to_replica(entry: BackendEntry, payload: bytes, headers: dict[str, str],
                              rec: TelemetryRecorder, client: httpx.AsyncClient) -> httpx.Response:
        """Least-in-flight replica; fails over only when the connection was never established."""
        await registry.refresh_replicas(entry)
        tried: list[str] = []
        while True:
            rep = entry.acquire(exclude=tried)
            rep.in_flight += 1
            rep.dispatched += 1
            try:
                resp = await client.post(rep.execute_url, content=payload, headers=headers)
                rep.last_error = None
                rec.replica = rep.url
                return resp
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                rep.failures += 1
                rep.last_error = type(exc).__name__
                tried.append(rep.url)
                if len(tried) >= len(entry.replicas):
                    raise
            finally:
                rep.in_flight -= 1

    async def dispatch(backend: str, entry: BackendEntry, payload: bytes, headers: dict[str, str],
                       rec: TelemetryRecorder, client: httpx.AsyncClient) -> dict[str, Any]:
        """Dispatch + response assembly; returns the COMPLETED body or raises QPUTwinError."""
        try:
            with rec.span("dispatch"):
                resp = await post_to_replica(entry, payload, headers, rec, client)
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError(f"backend {backend!r} timed out", details={"backend": backend}) from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(f"backend {backend!r} is unreachable: {type(exc).__name__}",
                                          details={"backend": backend, "url": entry.url,
                                                   "replicas": [r.url for r in entry.replicas]}) from exc
        with rec.span("response_assembly"):
            try:
                body = resp.json()
            except ValueError as exc:
                raise BackendProtocolError(f"backend {backend!r} returned non-JSON (HTTP {resp.status_code})",
                                           details={"backend": backend}) from exc
            rec.worker_timing = body.get("timing") or {}
            rec.worker_env = body.get("worker") or {}
            rec.engine_invoked = bool(body.get("engine_invoked", False))
            rec.workload = body.get("workload") or {}
            if resp.status_code >= 400 or "error" in body:
                raise error_from_dict(body.get("error") or {"type": "QPUTwinError", "message": "worker error"},
                                      status=resp.status_code)
            result = body["result"]
            compilation = body["compilation"]
            rec.status = "COMPLETED"
            rec.http_status = 200
        telemetry = rec.finalize()
        return {"job_id": rec.job_id, "backend": backend, "status": "COMPLETED",
                "result": result, "compilation": compilation, "telemetry": telemetry}

    async def run_async_job(backend: str, payload: bytes, rec: TelemetryRecorder) -> tuple[int, dict[str, Any]]:
        try:
            entry = registry.get(backend)
            headers = {"content-type": "application/json", "x-qputwin-job-id": rec.job_id or ""}
            return 200, await dispatch(backend, entry, payload, headers, rec, app.state.client)
        except QPUTwinError as err:
            return error_body(rec, err)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("unhandled orchestrator error (async job)")
            return error_body(rec, QPUTwinError(f"internal orchestrator error: {type(exc).__name__}"))

    jobs = JobManager(registry, store, make_store(settings.job_store, settings.job_store_path), run_async_job)
    app.state.jobs = jobs
    app.state.metrics = metrics

    # ---------------------------------------------------------------- jobs
    @app.post("/v1/jobs", tags=["jobs"], response_model=JobResponse,
              responses={**_ERROR_RESPONSES, 202: {"model": JobAccepted,
                                                   "description": "Accepted for asynchronous execution"}})
    async def submit_job(job: JobRequest, request: Request,
                         mode: Literal["sync", "async"] | None = Query(
                             None, description="'async' returns 202 and a job to poll (also selected by "
                                               "the header `Prefer: respond-async`)."),
                         ident: tuple[str, int] = Depends(authenticate)) -> JSONResponse:
        rec = recorder(request)
        rec.enter_handler()
        rec.job_id = uuid.uuid4().hex
        rec.backend = job.backend
        if mode is None:
            mode = "async" if "respond-async" in request.headers.get("prefer", "").lower() else "sync"
        rec.execution_mode = mode
        client: httpx.AsyncClient = request.app.state.client
        try:
            entry, payload, headers = await validate(job, rec, client)
            if mode == "async":
                rec.deferred_record = True
                try:
                    row, position = await jobs.submit(rec.job_id, job.backend, payload, rec, client=ident[0],
                                                      priority=min(job.priority, ident[1]))
                except QPUTwinError:
                    rec.deferred_record = False
                    raise
                accepted = {**row.summary(), "queue_position": position}
                for k in ("started_at", "finished_at", "http_status", "attempts"):
                    accepted.pop(k, None)
                return JSONResponse(accepted, status_code=202,
                                    headers={"location": f"/v1/jobs/{rec.job_id}", "retry-after": "1"})
            return JSONResponse(await dispatch(job.backend, entry, payload, headers, rec, client))
        except QPUTwinError as err:
            return error_response(rec, err)
        except Exception as exc:  # pragma: no cover - defensive: never leak a bare 500
            log.exception("unhandled orchestrator error")
            return error_response(rec, QPUTwinError(f"internal orchestrator error: {type(exc).__name__}"))

    @app.get("/v1/jobs", tags=["jobs"])
    async def list_jobs(status: str | None = None, backend: str | None = None,
                        limit: int = Query(100, ge=1, le=10_000),
                        ident: tuple[str, int] = Depends(authenticate)) -> dict[str, Any]:
        return {"jobs": await jobs.list(status=status, backend=backend, client=scope(ident), limit=limit),
                "queues": await jobs.stats()}

    @app.get("/v1/jobs/{job_id}", tags=["jobs"], response_model=JobStatus)
    async def get_job(job_id: str, wait: float = Query(
            0.0, ge=0.0, description="Long-poll: wait up to this many seconds for the job to finish."),
                      ident: tuple[str, int] = Depends(authenticate)) -> JSONResponse:
        try:
            row = await jobs.wait(job_id, min(wait, settings.max_poll_wait_s), client=scope(ident))
        except QPUTwinError as err:
            return JSONResponse({"error": err.to_dict()}, status_code=err.http_status)
        headers = {} if row.finished else {"retry-after": "1"}
        return JSONResponse(await jobs.view(row), headers=headers)

    @app.delete("/v1/jobs/{job_id}", tags=["jobs"], response_model=JobStatus)
    async def cancel_job(job_id: str, ident: tuple[str, int] = Depends(authenticate)) -> JSONResponse:
        try:
            return JSONResponse(await jobs.view(await jobs.cancel(job_id, client=scope(ident))))
        except QPUTwinError as err:
            return JSONResponse({"error": err.to_dict()}, status_code=err.http_status)

    @app.get("/v1/telemetry", tags=["telemetry"], dependencies=[Depends(authenticate)])
    async def telemetry_recent(limit: int = Query(100, ge=1, le=10_000), backend: str | None = None) -> dict[str, Any]:
        return {"records": store.recent(limit, backend)}

    @app.get("/v1/telemetry/summary", tags=["telemetry"], dependencies=[Depends(authenticate)])
    async def telemetry_summary() -> dict[str, Any]:
        return {"summary": store.summary()}

    @app.exception_handler(QPUTwinError)
    async def domain_error(request: Request, exc: QPUTwinError) -> JSONResponse:
        """Errors raised before a handler body runs (authentication): job submissions keep the envelope."""
        headers = {"www-authenticate": "Bearer"} if exc.http_status == 401 else None
        if request.method == "POST" and request.url.path == "/v1/jobs":
            resp = error_response(recorder(request), exc)
            if headers:
                resp.headers.update(headers)
            return resp
        return JSONResponse({"error": exc.to_dict()}, status_code=exc.http_status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        rec = recorder(request)
        rec.backend = None
        err = InvalidRequestError("request body failed validation",
                                  details={"errors": json.loads(json.dumps(exc.errors(), default=str))})
        return error_response(rec, err)

    return app


app = create_app()
