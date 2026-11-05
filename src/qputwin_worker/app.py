"""Vendor-emulator worker API (FastAPI) wrapping one QPUTwin physics engine.

``create_worker_app(spec, simulator_cls)`` builds the HTTP service for one vendor;
each module in :mod:`qputwin_worker.vendors` instantiates it for its device.

Endpoints
---------
``GET /health``      liveness + vendor id
``GET /v1/device``   full device specification (topology, natives, emulator limits)
``POST /v1/execute`` run an OpenQASM job; returns ``result``, ``compilation``,
                     ``workload`` and a ``timing`` block of *durations* measured
                     inside this process (``service_ms`` spans ASGI entry → response
                     ready, so the orchestrator can derive network transit exactly).

At start-up the device spec is passed through
:func:`~qputwin_core.devices.apply_overrides`, so ``QPUTWIN_COUPLING_MAP`` /
``QPUTWIN_CALIBRATION`` load a calibrated coupling map or calibration snapshot.

Jobs execute one at a time per worker by default (``QPUTWIN_WORKER_CONCURRENCY=1``),
modelling a QPU's serial job queue; time spent waiting is reported as
``queue_ms``. The CPU-bound engine runs in Starlette's threadpool (sync endpoint),
keeping the event loop responsive for health checks.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import threading
import time
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from qputwin_core import __author__, __email__, __version__
from qputwin_core.devices import DeviceSpec, apply_overrides
from qputwin_core.engine import FUSION_MAX_QUBITS
from qputwin_core.errors import InvalidRequestError, QPUTwinError
from qputwin_core.fused import engine_threads
from qputwin_core.simulators import ExecutionOptions, QPUSimulator, SimulatorConfig
from qputwin_core.timing import SpanTimer, ns_to_ms

log = logging.getLogger("qputwin.worker")
_T0 = "qputwin_t0_ns"


class ExecuteOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    compilation: Literal["native", "strict"] = "native"
    return_statevector: bool = False
    memory: bool = False
    precision: Literal["complex128", "complex64"] | None = None
    # validated in qputwin_core (RoutingOptions / NoiseOptions.from_dict), which owns their semantics
    routing: dict[str, Any] | None = None
    noise: dict[str, Any] | None = None
    engine: Literal["auto", "statevector", "density_matrix", "trajectory"] = "auto"
    return_density_matrix: bool = False


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qasm: str = Field(..., min_length=1)
    shots: int = Field(1024, ge=1)
    seed: int | None = Field(None, ge=0, le=2 ** 63 - 1)
    options: ExecuteOptions = Field(default_factory=ExecuteOptions)
    job_id: str | None = None


class ServiceTimingMiddleware:
    """Pure-ASGI middleware stamping the request's entry time (perf_counter_ns)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})[_T0] = time.perf_counter_ns()
        await self.app(scope, receive, send)


def _cpu_limit() -> float | None:
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        return None


def worker_environment(spec: DeviceSpec, sim: QPUSimulator) -> dict[str, Any]:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        cpus = os.cpu_count() or 1
    return {
        "worker_id": os.environ.get("QPUTWIN_WORKER_ID") or socket.gethostname(),
        "vendor": spec.id,
        "simulator_class": type(sim).__name__,
        "qputwin_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "precision": sim.config.precision,
        "max_sim_qubits": sim.max_sim_qubits,
        "blas_threads": os.environ.get("OMP_NUM_THREADS"),
        "engine_threads": engine_threads(),
        "fusion_max_qubits": FUSION_MAX_QUBITS,
        "cpu_affinity": cpus,
        "cgroup_cpu_limit": _cpu_limit(),
        "concurrency": int(os.environ.get("QPUTWIN_WORKER_CONCURRENCY", "1")),
        "coupling_map": os.environ.get("QPUTWIN_COUPLING_MAP") or "packaged",
        "calibration": (spec.calibration.origin if spec.calibration is not None else None),
    }


def create_worker_app(spec: DeviceSpec, simulator_cls: type[QPUSimulator],
                      config: SimulatorConfig | None = None, *, overrides: bool = True) -> FastAPI:
    spec = apply_overrides(spec) if overrides else spec
    sim = simulator_cls(spec, config or SimulatorConfig.from_env())
    env = worker_environment(spec, sim)
    gate = threading.BoundedSemaphore(max(1, env["concurrency"]))

    app = FastAPI(
        title=f"QPUTwin worker — {spec.vendor} {spec.name}",
        version=__version__,
        description=f"{spec.description}\n\n{__author__} <{__email__}>.",
        contact={"name": __author__, "email": __email__},
    )
    app.add_middleware(ServiceTimingMiddleware)
    app.state.simulator = sim

    def finish(request: Request, timer: SpanTimer, status: int, body: dict[str, Any]) -> JSONResponse:
        t0 = getattr(request.state, _T0, None)
        spans = timer.as_ms()
        service = ns_to_ms(time.perf_counter_ns() - t0) if t0 is not None else sum(spans.values())
        timing = {
            "service_ms": service,
            "queue_ms": spans.get("queue", 0.0),
            "parse_ms": spans.get("parse", 0.0),
            "route_ms": spans.get("route", 0.0),
            "compile_ms": spans.get("compile", 0.0),
            "simulation_ms": spans.get("simulation", 0.0),
            "sampling_ms": spans.get("sampling", 0.0),
        }
        timing["unattributed_ms"] = service - sum(v for k, v in timing.items() if k != "service_ms")
        body["timing"] = timing
        body["worker"] = env
        return JSONResponse(body, status_code=status)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "worker", "vendor": spec.id, "version": __version__}

    @app.get("/v1/device")
    def device() -> dict[str, Any]:
        return sim.describe()

    @app.post("/v1/execute")
    def execute(req: ExecuteRequest, request: Request) -> JSONResponse:
        timer = SpanTimer()
        with timer.span("queue"):
            gate.acquire()
        try:
            opts = ExecutionOptions(shots=req.shots, seed=req.seed, compilation=req.options.compilation,
                                    return_statevector=req.options.return_statevector,
                                    memory=req.options.memory, precision=req.options.precision,
                                    routing=req.options.routing, noise=req.options.noise,
                                    engine=req.options.engine,
                                    return_density_matrix=req.options.return_density_matrix)
            result = sim.execute(req.qasm, opts, timer)
            return finish(request, timer, 200, result.to_dict())
        except QPUTwinError as err:
            return finish(request, timer, err.http_status,
                          {"error": err.to_dict(), "engine_invoked": "simulation" in timer.spans_ns})
        except MemoryError:
            err = QPUTwinError("worker ran out of memory during simulation")
            err.http_status = 507
            return finish(request, timer, 507, {"error": err.to_dict(), "engine_invoked": True})
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("unhandled worker error")
            err = QPUTwinError(f"internal worker error: {type(exc).__name__}: {exc}")
            return finish(request, timer, 500, {"error": err.to_dict(),
                                                "engine_invoked": "simulation" in timer.spans_ns})
        finally:
            gate.release()

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        err = InvalidRequestError("request body failed validation",
                                  details={"errors": [str(e) for e in exc.errors()]})
        return finish(request, SpanTimer(), 400, {"error": err.to_dict(), "engine_invoked": False})

    return app


def load_vendor_app(vendor: str) -> FastAPI:
    """Import ``qputwin_worker.vendors.<vendor>`` and return its ``app``."""
    import importlib

    module = importlib.import_module(f"qputwin_worker.vendors.{vendor}")
    return module.app
