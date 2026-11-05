"""Backend registry: maps backend IDs to worker URLs and caches device specifications.

Configuration is read from a TOML file (``QPUTWIN_BACKENDS_CONFIG``, default
``config/backends.toml``) and can be overridden per backend with environment
variables ``QPUTWIN_BACKEND_URL_<BACKEND_ID_UPPERCASE>`` — e.g.
``QPUTWIN_BACKEND_URL_IONQ_FORTE=http://localhost:8102`` — which is how the test-suite
and local development point the orchestrator at differently-hosted workers.

Horizontal scaling
------------------
A backend may be served by several identical worker **replicas**:

* *static* — ``urls = ["http://a:8001", "http://b:8001"]`` in the TOML file, or a
  comma-separated ``QPUTWIN_BACKEND_URL_<ID>``;
* *DNS discovery* — with ``discovery = "dns"`` each configured URL's host name is
  resolved to **all** of its addresses (re-resolved every
  ``discovery_interval_s``), so ``docker compose up --scale worker-ionq-forte=3``
  is picked up without reconfiguration.

Each job goes to the replica with the fewest in-flight requests (round-robin
among ties). A replica that refuses the connection is skipped and the job fails
over to the next one; a timeout is never retried, because the worker may already
be executing the job.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from qputwin_core.errors import BackendProtocolError, BackendUnavailableError, UnknownBackendError

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "backends.toml"


@dataclass(slots=True)
class OrchestratorSettings:
    request_timeout_s: float = 600.0
    connect_timeout_s: float = 5.0
    max_qasm_bytes: int = 1 << 20
    max_shots: int = 1_000_000
    telemetry_buffer: int = 10_000
    telemetry_path: str | None = None
    validate_in_threadpool_bytes: int = 64 * 1024
    discovery: str = "static"                  # "static" | "dns"
    discovery_interval_s: float = 10.0
    async_max_queued: int = 10_000             # jobs waiting in the orchestrator queue, all backends
    async_retention: int = 10_000              # finished async jobs kept for polling
    async_ttl_s: float = 3600.0                # finished async jobs expire after this long
    async_per_replica_concurrency: int = 1     # dispatch slots per replica (matches the worker's serial queue)
    max_poll_wait_s: float = 30.0              # upper bound for GET /v1/jobs/{id}?wait=
    job_store: str = "memory"                  # "memory" | "sqlite"
    job_store_path: str | None = None          # sqlite file, shared by every orchestrator instance
    job_lease_s: float = 30.0                  # a running job's lease; renewed every lease/3
    job_max_attempts: int = 3                  # executions before an abandoned job is failed
    job_poll_interval_s: float = 0.2           # queue polling period (other instances' submissions)
    api_keys: dict[str, tuple[str, int]] = field(default_factory=dict)   # key -> (client, max priority)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


def parse_api_keys(spec: str) -> dict[str, tuple[str, int]]:
    """``name:key[:max_priority]`` entries separated by commas or newlines (``#`` comments allowed)."""
    keys: dict[str, tuple[str, int]] = {}
    for raw in spec.replace(",", "\n").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) not in (2, 3) or not parts[0] or len(parts[1]) < 16:
            raise ValueError("API keys must be 'name:key[:max_priority]' with keys of at least 16 characters")
        maxp = int(parts[2]) if len(parts) == 3 else 9
        if not 0 <= maxp <= 9:
            raise ValueError("max_priority must be in [0, 9]")
        keys[parts[1]] = (parts[0], maxp)
    return keys


@dataclass(slots=True)
class Replica:
    url: str
    in_flight: int = 0
    dispatched: int = 0
    failures: int = 0
    last_error: str | None = None

    @property
    def execute_url(self) -> str:
        return f"{self.url}/v1/execute"

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "in_flight": self.in_flight, "dispatched": self.dispatched,
                "failures": self.failures, "last_error": self.last_error}


@dataclass(slots=True)
class BackendEntry:
    id: str
    url: str                                   # primary configured URL
    description: str = ""
    device: dict[str, Any] | None = None
    last_error: str | None = None
    urls: list[str] = field(default_factory=list)          # configured seed URLs
    replicas: list[Replica] = field(default_factory=list)  # current replica set
    discovered_at: float = float("-inf")
    _rr: int = 0

    def __post_init__(self) -> None:
        if not self.urls:
            self.urls = [self.url]
        if not self.replicas:
            self.replicas = [Replica(u) for u in self.urls]

    @property
    def execute_url(self) -> str:
        return f"{self.url}/v1/execute"

    def set_replica_urls(self, urls: Iterable[str]) -> None:
        """Replace the replica set, keeping the counters of replicas that persist."""
        old = {r.url: r for r in self.replicas}
        new = [old.get(u) or Replica(u) for u in dict.fromkeys(urls)]
        if new:
            self.replicas = new

    def acquire(self, exclude: Iterable[str] = ()) -> Replica:
        """Least-in-flight replica (round-robin among ties), excluding ``exclude``."""
        skip = set(exclude)
        cands = [r for r in self.replicas if r.url not in skip]
        if not cands:
            raise BackendUnavailableError(f"backend {self.id!r} has no reachable replica",
                                          details={"backend": self.id, "replicas": [r.url for r in self.replicas]})
        least = min(r.in_flight for r in cands)
        ties = [r for r in cands if r.in_flight == least]
        rep = ties[self._rr % len(ties)]
        self._rr += 1
        return rep


@dataclass(slots=True)
class BackendRegistry:
    entries: dict[str, BackendEntry]
    settings: OrchestratorSettings = field(default_factory=OrchestratorSettings)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    # ------------------------------------------------------------- loading
    @classmethod
    def from_config(cls, path: str | Path | None = None) -> BackendRegistry:
        path = Path(path or os.environ.get("QPUTWIN_BACKENDS_CONFIG") or DEFAULT_CONFIG)
        doc = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        o = doc.get("orchestrator", {})
        settings = OrchestratorSettings(
            request_timeout_s=float(os.environ.get("QPUTWIN_REQUEST_TIMEOUT_S", o.get("request_timeout_s", 600.0))),
            connect_timeout_s=float(os.environ.get("QPUTWIN_CONNECT_TIMEOUT_S", o.get("connect_timeout_s", 5.0))),
            max_qasm_bytes=int(os.environ.get("QPUTWIN_MAX_QASM_BYTES", o.get("max_qasm_bytes", 1 << 20))),
            max_shots=int(os.environ.get("QPUTWIN_MAX_SHOTS", o.get("max_shots", 1_000_000))),
            telemetry_buffer=int(os.environ.get("QPUTWIN_TELEMETRY_BUFFER", o.get("telemetry_buffer", 10_000))),
            telemetry_path=os.environ.get("QPUTWIN_TELEMETRY_PATH", o.get("telemetry_path")) or None,
            discovery=str(os.environ.get("QPUTWIN_DISCOVERY", o.get("discovery", "static"))).strip().lower(),
            discovery_interval_s=float(os.environ.get("QPUTWIN_DISCOVERY_INTERVAL_S", o.get("discovery_interval_s", 10.0))),
            async_max_queued=int(os.environ.get("QPUTWIN_ASYNC_MAX_QUEUED", o.get("async_max_queued", 10_000))),
            async_retention=int(os.environ.get("QPUTWIN_ASYNC_RETENTION", o.get("async_retention", 10_000))),
            async_ttl_s=float(os.environ.get("QPUTWIN_ASYNC_TTL_S", o.get("async_ttl_s", 3600.0))),
            async_per_replica_concurrency=int(os.environ.get(
                "QPUTWIN_ASYNC_PER_REPLICA_CONCURRENCY", o.get("async_per_replica_concurrency", 1))),
            max_poll_wait_s=float(os.environ.get("QPUTWIN_MAX_POLL_WAIT_S", o.get("max_poll_wait_s", 30.0))),
            job_store=str(os.environ.get("QPUTWIN_JOB_STORE", o.get("job_store", "memory"))).strip().lower(),
            job_store_path=os.environ.get("QPUTWIN_JOB_STORE_PATH", o.get("job_store_path")) or None,
            job_lease_s=float(os.environ.get("QPUTWIN_JOB_LEASE_S", o.get("job_lease_s", 30.0))),
            job_max_attempts=int(os.environ.get("QPUTWIN_JOB_MAX_ATTEMPTS", o.get("job_max_attempts", 3))),
            job_poll_interval_s=float(os.environ.get("QPUTWIN_JOB_POLL_INTERVAL_S", o.get("job_poll_interval_s", 0.2))),
        )
        keys = os.environ.get("QPUTWIN_API_KEYS", "")
        key_file = os.environ.get("QPUTWIN_API_KEYS_FILE")
        if key_file:
            keys += "\n" + Path(key_file).read_text(encoding="utf-8")
        settings.api_keys = parse_api_keys(keys)
        if settings.discovery not in ("static", "dns"):
            raise ValueError(f"discovery must be 'static' or 'dns', got {settings.discovery!r}")
        entries: dict[str, BackendEntry] = {}
        for bid, b in doc.get("backends", {}).items():
            env = os.environ.get(f"QPUTWIN_BACKEND_URL_{bid.upper()}")
            if env:
                urls = [u.strip().rstrip("/") for u in env.split(",") if u.strip()]
            else:
                raw = b.get("urls") or [b["url"]]
                urls = [str(u).rstrip("/") for u in raw]
            entries[bid] = BackendEntry(id=bid, url=urls[0], description=b.get("description", ""), urls=urls)
        return cls(entries=entries, settings=settings)

    # ------------------------------------------------------------- queries
    def get(self, backend_id: str) -> BackendEntry:
        entry = self.entries.get(backend_id)
        if entry is None:
            raise UnknownBackendError(f"unknown backend {backend_id!r}",
                                      details={"backend": backend_id, "available": sorted(self.entries)})
        return entry

    async def device(self, client: httpx.AsyncClient, backend_id: str, *, refresh: bool = False) -> dict[str, Any]:
        entry = self.get(backend_id)
        if entry.device is not None and not refresh:
            return entry.device
        lock = self._locks.setdefault(backend_id, asyncio.Lock())
        async with lock:
            if entry.device is not None and not refresh:
                return entry.device
            await self.refresh_replicas(entry)
            resp = None
            for rep in list(entry.replicas):
                try:
                    resp = await client.get(f"{rep.url}/v1/device", timeout=self.settings.connect_timeout_s + 5)
                    break
                except httpx.HTTPError as exc:
                    entry.last_error = f"{type(exc).__name__}: {exc}"
            if resp is None:
                raise BackendUnavailableError(f"backend {backend_id!r} is unreachable",
                                              details={"backend": backend_id, "url": entry.url,
                                                       "reason": entry.last_error})
            if resp.status_code != 200:
                raise BackendProtocolError(f"backend {backend_id!r} returned HTTP {resp.status_code} for /v1/device",
                                           details={"backend": backend_id})
            entry.device = resp.json()
            entry.last_error = None
            return entry.device

    async def prefetch(self, client: httpx.AsyncClient) -> None:
        async def one(bid: str) -> None:
            try:
                await self.device(client, bid)
            except Exception:  # noqa: BLE001 - workers may still be starting; fetched lazily later
                pass
        await asyncio.gather(*(one(b) for b in self.entries))

    async def health(self, client: httpx.AsyncClient, backend_id: str) -> dict[str, Any]:
        """Aggregate health: ``online`` if any replica answers; per-replica detail in ``replicas``."""
        entry = self.get(backend_id)
        await self.refresh_replicas(entry)

        async def one(rep: Replica) -> dict[str, Any]:
            try:
                resp = await client.get(f"{rep.url}/health", timeout=2.0)
                ok = resp.status_code == 200
                info = resp.json() if ok else {}
                return {**info, "worker_status": info.get("status"),
                        "status": "online" if ok else f"http_{resp.status_code}"}
            except httpx.HTTPError as exc:
                return {"status": "offline", "reason": f"{type(exc).__name__}"}

        reps = list(entry.replicas)
        results = await asyncio.gather(*(one(r) for r in reps))
        online = [h for h in results if h["status"] == "online"]
        head = online[0] if online else results[0]
        return {**head, "status": "online" if online else head["status"],
                "replicas": [{**r.to_dict(), "status": h["status"]} for r, h in zip(reps, results, strict=True)]}

    # ----------------------------------------------------------- replicas
    async def refresh_replicas(self, entry: BackendEntry, *, force: bool = False) -> None:
        """DNS discovery: resolve every seed URL to all of its addresses (no-op for static)."""
        if self.settings.discovery != "dns":
            return
        now = time.monotonic()
        if not force and now - entry.discovered_at < self.settings.discovery_interval_s:
            return
        entry.discovered_at = now
        loop = asyncio.get_running_loop()
        urls: list[str] = []
        for seed in entry.urls:
            parts = urlsplit(seed)
            host, port = parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
            if not host:
                continue
            try:
                infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except OSError:
                continue
            for family, *_rest, sockaddr in infos:
                ip = sockaddr[0]
                netloc = f"[{ip}]:{port}" if family == socket.AF_INET6 else f"{ip}:{port}"
                urls.append(f"{parts.scheme}://{netloc}")
        if urls:                                   # resolution failure keeps the last known set
            entry.set_replica_urls(sorted(set(urls)))

    def capacity(self, backend_id: str) -> int:
        """Dispatch slots for asynchronous jobs: replicas × per-replica concurrency."""
        entry = self.get(backend_id)
        return max(1, len(entry.replicas)) * max(1, self.settings.async_per_replica_concurrency)
