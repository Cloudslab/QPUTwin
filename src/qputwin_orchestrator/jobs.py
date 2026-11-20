"""Asynchronous job queue with a durable SQLite job store.

Usage: from qputwin_orchestrator.jobs import JobManager

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qputwin_core.errors import JobNotFoundError, JobStateError, QPUTwinError, QueueFullError
from qputwin_orchestrator.registry import BackendRegistry
from qputwin_orchestrator.telemetry import TelemetryRecorder, TelemetryStore

__all__ = ["JobRow", "JobStore", "MemoryJobStore", "SqliteJobStore", "JobManager", "FINAL_STATES",
           "make_store"]

log = logging.getLogger("qputwin.orchestrator.jobs")

FINAL_STATES = frozenset({"COMPLETED", "REJECTED", "FAILED", "CANCELLED"})
PRUNE_INTERVAL_S = 2.0           # retention runs at most this often (it was once per submission)
PRE_SPANS = ("ingress", "validation", "routing")


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ======================================================================= rows
@dataclass(slots=True)
class JobRow:
    job_id: str
    backend: str
    payload: str                                   # worker request body (JSON, seed fixed)
    client: str = "anonymous"
    priority: int = 5
    status: str = "QUEUED"
    submitted_at: str = field(default_factory=utcnow)
    submitted_wall: float = field(default_factory=time.time)
    started_at: str | None = None
    finished_at: str | None = None
    finished_wall: float | None = None
    http_status: int | None = None
    body: str | None = None                        # final response body (JSON)
    pre_spans: str = "{}"                          # {"ingress": ns, "validation": ns, "routing": ns}
    owner: str | None = None
    lease_until: float | None = None
    attempts: int = 0
    seq: int = 0

    @property
    def finished(self) -> bool:
        return self.status in FINAL_STATES

    def summary(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "backend": self.backend, "status": self.status, "execution_mode": "async",
                "client": self.client, "priority": self.priority, "submitted_at": self.submitted_at,
                "started_at": self.started_at, "finished_at": self.finished_at, "http_status": self.http_status,
                "attempts": self.attempts, "links": {"self": f"/v1/jobs/{self.job_id}"}}


# ====================================================================== stores
class JobStore:
    """Interface of a job store. All methods are synchronous and individually atomic."""

    blocking = False

    def insert(self, row: JobRow, max_queued: int) -> int: ...                      # -> queue position
    def get(self, job_id: str) -> JobRow | None: ...
    def claim(self, backend: str, capacity: int, owner: str, lease_s: float) -> JobRow | None: ...
    def renew(self, job_ids: list[str], owner: str, lease_s: float) -> None: ...
    def finish(self, job_id: str, owner: str | None, status: str, http_status: int | None, body: str,
               *, from_states: tuple[str, ...] = ("RUNNING",)) -> bool: ...
    def requeue_expired(self, max_attempts: int) -> list[JobRow]: ...              # -> rows that ran out of attempts
    def release(self, owner: str) -> int: ...
    def position(self, row: JobRow) -> int | None: ...
    def stats(self) -> dict[str, dict[str, int]]: ...
    def list(self, status: str | None, backend: str | None, client: str | None, limit: int) -> list[JobRow]: ...
    def prune(self, ttl_s: float, keep: int) -> int: ...
    def ping(self) -> None: ...
    def close(self) -> None: ...


def _order_key(running_by_client: dict[str, int], last_served: dict[str, int]
               ) -> Callable[[JobRow], tuple[Any, ...]]:
    """Priority (high first), then fair share, then FIFO.

    Fair share ranks a client by its running jobs on the backend and then by when it was last
    served there (never served first). The second key matters with one dispatch slot, where
    every client has zero running jobs at claim time: it turns the order into round-robin
    between clients instead of FIFO behind whichever client queued the most.
    """
    return lambda r: (-r.priority, running_by_client.get(r.client, 0), last_served.get(r.client, 0), r.seq)


class MemoryJobStore(JobStore):
    """In-process store (single orchestrator instance; jobs do not survive a restart)."""

    def __init__(self) -> None:
        self._rows: dict[str, JobRow] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._served: dict[tuple[str, str], int] = {}       # (backend, client) -> serve counter of its last claim
        self._serve_seq = 0

    def insert(self, row: JobRow, max_queued: int) -> int:
        with self._lock:
            queued = [r for r in self._rows.values() if r.status == "QUEUED"]
            if len(queued) >= max_queued:
                raise QueueFullError(f"the asynchronous queue is full ({len(queued)} jobs waiting); retry later",
                                     details={"queued": len(queued), "max_queued": max_queued})
            self._seq += 1
            row.seq = self._seq
            self._rows[row.job_id] = row
            return sum(1 for r in queued if r.backend == row.backend) + 1

    def get(self, job_id: str) -> JobRow | None:
        with self._lock:
            r = self._rows.get(job_id)
            return replace(r) if r is not None else None

    def claim(self, backend: str, capacity: int, owner: str, lease_s: float) -> JobRow | None:
        with self._lock:
            rows = [r for r in self._rows.values() if r.backend == backend]
            running = [r for r in rows if r.status == "RUNNING"]
            if len(running) >= capacity:
                return None
            queued = [r for r in rows if r.status == "QUEUED"]
            if not queued:
                return None
            by_client: dict[str, int] = {}
            for r in running:
                by_client[r.client] = by_client.get(r.client, 0) + 1
            last = {c: n for (b, c), n in self._served.items() if b == backend}
            r = min(queued, key=_order_key(by_client, last))
            r.status, r.owner, r.lease_until = "RUNNING", owner, time.time() + lease_s
            r.started_at, r.attempts = utcnow(), r.attempts + 1
            self._serve_seq += 1
            self._served[(backend, r.client)] = self._serve_seq
            return replace(r)

    def renew(self, job_ids: list[str], owner: str, lease_s: float) -> None:
        with self._lock:
            for j in job_ids:
                r = self._rows.get(j)
                if r is not None and r.owner == owner and r.status == "RUNNING":
                    r.lease_until = time.time() + lease_s

    def finish(self, job_id: str, owner: str | None, status: str, http_status: int | None, body: str,
               *, from_states: tuple[str, ...] = ("RUNNING",)) -> bool:
        with self._lock:
            r = self._rows.get(job_id)
            if r is None or r.status not in from_states or (owner is not None and r.owner != owner
                                                            and r.status == "RUNNING"):
                return False
            r.status, r.http_status, r.body = status, http_status, body
            r.finished_at, r.finished_wall, r.lease_until = utcnow(), time.time(), None
            return True

    def requeue_expired(self, max_attempts: int) -> list[JobRow]:
        now = time.time()
        exhausted = []
        with self._lock:
            for r in self._rows.values():
                if r.status == "RUNNING" and r.lease_until is not None and r.lease_until < now:
                    if r.attempts >= max_attempts:
                        exhausted.append(replace(r))
                    else:
                        r.status, r.owner, r.lease_until = "QUEUED", None, None
        return exhausted

    def release(self, owner: str) -> int:
        with self._lock:
            n = 0
            for r in self._rows.values():
                if r.status == "RUNNING" and r.owner == owner:
                    r.status, r.owner, r.lease_until = "QUEUED", None, None
                    r.attempts = max(0, r.attempts - 1)
                    n += 1
            return n

    def position(self, row: JobRow) -> int | None:
        if row.status != "QUEUED":
            return None
        with self._lock:
            ahead = [r for r in self._rows.values() if r.backend == row.backend and r.status == "QUEUED"]
        ahead.sort(key=lambda r: (-r.priority, r.seq))
        return next((k + 1 for k, r in enumerate(ahead) if r.job_id == row.job_id), None)

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            out: dict[str, dict[str, int]] = {}
            for r in self._rows.values():
                if r.status in ("QUEUED", "RUNNING"):
                    d = out.setdefault(r.backend, {"queued": 0, "running": 0})
                    d["queued" if r.status == "QUEUED" else "running"] += 1
            return out

    def list(self, status: str | None, backend: str | None, client: str | None, limit: int) -> list[JobRow]:
        with self._lock:
            rows = [replace(r) for r in self._rows.values()
                    if (status is None or r.status == status) and (backend is None or r.backend == backend)
                    and (client is None or r.client == client)]
        rows.sort(key=lambda r: r.seq)
        return rows[-limit:]

    def prune(self, ttl_s: float, keep: int) -> int:
        now = time.time()
        with self._lock:
            fin = sorted((r for r in self._rows.values() if r.finished), key=lambda r: r.seq)
            excess = len(fin) - keep
            gone = 0
            for r in fin:
                if (r.finished_wall is not None and now - r.finished_wall > ttl_s) or excess > 0:
                    del self._rows[r.job_id]
                    excess -= 1
                    gone += 1
            return gone

    def ping(self) -> None:
        return None

    def close(self) -> None:
        return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL, backend TEXT NOT NULL, payload TEXT NOT NULL,
    client TEXT NOT NULL, priority INTEGER NOT NULL, status TEXT NOT NULL,
    submitted_at TEXT NOT NULL, submitted_wall REAL NOT NULL,
    started_at TEXT, finished_at TEXT, finished_wall REAL, http_status INTEGER, body TEXT,
    pre_spans TEXT NOT NULL, owner TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_backend_status ON jobs (backend, status, priority, seq);
CREATE INDEX IF NOT EXISTS jobs_finished ON jobs (status, finished_wall);
DROP INDEX IF EXISTS jobs_client_started;
CREATE INDEX IF NOT EXISTS jobs_done_wall ON jobs (finished_wall) WHERE finished_wall IS NOT NULL;
CREATE INDEX IF NOT EXISTS jobs_done_seq ON jobs (seq) WHERE finished_wall IS NOT NULL;
CREATE TABLE IF NOT EXISTS client_usage (
    backend TEXT NOT NULL, client TEXT NOT NULL, served INTEGER NOT NULL,
    PRIMARY KEY (backend, client)
);
"""
_COLS = ("job_id", "backend", "payload", "client", "priority", "status", "submitted_at", "submitted_wall",
         "started_at", "finished_at", "finished_wall", "http_status", "body", "pre_spans", "owner",
         "lease_until", "attempts", "seq")


class SqliteJobStore(JobStore):
    """Durable store in one SQLite file (WAL). Safe for several orchestrator processes on one host."""

    blocking = True

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._db().executescript(_SCHEMA)          # executescript manages its own transaction

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=True)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA busy_timeout=30000")
            self._local.db = db
        return db

    @contextlib.contextmanager
    def _tx(self) -> Any:
        db = self._db()
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise

    @staticmethod
    def _row(t: tuple[Any, ...] | None) -> JobRow | None:
        return None if t is None else JobRow(**dict(zip(_COLS, t, strict=True)))

    _SELECT = f"SELECT {', '.join(_COLS)} FROM jobs"

    def insert(self, row: JobRow, max_queued: int) -> int:
        with self._tx() as db:
            queued = db.execute("SELECT COUNT(*) FROM jobs WHERE status='QUEUED'").fetchone()[0]
            if queued >= max_queued:
                raise QueueFullError(f"the asynchronous queue is full ({queued} jobs waiting); retry later",
                                     details={"queued": queued, "max_queued": max_queued})
            cols = [c for c in _COLS if c != "seq"]
            cur = db.execute(f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                             [getattr(row, c) for c in cols])
            row.seq = int(cur.lastrowid or 0)
            return db.execute("SELECT COUNT(*) FROM jobs WHERE backend=? AND status='QUEUED'",
                              (row.backend,)).fetchone()[0]

    def get(self, job_id: str) -> JobRow | None:
        return self._row(self._db().execute(f"{self._SELECT} WHERE job_id=?", (job_id,)).fetchone())

    def claim(self, backend: str, capacity: int, owner: str, lease_s: float) -> JobRow | None:
        with self._tx() as db:
            running = db.execute("SELECT COUNT(*) FROM jobs WHERE backend=? AND status='RUNNING'",
                                 (backend,)).fetchone()[0]
            if running >= capacity:
                return None
            t = db.execute(
                f"SELECT {', '.join('j.' + c for c in _COLS)} FROM jobs AS j "
                "LEFT JOIN client_usage AS u ON u.backend = j.backend AND u.client = j.client "
                "WHERE j.backend=? AND j.status='QUEUED' ORDER BY j.priority DESC, "
                "(SELECT COUNT(*) FROM jobs r WHERE r.backend=j.backend AND r.status='RUNNING' "
                "AND r.client=j.client) ASC, IFNULL(u.served, 0) ASC, j.seq ASC LIMIT 1", (backend,)).fetchone()
            row = self._row(t)
            if row is None:
                return None
            row.status, row.owner, row.lease_until = "RUNNING", owner, time.time() + lease_s
            row.started_at, row.attempts = utcnow(), row.attempts + 1
            db.execute("UPDATE jobs SET status='RUNNING', owner=?, lease_until=?, started_at=?, attempts=? "
                       "WHERE job_id=?", (owner, row.lease_until, row.started_at, row.attempts, row.job_id))
            db.execute("INSERT INTO client_usage (backend, client, served) "              # fair-share bookkeeping
                       "VALUES (?, ?, (SELECT IFNULL(MAX(served), 0) + 1 FROM client_usage)) "
                       "ON CONFLICT (backend, client) DO UPDATE SET served = excluded.served",
                       (backend, row.client))
            return row

    def renew(self, job_ids: list[str], owner: str, lease_s: float) -> None:
        if not job_ids:
            return
        with self._tx() as db:
            db.executemany("UPDATE jobs SET lease_until=? WHERE job_id=? AND owner=? AND status='RUNNING'",
                           [(time.time() + lease_s, j, owner) for j in job_ids])

    def finish(self, job_id: str, owner: str | None, status: str, http_status: int | None, body: str,
               *, from_states: tuple[str, ...] = ("RUNNING",)) -> bool:
        with self._tx() as db:
            q = (f"UPDATE jobs SET status=?, http_status=?, body=?, finished_at=?, finished_wall=?, lease_until=NULL "
                 f"WHERE job_id=? AND status IN ({', '.join('?' * len(from_states))})")
            args: list[Any] = [status, http_status, body, utcnow(), time.time(), job_id, *from_states]
            if owner is not None:
                q += " AND (status != 'RUNNING' OR owner=?)"
                args.append(owner)
            return db.execute(q, args).rowcount == 1

    def requeue_expired(self, max_attempts: int) -> list[JobRow]:
        now = time.time()
        with self._tx() as db:
            exhausted = [self._row(t) for t in db.execute(
                f"{self._SELECT} WHERE status='RUNNING' AND lease_until < ? AND attempts >= ?",
                (now, max_attempts)).fetchall()]
            db.execute("UPDATE jobs SET status='QUEUED', owner=NULL, lease_until=NULL "
                       "WHERE status='RUNNING' AND lease_until < ? AND attempts < ?", (now, max_attempts))
        return [r for r in exhausted if r is not None]

    def release(self, owner: str) -> int:
        with self._tx() as db:
            return db.execute("UPDATE jobs SET status='QUEUED', owner=NULL, lease_until=NULL, "
                              "attempts=MAX(0, attempts - 1) WHERE status='RUNNING' AND owner=?", (owner,)).rowcount

    def position(self, row: JobRow) -> int | None:
        if row.status != "QUEUED":
            return None
        return self._db().execute(
            "SELECT COUNT(*) FROM jobs WHERE backend=? AND status='QUEUED' AND "
            "(priority > ? OR (priority = ? AND seq <= ?))",
            (row.backend, row.priority, row.priority, row.seq)).fetchone()[0]

    def stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for backend, status, n in self._db().execute(
                "SELECT backend, status, COUNT(*) FROM jobs WHERE status IN ('QUEUED', 'RUNNING') "
                "GROUP BY backend, status"):
            out.setdefault(backend, {"queued": 0, "running": 0})["queued" if status == "QUEUED" else "running"] = n
        return out

    def list(self, status: str | None, backend: str | None, client: str | None, limit: int) -> list[JobRow]:
        where, args = [], []
        for col, v in (("status", status), ("backend", backend), ("client", client)):
            if v is not None:
                where.append(f"{col}=?")
                args.append(v)
        q = self._SELECT + (f" WHERE {' AND '.join(where)}" if where else "") + " ORDER BY seq DESC LIMIT ?"
        rows = [self._row(t) for t in self._db().execute(q, [*args, limit]).fetchall()]
        return [r for r in reversed(rows) if r is not None]

    def prune(self, ttl_s: float, keep: int) -> int:
        # both statements are index range operations on the partial indexes over finished jobs
        with self._tx() as db:
            gone = db.execute("DELETE FROM jobs WHERE finished_wall IS NOT NULL AND finished_wall < ?",
                              (time.time() - ttl_s,)).rowcount
            done = db.execute("SELECT COUNT(*) FROM jobs WHERE finished_wall IS NOT NULL").fetchone()[0]
            if done > keep:
                gone += db.execute(
                    "DELETE FROM jobs WHERE seq IN (SELECT seq FROM jobs WHERE finished_wall IS NOT NULL "
                    "ORDER BY seq LIMIT ?)", (done - keep,)).rowcount
            return gone

    def ping(self) -> None:
        self._db().execute("SELECT 1").fetchone()

    def close(self) -> None:
        db = getattr(self._local, "db", None)
        if db is not None:
            db.close()
            self._local.db = None


def make_store(kind: str, path: str | None) -> JobStore:
    if kind == "memory":
        return MemoryJobStore()
    if kind == "sqlite":
        if not path:
            raise ValueError("job_store = 'sqlite' needs job_store_path (QPUTWIN_JOB_STORE_PATH)")
        return SqliteJobStore(path)
    raise ValueError(f"job_store must be 'memory' or 'sqlite', got {kind!r}")


# ===================================================================== manager
RunJob = Callable[[str, bytes, TelemetryRecorder], Awaitable[tuple[int, dict[str, Any]]]]


class JobManager:
    def __init__(self, registry: BackendRegistry, telemetry: TelemetryStore, store: JobStore, run_job: RunJob) -> None:
        self.registry = registry
        self.telemetry = telemetry
        self.store = store
        self.settings = registry.settings
        self._run_job = run_job
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self._local: dict[str, tuple[TelemetryRecorder, int]] = {}      # job_id -> (recorder, enqueue ns)
        self._events: dict[str, asyncio.Event] = {}
        self._wake: dict[str, asyncio.Event] = {}
        self._running: dict[str, set[str]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._pumps: list[asyncio.Task[None]] = []
        self._stopping = False
        self._pruned_at = 0.0

    async def _maybe_prune(self) -> None:
        """Apply retention at most every ``PRUNE_INTERVAL_S`` (not on every request)."""
        now = time.monotonic()
        if now - self._pruned_at >= PRUNE_INTERVAL_S:
            self._pruned_at = now
            await self._call(self.store.prune, self.settings.async_ttl_s, self.settings.async_retention)

    async def _call(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        if self.store.blocking:
            return await asyncio.to_thread(fn, *args, **kw)
        return fn(*args, **kw)

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        for b in self.registry.entries:
            self._ensure_pump(b)
        self._pumps.append(asyncio.create_task(self._housekeeping(), name="qputwin-jobs-housekeeping"))

    def _ensure_pump(self, backend: str) -> None:
        if backend in self._wake:
            return
        self._wake[backend] = asyncio.Event()
        self._running[backend] = set()
        self._pumps.append(asyncio.create_task(self._pump(backend), name=f"qputwin-pump-{backend}"))

    async def stop(self) -> None:
        self._stopping = True
        for t in [*self._pumps, *self._tasks]:
            t.cancel()
        for t in [*self._pumps, *self._tasks]:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._pumps.clear()
        with contextlib.suppress(Exception):                # hand unfinished work to the other instances
            released = await self._call(self.store.release, self.owner)
            if released:
                log.info("released %d running job(s) back to the queue on shutdown", released)
        with contextlib.suppress(Exception):
            self.store.close()

    # ----------------------------------------------------------- submission
    async def submit(self, job_id: str, backend: str, payload: bytes, rec: TelemetryRecorder, *,
                     client: str, priority: int) -> tuple[JobRow, int]:
        """Store a validated job; returns the row and its 1-based position in the backend's queue."""
        await self._maybe_prune()
        pre = {k: int(v) for k, v in (("ingress", (rec.handler_entry_ns or rec.t0_ns) - rec.t0_ns),
                                      ("validation", rec.spans_ns.get("validation", 0)),
                                      ("routing", rec.spans_ns.get("routing", 0)))}
        row = JobRow(job_id=job_id, backend=backend, payload=payload.decode("utf-8"), client=client,
                     priority=priority, pre_spans=json.dumps(pre))
        self._local[job_id] = (rec, time.perf_counter_ns())
        try:
            position = await self._call(self.store.insert, row, self.settings.async_max_queued)
        except BaseException:
            self._local.pop(job_id, None)
            raise
        self._events[job_id] = asyncio.Event()
        self._ensure_pump(backend)
        self._wake[backend].set()
        return row, position

    # ------------------------------------------------------------- dispatch
    async def _pump(self, backend: str) -> None:
        wake = self._wake[backend]
        interval = self.settings.job_poll_interval_s
        while not self._stopping:
            try:
                while True:
                    cap = self.registry.capacity(backend) if backend in self.registry.entries else 1
                    if len(self._running[backend]) >= cap:
                        break
                    row = await self._call(self.store.claim, backend, cap, self.owner, self.settings.job_lease_s)
                    if row is None:
                        break
                    self._running[backend].add(row.job_id)
                    task = asyncio.create_task(self._execute(row), name=f"qputwin-job-{row.job_id}")
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
            except Exception:  # pragma: no cover - store hiccup: retry on the next tick
                log.exception("job pump for %s failed; retrying", backend)
            wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wake.wait(), timeout=interval)

    async def _housekeeping(self) -> None:
        s = self.settings
        while not self._stopping:
            await asyncio.sleep(max(0.05, s.job_lease_s / 3))
            try:
                ids = [j for js in self._running.values() for j in js]
                await self._call(self.store.renew, ids, self.owner, s.job_lease_s)
                for row in await self._call(self.store.requeue_expired, s.job_max_attempts):
                    err = QPUTwinError(f"job abandoned: its orchestrator instance stopped renewing the lease "
                                       f"{row.attempts} time(s)", details={"attempts": row.attempts})
                    rec = self._recorder(row)
                    rec.status, rec.http_status, rec.error_type = "FAILED", 500, err.type
                    body = {"job_id": row.job_id, "backend": row.backend, "status": "FAILED",
                            "error": err.to_dict(), "telemetry": rec.finalize()}
                    if await self._call(self.store.finish, row.job_id, None, "FAILED", 500, json.dumps(body)):
                        self._record(rec)
                for w in self._wake.values():
                    w.set()
            except Exception:  # pragma: no cover
                log.exception("job housekeeping failed")

    def _recorder(self, row: JobRow) -> TelemetryRecorder:
        """The submitting recorder if this instance accepted the job, else one rebuilt from stored spans."""
        local = self._local.pop(row.job_id, None)
        if local is not None and row.attempts <= 1:
            rec, enq_ns = local
            rec.spans_ns["queue"] = time.perf_counter_ns() - enq_ns
            rec.queue_clock = "perf_counter"
            return rec
        pre = json.loads(row.pre_spans or "{}")
        queue_ns = max(0, int((time.time() - row.submitted_wall) * 1e9) - sum(pre.values()))
        now = time.perf_counter_ns()
        t0 = now - (sum(pre.values()) + queue_ns)
        rec = TelemetryRecorder(t0)
        rec.handler_entry_ns = t0 + pre.get("ingress", 0)
        rec.spans_ns.update({"validation": pre.get("validation", 0), "routing": pre.get("routing", 0),
                             "queue": queue_ns})
        rec.job_id, rec.backend = row.job_id, row.backend
        rec.execution_mode, rec.deferred_record = "async", True
        rec.queue_clock = "realtime"
        rec.timestamp_utc = row.submitted_at
        return rec

    async def _execute(self, row: JobRow) -> None:
        rec = self._recorder(row)
        rec.attempt = row.attempts
        try:
            status, body = await self._run_job(row.backend, row.payload.encode("utf-8"), rec)
        except Exception as exc:  # pragma: no cover - run_job converts every error into a body
            log.exception("unhandled error in async job %s", row.job_id)
            err = QPUTwinError(f"internal orchestrator error: {type(exc).__name__}")
            rec.status, rec.http_status, rec.error_type = "FAILED", 500, err.type
            status, body = 500, {"job_id": row.job_id, "backend": row.backend, "status": "FAILED",
                                 "error": err.to_dict(), "telemetry": rec.finalize()}
        except asyncio.CancelledError:
            raise                                   # shutdown: stop() releases the lease
        finally:
            self._running.get(row.backend, set()).discard(row.job_id)
        ok = await self._call(self.store.finish, row.job_id, self.owner, str(body.get("status", "FAILED")),
                              status, json.dumps(body, separators=(",", ":")))
        if ok:
            self._record(rec)
        else:
            log.warning("job %s finished after losing its lease; result discarded (re-run elsewhere)", row.job_id)
        ev = self._events.pop(row.job_id, None)
        if ev is not None:
            ev.set()
        self._wake[row.backend].set()

    def _record(self, rec: TelemetryRecorder) -> None:
        record = rec.dataset_record()
        if record is not None:
            self.telemetry.add(record)

    # ---------------------------------------------------------------- queries
    async def get(self, job_id: str, *, client: str | None = None) -> JobRow:
        row = await self._call(self.store.get, job_id)
        if row is None or (client is not None and row.client != client):
            raise JobNotFoundError(f"no asynchronous job {job_id!r} (unknown, or expired after "
                                   f"{self.settings.async_ttl_s:g} s)", details={"job_id": job_id})
        return row

    async def wait(self, job_id: str, timeout: float, *, client: str | None = None) -> JobRow:
        row = await self.get(job_id, client=client)
        deadline = time.monotonic() + timeout
        while not row.finished and time.monotonic() < deadline:
            ev = self._events.get(job_id)
            remaining = deadline - time.monotonic()
            with contextlib.suppress(TimeoutError):
                # same-instance jobs wake immediately; others are polled
                await asyncio.wait_for((ev.wait() if ev is not None else asyncio.sleep(remaining)),
                                       timeout=min(remaining, self.settings.job_poll_interval_s * 2
                                                   if ev is None else remaining))
            row = await self.get(job_id, client=client)
        return row

    async def view(self, row: JobRow) -> dict[str, Any]:
        out = row.summary()
        if row.finished and row.body is not None:
            body = json.loads(row.body)
            return {**body, **{k: out[k] for k in ("execution_mode", "client", "priority", "submitted_at",
                                                    "started_at", "finished_at", "http_status", "attempts",
                                                    "links")}}
        out["queue_position"] = await self._call(self.store.position, row)
        return out

    async def list(self, *, status: str | None = None, backend: str | None = None, client: str | None = None,
                   limit: int = 100) -> list[dict[str, Any]]:
        await self._call(self.store.prune, self.settings.async_ttl_s, self.settings.async_retention)   # admin: exact
        return [r.summary() for r in await self._call(self.store.list, status, backend, client, limit)]

    async def stats(self) -> dict[str, Any]:
        raw = await self._call(self.store.stats)
        backends = sorted(set(raw) | set(self.registry.entries))
        return {b: {**raw.get(b, {"queued": 0, "running": 0}),
                    "capacity": self.registry.capacity(b) if b in self.registry.entries else None}
                for b in backends}

    # --------------------------------------------------------------- control
    async def cancel(self, job_id: str, *, client: str | None = None) -> JobRow:
        row = await self.get(job_id, client=client)
        if row.status != "QUEUED":
            raise JobStateError(f"job {job_id!r} is {row.status}; only QUEUED jobs can be cancelled",
                                details={"job_id": job_id, "status": row.status})
        rec = self._recorder(row)
        rec.status, rec.http_status = "CANCELLED", None
        body = {"job_id": row.job_id, "backend": row.backend, "status": "CANCELLED", "telemetry": rec.finalize()}
        if not await self._call(self.store.finish, job_id, None, "CANCELLED", None, json.dumps(body),
                                from_states=("QUEUED",)):
            current = await self.get(job_id, client=client)
            raise JobStateError(f"job {job_id!r} is {current.status}; only QUEUED jobs can be cancelled",
                                details={"job_id": job_id, "status": current.status})
        self._record(rec)
        ev = self._events.pop(job_id, None)
        if ev is not None:
            ev.set()
        return await self.get(job_id, client=client)
