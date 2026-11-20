"""Tests the durable job store: restarts, leases, priorities and fair share.

Usage: uv run pytest tests/test_job_store.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from qputwin_orchestrator.jobs import JobRow, MemoryJobStore, SqliteJobStore
from qputwin_orchestrator.registry import parse_api_keys

from test_async_jobs import SLOW_S, _identities, _make_client, _wait_for_status
from test_simulators import star_ghz

JOB = {"backend": "slow", "qasm": star_ghz(2), "shots": 5}
ALICE, BOB = "alice-key-0123456789", "bob-key-0123456789ab"


def _submit(c, body=None, **headers) -> dict:
    r = c.post("/v1/jobs", params={"mode": "async"}, json=body or JOB, headers=headers)
    assert r.status_code == 202, r.text
    return r.json()


def _done(c, job_id: str, **headers) -> dict:
    body = c.get(f"/v1/jobs/{job_id}", params={"wait": 10}, headers=headers).json()
    assert body["status"] not in ("QUEUED", "RUNNING"), body
    return body


# ===================================================================== stores
@pytest.mark.parametrize("make", [MemoryJobStore, lambda: SqliteJobStore(Path(pytest.tmp) / "s.db")])
def test_store_contract(make, tmp_path: Path) -> None:
    pytest.tmp = str(tmp_path)                                     # type: ignore[attr-defined]
    st = make()
    rows = [JobRow(job_id=f"j{k}", backend="b", payload="{}", client=c, priority=p)
            for k, (c, p) in enumerate([("a", 5), ("a", 5), ("b", 5), ("a", 9)])]
    for r in rows:
        st.insert(r, max_queued=10)
    assert st.position(st.get("j3")) == 1 and st.position(st.get("j0")) == 2
    first = st.claim("b", 2, "me", 30)
    assert first.job_id == "j3"                                    # priority first
    second = st.claim("b", 2, "me", 30)
    assert second.job_id == "j2"                                   # fair share: client b has nothing running
    assert st.claim("b", 2, "me", 30) is None                      # global capacity reached
    assert st.stats() == {"b": {"queued": 2, "running": 2}}
    assert not st.finish("j3", "someone-else", "COMPLETED", 200, "{}")   # stale owner ignored
    assert st.finish("j3", "me", "COMPLETED", 200, '{"ok": 1}') and st.get("j3").finished
    assert st.release("me") == 1 and st.get("j2").status == "QUEUED" and st.get("j2").attempts == 0
    with pytest.raises(Exception, match="full"):
        st.insert(JobRow(job_id="x", backend="b", payload="{}"), max_queued=3)
    assert st.prune(ttl_s=0.0, keep=100) == 1 and st.get("j3") is None
    st.ping()
    st.close()


def test_api_key_parsing() -> None:
    keys = parse_api_keys(f"alice:{ALICE}:9, bob:{BOB}:3 # comment")
    assert keys == {ALICE: ("alice", 9), BOB: ("bob", 3)}
    for bad in ("alice:short", "noseparator", f"x:{ALICE}:12"):
        with pytest.raises(ValueError):
            parse_api_keys(bad)


# ========================================================= durability / HA
def test_jobs_survive_an_orchestrator_restart(tmp_path: Path) -> None:
    db = str(tmp_path / "jobs.db")
    with _make_client(job_store="sqlite", job_store_path=db, job_poll_interval_s=0.02) as a:
        running = _submit(a)["job_id"]
        _wait_for_status(a, running, "RUNNING")
        queued = _submit(a)["job_id"]
    # instance A stopped mid-execution: its running job was handed back to the queue
    with _make_client(job_store="sqlite", job_store_path=db, job_poll_interval_s=0.02) as b:
        for jid in (running, queued):
            body = _done(b, jid)
            counts = body["result"]["counts"]               # 5 shots: both outcomes need not appear
            assert body["status"] == "COMPLETED" and set(counts) <= {"00", "11"} and sum(counts.values()) == 5
            t = body["telemetry"]
            assert t["queue_clock"] == "realtime"                  # executed by another instance
            _identities(t)
            assert t["breakdown"]["validation_ms"] > 0             # carried over from the submitting instance


def test_two_instances_share_one_queue(tmp_path: Path) -> None:
    db = str(tmp_path / "jobs.db")
    kw = {"job_store": "sqlite", "job_store_path": db, "job_poll_interval_s": 0.02}
    with _make_client(**kw) as a, _make_client(**kw) as b:
        ids = [_submit(a if k % 2 else b)["job_id"] for k in range(6)]
        bodies = [_done(b if k % 2 else a, j) for k, j in enumerate(ids)]     # poll the *other* instance
        assert all(x["status"] == "COMPLETED" for x in bodies)
        # global capacity (1 slot for 'slow') held across both instances: executions never overlapped
        spans = sorted((x["started_at"], x["finished_at"]) for x in bodies)
        assert all(spans[k][1] <= spans[k + 1][0] for k in range(len(spans) - 1)), spans
        assert {x["telemetry"]["queue_clock"] for x in bodies} <= {"perf_counter", "realtime"}
        assert len({j["job_id"] for j in a.get("/v1/jobs").json()["jobs"]}) == 6


def _payload(job_id: str) -> str:
    return json.dumps({"qasm": star_ghz(2), "shots": 10, "seed": 3, "options": {}, "job_id": job_id})


def test_lease_expiry_recovery_and_abandonment(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    st = SqliteJobStore(db)
    stale = time.time() - 60
    for jid, attempts in (("lost", 1), ("doomed", 3)):
        st.insert(JobRow(job_id=jid, backend="ionq_forte", payload=_payload(jid), pre_spans="{}"), 10)
        with st._tx() as conn:
            conn.execute("UPDATE jobs SET status='RUNNING', owner='dead-instance', lease_until=?, attempts=? "
                         "WHERE job_id=?", (stale, attempts, jid))
    st.close()
    with _make_client(job_store="sqlite", job_store_path=str(db), job_lease_s=0.3, job_max_attempts=3,
                      job_poll_interval_s=0.02) as c:
        lost = _done(c, "lost")
        assert lost["status"] == "COMPLETED" and lost["attempts"] == 2
        assert lost["telemetry"]["attempt"] == 2 and lost["telemetry"]["queue_clock"] == "realtime"
        assert lost["result"]["seed"] == 3                         # same seed → identical re-execution
        doomed = _done(c, "doomed")
        assert doomed["status"] == "FAILED" and "abandoned" in doomed["error"]["message"]


# ==================================================== priority & fair share
def test_priority_jumps_the_queue() -> None:
    with _make_client() as c:
        blocker = _submit(c)["job_id"]
        _wait_for_status(c, blocker, "RUNNING")
        low = _submit(c, {**JOB, "priority": 1})["job_id"]
        high = _submit(c, {**JOB, "priority": 9})["job_id"]
        assert c.get(f"/v1/jobs/{high}").json()["queue_position"] == 1
        lo, hi = _done(c, low), _done(c, high)
        assert hi["started_at"] < lo["started_at"] and hi["priority"] == 9


def test_fair_share_between_clients() -> None:
    with _make_client() as c:
        job = {**JOB, "backend": "slowpair"}                         # capacity 2
        blockers = [_submit(c, job, **{"x-qputwin-client": "carol"})["job_id"] for _ in range(2)]
        for b in blockers:
            _wait_for_status(c, b, "RUNNING")
        a1, a2 = (_submit(c, job, **{"x-qputwin-client": "alice"})["job_id"] for _ in range(2))
        b1 = _submit(c, job, **{"x-qputwin-client": "bob"})["job_id"]
        rows = {j: _done(c, j) for j in (a1, a2, b1)}
        assert rows[b1]["started_at"] < rows[a2]["started_at"]        # bob is not starved behind alice
        assert rows[a1]["client"] == "alice" and rows[b1]["client"] == "bob"


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_fair_share_round_robin_with_one_slot(kind: str, tmp_path: Path) -> None:
    """With one dispatch slot nobody is running at claim time; least-recently-served breaks the tie."""
    st = MemoryJobStore() if kind == "memory" else SqliteJobStore(tmp_path / "rr.db")
    for k in range(6):
        st.insert(JobRow(job_id=f"h{k}", backend="b", payload="{}", client="heavy"), max_queued=100)
    for k in range(2):
        st.insert(JobRow(job_id=f"l{k}", backend="b", payload="{}", client="light"), max_queued=100)
    order = []
    while (row := st.claim("b", 1, "me", 30)) is not None:
        order.append(row.job_id)
        assert st.finish(row.job_id, "me", "COMPLETED", 200, "{}")
        time.sleep(0.002)                                          # distinct millisecond timestamps
    assert order == ["h0", "l0", "h1", "l1", "h2", "h3", "h4", "h5"]



@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_retention_keeps_the_newest_finished_jobs(kind: str, tmp_path: Path) -> None:
    st = MemoryJobStore() if kind == "memory" else SqliteJobStore(tmp_path / "keep.db")
    for k in range(5):
        st.insert(JobRow(job_id=f"f{k}", backend="b", payload="{}"), max_queued=100)
        row = st.claim("b", 1, "me", 30)
        assert st.finish(row.job_id, "me", "COMPLETED", 200, "{}")
    st.insert(JobRow(job_id="waiting", backend="b", payload="{}"), max_queued=100)
    assert st.prune(ttl_s=3600.0, keep=2) == 3
    assert [st.get(f"f{k}") is not None for k in range(5)] == [False, False, False, True, True]
    assert st.get("waiting").status == "QUEUED"                    # unfinished jobs are never pruned


def test_submissions_do_not_prune_every_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retention scans the store, so the submission hot path runs it at most every PRUNE_INTERVAL_S."""
    with _make_client() as c:
        store = c.app.state.jobs.store
        calls = []
        original = store.prune
        monkeypatch.setattr(store, "prune", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
        for k in range(20):
            assert c.post("/v1/jobs", params={"mode": "async"},
                          json={"backend": "ionq_forte", "qasm": star_ghz(2), "shots": 5, "seed": k}).status_code == 202
        assert len(calls) <= 1

# ============================================================== auth
def test_api_keys_isolate_clients() -> None:
    keys = parse_api_keys(f"alice:{ALICE}:9,bob:{BOB}:3")
    with _make_client(api_keys=keys) as c:
        r = c.post("/v1/jobs", json={"backend": "ionq_forte", "qasm": star_ghz(2), "shots": 5})
        assert r.status_code == 401 and r.json()["error"]["type"] == "AuthenticationError"
        assert r.headers["www-authenticate"] == "Bearer" and "telemetry" in r.json()
        assert c.get("/v1/backends").status_code == 401
        assert c.get("/health").status_code == 200 and c.get("/metrics").status_code == 200
        auth_a = {"authorization": f"Bearer {ALICE}"}
        assert c.post("/v1/jobs", json={"backend": "ionq_forte", "qasm": star_ghz(2), "shots": 5},
                      headers=auth_a).status_code == 200
        acc = _submit(c, {**JOB, "priority": 9}, **{"x-api-key": BOB})
        assert acc["client"] == "bob" and acc["priority"] == 3            # capped by bob's key
        assert c.get(f"/v1/jobs/{acc['job_id']}", headers=auth_a).status_code == 404    # not alice's job
        assert c.delete(f"/v1/jobs/{acc['job_id']}", headers=auth_a).status_code == 404
        assert _done(c, acc["job_id"], **{"x-api-key": BOB})["status"] == "COMPLETED"
        assert c.get("/v1/jobs", headers=auth_a).json()["jobs"] == []


# ======================================================= metrics / readiness
def test_metrics_and_readiness() -> None:
    with _make_client() as c:
        c.post("/v1/jobs", json={"backend": "ionq_forte", "qasm": star_ghz(3), "shots": 10})
        c.post("/v1/jobs", json={"backend": "rigetti_ankaa_3", "qasm": star_ghz(12), "shots": 10})
        _done(c, _submit(c)["job_id"])
        text = c.get("/metrics").text
        assert 'qputwin_jobs_total{backend="ionq_forte",status="COMPLETED",mode="sync"} 1' in text
        assert 'qputwin_jobs_total{backend="rigetti_ankaa_3",status="REJECTED",mode="sync"} 1' in text
        assert 'qputwin_jobs_total{backend="slow",status="COMPLETED",mode="async"} 1' in text
        assert 'qputwin_job_roundtrip_ms_bucket{backend="ionq_forte",le="+Inf"} 1' in text
        assert 'qputwin_async_capacity{backend="slowpair"} 2' in text
        assert "# TYPE qputwin_job_qpu_compute_ms histogram" in text
        ready = c.get("/ready")
        assert ready.status_code == 200 and ready.json()["job_store"] == "ok"
        assert math.isfinite(SLOW_S)
