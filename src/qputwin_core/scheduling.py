"""Gate-duration model: an ASAP schedule of a native circuit → estimated hardware wall-clock time.

Every native operation gets a duration from the device
:class:`~qputwin_core.calibration.Calibration`: virtual gates (``rz`` frame
changes) take zero time, physical gates their calibrated ``duration_ns``
(per-coupler / per-qubit overrides first), measurements the readout time and
resets the reset time. Operations are then placed **as soon as possible**:

* an operation starts when every qubit it touches is free;
* a barrier synchronises its qubits (zero duration);
* a classically-controlled operation additionally waits for the measurements
  that wrote its condition bits, plus ``feedforward_latency_ns`` when known;
* **final measurements** — the last operation on their qubit, whose bits no
  later condition reads — are aligned to start together once every other
  operation has finished, as on hardware that reads out all qubits at the end.
  A qubit that finishes early therefore idles (and, with the noise model,
  relaxes) until read-out;
* **parallelism limits** (``parallelism.max_parallel_one_qubit`` /
  ``max_parallel_two_qubit``) delay an operation until a slot of its kind is
  free, e.g. one entangling gate at a time on a shared motional bus;
* **QCCD transport and cooling** (``qccd.transport_ns_per_two_qubit_gate``,
  ``cooling_ns_per_two_qubit_gate``) precede every two-qubit gate: the ions are
  busy being shuttled and re-cooled, so the gate starts that much later, and the
  noise model treats the window as idle time (relaxation, detuning).

The makespan is the estimated duration of **one shot**. The job estimate is
``shots × (makespan + shot_overhead)``, where the per-shot overhead (cooling,
state preparation, repetition delay) is included only if the calibration
provides it. The flags in the report say exactly what is and is not included.

The same schedule supplies the idle windows used by the noise model's idle
relaxation (:mod:`qputwin_core.noise`).

Assumption (stated in every report): the controller runs a fixed timeline, so a
classically-controlled operation reserves its slot whether or not it fires.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from qputwin_core.calibration import Calibration

if TYPE_CHECKING:  # pragma: no cover
    from qputwin_core.synthesis import NativeOp

__all__ = ["Schedule", "asap_schedule", "duration_report"]


@dataclass(slots=True)
class Schedule:
    starts_ns: list[float]                     # per native op (same index as the op list)
    durations_ns: list[float]
    makespan_ns: float
    qubit_end_ns: dict[int, float]             # when each qubit becomes free for the last time
    busy_ns: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    conditional_ops: int = 0


def _op_duration(op: NativeOp, cal: Calibration, missing: set[str]) -> float:
    if op.name == "measure":
        d = cal.readout_duration_ns
    elif op.name == "reset":
        d = cal.reset_duration_ns
    elif op.virtual:
        return 0.0
    else:
        d = cal.gate_duration_ns(op.name, op.qubits)
    if d is None:
        missing.add(op.name)
        return 0.0
    return d


def _kind(op: NativeOp) -> str:
    if op.name in ("measure", "reset"):
        return op.name
    if op.virtual:
        return "virtual"
    return "two_qubit" if len(op.qubits) == 2 else "one_qubit"


def final_measurements(ops: Sequence[NativeOp]) -> set[int]:
    """Indices of measurements that are the last op on their qubit and whose bits are never read later."""
    seen_q: set[int] = set()
    read_later: set[int] = set()
    final: set[int] = set()
    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]
        if op.name == "measure" and op.condition is None and not (set(op.qubits) & seen_q) \
                and not (set(op.clbits) & read_later):
            final.add(i)
        if op.name != "barrier":
            seen_q.update(op.qubits)
        if op.condition is not None:
            read_later.update(op.condition.clbits)
    return final


# Booked intervals are pruned once a list holds more than this many (see asap_schedule).
_PRUNE_MIN = 64


def _earliest_slot(intervals: list[tuple[float, float]], t: float, d: float, limit: int) -> float:
    """Earliest start ≥ ``t`` at which fewer than ``limit`` intervals overlap ``[start, start + d)``."""
    if d <= 0:
        return t
    for c in sorted({t, *(e for _, e in intervals if e > t)}):
        if sum(1 for s, e in intervals if s < c + d and e > c) < limit:
            return c
    return max(e for _, e in intervals)  # pragma: no cover - the last end always frees a slot


def asap_schedule(ops: Sequence[NativeOp], cal: Calibration, *, align_final_measurements: bool = True) -> Schedule:
    final = final_measurements(ops) if align_final_measurements else set()
    limits = {"one_qubit": cal.max_parallel_one_qubit, "two_qubit": cal.max_parallel_two_qubit}
    occupied: dict[str, list[tuple[float, float]]] = {"one_qubit": [], "two_qubit": []}
    prune_at = {"one_qubit": _PRUNE_MIN, "two_qubit": _PRUNE_MIN}
    all_qubits = {q for op in ops for q in op.qubits}
    pre_2q = (cal.transport_ns_per_two_qubit_gate or 0.0) + (cal.cooling_ns_per_two_qubit_gate or 0.0)
    deferred: list[int] = []
    clock: dict[int, float] = {}
    clbit_ready: dict[int, float] = {}
    ff = cal.feedforward_latency_ns or 0.0
    missing: set[str] = set()
    starts: list[float] = []
    durs: list[float] = []
    busy: dict[str, float] = {"one_qubit": 0.0, "two_qubit": 0.0, "measure": 0.0, "reset": 0.0, "transport": 0.0}
    conditional = 0
    for i, op in enumerate(ops):
        if i in final:
            deferred.append(i)
            starts.append(0.0)
            durs.append(0.0)
            continue
        if op.name == "barrier":
            t = max((clock.get(q, 0.0) for q in op.qubits), default=0.0)
            for q in op.qubits:
                clock[q] = t
            starts.append(t)
            durs.append(0.0)
            continue
        d = _op_duration(op, cal, missing)
        t = max((clock.get(q, 0.0) for q in op.qubits), default=0.0)
        if op.condition is not None:
            conditional += 1
            t = max(t, max((clbit_ready.get(c, 0.0) for c in op.condition.clbits), default=0.0) + ff)
        k = _kind(op)
        if k == "two_qubit" and pre_2q > 0:
            t += pre_2q                                   # shuttle + re-cool, then the gate
            busy["transport"] += pre_2q
        lim = limits.get(k)
        if lim is not None:
            occ = occupied[k]
            if len(occ) > prune_at[k]:
                # No later op starts before the earliest qubit clock, so intervals that end by then can
                # neither overlap nor bound a future slot: dropping them keeps the schedule identical and
                # turns the O(N²) scan over every booked interval into an O(N) one.
                floor = min((clock.get(q, 0.0) for q in all_qubits), default=0.0)
                occ[:] = [iv for iv in occ if iv[1] > floor]
                prune_at[k] = max(_PRUNE_MIN, 2 * len(occ))
            t = _earliest_slot(occ, t, d, lim)
            occ.append((t, t + d))
        end = t + d
        for q in op.qubits:
            clock[q] = end
        if op.name == "measure":
            for c in op.clbits:
                clbit_ready[c] = end
        if k in busy:
            busy[k] += d
        starts.append(t)
        durs.append(d)
    if deferred:
        t_read = max(clock.values(), default=0.0)
        for i in deferred:
            d = _op_duration(ops[i], cal, missing)
            starts[i], durs[i] = t_read, d
            busy["measure"] += d
            for q in ops[i].qubits:
                clock[q] = t_read + d
    makespan = max(clock.values(), default=0.0)
    return Schedule(starts, durs, makespan, dict(clock), busy, sorted(missing), conditional)


def duration_report(sched: Schedule, cal: Calibration, shots: int) -> dict[str, Any]:
    """The ``hardware_time_estimate`` block reported with every compiled job."""
    overhead_us = cal.shot_overhead_us
    per_shot_ns = sched.makespan_ns + (overhead_us * 1e3 if overhead_us is not None else 0.0)
    return {
        "model": "asap_list_schedule",
        "final_measurements": "aligned",
        "circuit_duration_us": sched.makespan_ns / 1e3,
        "shot_overhead_us": overhead_us,
        "estimated_job_duration_s": shots * per_shot_ns / 1e9,
        "includes_shot_overhead": overhead_us is not None,
        "includes_feedforward_latency": cal.feedforward_latency_ns is not None,
        "includes_qccd_transport": bool(cal.transport_ns_per_two_qubit_gate or cal.cooling_ns_per_two_qubit_gate),
        "parallelism": {"max_parallel_one_qubit": cal.max_parallel_one_qubit,
                        "max_parallel_two_qubit": cal.max_parallel_two_qubit},
        "conditional_timing": "fixed_timeline",
        "busy_time_us": {k: v / 1e3 for k, v in sched.busy_ns.items()},
        "conditional_ops_scheduled_as_executed": sched.conditional_ops,
        "missing_durations": sched.missing,
        "complete": not sched.missing and math.isfinite(sched.makespan_ns),
        "calibration": cal.label or cal.origin,
    }
