"""Device calibration data: gate fidelities, gate durations, coherence times, readout errors.

A :class:`Calibration` is the single source of numbers for both the **noise
model** (:mod:`qputwin_core.noise`) and the **gate-duration model**
(:mod:`qputwin_core.scheduling`). Each device ships a packaged file
``devices_data/calibration/<device>.json`` with *nominal medians*; a worker can
load a different snapshot at runtime from ``QPUTWIN_CALIBRATION`` (a vendor's
published calibration, a historical snapshot, a what-if study…), with no code
change.

File format (all keys except ``device`` optional)::

    {
      "device": "iqm_emerald",
      "label": "nominal medians",
      "snapshot_date": "2026-08-01",
      "fidelity_convention": "average_gate_fidelity",
      "gates":    {"prx": {"fidelity": 0.9993, "duration_ns": 40}, "cz": {...}},
      "readout":  {"fidelity": 0.97, "duration_ns": 1500},          # or p01 / p10
      "reset":    {"duration_ns": 2000},
      "coherence": {"t1_us": 54.7, "t2_us": 16.6},
      "feedforward_latency_ns": null,
      "shot_overhead_us": null,
      "crosstalk": {"zz_rad_per_two_qubit_gate": 0.002, "pairs": {"3-4": 0.004}},
      "leakage":  {"seepage_per_gate": 0.01},           # gates.<g>.leakage = leak probability
      "readout_correlations": [{"qubits": [0, 1], "matrix": [[…4×4 column-stochastic…]]}],
      "parallelism": {"max_parallel_one_qubit": null, "max_parallel_two_qubit": 1},
      "qccd": {"transport_ns_per_two_qubit_gate": 0, "cooling_ns_per_two_qubit_gate": 0},
      "qubits":   {"7": {"t1_us": 31.0, "readout_fidelity": 0.93,
                         "gates": {"prx": {"fidelity": 0.998}}}},
      "couplers": {"6-7": {"cz": {"fidelity": 0.981, "duration_ns": 70}}},
      "provenance": {"gates.cz.fidelity": "published: …", "coherence.t1_us": "assumed: …"},
      "sources": ["https://…"]
    }

Per-qubit and per-coupler entries override the device-wide defaults. Readout
errors are ``p01 = P(read 1 | prepared 0)`` and ``p10 = P(read 0 | prepared 1)``;
a symmetric ``fidelity`` F gives ``p01 = p10 = 1 − F``.

Gate fidelities follow ``fidelity_convention``: ``average_gate_fidelity`` (what
randomized / interleaved benchmarking report; the default) or
``process_fidelity`` (entanglement fidelity ``F_e``), converted exactly with
``F_avg = (d·F_e + 1)/(d + 1)``. :meth:`Calibration.gate_fidelity` always
returns the *average* gate fidelity.

Optional per-gate error terms (in ``gates.<g>``, ``qubits.<q>.gates.<g>`` or
``couplers.<a-b>.<g>``): ``overrotation`` (fractional coherent over-rotation ε,
the gate becomes ``U^{1+ε}``), ``coherent_z_rad`` (a residual Z / ZZ phase after
the gate) and ``leakage`` (probability of leaving the computational subspace per
gate and qubit). Device-level blocks add spectator ZZ ``crosstalk``,
``leakage.seepage_per_gate``, quasi-static detuning
(``coherence.quasi_static_detuning_khz``), ``reset.fidelity``, joint
``readout_correlations``, gate ``parallelism`` limits and ``qccd`` transport /
cooling times. Everything absent is simply not modelled.

**Research honesty.** Every number carries a provenance tag. Values marked
``assumed`` are placeholders typical of the technology class, *not* vendor data,
and must be replaced before being quoted in a publication.

Pure Python (no numpy), so the orchestrator can parse calibration files too.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = ["GateCalibration", "QubitCalibration", "ReadoutCorrelation", "Calibration", "load_calibration",
           "load_calibration_file"]

_CONVENTIONS = ("average_gate_fidelity", "process_fidelity")


def _opt_float(d: dict[str, Any], key: str, where: str, lo: float | None = None, hi: float | None = None,
               *, lo_open: bool = False) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    x = float(v)
    if (lo is not None and (x <= lo if lo_open else x < lo)) or (hi is not None and x > hi):
        raise ValueError(f"{where}.{key} = {v!r} is out of range")
    return x


@dataclass(frozen=True, slots=True)
class GateCalibration:
    fidelity: float | None = None
    duration_ns: float | None = None
    overrotation: float | None = None          # fractional coherent over-rotation ε: U → U^{1+ε}
    coherent_z_rad: float | None = None        # residual Z (1Q) / ZZ (2Q) phase after the gate
    leakage: float | None = None               # leak probability per gate and qubit

    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str) -> GateCalibration:
        return cls(_opt_float(d, "fidelity", where, 0.0, 1.0, lo_open=True),
                   _opt_float(d, "duration_ns", where, 0.0),
                   _opt_float(d, "overrotation", where, -0.5, 0.5),
                   _opt_float(d, "coherent_z_rad", where, -math.pi, math.pi),
                   _opt_float(d, "leakage", where, 0.0, 0.5))

    def merged(self, base: GateCalibration | None) -> GateCalibration:
        if base is None:
            return self
        return GateCalibration(*(a if a is not None else b for a, b in zip(
            (self.fidelity, self.duration_ns, self.overrotation, self.coherent_z_rad, self.leakage),
            (base.fidelity, base.duration_ns, base.overrotation, base.coherent_z_rad, base.leakage), strict=True)))

    def to_dict(self) -> dict[str, Any]:
        out = {"fidelity": self.fidelity, "duration_ns": self.duration_ns}
        for k in ("overrotation", "coherent_z_rad", "leakage"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        return out


@dataclass(frozen=True, slots=True)
class ReadoutCorrelation:
    """Joint assignment matrix ``M[recorded, true]`` for a group of qubits (bit k ↔ ``qubits[k]``)."""

    qubits: tuple[int, ...]
    matrix: tuple[tuple[float, ...], ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str) -> ReadoutCorrelation:
        qs = tuple(int(q) for q in d["qubits"])
        m = tuple(tuple(float(x) for x in row) for row in d["matrix"])
        n = 1 << len(qs)
        if len(set(qs)) != len(qs) or len(qs) < 2:
            raise ValueError(f"{where}: needs at least two distinct qubits")
        if len(m) != n or any(len(r) != n for r in m):
            raise ValueError(f"{where}: matrix must be {n}×{n} for {len(qs)} qubits")
        for c in range(n):
            col = [m[r][c] for r in range(n)]
            if any(x < 0 for x in col) or abs(sum(col) - 1.0) > 1e-9:
                raise ValueError(f"{where}: column {c} is not a probability distribution")
        return cls(qs, m)


@dataclass(frozen=True, slots=True)
class QubitCalibration:
    t1_us: float | None = None
    t2_us: float | None = None
    readout_p01: float | None = None
    readout_p10: float | None = None
    gates: dict[str, GateCalibration] = field(default_factory=dict)
    quasi_static_detuning_khz: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"t1_us": self.t1_us, "t2_us": self.t2_us, "readout_p01": self.readout_p01,
                "readout_p10": self.readout_p10, "quasi_static_detuning_khz": self.quasi_static_detuning_khz,
                "gates": {k: v.to_dict() for k, v in self.gates.items()}}


def _readout_pair(d: dict[str, Any], where: str, prefix: str = "") -> tuple[float | None, float | None]:
    f = d.get(f"{prefix}fidelity")
    p01 = d.get(f"{prefix}p01")
    p10 = d.get(f"{prefix}p10")
    if f is not None:
        if not 0.0 < float(f) <= 1.0:
            raise ValueError(f"{where}: readout fidelity must be in (0, 1], got {f!r}")
        p01 = 1.0 - float(f) if p01 is None else p01
        p10 = 1.0 - float(f) if p10 is None else p10
    for name, p in (("p01", p01), ("p10", p10)):
        if p is not None and not 0.0 <= float(p) <= 0.5:
            raise ValueError(f"{where}: readout {name} must be in [0, 0.5], got {p!r}")
    return (None if p01 is None else float(p01)), (None if p10 is None else float(p10))


def _coherence(t1: Any, t2: Any, where: str) -> tuple[float | None, float | None]:
    t1 = None if t1 is None else float(t1)
    t2 = None if t2 is None else float(t2)
    if t1 is not None and t1 <= 0:
        raise ValueError(f"{where}: t1_us must be > 0")
    if t2 is not None and t2 <= 0:
        raise ValueError(f"{where}: t2_us must be > 0")
    if t1 is not None and t2 is not None and t2 > 2 * t1 * (1 + 1e-9):
        raise ValueError(f"{where}: unphysical coherence, T2 = {t2} us > 2·T1 = {2 * t1} us")
    return t1, t2


def _edge_key(a: int, b: int) -> str:
    return f"{min(a, b)}-{max(a, b)}"


@dataclass(frozen=True, slots=True)
class Calibration:
    device: str
    label: str = ""
    snapshot_date: str | None = None
    fidelity_convention: str = "average_gate_fidelity"
    gates: dict[str, GateCalibration] = field(default_factory=dict)
    readout_p01: float | None = None
    readout_p10: float | None = None
    readout_duration_ns: float | None = None
    reset_duration_ns: float | None = None
    t1_us: float | None = None
    t2_us: float | None = None
    feedforward_latency_ns: float | None = None
    shot_overhead_us: float | None = None
    reset_fidelity: float | None = None
    quasi_static_detuning_khz: float | None = None
    crosstalk_zz_rad: float | None = None                       # spectator ZZ per 2Q gate (default pair value)
    crosstalk_pairs: dict[str, float] = field(default_factory=dict)
    seepage_per_gate: float | None = None
    readout_correlations: tuple[ReadoutCorrelation, ...] = ()
    max_parallel_one_qubit: int | None = None
    max_parallel_two_qubit: int | None = None
    transport_ns_per_two_qubit_gate: float | None = None
    cooling_ns_per_two_qubit_gate: float | None = None
    qubits: dict[int, QubitCalibration] = field(default_factory=dict)
    couplers: dict[str, dict[str, GateCalibration]] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    origin: str = "packaged"

    # ------------------------------------------------------------- parsing
    @classmethod
    def from_dict(cls, d: dict[str, Any], *, origin: str = "packaged") -> Calibration:
        if "device" not in d:
            raise ValueError("calibration file must name its 'device'")
        convention = d.get("fidelity_convention", "average_gate_fidelity")
        if convention not in _CONVENTIONS:
            raise ValueError(f"fidelity_convention must be one of {_CONVENTIONS}, got {convention!r}")
        gates = {g: GateCalibration.from_dict(v, f"gates.{g}") for g, v in (d.get("gates") or {}).items()}
        ro = d.get("readout") or {}
        p01, p10 = _readout_pair(ro, "readout")
        coh = d.get("coherence") or {}
        t1, t2 = _coherence(coh.get("t1_us"), coh.get("t2_us"), "coherence")
        qubits: dict[int, QubitCalibration] = {}
        for k, v in (d.get("qubits") or {}).items():
            where = f"qubits.{k}"
            qt1, qt2 = _coherence(v.get("t1_us"), v.get("t2_us"), where)
            q01, q10 = _readout_pair(v, where, prefix="readout_")
            qubits[int(k)] = QubitCalibration(
                qt1, qt2, q01, q10,
                {g: GateCalibration.from_dict(gv, f"{where}.gates.{g}") for g, gv in (v.get("gates") or {}).items()},
                _opt_float(v, "quasi_static_detuning_khz", where, 0.0))
        couplers: dict[str, dict[str, GateCalibration]] = {}
        for k, v in (d.get("couplers") or {}).items():
            a, b = (int(x) for x in str(k).split("-"))
            couplers[_edge_key(a, b)] = {g: GateCalibration.from_dict(gv, f"couplers.{k}.{g}") for g, gv in v.items()}

        def opt(x: Any) -> float | None:
            return None if x is None else float(x)

        xt = d.get("crosstalk") or {}
        pairs: dict[str, float] = {}
        for k, v in (xt.get("pairs") or {}).items():
            a, b = (int(x) for x in str(k).split("-"))
            pairs[_edge_key(a, b)] = float(v)
        corr = tuple(ReadoutCorrelation.from_dict(c, f"readout_correlations[{i}]")
                     for i, c in enumerate(d.get("readout_correlations") or ()))
        seen: set[int] = set()
        for c in corr:
            if seen & set(c.qubits):
                raise ValueError("readout_correlations groups must be disjoint")
            seen |= set(c.qubits)
        par = d.get("parallelism") or {}
        qccd = d.get("qccd") or {}
        for k in ("max_parallel_one_qubit", "max_parallel_two_qubit"):
            if par.get(k) is not None and int(par[k]) < 1:
                raise ValueError(f"parallelism.{k} must be >= 1")

        return cls(
            device=str(d["device"]), label=str(d.get("label", "")), snapshot_date=d.get("snapshot_date"),
            gates=gates, readout_p01=p01, readout_p10=p10, readout_duration_ns=opt(ro.get("duration_ns")),
            reset_duration_ns=opt((d.get("reset") or {}).get("duration_ns")), t1_us=t1, t2_us=t2,
            feedforward_latency_ns=opt(d.get("feedforward_latency_ns")),
            shot_overhead_us=opt(d.get("shot_overhead_us")),
            reset_fidelity=_opt_float(d.get("reset") or {}, "fidelity", "reset", 0.0, 1.0, lo_open=True),
            quasi_static_detuning_khz=_opt_float(coh, "quasi_static_detuning_khz", "coherence", 0.0),
            crosstalk_zz_rad=_opt_float(xt, "zz_rad_per_two_qubit_gate", "crosstalk", -math.pi, math.pi),
            crosstalk_pairs=pairs,
            seepage_per_gate=_opt_float(d.get("leakage") or {}, "seepage_per_gate", "leakage", 0.0, 1.0),
            readout_correlations=corr,
            max_parallel_one_qubit=None if par.get("max_parallel_one_qubit") is None else int(par["max_parallel_one_qubit"]),
            max_parallel_two_qubit=None if par.get("max_parallel_two_qubit") is None else int(par["max_parallel_two_qubit"]),
            transport_ns_per_two_qubit_gate=_opt_float(qccd, "transport_ns_per_two_qubit_gate", "qccd", 0.0),
            cooling_ns_per_two_qubit_gate=_opt_float(qccd, "cooling_ns_per_two_qubit_gate", "qccd", 0.0),
            fidelity_convention=convention, qubits=qubits, couplers=couplers,
            provenance={str(k): str(v) for k, v in (d.get("provenance") or {}).items()},
            sources=tuple(str(s) for s in d.get("sources") or ()), origin=origin,
        )

    # ------------------------------------------------------------- queries
    def gate(self, name: str, qubits: tuple[int, ...] = ()) -> GateCalibration | None:
        """Calibration of ``name`` on ``qubits``: coupler / qubit override, then device default."""
        base = self.gates.get(name)
        if len(qubits) == 2:
            over = self.couplers.get(_edge_key(*qubits), {}).get(name)
            if over is not None:
                return over.merged(base)
        elif len(qubits) == 1:
            qc = self.qubits.get(qubits[0])
            if qc is not None and name in qc.gates:
                return qc.gates[name].merged(base)
        return base

    def gate_fidelity(self, name: str, qubits: tuple[int, ...] = ()) -> float | None:
        """*Average* gate fidelity of ``name`` on ``qubits`` (process fidelities are converted)."""
        g = self.gate(name, qubits)
        if g is None or g.fidelity is None:
            return None
        if self.fidelity_convention == "process_fidelity":
            d = 1 << max(1, len(qubits))
            return (d * g.fidelity + 1.0) / (d + 1.0)
        return g.fidelity

    def gate_duration_ns(self, name: str, qubits: tuple[int, ...] = ()) -> float | None:
        g = self.gate(name, qubits)
        return None if g is None else g.duration_ns

    def t1(self, q: int) -> float | None:
        qc = self.qubits.get(q)
        return qc.t1_us if qc is not None and qc.t1_us is not None else self.t1_us

    def t2(self, q: int) -> float | None:
        qc = self.qubits.get(q)
        return qc.t2_us if qc is not None and qc.t2_us is not None else self.t2_us

    def readout_errors(self, q: int) -> tuple[float, float]:
        """``(p01, p10)`` for qubit ``q`` (0 when the calibration has no readout data)."""
        qc = self.qubits.get(q)
        p01 = qc.readout_p01 if qc is not None and qc.readout_p01 is not None else self.readout_p01
        p10 = qc.readout_p10 if qc is not None and qc.readout_p10 is not None else self.readout_p10
        return (p01 or 0.0), (p10 or 0.0)

    def quasi_static_khz(self, q: int) -> float | None:
        qc = self.qubits.get(q)
        if qc is not None and qc.quasi_static_detuning_khz is not None:
            return qc.quasi_static_detuning_khz
        return self.quasi_static_detuning_khz

    def crosstalk_rad(self, a: int, b: int) -> float | None:
        """Spectator ZZ angle between ``a`` (gate qubit) and ``b`` (spectator) per 2Q gate."""
        return self.crosstalk_pairs.get(_edge_key(a, b), self.crosstalk_zz_rad)

    # ------------------------------------------------------------ provenance
    def provenance_of(self, path: str) -> str | None:
        """Provenance tag for a dotted field path (most specific prefix wins; ``*`` = whole file)."""
        parts = path.split(".")
        for k in range(len(parts), 0, -1):
            tag = self.provenance.get(".".join(parts[:k]))
            if tag is not None:
                return tag
        return self.provenance.get("*")

    def provenance_class(self, path: str) -> str:
        """``published`` | ``assumed`` | ``untagged`` for a field path."""
        tag = (self.provenance_of(path) or "").strip().lower()
        if tag.startswith("published"):
            return "published"
        if not tag:
            return "untagged"
        return "assumed"

    def missing_durations(self, gate_names: set[str]) -> list[str]:
        return sorted(g for g in gate_names if self.gate_duration_ns(g) is None)

    def to_dict(self) -> dict[str, Any]:
        def num(x: float | None) -> float | None:
            return None if x is None or not math.isfinite(x) else x

        return {
            "device": self.device, "label": self.label, "snapshot_date": self.snapshot_date,
            "origin": self.origin, "fidelity_convention": self.fidelity_convention,
            "gates": {k: v.to_dict() for k, v in sorted(self.gates.items())},
            "readout": {"p01": num(self.readout_p01), "p10": num(self.readout_p10),
                        "duration_ns": num(self.readout_duration_ns)},
            "reset": {"duration_ns": num(self.reset_duration_ns)},
            "coherence": {"t1_us": num(self.t1_us), "t2_us": num(self.t2_us)},
            "feedforward_latency_ns": num(self.feedforward_latency_ns),
            "shot_overhead_us": num(self.shot_overhead_us),
            "qubit_overrides": len(self.qubits), "coupler_overrides": len(self.couplers),
            "reset_fidelity": num(self.reset_fidelity),
            "quasi_static_detuning_khz": num(self.quasi_static_detuning_khz),
            "crosstalk": {"zz_rad_per_two_qubit_gate": num(self.crosstalk_zz_rad), "pairs": dict(self.crosstalk_pairs)},
            "leakage": {"seepage_per_gate": num(self.seepage_per_gate)},
            "readout_correlations": [{"qubits": list(c.qubits), "matrix": [list(r) for r in c.matrix]}
                                     for c in self.readout_correlations],
            "parallelism": {"max_parallel_one_qubit": self.max_parallel_one_qubit,
                            "max_parallel_two_qubit": self.max_parallel_two_qubit},
            "qccd": {"transport_ns_per_two_qubit_gate": num(self.transport_ns_per_two_qubit_gate),
                     "cooling_ns_per_two_qubit_gate": num(self.cooling_ns_per_two_qubit_gate)},
            "provenance": dict(self.provenance), "sources": list(self.sources),
        }


def load_calibration(device_id: str) -> Calibration | None:
    """Packaged nominal calibration for ``device_id`` (None if the device ships none)."""
    path = resources.files("qputwin_core.devices_data").joinpath("calibration", f"{device_id}.json")
    if not path.is_file():
        return None
    return Calibration.from_dict(json.loads(path.read_text(encoding="utf-8")), origin="packaged")


def load_calibration_file(path: str | Path) -> Calibration:
    """Load a calibration snapshot from an arbitrary path (runtime override, ``QPUTWIN_CALIBRATION``)."""
    p = Path(path)
    return Calibration.from_dict(json.loads(p.read_text(encoding="utf-8")), origin=str(p))
