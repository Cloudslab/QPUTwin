"""Calibration-driven noise model.

Channels and error terms (applied by :mod:`qputwin_core.density` and :mod:`qputwin_core.trajectory`)
------------------------------------------------------------------------------------------------------
After every **physical** gate ``U`` the error channel ``E = D_p ∘ R ∘ V`` is applied:

* ``V`` — **coherent error** (unitary): a fractional over-rotation ``U^ε``
  (``overrotation`` ε, via the principal matrix logarithm, so a rotation by θ
  becomes a rotation by θ(1+ε)) followed by a residual Z (1Q) or ZZ (2Q) phase
  ``coherent_z_rad``;
* ``R`` — **thermal relaxation** on each gate qubit for the gate duration ``t``:
  amplitude damping ``γ = 1 − e^{−t/T1}`` then pure dephasing with
  ``√(1−λ) = e^{−t/T2} / e^{−t/(2T1)}``, so coherences decay exactly as
  ``e^{−t/T2}`` (requires ``T2 ≤ 2·T1``)::

      K₀ᴬᴰ = [[1, 0], [0, √(1−γ)]],  K₁ᴬᴰ = [[0, √γ], [0, 0]]
      K₀ᴾᴰ = [[1, 0], [0, √(1−λ)]],  K₁ᴾᴰ = [[0, 0], [0, √λ]]

* ``D_p`` — **depolarising** ``(1−p)·ρ + p·Tr_Q(ρ) ⊗ I/d``, with ``p`` fixed so
  that the *whole* error channel reproduces the published average gate fidelity
  ``F`` exactly. With ``F_e = ((d+1)·F − 1)/d`` and the exact identities
  ``F_e(D_p ∘ X) = (1−p)·F_e(X) + p/d²`` and ``F_e(R ∘ V) = Σ_k |Tr(K_k V)|² / d²``::

      p = (F_e(R∘V) − F_e(target)) / (F_e(R∘V) − 1/d²)

  so neither relaxation nor coherent error is double counted. If they alone
  exceed the published error, ``p = 0`` and the gate is listed under
  ``budget_limited_gates``.

Further terms (each only if calibrated or overridden, each with its own toggle):

* **Crosstalk** — during a 2Q gate on ``(a, b)`` every coupled spectator ``n``
  picks up ``exp(−iθ/2·Z_x Z_n)`` (``x`` the adjacent gate qubit). An unsimulated
  spectator is exactly ``|0⟩``, so its term reduces to ``Rz(θ)`` on ``x``.
* **Idle relaxation** in every idle window of the ASAP schedule. Controllers run a
  fixed timeline, so a classically-controlled gate that does *not* fire still
  occupies its slot: its qubits then relax for the slot duration (negated condition).
* **Reset error** — after a reset the qubit is ``|1⟩`` with probability ``1 − F_reset``.
* **Readout** — per-qubit confusion ``[[1−p01, p10], [p01, 1−p10]]`` or, for a
  group of simultaneously read qubits, a joint ``readout_correlations`` matrix.
* **Leakage** (trajectory engine) — a phenomenological model: with probability
  ``L`` per gate a qubit leaves the computational subspace; while leaked, gates on
  it are skipped, a 2Q partner is fully depolarised, it reads out as ``1``, and it
  returns to ``|1⟩`` with probability ``seepage_per_gate`` per gate.
* **Quasi-static (non-Markovian) detuning** (trajectory engine) — each
  trajectory draws a frequency offset ``δ_q ~ N(0, σ_q)`` per qubit, constant over
  the shot, giving phase ``2π·δ_q·t`` over every window: correlated in time, so it
  is not representable as a product of per-window channels.

Virtual gates (``rz`` frame changes) are noiseless and take zero time. ``scale``
multiplies every error rate, decay rate and coherent angle, for noise sweeps.
Every calibration parameter a job uses is reported with its provenance class
(``published`` / ``assumed`` / ``untagged``).

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import linalg as sla

from qputwin_core import gates as G
from qputwin_core.calibration import Calibration
from qputwin_core.errors import InvalidRequestError
from qputwin_core.ir import Condition
from qputwin_core.scheduling import Schedule
from qputwin_core.synthesis import NativeCircuit, NativeOp
from qputwin_core.topology import CouplingMap

__all__ = ["NoiseOptions", "NoiseModel", "NoisyInstr", "relaxation_kraus", "superoperator",
           "entanglement_fidelity", "fidelity_with_kraus", "depolarizing_parameter", "build_noisy_program",
           "TRAJECTORY_ONLY"]

_C = np.complex128
_I2 = np.eye(2, dtype=_C)
_X = np.array([[0, 1], [1, 0]], dtype=_C)
TRAJECTORY_ONLY = frozenset({"leakage", "quasi_static"})


# ===================================================================== options
_BOOL_FIELDS = ("depolarizing", "relaxation", "idle", "readout", "coherent", "crosstalk", "reset_errors",
                "correlated_readout", "leakage", "quasi_static")
_FLOAT_FIELDS: dict[str, tuple[float, float, bool]] = {   # name -> (lo, hi, lo_open)
    "scale": (0.0, 100.0, False),
    "one_qubit_fidelity": (0.5, 1.0, False), "two_qubit_fidelity": (0.5, 1.0, False),
    "readout_fidelity": (0.5, 1.0, False), "reset_fidelity": (0.5, 1.0, False),
    "t1_us": (0.0, math.inf, True), "t2_us": (0.0, math.inf, True),
    "overrotation": (-0.5, 0.5, False), "coherent_z_rad": (-math.pi, math.pi, False),
    "crosstalk_zz_rad": (-math.pi, math.pi, False), "leakage_rate": (0.0, 0.5, False),
    "seepage_rate": (0.0, 1.0, False), "quasi_static_khz": (0.0, 1e6, False),
}


@dataclass(frozen=True, slots=True)
class NoiseOptions:
    model: str = "device"
    scale: float = 1.0
    depolarizing: bool = True
    relaxation: bool = True
    idle: bool = True
    readout: bool = True
    coherent: bool = True
    crosstalk: bool = True
    reset_errors: bool = True
    correlated_readout: bool = True
    leakage: bool = True
    quasi_static: bool = True
    one_qubit_fidelity: float | None = None
    two_qubit_fidelity: float | None = None
    readout_fidelity: float | None = None
    reset_fidelity: float | None = None
    t1_us: float | None = None
    t2_us: float | None = None
    overrotation: float | None = None
    coherent_z_rad: float | None = None
    crosstalk_zz_rad: float | None = None
    leakage_rate: float | None = None
    seepage_rate: float | None = None
    quasi_static_khz: float | None = None
    trajectories: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> NoiseOptions | None:
        """Validate a request's ``options.noise`` (``None`` or ``{"model": "ideal"}`` = noiseless)."""
        if d is None:
            return None
        if not isinstance(d, dict):
            raise InvalidRequestError("options.noise must be an object or null")
        d = dict(d)
        model = d.pop("model", "device")
        if model == "ideal":
            return None
        if model != "device":
            raise InvalidRequestError(f"unknown noise model {model!r} (supported: 'ideal', 'device')")
        kw: dict[str, Any] = {}
        for k in _BOOL_FIELDS:
            if k in d:
                v = d.pop(k)
                if not isinstance(v, bool):
                    raise InvalidRequestError(f"options.noise.{k} must be a boolean")
                kw[k] = v
        for k, (lo, hi, lo_open) in _FLOAT_FIELDS.items():
            v = d.pop(k, None)
            if v is None:
                continue
            if not isinstance(v, int | float) or isinstance(v, bool) or not (
                    (float(v) > lo if lo_open else float(v) >= lo) and float(v) <= hi):
                raise InvalidRequestError(f"options.noise.{k} must be a number in "
                                          f"{'(' if lo_open else '['}{lo:g}, {hi:g}]")
            kw[k] = float(v)
        t = d.pop("trajectories", None)
        if t is not None:
            if not isinstance(t, int) or isinstance(t, bool) or not 1 <= t <= 1_000_000:
                raise InvalidRequestError("options.noise.trajectories must be an integer in [1, 1000000]")
            kw["trajectories"] = t
        if kw.get("t1_us") and kw.get("t2_us") and kw["t2_us"] > 2 * kw["t1_us"]:
            raise InvalidRequestError("options.noise: unphysical coherence, t2_us > 2·t1_us")
        if d:
            raise InvalidRequestError(f"unknown options.noise field(s): {sorted(d)}")
        return cls(model=model, **kw)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}  # type: ignore[attr-defined]


# =================================================================== channels
def relaxation_kraus(t_ns: float, t1_us: float | None, t2_us: float | None) -> list[np.ndarray]:
    """Kraus operators of amplitude damping (T1) followed by pure dephasing (T2), duration ``t_ns``."""
    t = t_ns * 1e-3                                         # µs
    kraus = [np.eye(2, dtype=_C)]
    coh = 1.0                                              # coherence factor from amplitude damping
    if t1_us is not None and math.isfinite(t1_us) and t > 0:
        gamma = -math.expm1(-t / t1_us)
        coh = math.sqrt(1.0 - gamma)
        kraus = [np.array([[1, 0], [0, coh]], dtype=_C), np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=_C)]
    if t2_us is not None and math.isfinite(t2_us) and t > 0:
        target = math.exp(-t / t2_us)
        r = min(1.0, target / coh) if coh > 0 else 1.0     # √(1−λ)
        lam = 1.0 - r * r
        if lam > 0:
            pd = [np.array([[1, 0], [0, r]], dtype=_C), np.array([[0, 0], [0, math.sqrt(lam)]], dtype=_C)]
            kraus = [p @ k for p in pd for k in kraus]
    return kraus


def bitflip_kraus(p: float) -> list[np.ndarray]:
    return [math.sqrt(1.0 - p) * _I2, math.sqrt(p) * _X]


def superoperator(kraus: list[np.ndarray]) -> np.ndarray:
    """``S = Σ_k K_k ⊗ conj(K_k)`` acting on the vectorised (row, column) pair of one qubit."""
    s = np.zeros((4, 4), dtype=_C)
    for k in kraus:
        s += np.kron(k, k.conj())
    return s


def entanglement_fidelity(s: np.ndarray, d: int = 2) -> float:
    """``F_e = Σ_k |Tr K_k|² / d² = Tr(S) / d²`` for a channel with superoperator ``S``."""
    return float(np.trace(s).real) / (d * d)


def fidelity_with_kraus(per_qubit_kraus: list[list[np.ndarray]], v: np.ndarray | None) -> float:
    """``F_e(R ∘ V) = Σ_k |Tr(K_k V)|² / d²`` for ``R = ⊗_q R_q`` (first qubit most significant)."""
    d = 1 << len(per_qubit_kraus)
    total = 0.0
    for combo in np.ndindex(*(len(k) for k in per_qubit_kraus)):
        k = per_qubit_kraus[0][combo[0]]
        for j in range(1, len(per_qubit_kraus)):
            k = np.kron(k, per_qubit_kraus[j][combo[j]])
        total += float(abs(np.trace(k if v is None else k @ v))) ** 2
    return float(total / (d * d))


def depolarizing_parameter(avg_fidelity: float, d: int, fe_other: float) -> tuple[float, bool]:
    """``(p, budget_limited)`` making ``D_p ∘ X`` hit ``avg_fidelity`` exactly, given ``F_e(X)``."""
    fe_target = ((d + 1) * avg_fidelity - 1.0) / d
    if fe_other <= fe_target + 1e-15:
        return 0.0, fe_other < fe_target - 1e-12
    p = (fe_other - fe_target) / (fe_other - 1.0 / (d * d))
    return min(1.0, max(0.0, p)), False


def _diag_or_none(m: np.ndarray) -> np.ndarray | None:
    return np.diag(m).copy() if np.all(np.abs(m - np.diag(np.diag(m))) <= 1e-15) else None


# ======================================================================= model
@dataclass(frozen=True, slots=True)
class NoisyInstr:
    kind: str           # gate | channel | depolarize | measure | reset | leak | detune
    qubits: tuple[int, ...]                     # *simulated* qubit indices
    matrix: np.ndarray | None = None            # unitary (gate) or 4×4 superoperator (channel)
    diagonal: np.ndarray | None = None
    p: float = 0.0
    clbits: tuple[int, ...] = ()
    condition: Any = None
    index: int = -1                             # originating native op
    kraus: tuple[np.ndarray, ...] | None = None
    duration_ns: float = 0.0
    physical: bool = False                      # a physical (non-virtual) native gate


@dataclass(slots=True)
class _GateNoise:
    coherent: np.ndarray | None                 # V on the gate's qubits
    channels: list[tuple[int, list[np.ndarray]]]    # (physical qubit, Kraus) relaxation
    p: float
    fidelity: float | None
    budget_limited: bool
    leak: float


@dataclass(slots=True)
class NoiseModel:
    """Noise parameters for one device calibration, coupling map and set of request options."""

    calibration: Calibration
    options: NoiseOptions
    coupling: CouplingMap | None = None
    _gate_cache: dict[tuple[Any, ...], _GateNoise] = field(default_factory=dict)
    _idle_cache: dict[tuple[int, float], list[np.ndarray] | None] = field(default_factory=dict)
    used: dict[str, str] = field(default_factory=dict)            # calibration path -> provenance class
    features: set[str] = field(default_factory=set)
    missing_fidelities: set[str] = field(default_factory=set)
    missing_durations: set[str] = field(default_factory=set)
    budget_limited: set[str] = field(default_factory=set)

    # ------------------------------------------------------------ helpers
    def _use(self, path: str) -> None:
        if path not in self.used:
            self.used[path] = self.calibration.provenance_class(path)

    def _gate_path(self, name: str, qubits: tuple[int, ...], attr: str) -> str | None:
        cal = self.calibration
        if len(qubits) == 2:
            key = f"{min(qubits)}-{max(qubits)}"
            g = cal.couplers.get(key, {}).get(name)
            if g is not None and getattr(g, attr) is not None:
                return f"couplers.{key}.{name}.{attr}"
        elif len(qubits) == 1:
            qc = cal.qubits.get(qubits[0])
            if qc is not None and name in qc.gates and getattr(qc.gates[name], attr) is not None:
                return f"qubits.{qubits[0]}.gates.{name}.{attr}"
        base = cal.gates.get(name)
        if base is not None and getattr(base, attr) is not None:
            return f"gates.{name}.{attr}"
        return None

    def _gate_attr(self, op: NativeOp, attr: str) -> float | None:
        path = self._gate_path(op.name, op.qubits, attr)
        if path is None:
            return None
        self._use(path)
        g = self.calibration.gate(op.name, op.qubits)
        return getattr(g, attr) if g is not None else None

    def _scaled_time(self, t: float | None) -> float | None:
        s = self.options.scale
        if t is None or s == 0:
            return None
        return t / s

    # --------------------------------------------------------- parameters
    def t1(self, q: int) -> float | None:
        if self.options.t1_us is not None:
            return self._scaled_time(self.options.t1_us)
        t = self.calibration.t1(q)
        if t is not None:
            self._use(f"qubits.{q}.t1_us" if q in self.calibration.qubits and
                      self.calibration.qubits[q].t1_us is not None else "coherence.t1_us")
        return self._scaled_time(t)

    def t2(self, q: int) -> float | None:
        if self.options.t2_us is not None:
            return self._scaled_time(self.options.t2_us)
        t = self.calibration.t2(q)
        if t is not None:
            self._use(f"qubits.{q}.t2_us" if q in self.calibration.qubits and
                      self.calibration.qubits[q].t2_us is not None else "coherence.t2_us")
        return self._scaled_time(t)

    def gate_fidelity(self, op: NativeOp) -> float | None:
        o = self.options
        override = o.one_qubit_fidelity if len(op.qubits) == 1 else o.two_qubit_fidelity
        if override is not None:
            f: float | None = override
        else:
            path = self._gate_path(op.name, op.qubits, "fidelity")
            if path is not None:
                self._use(path)
            f = self.calibration.gate_fidelity(op.name, op.qubits)
        if f is None:
            return None
        return max(0.0, 1.0 - (1.0 - f) * o.scale)

    def readout_errors(self, q: int) -> tuple[float, float]:
        o = self.options
        if not o.readout:
            return 0.0, 0.0
        if o.readout_fidelity is not None:
            p01 = p10 = 1.0 - o.readout_fidelity
        else:
            qc = self.calibration.qubits.get(q)
            self._use(f"qubits.{q}.readout_fidelity" if qc is not None and qc.readout_p01 is not None
                      else "readout.fidelity")
            p01, p10 = self.calibration.readout_errors(q)
        if p01 or p10:
            self.features.add("readout")
        return min(0.5, p01 * o.scale), min(0.5, p10 * o.scale)

    def readout_groups(self, measured: list[int]) -> list[tuple[tuple[int, ...], np.ndarray]]:
        """Joint confusion matrices for correlated groups whose qubits are all being read together."""
        o = self.options
        if not (o.readout and o.correlated_readout) or o.readout_fidelity is not None:
            return []
        have = set(measured)
        out = []
        for c in self.calibration.readout_correlations:
            if set(c.qubits) <= have:
                m = np.array(c.matrix, dtype=float)
                m = np.eye(len(m)) + o.scale * (m - np.eye(len(m)))
                m = np.clip(m, 0.0, None)
                m /= m.sum(axis=0, keepdims=True)
                out.append((c.qubits, m))
                self._use("readout_correlations")
                self.features.add("correlated_readout")
        return out

    def reset_error(self) -> float:
        o = self.options
        if not o.reset_errors:
            return 0.0
        f = o.reset_fidelity
        if f is None:
            f = self.calibration.reset_fidelity
            if f is not None:
                self._use("reset.fidelity")
        if f is None:
            return 0.0
        return min(0.5, (1.0 - f) * o.scale)

    def seepage(self) -> float:
        s = self.options.seepage_rate
        if s is None:
            s = self.calibration.seepage_per_gate
            if s is not None:
                self._use("leakage.seepage_per_gate")
        return min(1.0, (s or 0.0) * self.options.scale)

    def quasi_static_khz(self, q: int) -> float:
        o = self.options
        if not o.quasi_static:
            return 0.0
        s = o.quasi_static_khz
        if s is None:
            s = self.calibration.quasi_static_khz(q)
            if s is not None:
                qc = self.calibration.qubits.get(q)
                self._use(f"qubits.{q}.quasi_static_detuning_khz" if qc is not None and
                          qc.quasi_static_detuning_khz is not None else "coherence.quasi_static_detuning_khz")
        return (s or 0.0) * o.scale

    def _relax_kraus(self, q: int, t_ns: float) -> list[np.ndarray] | None:
        t1, t2 = self.t1(q), self.t2(q)
        if t_ns <= 0 or (t1 is None and t2 is None):
            return None
        return relaxation_kraus(t_ns, t1, t2)

    def _coherent(self, op: NativeOp) -> np.ndarray | None:
        o = self.options
        if not o.coherent or o.scale == 0:
            return None
        eps = o.overrotation if o.overrotation is not None else self._gate_attr(op, "overrotation")
        phi = o.coherent_z_rad if o.coherent_z_rad is not None else self._gate_attr(op, "coherent_z_rad")
        v = None
        if eps:
            v = sla.expm(eps * o.scale * sla.logm(op.matrix)).astype(_C)          # U^ε (principal branch)
        if phi:
            z = G.rz(phi * o.scale) if len(op.qubits) == 1 else G.rzz(phi * o.scale)
            v = z if v is None else z @ v
        if v is not None:
            self.features.add("coherent")
        return v

    # ----------------------------------------------------------- per-op noise
    def gate_noise(self, op: NativeOp) -> _GateNoise:
        key = (op.name, op.qubits, op.params)
        cached = self._gate_cache.get(key)
        if cached is not None:
            return cached
        o = self.options
        d = 1 << len(op.qubits)
        v = self._coherent(op)
        channels: list[tuple[int, list[np.ndarray]]] = []
        per_qubit: list[list[np.ndarray]] = []
        if o.relaxation:
            dur = self.calibration.gate_duration_ns(op.name, op.qubits)
            if dur is None:
                self.missing_durations.add(op.name)
            else:
                path = self._gate_path(op.name, op.qubits, "duration_ns")
                if path is not None:
                    self._use(path)
            for q in op.qubits:
                k = self._relax_kraus(q, dur) if dur is not None else None
                if k is not None:
                    channels.append((q, k))
                    self.features.add("relaxation")
                per_qubit.append(k if k is not None else [_I2])
        else:
            per_qubit = [[_I2] for _ in op.qubits]
        fe_other = fidelity_with_kraus(per_qubit, v)
        fid = self.gate_fidelity(op)
        p, limited = 0.0, False
        if fid is None:
            self.missing_fidelities.add(op.name)
        elif o.depolarizing:
            p, limited = depolarizing_parameter(fid, d, fe_other)
            if p > 0:
                self.features.add("depolarizing")
            if limited:
                self.budget_limited.add(op.name)
        leak = 0.0
        if o.leakage:
            lr = o.leakage_rate if o.leakage_rate is not None else self._gate_attr(op, "leakage")
            leak = min(0.5, (lr or 0.0) * o.scale)
            if leak > 0:
                self.features.add("leakage")
        g = _GateNoise(v, channels, p, fid, limited, leak)
        self._gate_cache[key] = g
        return g

    def crosstalk_terms(self, op: NativeOp) -> list[tuple[int, int | None, float]]:
        """``(gate qubit x, spectator n or None for |0⟩ spectators, angle)`` during a 2Q gate."""
        o = self.options
        if not o.crosstalk or len(op.qubits) != 2 or self.coupling is None or o.scale == 0:
            return []
        out: list[tuple[int, int | None, float]] = []
        for x in op.qubits:
            for n in self.coupling.neighbors(x):
                if n in op.qubits:
                    continue
                if o.crosstalk_zz_rad is not None:
                    theta = o.crosstalk_zz_rad
                else:
                    theta = self.calibration.crosstalk_rad(x, n)
                    if theta is not None:
                        key = f"{min(x, n)}-{max(x, n)}"
                        self._use(f"crosstalk.pairs.{key}" if key in self.calibration.crosstalk_pairs
                                  else "crosstalk.zz_rad_per_two_qubit_gate")
                if theta:
                    out.append((x, n, theta * o.scale))
        if out:
            self.features.add("crosstalk")
        return out

    def idle_kraus(self, q: int, t_ns: float) -> list[np.ndarray] | None:
        if not (self.options.idle and self.options.relaxation) or t_ns <= 1e-9:
            return None
        key = (q, round(t_ns, 6))
        if key not in self._idle_cache:
            self._idle_cache[key] = self._relax_kraus(q, t_ns)
        if self._idle_cache[key] is not None:
            self.features.add("idle_relaxation")
        return self._idle_cache[key]

    # ---------------------------------------------------------------- report
    def report(self, native: NativeCircuit, measured: list[int]) -> dict[str, Any]:
        per_gate: dict[str, dict[str, Any]] = {}
        esp = 1.0
        for op in native.ops:
            if not op.is_gate or op.virtual:
                continue
            g = self.gate_noise(op)
            if g.fidelity is not None:
                esp *= g.fidelity
            e = per_gate.setdefault(op.name, {"count": 0, "fidelity": g.fidelity, "depolarizing_p": g.p,
                                              "budget_limited": g.budget_limited, "coherent": g.coherent is not None,
                                              "leakage": g.leak})
            e["count"] += 1
        for q in measured:
            p01, p10 = self.readout_errors(q)
            esp *= 1.0 - 0.5 * (p01 + p10)
        cal = self.calibration
        by_class: dict[str, list[str]] = {"published": [], "assumed": [], "untagged": []}
        for path, cls in sorted(self.used.items()):
            by_class[cls].append(path)
        return {
            "model": self.options.model, "options": self.options.to_dict(),
            "calibration": cal.label or cal.origin, "calibration_origin": cal.origin,
            "fidelity_convention": cal.fidelity_convention,
            "features": sorted(self.features),
            "per_gate": dict(sorted(per_gate.items())),
            "estimated_success_probability": esp,
            "parameters_used": by_class,
            "assumed_parameters": by_class["assumed"] + by_class["untagged"],
            "missing_fidelities": sorted(self.missing_fidelities),
            "missing_durations": sorted(self.missing_durations),
            "budget_limited_gates": sorted(self.budget_limited),
            "relaxation_limited_gates": sorted(self.budget_limited),       # alias kept for schema 1.1 clients
        }


# ================================================================== builder
def _channel(q: int, kraus: list[np.ndarray], condition: Any, index: int) -> NoisyInstr:
    s = superoperator(kraus)
    return NoisyInstr("channel", (q,), s, _diag_or_none(s), condition=condition, index=index, kraus=tuple(kraus))


def _negate(c: Condition) -> Condition:
    return Condition(c.clbits, c.value, not c.equal)


def build_noisy_program(native: NativeCircuit, qmap: dict[int, int], model: NoiseModel | None,
                        schedule: Schedule | None) -> list[NoisyInstr]:
    """Interleave the native circuit with its noise channels (on simulated qubit indices).

    ``model=None`` yields the ideal (noiseless) program; ``schedule`` drives idle
    relaxation and quasi-static detuning and may be ``None`` to disable them.
    """
    out: list[NoisyInstr] = []
    free_at: dict[int, float] = {q: 0.0 for q in qmap}
    detuned = model is not None and any(model.quasi_static_khz(q) > 0 for q in qmap)
    if detuned:
        assert model is not None
        model.features.add("quasi_static")
    reset_p = model.reset_error() if model is not None else 0.0
    for i, op in enumerate(native.ops):
        if op.name == "barrier":
            continue
        timed = model is not None and schedule is not None
        if timed:
            start = schedule.starts_ns[i]                                   # type: ignore[union-attr]
            for q in op.qubits:
                if q not in qmap:
                    continue
                gap = start - free_at[q]
                k = model.idle_kraus(q, gap)                                # type: ignore[union-attr]
                if k is not None:
                    out.append(_channel(qmap[q], k, None, i))
                if detuned and gap > 1e-9:
                    out.append(NoisyInstr("detune", (qmap[q],), index=i, duration_ns=gap))
                free_at[q] = start + schedule.durations_ns[i]               # type: ignore[union-attr]
        if op.name in ("measure", "reset"):
            sq = tuple(qmap.get(q, -1) for q in op.qubits)
            out.append(NoisyInstr(op.name, sq, clbits=op.clbits, condition=op.condition, index=i))
            if op.name == "reset" and reset_p > 0 and sq[0] >= 0:
                out.append(_channel(sq[0], bitflip_kraus(reset_p), op.condition, i))
                model.features.add("reset_errors")                          # type: ignore[union-attr]
            continue
        sq = tuple(qmap[q] for q in op.qubits)
        out.append(NoisyInstr("gate", sq, op.matrix, op.diagonal, condition=op.condition, index=i,
                              physical=not op.virtual))
        if op.virtual or model is None:
            continue
        g = model.gate_noise(op)
        if g.coherent is not None:
            out.append(NoisyInstr("gate", sq, g.coherent, _diag_or_none(g.coherent), condition=op.condition,
                                  index=i))
        for q, k in g.channels:
            out.append(_channel(qmap[q], k, op.condition, i))
        if g.p > 0:
            out.append(NoisyInstr("depolarize", sq, p=g.p, condition=op.condition, index=i))
        for x, n, theta in model.crosstalk_terms(op):
            if n in qmap:
                m = G.rzz(theta)
                out.append(NoisyInstr("gate", (qmap[x], qmap[n]), m, np.diag(m).copy(), condition=op.condition,
                                      index=i))
            else:                                       # spectator in |0⟩: ZZ reduces to Rz(θ) on x
                m = G.rz(theta)
                out.append(NoisyInstr("gate", (qmap[x],), m, np.diag(m).copy(), condition=op.condition, index=i))
        if g.leak > 0:
            for q in sq:
                out.append(NoisyInstr("leak", (q,), p=g.leak, condition=op.condition, index=i))
        if schedule is not None and detuned:
            dur = schedule.durations_ns[i]
            for q in sq:
                if dur > 0:
                    out.append(NoisyInstr("detune", (q,), index=i, duration_ns=dur))
        if op.condition is not None and schedule is not None:
            # fixed timeline: when the gate does not fire its qubits still sit out the slot
            dur = schedule.durations_ns[i]
            for q in op.qubits:
                k = model.idle_kraus(q, dur)
                if k is not None:
                    out.append(_channel(qmap[q], k, _negate(op.condition), i))
    return out
