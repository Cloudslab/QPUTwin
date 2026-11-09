"""Phase-exact native-gate compiler.

Pipeline (docs/guide/compilation.md)::

    IR{U, CX, gphase, device natives}
      ─► (strict) reject non-native gates / off-grid angles        → NativeGateError
      ─► fuse runs of one-qubit gates per qubit into one 2×2 matrix
      ─► CX → (L1⊗L2) · E · (R1⊗R2) · e^{iφ}  with E the device entangler
      ─► re-fuse; synthesise each 2×2 block in the vendor's one-qubit basis
      ─► NativeCircuit + statistics + numerical-equivalence certificate

**Phase exactness.** Every rewrite tracks its global phase, so the compiled
native circuit reproduces the source unitary *exactly* (not merely up to phase):
``U_src = e^{i·global_phase} · U_native``. Each one-qubit block's synthesis error
is measured on the fly and the maximum is reported as ``max_synthesis_error`` —
a per-job numerical certificate. The entangler identities are verified once at
import time.

One-qubit synthesis uses the phased-X normal form
``V = e^{iα} · Rz(ζ) · R(θ, φ)`` (θ ∈ [0, π]) obtained from a ZYZ Euler
decomposition ``V = e^{iα} Rz(β) Ry(γ) Rz(δ)`` with ``θ = γ``, ``φ = π/2 − δ``,
``ζ = β + δ``, then realised per vendor:

==========  ===============================================================
quantinuum  ``u1q(θ, φ)``, ``rz(ζ)``
iqm         ``prx(θ, φ)``, ``rz(ζ)``
ionq        ``gpi2(φ)`` (θ=π/2) · ``gpi(φ)`` (θ=π) · otherwise
            ``gpi2(φ−π/2) rz(θ) gpi2(φ+π/2)``; then ``rz(ζ)``
rigetti     ``rz(−φ) rx(π/2|π) rz(φ)`` or the two-pulse form
            ``rz(π/2−φ) rx(π/2) rz(θ−π) rx(π/2) rz(φ+π/2)``; then ``rz(ζ)``
ibm         as rigetti with ``sx`` for ``rx(π/2)`` and ``x`` for ``rx(π)``
            (IBM's ``rz · sx · rz · sx · rz`` form)
google      ``phased_xz(θ, φ, ζ)`` (one PhasedXZ pulse, Z folded in), or
            ``rz(ζ)`` alone when θ = 0
==========  ===============================================================

(``rz`` runs are merged and angles wrapped to (−π, π] with exact phase correction.)

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qputwin_core import gates as G
from qputwin_core.devices import DeviceSpec
from qputwin_core.errors import CompilationError, NativeGateError
from qputwin_core.ir import Circuit, Condition, Operation, SourceRef

__all__ = ["NativeOp", "NativeCircuit", "NativeCompiler", "zyz", "phased_x"]

_PI = math.pi
_TAU = 2 * math.pi
_ATOL = 1e-12


# --------------------------------------------------------------------------- data
@dataclass(frozen=True, slots=True)
class NativeOp:
    name: str
    qubits: tuple[int, ...]                      # physical qubits
    params: tuple[float, ...] = ()
    matrix: np.ndarray | None = None             # None for measure / reset
    diagonal: np.ndarray | None = None           # diag(matrix) when the gate is diagonal
    clbits: tuple[int, ...] = ()
    condition: Condition | None = None
    origin: SourceRef | None = None
    virtual: bool = False

    @property
    def is_gate(self) -> bool:
        return self.matrix is not None


@dataclass(slots=True)
class NativeCircuit:
    device_id: str
    mode: str
    num_qubits: int
    num_clbits: int
    ops: list[NativeOp]
    global_phase: float
    stats: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------ 1Q decompositions
def zyz(v: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(α, β, γ, δ)`` with ``v = e^{iα} Rz(β) Ry(γ) Rz(δ)``."""
    det = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
    alpha = float(np.angle(det)) / 2.0
    w = v * np.exp(-1j * alpha)                  # w ∈ SU(2)
    c, s = abs(w[0, 0]), abs(w[1, 0])
    gamma = 2.0 * math.atan2(s, c)
    if s < 1e-14:
        plus, minus = 2.0 * float(np.angle(w[1, 1])), 0.0
    elif c < 1e-14:
        plus, minus = 0.0, 2.0 * float(np.angle(w[1, 0]))
    else:
        plus, minus = 2.0 * float(np.angle(w[1, 1])), 2.0 * float(np.angle(w[1, 0]))
    beta, delta = (plus + minus) / 2.0, (plus - minus) / 2.0
    return alpha, beta, gamma, delta


def phased_x(v: np.ndarray) -> tuple[float, float, float]:
    """Return ``(θ, φ, ζ)`` with ``v ≅ Rz(ζ) · R(θ, φ)`` (up to global phase)."""
    _, beta, gamma, delta = zyz(v)
    return gamma, _PI / 2 - delta, beta + delta


def _wrap(angle: float) -> float:
    """Wrap an angle to (−π, π] (the caller re-derives the exact global phase numerically)."""
    w = math.remainder(angle, _TAU)            # in [−π, π]
    if w <= -_PI:
        w += _TAU
    return min(w, _PI)


def _near(a: float, b: float, tol: float = 1e-10) -> bool:
    d = (a - b) % _TAU
    return min(d, _TAU - d) < tol


# ------------------------------------------------------------------- compiler
class NativeCompiler:
    """Compiles flattened IR to a device's native gate set (phase-exact)."""

    def __init__(self, spec: DeviceSpec) -> None:
        self.spec = spec
        self.native_set = frozenset(spec.native_gates)
        self._mat_cache: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}
        self._cx_rule = self._derive_cx_rule()

    # --------------------------------------------------------- CX identities
    def _derive_cx_rule(self) -> tuple[np.ndarray, np.ndarray, str, tuple[float, ...],
                                       np.ndarray, np.ndarray, float]:
        """Return ``(R1, R2, E, E_params, L1, L2, φ)`` with ``CX = e^{iφ}(L1⊗L2)E(R1⊗R2)``."""
        H, I2 = G.H, G.I2
        rzz_local = G.rz(_PI / 2) @ G.Z           # CZ ≅ (Rz(π/2)Z ⊗ Rz(π/2)Z) · RZZ(π/2)
        basis = self.spec.basis
        if basis in ("rigetti", "iqm", "ibm", "google"):
            r1, r2, e_name, e_params, l1, l2 = I2, H, "cz", (), I2, H
        elif basis == "quantinuum":
            r1, r2, e_name, e_params, l1, l2 = I2, H, "zzmax", (), rzz_local, H @ rzz_local
        elif basis == "ionq":
            # RZZ(π/2) = (H⊗H) · MS(0,0,π/2) · (H⊗H)
            r1, r2, e_name, e_params = H, H @ H, "ms", (0.0, 0.0, _PI / 2)
            l1, l2 = rzz_local @ H, H @ rzz_local @ H
        else:  # pragma: no cover
            raise ValueError(basis)
        e = G.matrix(e_name, e_params)
        product = np.kron(l1, l2) @ e @ np.kron(r1, r2)
        phi, err = G.phase_difference(G.cx(), product)
        if err > 1e-12:  # pragma: no cover - guarded by tests
            raise AssertionError(f"CX identity for {basis} is wrong (err={err:.2e})")
        return r1, r2, e_name, e_params, l1, l2, phi

    # ----------------------------------------------------------- matrices
    def _matrix(self, name: str, params: tuple[float, ...]) -> np.ndarray:
        key = (name, params)
        m = self._mat_cache.get(key)
        if m is None:
            m = G.matrix(name, params)
            if len(self._mat_cache) < 65536:
                self._mat_cache[key] = m
        return m

    def _native_op(self, name: str, qubits: tuple[int, ...], params: tuple[float, ...] = (), *,
                   condition: Condition | None = None, origin: SourceRef | None = None) -> NativeOp:
        m = self._matrix(name, params)
        diag = np.diag(m).copy() if G.is_diagonal(m) else None
        return NativeOp(name, qubits, params, m, diag, (), condition, origin,
                        virtual=name in self.spec.virtual_gates)

    # --------------------------------------------------------- 1Q synthesis
    def synth_1q(self, v: np.ndarray) -> tuple[list[tuple[str, tuple[float, ...]]], float, float]:
        """Synthesise ``v`` in the vendor basis → ``(ops, phase, error)``, ``v = e^{i·phase}·∏ops``."""
        theta, phi, zeta = phased_x(v)
        seq: list[tuple[str, tuple[float, ...]]] = []
        basis = self.spec.basis
        if theta > 1e-12:
            if basis == "quantinuum":
                seq.append(("u1q", (theta, phi)))
            elif basis == "iqm":
                seq.append(("prx", (theta, phi)))
            elif basis == "ionq":
                if abs(theta - _PI / 2) < 1e-12:
                    seq.append(("gpi2", (phi,)))
                elif abs(theta - _PI) < 1e-12:
                    seq.append(("gpi", (phi,)))
                else:
                    seq += [("gpi2", (phi - _PI / 2,)), ("rz", (theta,)), ("gpi2", (phi + _PI / 2,))]
            elif basis == "rigetti":
                if abs(theta - _PI / 2) < 1e-12 or abs(theta - _PI) < 1e-12:
                    pulse = _PI / 2 if abs(theta - _PI / 2) < 1e-12 else _PI
                    seq += [("rz", (-phi,)), ("rx", (pulse,)), ("rz", (phi,))]
                else:
                    seq += [("rz", (_PI / 2 - phi,)), ("rx", (_PI / 2,)), ("rz", (theta - _PI,)),
                            ("rx", (_PI / 2,)), ("rz", (phi + _PI / 2,))]
            elif basis == "ibm":
                # sx ≅ rx(π/2) and x ≅ rx(π) up to global phase; the phase is re-derived below
                if abs(theta - _PI / 2) < 1e-12:
                    seq += [("rz", (-phi,)), ("sx", ()), ("rz", (phi,))]
                elif abs(theta - _PI) < 1e-12:
                    seq += [("rz", (-phi,)), ("x", ()), ("rz", (phi,))]
                else:
                    seq += [("rz", (_PI / 2 - phi,)), ("sx", ()), ("rz", (theta - _PI,)),
                            ("sx", ()), ("rz", (phi + _PI / 2,))]
            elif basis == "google":
                seq.append(("phased_xz", (theta, _wrap(phi), _wrap(zeta))))
                zeta = 0.0
        seq.append(("rz", (zeta,)))
        seq = self._normalise_rz(seq)
        prod = np.eye(2, dtype=np.complex128)
        for name, params in seq:
            prod = self._matrix(name, params) @ prod
        phase, err = G.phase_difference(v, prod)
        return seq, phase, err

    def _normalise_rz(self, seq: list[tuple[str, tuple[float, ...]]]) -> list[tuple[str, tuple[float, ...]]]:
        """Merge adjacent rz, wrap angles to (−π, π], drop identities (phase fixed by caller)."""
        merged: list[tuple[str, tuple[float, ...]]] = []
        for name, params in seq:
            if name == "rz" and merged and merged[-1][0] == "rz":
                merged[-1] = ("rz", (merged[-1][1][0] + params[0],))
            else:
                merged.append((name, params))
        out: list[tuple[str, tuple[float, ...]]] = []
        for name, params in merged:
            if name == "rz":
                w = _wrap(params[0])
                if abs(w) < 1e-12:
                    continue
                out.append(("rz", (w,)))
            else:
                out.append((name, tuple(float(p) for p in params)))
        return out

    # ------------------------------------------------------------- strict
    def _check_strict(self, op: Operation) -> None:
        if op.name == "gphase":
            return
        gate = op.origin.gate if op.origin else op.name
        line = op.origin.line if op.origin else None
        if op.name not in self.native_set:
            raise NativeGateError(
                f"'{gate}' (line {line}) lowers to non-native operation '{op.name}'; "
                f"{self.spec.id} natives are {sorted(self.native_set)}",
                details={"gate": gate, "primitive": op.name, "line": line,
                         "native_gates": sorted(self.native_set), "device": self.spec.id})
        grid = self.spec.param_grid.get(op.name)
        if grid is not None:
            for p in op.params:
                if not any(_near(p, g) for g in grid):
                    raise NativeGateError(
                        f"'{op.name}({p:.6g})' (line {line}) is not a calibrated angle on {self.spec.id}; "
                        f"allowed {[round(g, 6) for g in grid]}",
                        details={"gate": op.name, "angle": p, "allowed": list(grid), "line": line,
                                 "device": self.spec.id})

    # ------------------------------------------------------------ compile
    def compile(self, circuit: Circuit, mode: str = "native") -> NativeCircuit:
        if mode not in ("native", "strict"):
            raise CompilationError(f"unknown compilation mode {mode!r}")
        out: list[NativeOp] = []
        phase = 0.0
        max_err = 0.0
        pending: dict[int, np.ndarray] = {}

        def emit_1q(q: int, v: np.ndarray, *, condition: Condition | None = None,
                    origin: SourceRef | None = None) -> float:
            nonlocal max_err
            seq, ph, err = self.synth_1q(v)
            max_err = max(max_err, err)
            for name, params in seq:
                out.append(self._native_op(name, (q,), params, condition=condition, origin=origin))
            return ph

        def flush(qubits: tuple[int, ...] | list[int]) -> None:
            nonlocal phase
            for q in qubits:
                v = pending.pop(q, None)
                if v is not None:
                    phase += emit_1q(q, v)

        def lower_cx(c: int, t: int, *, condition: Condition | None, origin: SourceRef | None) -> float:
            r1, r2, e_name, e_params, l1, l2, ph = self._cx_rule
            if condition is None:
                pending[c] = r1 @ pending.get(c, G.I2)
                pending[t] = r2 @ pending.get(t, G.I2)
                flush((c, t))
                out.append(self._native_op(e_name, (c, t), e_params, origin=origin))
                pending[c] = l1
                pending[t] = l2
                return ph
            ph_total = ph + emit_1q(c, r1, condition=condition, origin=origin)
            ph_total += emit_1q(t, r2, condition=condition, origin=origin)
            out.append(self._native_op(e_name, (c, t), e_params, condition=condition, origin=origin))
            ph_total += emit_1q(c, l1, condition=condition, origin=origin)
            ph_total += emit_1q(t, l2, condition=condition, origin=origin)
            return ph_total

        for op in circuit.ops:
            if mode == "strict" and (op.is_gate or op.name == "gphase"):
                self._check_strict(op)
            if op.name == "gphase":
                if op.condition is None:
                    phase += op.params[0]
                continue
            if op.name == "barrier":
                flush(op.qubits)
                # kept in the native circuit so the gate-duration model can synchronise on it
                out.append(NativeOp("barrier", op.qubits, origin=op.origin))
                continue
            if op.name in ("measure", "reset"):
                flush(op.qubits)
                out.append(NativeOp(op.name, op.qubits, (), None, None, op.clbits, op.condition, op.origin))
                continue
            nq = len(op.qubits)
            if mode == "strict":
                flush(op.qubits)
                out.append(self._native_op(op.name, op.qubits, op.params, condition=op.condition,
                                           origin=op.origin))
                continue
            if op.condition is not None:
                flush(op.qubits)
                if nq == 1:
                    emit_1q(op.qubits[0], self._matrix(op.name, op.params), condition=op.condition,
                            origin=op.origin)
                elif op.name == "CX":
                    lower_cx(*op.qubits, condition=op.condition, origin=op.origin)
                elif op.name in self.native_set:
                    out.append(self._native_op(op.name, op.qubits, op.params, condition=op.condition,
                                               origin=op.origin))
                else:
                    raise CompilationError(f"no native lowering for conditional '{op.name}' on {self.spec.id}")
                # a phase acquired only on one classical branch is unobservable: discarded
                continue
            if nq == 1:
                q = op.qubits[0]
                pending[q] = self._matrix(op.name, op.params) @ pending.get(q, G.I2)
            elif op.name == "CX":
                # NB: evaluate first — lower_cx() itself updates `phase` via flush()
                cx_phase = lower_cx(*op.qubits, condition=None, origin=op.origin)
                phase += cx_phase
            elif nq == 2 and op.name in self.native_set:
                flush(op.qubits)
                out.append(self._native_op(op.name, op.qubits, op.params, origin=op.origin))
            else:
                raise CompilationError(
                    f"no native lowering for '{op.name}' on {self.spec.id}",
                    details={"gate": op.name, "device": self.spec.id})
        flush(sorted(pending))
        native = NativeCircuit(self.spec.id, mode, circuit.num_qubits, circuit.num_clbits, out,
                               math.remainder(phase, _TAU))
        native.stats = self.statistics(circuit, native, max_err)
        return native

    # ---------------------------------------------------------- statistics
    def statistics(self, circuit: Circuit, native: NativeCircuit, max_err: float) -> dict[str, Any]:
        gates = [op for op in native.ops if op.is_gate]
        hist = Counter(op.name for op in gates)
        level: dict[int, int] = {}
        level2: dict[int, int] = {}
        depth = depth2 = 0
        for op in native.ops:
            if (op.is_gate and op.virtual) or op.name == "barrier":
                continue
            d = max((level.get(q, 0) for q in op.qubits), default=0) + 1
            for q in op.qubits:
                level[q] = d
            depth = max(depth, d)
            if op.is_gate and len(op.qubits) == 2:
                d2 = max(level2.get(q, 0) for q in op.qubits) + 1
                for q in op.qubits:
                    level2[q] = d2
                depth2 = max(depth2, d2)
        return {
            "input_gate_count": circuit.gate_count,
            "input_multi_qubit_gate_count": circuit.multi_qubit_gate_count,
            "native_op_count": len(gates),
            "native_physical_op_count": sum(1 for op in gates if not op.virtual),
            "native_virtual_op_count": sum(1 for op in gates if op.virtual),
            "native_two_qubit_count": sum(1 for op in gates if len(op.qubits) == 2),
            "measurement_count": sum(1 for op in native.ops if op.name == "measure"),
            "reset_count": sum(1 for op in native.ops if op.name == "reset"),
            "native_depth": depth,
            "two_qubit_depth": depth2,
            "native_gate_histogram": dict(sorted(hist.items())),
            "global_phase": native.global_phase,
            "max_synthesis_error": max_err,
        }
