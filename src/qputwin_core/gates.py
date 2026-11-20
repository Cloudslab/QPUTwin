"""Unitary matrices of the supported quantum gates.

Usage: from qputwin_core.gates import rx, ry, rz

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from qputwin_core.errors import QASMSemanticError

__all__ = [
    "I2", "X", "Y", "Z", "H", "u_gate", "rx", "ry", "rz", "r_gate", "phased_xz", "fsim", "gpi", "gpi2", "ms", "rzz",
    "rxx", "ryy", "cx", "cz", "iswap", "matrix", "num_qubits_of", "GATE_TABLE", "is_diagonal",
    "allclose_up_to_phase", "phase_difference",
]

C = np.complex128
SQ2 = np.sqrt(2.0)

I2 = np.eye(2, dtype=C)
X = np.array([[0, 1], [1, 0]], dtype=C)
Y = np.array([[0, -1j], [1j, 0]], dtype=C)
Z = np.array([[1, 0], [0, -1]], dtype=C)
H = np.array([[1, 1], [1, -1]], dtype=C) / SQ2
S = np.diag([1, 1j]).astype(C)
SDG = np.diag([1, -1j]).astype(C)
T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(C)
TDG = np.diag([1, np.exp(-1j * np.pi / 4)]).astype(C)
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=C)
SXDG = SX.conj().T


def u_gate(theta: float, phi: float, lam: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=C)


def rx(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=C)


def ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=C)


def rz(t: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)]).astype(C)


def phase(lam: float) -> np.ndarray:
    return np.diag([1.0, np.exp(1j * lam)]).astype(C)


def r_gate(theta: float, phi: float) -> np.ndarray:
    """``R(θ, φ) = exp(-i θ/2 (cos φ X + sin φ Y))`` — Quantinuum U1q, IQM PRX."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * np.exp(-1j * phi) * s],
                     [-1j * np.exp(1j * phi) * s, c]], dtype=C)


def phased_xz(theta: float, phi: float, zeta: float) -> np.ndarray:
    """Google PhasedXZ in radians: ``Rz(ζ) · R(θ, φ)``.

    Equals ``cirq.PhasedXZGate(x_exponent=θ/π, axis_phase_exponent=φ/π, z_exponent=ζ/π)``
    up to global phase: one microwave pulse about the equatorial axis φ, followed by a
    virtual Z.
    """
    return rz(zeta) @ r_gate(theta, phi)


def fsim(theta: float, phi: float) -> np.ndarray:
    """``cirq.FSimGate(θ, φ)`` = ``cirq.PhasedFSimGate(θ, 0, 0, 0, φ)``: partial swap θ, conditional phase φ."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]], dtype=C)


def gpi(phi: float) -> np.ndarray:
    """IonQ GPI(φ) = [[0, e^{-iφ}], [e^{iφ}, 0]]."""
    return np.array([[0, np.exp(-1j * phi)], [np.exp(1j * phi), 0]], dtype=C)


def gpi2(phi: float) -> np.ndarray:
    """IonQ GPI2(φ) = R(π/2, φ)."""
    return r_gate(np.pi / 2, phi)


def _sigma(phi: float) -> np.ndarray:
    return np.cos(phi) * X + np.sin(phi) * Y


def ms(phi0: float, phi1: float, theta: float = np.pi / 2) -> np.ndarray:
    """IonQ Mølmer–Sørensen ``exp(-i θ/2 · σ_φ0 ⊗ σ_φ1)`` (closed form, A² = I)."""
    a = np.kron(_sigma(phi0), _sigma(phi1))
    return (np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * a).astype(C)


def rzz(t: float) -> np.ndarray:
    e, f = np.exp(-0.5j * t), np.exp(0.5j * t)
    return np.diag([e, f, f, e]).astype(C)


def rxx(t: float) -> np.ndarray:
    return (np.cos(t / 2) * np.eye(4) - 1j * np.sin(t / 2) * np.kron(X, X)).astype(C)


def ryy(t: float) -> np.ndarray:
    return (np.cos(t / 2) * np.eye(4) - 1j * np.sin(t / 2) * np.kron(Y, Y)).astype(C)


def controlled(u: np.ndarray) -> np.ndarray:
    """|0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U, control is the first (most significant) qubit."""
    d = u.shape[0]
    out = np.eye(2 * d, dtype=C)
    out[d:, d:] = u
    return out


def cx() -> np.ndarray:
    return controlled(X)


def cz() -> np.ndarray:
    return np.diag([1, 1, 1, -1]).astype(C)


def iswap() -> np.ndarray:
    return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=C)


def swap() -> np.ndarray:
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=C)


def _const(m: np.ndarray) -> Callable[..., np.ndarray]:
    m = m.copy()
    m.setflags(write=False)
    return lambda: m


def _cu(theta: float, phi: float, lam: float, gamma: float = 0.0) -> np.ndarray:
    return controlled(np.exp(1j * gamma) * u_gate(theta, phi, lam))


def _ccx() -> np.ndarray:
    out = np.eye(8, dtype=C)
    out[6:, 6:] = X
    return out


def _cswap() -> np.ndarray:
    out = np.eye(8, dtype=C)
    out[[5, 6]] = out[[6, 5]]
    return out


# name -> (num_qubits, num_params, factory). Canonical (phase-exact) semantics.
GATE_TABLE: dict[str, tuple[int, int, Callable[..., np.ndarray]]] = {
    # OpenQASM builtins
    "U": (1, 3, u_gate),
    "CX": (2, 0, cx),
    # one-qubit standard gates
    "id": (1, 0, _const(I2)), "x": (1, 0, _const(X)), "y": (1, 0, _const(Y)), "z": (1, 0, _const(Z)),
    "h": (1, 0, _const(H)), "s": (1, 0, _const(S)), "sdg": (1, 0, _const(SDG)),
    "t": (1, 0, _const(T)), "tdg": (1, 0, _const(TDG)), "sx": (1, 0, _const(SX)), "sxdg": (1, 0, _const(SXDG)),
    "rx": (1, 1, rx), "ry": (1, 1, ry), "rz": (1, 1, rz),
    "p": (1, 1, phase), "phase": (1, 1, phase), "u1": (1, 1, phase),
    "u2": (1, 2, lambda phi, lam: u_gate(np.pi / 2, phi, lam)),
    "u3": (1, 3, u_gate), "u": (1, 3, u_gate),
    # vendor-native one-qubit gates
    "u1q": (1, 2, r_gate), "prx": (1, 2, r_gate), "gpi": (1, 1, gpi), "gpi2": (1, 1, gpi2),
    "phased_xz": (1, 3, phased_xz),
    # two-qubit
    "cx": (2, 0, cx), "cz": (2, 0, cz), "cy": (2, 0, lambda: controlled(Y)),
    "ch": (2, 0, lambda: controlled(H)), "swap": (2, 0, swap), "iswap": (2, 0, iswap),
    "cp": (2, 1, lambda lam: controlled(phase(lam))), "cphase": (2, 1, lambda lam: controlled(phase(lam))),
    "cu1": (2, 1, lambda lam: controlled(phase(lam))),
    "crx": (2, 1, lambda t: controlled(rx(t))), "cry": (2, 1, lambda t: controlled(ry(t))),
    "crz": (2, 1, lambda t: controlled(rz(t))),
    "cu3": (2, 3, _cu), "cu": (2, 4, _cu), "csx": (2, 0, lambda: controlled(SX)),
    "rzz": (2, 1, rzz), "rxx": (2, 1, rxx), "ryy": (2, 1, ryy),
    "zz": (2, 1, rzz), "zzmax": (2, 0, lambda: rzz(np.pi / 2)), "ms": (2, 3, ms),
    # three-qubit
    "ccx": (3, 0, _ccx), "cswap": (3, 0, _cswap),
}


def num_qubits_of(name: str) -> int:
    try:
        return GATE_TABLE[name][0]
    except KeyError as exc:
        raise QASMSemanticError(f"no closed-form matrix for gate {name!r}") from exc


def matrix(name: str, params: tuple[float, ...] | list[float] = ()) -> np.ndarray:
    """Canonical unitary of ``name`` with ``params`` (first qubit = most significant)."""
    try:
        _, n_params, factory = GATE_TABLE[name]
    except KeyError as exc:
        raise QASMSemanticError(f"no closed-form matrix for gate {name!r}") from exc
    if len(params) != n_params:
        raise QASMSemanticError(f"gate {name!r} expects {n_params} parameter(s), got {len(params)}")
    return np.asarray(factory(*params), dtype=C)


def is_diagonal(m: np.ndarray, tol: float = 1e-14) -> bool:
    return bool(np.all(np.abs(m - np.diag(np.diag(m))) <= tol))


def phase_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return ``(φ, err)`` minimising ``‖a − e^{iφ} b‖``; ``err`` is the residual max-norm."""
    inner = np.vdot(b, a)                      # Σ conj(b) a
    phi = float(np.angle(inner)) if abs(inner) > 1e-300 else 0.0
    err = float(np.max(np.abs(a - np.exp(1j * phi) * b)))
    return phi, err


def allclose_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> bool:
    return phase_difference(a, b)[1] <= atol
