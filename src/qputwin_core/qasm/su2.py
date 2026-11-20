"""Small helpers for 2x2 unitary matrices without numpy.

Usage: from qputwin_core.qasm.su2 import u_matrix

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import cmath
import math

M2 = tuple[complex, complex, complex, complex]

IDENTITY: M2 = (1 + 0j, 0j, 0j, 1 + 0j)
PAULI_X: M2 = (0j, 1 + 0j, 1 + 0j, 0j)
_EPS = 1e-12


def u_matrix(theta: float, phi: float, lam: float) -> M2:
    """OpenQASM ``U(θ, φ, λ)`` (the OpenQASM 3 phase convention)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return (complex(c), -cmath.exp(1j * lam) * s, cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c)


def matmul(a: M2, b: M2) -> M2:
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3])


def dagger(a: M2) -> M2:
    return (a[0].conjugate(), a[2].conjugate(), a[1].conjugate(), a[3].conjugate())


def scale(a: M2, z: complex) -> M2:
    return (a[0] * z, a[1] * z, a[2] * z, a[3] * z)


def is_identity(a: M2, tol: float = 1e-12) -> bool:
    return (abs(a[0] - 1) < tol and abs(a[1]) < tol and abs(a[2]) < tol and abs(a[3] - 1) < tol)


def to_u_params(m: M2) -> tuple[float, float, float, float]:
    """Return ``(γ, θ, φ, λ)`` with ``m == e^{iγ} · U(θ, φ, λ)`` and θ ∈ [0, π]."""
    c, s = abs(m[0]), abs(m[2])
    theta = 2.0 * math.atan2(s, c)
    if s < _EPS:                 # diagonal: e^{iγ} diag(1, e^{iλ})
        gamma = cmath.phase(m[0])
        return gamma, 0.0, 0.0, cmath.phase(m[3]) - gamma
    if c < _EPS:                 # anti-diagonal: choose λ = 0
        gamma = cmath.phase(-m[1])
        return gamma, math.pi, cmath.phase(m[2]) - gamma, 0.0
    gamma = cmath.phase(m[0])
    phi = cmath.phase(m[2]) - gamma
    lam = cmath.phase(-m[1]) - gamma
    return gamma, theta, phi, lam


def sqrt_unitary(m: M2) -> M2:
    """A principal square root ``V`` (``V @ V == m``) of a 2x2 unitary."""
    det = m[0] * m[3] - m[1] * m[2]
    half_gamma = cmath.phase(det) / 2.0          # m = e^{iγ} W, W ∈ SU(2), γ = arg(det)/2
    w = scale(m, cmath.exp(-1j * half_gamma))
    cos_a = max(-1.0, min(1.0, ((w[0] + w[3]) / 2).real))
    a = math.acos(cos_a)
    sin_a = math.sin(a)
    if abs(sin_a) < 1e-9:
        if cos_a > 0:                            # W = I
            root: M2 = IDENTITY
        else:                                    # W = -I  ->  (-iZ)^2 = -I
            root = (-1j, 0j, 0j, 1j)
    else:
        # W = cos(a) I - i sin(a) N  =>  N = (cos(a) I - W) / (i sin(a))
        k = 1.0 / (1j * sin_a)
        n = ((cos_a - w[0]) * k, -w[1] * k, -w[2] * k, (cos_a - w[3]) * k)
        ch, sh = math.cos(a / 2), math.sin(a / 2)
        root = (ch - 1j * sh * n[0], -1j * sh * n[1], -1j * sh * n[2], ch - 1j * sh * n[3])
    return scale(root, cmath.exp(1j * half_gamma / 2.0))
