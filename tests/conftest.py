"""Shared helpers for the QPUTwin tests.

Usage: loaded automatically by pytest

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import numpy as np
import pytest

from qputwin_core import gates as G
from qputwin_core.ir import Circuit


def apply_to_unitary(u: np.ndarray, m: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Left-multiply the 2^n×2^n matrix ``u`` by gate ``m`` on ``qubits`` (first = MSB of ``m``).

    State index bit q ↔ qubit q (little-endian), so tensor axis (n-1-q) ↔ qubit q.
    """
    k = len(qubits)
    t = u.reshape((2,) * n + (-1,))
    axes = [n - 1 - q for q in qubits]
    g = m.reshape((2,) * (2 * k))
    t = np.tensordot(g, t, axes=(list(range(k, 2 * k)), axes))
    t = np.moveaxis(t, list(range(k)), axes)
    return t.reshape(1 << n, -1)


def circuit_unitary(circuit: Circuit, n: int | None = None) -> np.ndarray:
    n = circuit.num_qubits if n is None else n
    u = np.eye(1 << n, dtype=np.complex128)
    phase = 0.0
    for op in circuit.ops:
        if op.name == "gphase":
            phase += op.params[0]
        elif op.name == "barrier":
            continue
        else:
            assert op.name not in ("measure", "reset"), "unitary only for measurement-free circuits"
            u = apply_to_unitary(u, G.matrix(op.name, op.params), op.qubits, n)
    return np.exp(1j * phase) * u


def native_unitary(native, n: int) -> np.ndarray:
    u = np.eye(1 << n, dtype=np.complex128)
    for op in native.ops:
        if op.is_gate:
            u = apply_to_unitary(u, op.matrix, op.qubits, n)
    return np.exp(1j * native.global_phase) * u


def random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261021)
