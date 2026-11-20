"""Tests the gate matrices against their generators.

Usage: uv run pytest tests/test_gates.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from qputwin_core import gates as G

ANGLES = [0.0, 0.3, -1.1, math.pi / 2, math.pi, 2.7, -math.pi]


@pytest.mark.parametrize("name", sorted(G.GATE_TABLE))
def test_every_gate_is_unitary(name: str, rng: np.random.Generator) -> None:
    nq, npar, _ = G.GATE_TABLE[name]
    m = G.matrix(name, tuple(rng.uniform(-4, 4, size=npar)))
    assert m.shape == (2 ** nq, 2 ** nq)
    np.testing.assert_allclose(m.conj().T @ m, np.eye(2 ** nq), atol=1e-12)


@pytest.mark.parametrize("t", ANGLES)
def test_pauli_rotations_match_expm(t: float) -> None:
    np.testing.assert_allclose(G.rx(t), expm(-0.5j * t * G.X), atol=1e-12)
    np.testing.assert_allclose(G.ry(t), expm(-0.5j * t * G.Y), atol=1e-12)
    np.testing.assert_allclose(G.rz(t), expm(-0.5j * t * G.Z), atol=1e-12)
    np.testing.assert_allclose(G.rzz(t), expm(-0.5j * t * np.kron(G.Z, G.Z)), atol=1e-12)
    np.testing.assert_allclose(G.rxx(t), expm(-0.5j * t * np.kron(G.X, G.X)), atol=1e-12)
    np.testing.assert_allclose(G.ryy(t), expm(-0.5j * t * np.kron(G.Y, G.Y)), atol=1e-12)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("phi", [0.0, 0.4, -2.0, math.pi / 2])
def test_equatorial_rotation_and_ms_match_expm(theta: float, phi: float) -> None:
    sig = math.cos(phi) * G.X + math.sin(phi) * G.Y
    np.testing.assert_allclose(G.r_gate(theta, phi), expm(-0.5j * theta * sig), atol=1e-12)
    sig2 = math.cos(phi / 3) * G.X + math.sin(phi / 3) * G.Y
    np.testing.assert_allclose(G.ms(phi, phi / 3, theta), expm(-0.5j * theta * np.kron(sig, sig2)), atol=1e-12)


@pytest.mark.parametrize("phi", [0.0, 0.7, math.pi / 3, -2.2])
def test_ionq_natives(phi: float) -> None:
    sig = math.cos(phi) * G.X + math.sin(phi) * G.Y
    np.testing.assert_allclose(G.gpi(phi), sig, atol=1e-12)                       # GPI = σ_φ
    np.testing.assert_allclose(G.gpi2(phi), G.r_gate(math.pi / 2, phi), atol=1e-12)
    np.testing.assert_allclose(G.gpi2(phi) @ G.gpi2(phi), -1j * G.gpi(phi), atol=1e-12)


@pytest.mark.parametrize("theta,phi,lam", [(0.3, 0.2, -0.5), (math.pi, 0, math.pi), (1.0, -2.0, 2.5)])
def test_openqasm_u_definition(theta: float, phi: float, lam: float) -> None:
    zyz = G.rz(phi) @ G.ry(theta) @ G.rz(lam)
    np.testing.assert_allclose(G.u_gate(theta, phi, lam), np.exp(0.5j * (phi + lam)) * zyz, atol=1e-12)


def test_known_identities() -> None:
    np.testing.assert_allclose(G.H @ G.X @ G.H, G.Z, atol=1e-12)
    np.testing.assert_allclose(G.SX @ G.SX, G.X, atol=1e-12)
    np.testing.assert_allclose(G.matrix("zzmax"), G.rzz(math.pi / 2), atol=1e-12)
    # MS(0,0,π/2) is maximally entangling: maps |00> to (|00> - i|11>)/√2
    np.testing.assert_allclose(G.ms(0, 0)[:, 0], np.array([1, 0, 0, -1j]) / math.sqrt(2), atol=1e-12)
    # iSWAP = exp(iπ/4 (XX + YY))
    np.testing.assert_allclose(G.iswap(), expm(0.25j * math.pi * (np.kron(G.X, G.X) + np.kron(G.Y, G.Y))),
                               atol=1e-12)


def test_phase_difference() -> None:
    a = G.H
    phi, err = G.phase_difference(np.exp(0.7j) * a, a)
    assert abs(phi - 0.7) < 1e-12 and err < 1e-12
    assert not G.allclose_up_to_phase(G.X, G.Z)
    assert G.is_diagonal(G.rzz(0.2)) and not G.is_diagonal(G.ms(0, 0))
