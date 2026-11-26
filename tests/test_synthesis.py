"""Tests that native compilation is exact.

Usage: uv run pytest tests/test_synthesis.py

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qputwin_core import gates as G
from qputwin_core.devices import DEVICES
from qputwin_core.errors import NativeGateError
from qputwin_core.ir import Circuit, Operation
from qputwin_core.qasm import load_circuit
from qputwin_core.synthesis import NativeCompiler, phased_x, zyz

from conftest import circuit_unitary, native_unitary, random_unitary

VENDORS = sorted(DEVICES)
SPECIAL_1Q = [G.I2, -G.I2, G.X, G.Y, G.Z, G.H, G.S, G.T, G.SX, G.rz(2 * math.pi), G.rx(math.pi / 2),
              G.ry(math.pi), 1j * G.X, G.gpi(0.3), G.u_gate(1e-14, 0.2, 0.3)]


def _random_circuit(n: int, depth: int, rng: np.random.Generator, natives: tuple[str, ...] = ()) -> Circuit:
    ops: list[Operation] = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.45:
            ops.append(Operation("U", (int(rng.integers(n)),), tuple(rng.uniform(-math.pi, math.pi, 3))))
        elif r < 0.85 or not natives:
            a, b = rng.choice(n, size=2, replace=False)
            ops.append(Operation("CX", (int(a), int(b))))
        else:
            name = natives[int(rng.integers(len(natives)))]
            nq, npar, _ = G.GATE_TABLE[name]
            qs = tuple(int(x) for x in rng.choice(n, size=nq, replace=False))
            ops.append(Operation(name, qs, tuple(rng.uniform(-math.pi, math.pi, npar))))
        if rng.random() < 0.05:
            ops.append(Operation("gphase", (), (float(rng.uniform(-3, 3)),)))
    return Circuit(version="3.0", num_qubits=n, num_clbits=0, ops=ops)


@pytest.mark.parametrize("vendor", VENDORS)
@pytest.mark.parametrize("seed", range(12))
def test_random_circuits_compile_exactly(vendor: str, seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    spec = DEVICES[vendor]
    n = int(rng.integers(2, 6))
    circ = _random_circuit(n, 40, rng, natives=spec.native_gates)
    native = NativeCompiler(spec).compile(circ)
    assert all(op.name in spec.native_gates for op in native.ops if op.is_gate)
    np.testing.assert_allclose(native_unitary(native, n), circuit_unitary(circ, n), atol=1e-9)
    assert native.stats["max_synthesis_error"] < 1e-12


@pytest.mark.parametrize("vendor", VENDORS)
def test_one_qubit_synthesis_special_and_random(vendor: str, rng: np.random.Generator) -> None:
    comp = NativeCompiler(DEVICES[vendor])
    for v in SPECIAL_1Q + [random_unitary(2, rng) for _ in range(200)]:
        seq, phase, err = comp.synth_1q(v)
        assert err < 1e-12
        prod = np.eye(2, dtype=complex)
        for name, params in seq:
            assert name in DEVICES[vendor].native_1q
            if name == "rz":
                assert -math.pi < params[0] <= math.pi
            prod = G.matrix(name, params) @ prod
        np.testing.assert_allclose(np.exp(1j * phase) * prod, v, atol=1e-12)
        physical = [n for n, _ in seq if n != "rz"]
        assert len(physical) <= 2


def test_rigetti_uses_only_calibrated_rx_angles(rng: np.random.Generator) -> None:
    comp = NativeCompiler(DEVICES["rigetti_ankaa_3"])
    for _ in range(100):
        seq, _, _ = comp.synth_1q(random_unitary(2, rng))
        for name, params in seq:
            if name == "rx":
                assert any(abs(params[0] - a) < 1e-12 for a in (math.pi / 2, math.pi))


def test_zyz_and_phased_x_roundtrip(rng: np.random.Generator) -> None:
    for _ in range(100):
        v = random_unitary(2, rng)
        a, b, g, d = zyz(v)
        np.testing.assert_allclose(np.exp(1j * a) * G.rz(b) @ G.ry(g) @ G.rz(d), v, atol=1e-12)
        th, ph, ze = phased_x(v)
        assert 0 <= th <= math.pi + 1e-12
        assert G.allclose_up_to_phase(G.rz(ze) @ G.r_gate(th, ph), v, atol=1e-12)


@pytest.mark.parametrize("vendor,entangler", [("quantinuum_helios", "zzmax"), ("ionq_forte", "ms"),
                                              ("rigetti_ankaa_3", "cz"), ("iqm_emerald", "cz")])
def test_cx_costs_exactly_one_entangler(vendor: str, entangler: str) -> None:
    native = NativeCompiler(DEVICES[vendor]).compile(Circuit("2.0", 2, 0, ops=[Operation("CX", (0, 1))]))
    two_q = [op.name for op in native.ops if op.is_gate and len(op.qubits) == 2]
    assert two_q == [entangler]


def test_fusion_reduces_one_qubit_runs() -> None:
    ops = [Operation("U", (0,), (0.1 * k, 0.2, 0.3)) for k in range(30)]
    native = NativeCompiler(DEVICES["iqm_emerald"]).compile(Circuit("2.0", 1, 0, ops=ops))
    assert native.stats["native_physical_op_count"] <= 1


def test_strict_mode() -> None:
    ionq = NativeCompiler(DEVICES["ionq_forte"])
    ok = load_circuit('OPENQASM 2.0;\nqreg q[2];\ngpi2(0.1) q[0];\nms(0,0,pi/2) q[0],q[1];\nrz(0.2) q[1];\n',
                      primitives=DEVICES["ionq_forte"].native_gates)
    native = ionq.compile(ok, "strict")
    assert [op.name for op in native.ops] == ["gpi2", "ms", "rz"]
    bad = load_circuit('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\n',
                       primitives=DEVICES["ionq_forte"].native_gates)
    with pytest.raises(NativeGateError) as ei:
        ionq.compile(bad, "strict")
    assert ei.value.details["gate"] == "h"
    rig = NativeCompiler(DEVICES["rigetti_ankaa_3"])
    prims = DEVICES["rigetti_ankaa_3"].native_gates
    rig.compile(load_circuit("OPENQASM 2.0;\nqreg q[1];\nrx(-pi/2) q[0];\n", primitives=prims), "strict")
    with pytest.raises(NativeGateError, match="calibrated angle"):
        rig.compile(load_circuit("OPENQASM 2.0;\nqreg q[1];\nrx(0.3) q[0];\n", primitives=prims), "strict")


def test_native_two_qubit_gates_pass_through() -> None:
    spec = DEVICES["rigetti_ankaa_3"]
    circ = load_circuit('OPENQASM 2.0;\nqreg q[2];\niswap q[0], q[1];\n', primitives=spec.native_gates)
    native = NativeCompiler(spec).compile(circ)
    assert [op.name for op in native.ops] == ["iswap"]
    np.testing.assert_allclose(native_unitary(native, 2), circuit_unitary(circ, 2), atol=1e-12)
