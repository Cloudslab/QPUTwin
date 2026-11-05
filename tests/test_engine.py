"""Statevector engine: kernels, bit order, sampling statistics, mid-circuit semantics.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from qputwin_core import gates as G
from qputwin_core.devices import DEVICES
from qputwin_core.engine import StatevectorEngine
from qputwin_core.simulators import ExecutionOptions, simulator_for
from qputwin_core.synthesis import NativeOp

from conftest import apply_to_unitary, random_unitary

H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _op(m: np.ndarray, qubits: tuple[int, ...]) -> NativeOp:
    return NativeOp("g", qubits, (), m, np.diag(m).copy() if G.is_diagonal(m) else None)


@pytest.mark.parametrize("dtype,tol", [(np.complex128, 1e-12), (np.complex64, 2e-5)])
def test_kernels_match_reference(rng: np.random.Generator, dtype, tol: float) -> None:
    n = 6
    psi0 = random_unitary(1 << n, rng)[:, 0]
    eng = StatevectorEngine(n, dtype, psi0.astype(dtype))
    ref = psi0.reshape(-1, 1).astype(np.complex128)
    qmap = {q: q for q in range(n)}
    for _ in range(60):
        if rng.random() < 0.5:
            q = int(rng.integers(n))
            m = random_unitary(2, rng) if rng.random() < 0.7 else G.rz(rng.uniform(-3, 3))
            qs: tuple[int, ...] = (q,)
        else:
            qs = tuple(int(x) for x in rng.choice(n, size=2, replace=False))
            if rng.random() < 0.6:
                m = random_unitary(4, rng)
            else:   # structured gates exercise the sparse and diagonal fast paths
                name, params = [("cz", ()), ("rzz", (0.4,)), ("cx", ()), ("iswap", ()),
                                ("ms", (0.1, 0.2, 0.3))][int(rng.integers(5))]
                m = G.matrix(name, params)
        eng.apply(_op(m, qs), qmap)
        ref = apply_to_unitary(ref, m, qs, n)
    np.testing.assert_allclose(eng.psi, ref[:, 0], atol=tol)


def test_little_endian_bitstrings() -> None:
    sim = simulator_for(DEVICES["ionq_forte"])
    r = sim.execute(H2 + "qreg q[3]; creg c[3];\nx q[0];\nmeasure q -> c;\n", ExecutionOptions(shots=10, seed=1))
    assert r.counts == {"001": 10}
    r = sim.execute(H2 + "qreg q[2]; creg a[1]; creg b[2];\nx q[1];\nmeasure q[1] -> b[1];\n",
                    ExecutionOptions(shots=5, seed=1))
    assert r.counts == {"100": 5} and r.cregs == {"a": [0, 1], "b": [1, 2]}


def test_sampler_chi_square_and_reproducibility() -> None:
    sim = simulator_for(DEVICES["quantinuum_helios"])
    src = H2 + "qreg q[3]; creg c[3];\nry(0.9) q[0];\nry(2.1) q[1];\nh q[2];\nmeasure q -> c;\n"
    shots = 200_000
    r = sim.execute(src, ExecutionOptions(shots=shots, seed=123))
    p0 = [math.cos(0.45) ** 2, math.sin(0.45) ** 2]
    p1 = [math.cos(1.05) ** 2, math.sin(1.05) ** 2]
    expected = []
    keys = []
    for b2 in (0, 1):
        for b1 in (0, 1):
            for b0 in (0, 1):
                keys.append(f"{b2}{b1}{b0}")
                expected.append(shots * 0.5 * p1[b1] * p0[b0])
    observed = [r.counts.get(k, 0) for k in keys]
    assert stats.chisquare(observed, expected).pvalue > 1e-3
    again = sim.execute(src, ExecutionOptions(shots=shots, seed=123))
    assert again.counts == r.counts and again.seed == 123


def test_teleportation_with_mid_circuit_measurement_and_feed_forward() -> None:
    theta = 1.234
    src = H2 + f"""
    qreg q[3]; creg m0[1]; creg m1[1]; creg out[1];
    ry({theta}) q[0];
    h q[1]; cx q[1], q[2];
    cx q[0], q[1]; h q[0];
    measure q[0] -> m0[0]; measure q[1] -> m1[0];
    if (m1 == 1) x q[2];
    if (m0 == 1) z q[2];
    measure q[2] -> out[0];
    """
    for vendor in DEVICES:
        if DEVICES[vendor].technology != "trapped_ion":
            continue
        r = simulator_for(DEVICES[vendor]).execute(src, ExecutionOptions(shots=40_000, seed=9))
        ones = sum(k for key, k in r.counts.items() if key[0] == "1")
        p1 = math.sin(theta / 2) ** 2
        assert abs(ones / 40_000 - p1) < 5 * math.sqrt(p1 * (1 - p1) / 40_000)
        assert r.workload["execution_path"] == "shot_branching"
        assert r.workload["branches"] == 4


def test_reset_and_idle_measurements() -> None:
    sim = simulator_for(DEVICES["iqm_emerald"])
    r = sim.execute(H2 + "qreg q[3]; creg c[3];\nh q[0];\nreset q[0];\nx q[1];\nmeasure q -> c;\n",
                    ExecutionOptions(shots=1000, seed=3))
    assert r.counts == {"010": 1000}
    assert r.compilation["idle_measured_qubits"] == [2]
    assert r.compilation["simulated_qubits"] == 2


def test_overwritten_and_dead_measurements() -> None:
    sim = simulator_for(DEVICES["ionq_forte"])
    r = sim.execute(H2 + "qreg q[2]; creg c[1];\nx q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[0];\n",
                    ExecutionOptions(shots=50, seed=1))
    assert r.counts == {"1": 50}
    assert r.workload["execution_path"] == "terminal_sampling"


def test_complex64_precision_matches(rng: np.random.Generator) -> None:
    sim = simulator_for(DEVICES["quantinuum_helios"])
    src = H2 + "qreg q[4];\n" + "".join(f"u3({a:.4f},{b:.4f},{c:.4f}) q[{k % 4}];\ncx q[{k % 4}],q[{(k + 1) % 4}];\n"
                                        for k, (a, b, c) in enumerate(rng.uniform(-3, 3, (12, 3))))
    a = np.array(sim.execute(src, ExecutionOptions(shots=1, return_statevector=True)).statevector)
    b = np.array(sim.execute(src, ExecutionOptions(shots=1, return_statevector=True,
                                                   precision="complex64")).statevector)
    np.testing.assert_allclose(a, b, atol=1e-5)
