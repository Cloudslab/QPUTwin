"""Density-matrix engine and calibration-driven noise model.

The engine is cross-checked against an independent dense-matrix reference
(explicit ``UρU†``, ``Σ KρK†`` and partial traces with ``einsum``), and every
channel against its analytic action. The depolarising calibration is verified
end-to-end: the *composite* gate channel reproduces the published average gate
fidelity exactly.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from qputwin_core import gates as G
from qputwin_core.density import DensityMatrixEngine, apply_readout_confusion
from qputwin_core.devices import DEVICES
from qputwin_core.errors import InvalidRequestError, ResourceLimitError
from qputwin_core.noise import (
    NoiseOptions, depolarizing_parameter, entanglement_fidelity, relaxation_kraus, superoperator,
)
from qputwin_core.simulators import ExecutionOptions, SimulatorConfig, simulator_for

from conftest import apply_to_unitary, random_unitary

H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
VENDORS = sorted(DEVICES)
ONLY_READOUT = {"depolarizing": False, "relaxation": False, "idle": False}


def _within(k: int, n: int, p: float, sigmas: float = 5.0) -> bool:
    return abs(k - n * p) <= sigmas * math.sqrt(max(n * p * (1 - p), 1e-12)) + 1


def _rho(r) -> np.ndarray:
    a = np.array(r.density_matrix)
    return a[..., 0] + 1j * a[..., 1]


def _sv(r) -> np.ndarray:
    a = np.array(r.statevector)
    return a[:, 0] + 1j * a[:, 1]


def _random_rho(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(1 << n, 1 << n)) + 1j * rng.normal(size=(1 << n, 1 << n))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


# ================================================================== channels
@pytest.mark.parametrize("t_ns,t1,t2", [(100.0, 10.0, 5.0), (50.0, 3.0, 6.0), (500.0, None, 2.0),
                                        (500.0, 2.0, None), (0.0, 1.0, 1.0)])
def test_relaxation_kraus_is_cptp_and_matches_t1_t2(t_ns: float, t1, t2) -> None:
    ks = relaxation_kraus(t_ns, t1, t2)
    np.testing.assert_allclose(sum(k.conj().T @ k for k in ks), np.eye(2), atol=1e-14)
    rho = np.array([[0.3, 0.2 + 0.1j], [0.2 - 0.1j, 0.7]])
    out = sum(k @ rho @ k.conj().T for k in ks)
    t = t_ns * 1e-3
    p1 = 0.7 * (math.exp(-t / t1) if t1 else 1.0)
    coh = math.exp(-t / t2) if t2 else (math.exp(-t / (2 * t1)) if t1 else 1.0)
    assert abs(out[1, 1] - p1) < 1e-12
    assert abs(out[0, 1] - (0.2 + 0.1j) * coh) < 1e-12


def test_depolarizing_parameter_hits_published_fidelity_exactly() -> None:
    """F_avg of the composite channel D_p ∘ R equals the target, for d = 2 and d = 4."""
    for d, target in ((2, 0.9993), (4, 0.995), (4, 0.95)):
        s = superoperator(relaxation_kraus(60.0, 200.0, 150.0))     # relaxation error < budget
        fe_r = entanglement_fidelity(s) ** (2 if d == 4 else 1)        # one channel per qubit
        p, limited = depolarizing_parameter(target, d, fe_r)
        assert not limited and 0 < p < 1
        fe_total = (1 - p) * fe_r + p / d ** 2
        assert abs((d * fe_total + 1) / (d + 1) - target) < 1e-13
    p, limited = depolarizing_parameter(0.9999, 2, 0.99)       # relaxation alone too strong
    assert p == 0.0 and limited


def test_depolarizing_composite_channel_via_engine() -> None:
    """Choi-state check: (E ⊗ I)(|Φ⟩⟨Φ|) on a 2-qubit gate gives F_e = ⟨Φ|·|Φ⟩ = target."""
    target_avg, d = 0.98, 4
    s = superoperator(relaxation_kraus(80.0, 10.0, 8.0))
    fe_r = entanglement_fidelity(s) ** 2
    p, _ = depolarizing_parameter(target_avg, d, fe_r)
    phi = np.zeros(16, dtype=complex)                       # Σ_i |i⟩_sys |i⟩_ref, sys = qubits 0,1
    for i in range(4):
        phi[i + 4 * i] = 0.5
    eng = DensityMatrixEngine.from_statevector(phi)
    eng.apply_superop_1q(0, s)
    eng.apply_superop_1q(1, s)
    eng.depolarize((0, 1), p)
    fe = float(np.vdot(phi, eng.matrix @ phi).real)
    assert abs((d * fe + 1) / (d + 1) - target_avg) < 1e-12


# ================================================================== kernels
def test_density_kernels_match_dense_reference(rng: np.random.Generator) -> None:
    n = 4
    rho = _random_rho(n, rng)
    eng = DensityMatrixEngine(n, np.complex128, rho.reshape(-1).copy())
    for step in range(40):
        kind = step % 4
        if kind == 0:
            q = int(rng.integers(n))
            u = random_unitary(2, rng)
            big = apply_to_unitary(np.eye(1 << n, dtype=complex), u, (q,), n)
            rho = big @ rho @ big.conj().T
            eng.apply_unitary((q,), u, None)
        elif kind == 1:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            u = random_unitary(4, rng) if step % 8 == 1 else G.cz()
            big = apply_to_unitary(np.eye(1 << n, dtype=complex), u, (a, b), n)
            rho = big @ rho @ big.conj().T
            eng.apply_unitary((a, b), u, np.diag(u).copy() if G.is_diagonal(u) else None)
        elif kind == 2:
            q = int(rng.integers(n))
            ks = relaxation_kraus(float(rng.uniform(10, 300)), float(rng.uniform(0.5, 2)), float(rng.uniform(0.3, 1)))
            bigs = [apply_to_unitary(np.eye(1 << n, dtype=complex), k, (q,), n) for k in ks]
            rho = sum(k @ rho @ k.conj().T for k in bigs)
            eng.apply_superop_1q(q, superoperator(ks))
        else:
            qs = tuple(int(x) for x in rng.choice(n, int(rng.integers(1, 3)), replace=False))
            p = float(rng.uniform(0, 0.3))
            t = rho.reshape((2,) * (2 * n))
            # reference partial trace + re-embedding with einsum on row/column axes
            letters = "abcdefghijklmnop"
            rows, cols = list(letters[:n]), list(letters[n:2 * n])
            for q in qs:
                cols[n - 1 - q] = rows[n - 1 - q]
            red = np.einsum("".join(rows + cols) + "->" + "".join(
                [r for k, r in enumerate(rows) if (n - 1 - k) not in qs]
                + [c for k, c in enumerate(cols) if (n - 1 - k) not in qs]), t)
            k = len(qs)
            embed = np.zeros_like(t)
            for bits in np.ndindex(*(2,) * k):
                ix = [slice(None)] * (2 * n)
                for q, b in zip(qs, bits, strict=True):
                    ix[n - 1 - q] = b
                    ix[2 * n - 1 - q] = b
                embed[tuple(ix)] = red / (1 << k)
            rho = (1 - p) * rho + p * embed.reshape(rho.shape)
            eng.depolarize(qs, p)
        np.testing.assert_allclose(eng.matrix, rho, atol=1e-12)
    assert abs(eng.trace() - 1) < 1e-12 and abs(eng.purity() - np.trace(rho @ rho).real) < 1e-12


def test_reset_and_projection_channels() -> None:
    psi = np.array([0.6, 0.8j])
    eng = DensityMatrixEngine.from_statevector(psi)
    eng.reset(0)
    np.testing.assert_allclose(eng.matrix, [[1, 0], [0, 0]], atol=1e-15)
    eng = DensityMatrixEngine.from_statevector(psi)
    eng.project(0, 0.5, 2.0)
    np.testing.assert_allclose(eng.matrix, [[0.18, 0], [0, 1.28]], atol=1e-14)


def test_readout_confusion_is_exact() -> None:
    p = np.array([0.5, 0.0, 0.0, 0.5])                  # Bell populations
    out = apply_readout_confusion(p.copy(), [(0.1, 0.2), (0.05, 0.0)])
    a0 = np.array([[0.9, 0.2], [0.1, 0.8]])
    a1 = np.array([[0.95, 0.0], [0.05, 1.0]])
    np.testing.assert_allclose(out, np.kron(a1, a0) @ p, atol=1e-15)


# ======================================================= ideal DM == statevector
@pytest.mark.parametrize("vendor", VENDORS)
def test_ideal_density_matrix_equals_statevector(vendor: str) -> None:
    src = H2 + ("qreg q[3];\nh q[0];\ncx q[0],q[1];\nrx(0.31) q[2];\ncz q[1],q[2];\nu3(0.4,1.1,-0.7) q[0];\n"
                "cx q[1],q[2];\n")
    sim = simulator_for(DEVICES[vendor])
    psi = _sv(sim.execute(src, ExecutionOptions(shots=1, return_statevector=True)))
    r = sim.execute(src, ExecutionOptions(shots=1, return_density_matrix=True))
    np.testing.assert_allclose(_rho(r), np.outer(psi, psi.conj()), atol=1e-12)
    assert r.workload["engine"] == "density_matrix" and r.noise == {"model": "ideal", "engine": "density_matrix",
                                                                  "purity": pytest.approx(1.0), "trace":
                                                                  pytest.approx(1.0)}


def test_ideal_density_matrix_teleportation_statistics() -> None:
    theta = 1.234
    src = H2 + f"""qreg q[3]; creg m0[1]; creg m1[1]; creg out[1];
    ry({theta}) q[0]; h q[1]; cx q[1], q[2]; cx q[0], q[1]; h q[0];
    measure q[0] -> m0[0]; measure q[1] -> m1[0];
    if (m1 == 1) x q[2]; if (m0 == 1) z q[2];
    measure q[2] -> out[0];
    """
    r = simulator_for(DEVICES["ionq_forte"]).execute(src, ExecutionOptions(shots=40_000, seed=9,
                                                                          engine="density_matrix"))
    ones = sum(k for key, k in r.counts.items() if key[0] == "1")
    assert _within(ones, 40_000, math.sin(theta / 2) ** 2)
    assert r.workload["execution_path"] == "density_matrix_branching" and r.workload["branches"] == 4


# ===================================================================== noise
@pytest.mark.parametrize("vendor", VENDORS)
def test_single_gate_error_probability_equals_one_minus_fidelity(vendor: str) -> None:
    """X compiles to one physical pulse; depolarising alone gives P(error) = 1 − F exactly."""
    f, shots = 0.9, 100_000
    r = simulator_for(DEVICES[vendor]).execute(
        H2 + "qreg q[1]; creg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n",
        ExecutionOptions(shots=shots, seed=3, noise={"one_qubit_fidelity": f, "relaxation": False,
                                                     "idle": False, "readout": False}))
    assert r.compilation["native_physical_op_count"] == 1
    assert _within(r.counts.get("0", 0), shots, 1 - f)
    assert r.noise["estimated_success_probability"] == pytest.approx(f)


def test_amplitude_damping_during_gate() -> None:
    """IQM prx takes 40 ns; with T1 = 40 ns, |1⟩ decays with probability 1 − e^{−1}."""
    shots = 100_000
    r = simulator_for(DEVICES["iqm_emerald"]).execute(
        H2 + "qreg q[1]; creg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n",
        ExecutionOptions(shots=shots, seed=5, noise={"t1_us": 0.04, "depolarizing": False, "readout": False}))
    assert _within(r.counts.get("0", 0), shots, 1 - math.exp(-1))


def test_idle_relaxation_until_aligned_final_readout() -> None:
    """q0 idles while q1–q2 run ten 60 ns CZs; the aligned read-out sees T1 decay over 600 ns."""
    shots = 100_000
    body = "x q[0];\n" + "cz q[1],q[2];\n" * 10 + "measure q[0] -> c[0];\n"
    noise = {"t1_us": 0.6, "depolarizing": False, "readout": False}
    sim = simulator_for(DEVICES["iqm_emerald"])
    r = sim.execute(H2 + "qreg q[3]; creg c[1];\n" + body, ExecutionOptions(shots=shots, seed=6, noise=noise))
    est = r.compilation["hardware_time_estimate"]
    assert est["circuit_duration_us"] == pytest.approx(0.6 + 1.5)          # 10 CZ + readout
    assert _within(r.counts.get("0", 0), shots, 1 - math.exp(-1))
    r = sim.execute(H2 + "qreg q[3]; creg c[1];\n" + body,
                    ExecutionOptions(shots=shots, seed=6, noise={**noise, "idle": False}))
    assert _within(r.counts.get("0", 0), shots, 1 - math.exp(-0.04 / 0.6))  # only the 40 ns gate


def test_readout_error_on_bell_state() -> None:
    shots, p = 200_000, 0.03
    r = simulator_for(DEVICES["rigetti_ankaa_3"]).execute(
        H2 + "qreg q[2]; creg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n",
        ExecutionOptions(shots=shots, seed=8, noise=ONLY_READOUT))
    odd = r.counts.get("01", 0) + r.counts.get("10", 0)
    assert _within(odd, shots, 2 * p * (1 - p))


def test_mid_circuit_readout_error_and_feed_forward() -> None:
    """The recorded (possibly wrong) bit drives feed-forward; the true outcome is what collapses."""
    shots, f = 100_000, 0.9
    src = H2 + ("qreg q[2]; creg a[1]; creg b[1];\nx q[0];\nmeasure q[0] -> a[0];\n"
                "if (a == 1) x q[1];\nmeasure q[1] -> b[0];\n")
    r = simulator_for(DEVICES["ionq_forte"]).execute(
        src, ExecutionOptions(shots=shots, seed=4, noise={**ONLY_READOUT, "readout_fidelity": f}))
    expected = {"11": f * f, "01": f * (1 - f), "00": (1 - f) * f, "10": (1 - f) ** 2}   # key = b a
    obs = [r.counts.get(k, 0) for k in expected]
    assert stats.chisquare(obs, [shots * v for v in expected.values()]).pvalue > 1e-3
    # repeated measurement: each read errs independently, but the state collapsed to the true outcome
    src = H2 + "qreg q[1]; creg c[2];\nh q[0];\nmeasure q[0] -> c[0];\nmeasure q[0] -> c[1];\n"
    r = simulator_for(DEVICES["ionq_forte"]).execute(
        src, ExecutionOptions(shots=shots, seed=4, noise={**ONLY_READOUT, "readout_fidelity": f}))
    disagree = r.counts.get("01", 0) + r.counts.get("10", 0)
    assert _within(disagree, shots, 2 * f * (1 - f))


def test_idle_measured_qubit_still_suffers_readout_error() -> None:
    shots, f = 100_000, 0.92
    r = simulator_for(DEVICES["quantinuum_helios"]).execute(
        H2 + "qreg q[3]; creg c[3];\nx q[0];\nmeasure q -> c;\n",
        ExecutionOptions(shots=shots, seed=2, noise={**ONLY_READOUT, "readout_fidelity": f}))
    assert r.compilation["simulated_qubits"] == 1
    flipped_q2 = sum(k for key, k in r.counts.items() if key[0] == "1")
    assert _within(flipped_q2, shots, 1 - f)


def test_reset_is_a_channel_on_the_density_engine() -> None:
    r = simulator_for(DEVICES["iqm_emerald"]).execute(
        H2 + "qreg q[1]; creg c[1];\nh q[0];\nreset q[0];\nmeasure q[0] -> c[0];\n",
        ExecutionOptions(shots=1000, seed=1, engine="density_matrix"))
    assert r.counts == {"0": 1000} and r.workload["execution_path"] == "density_matrix"


def test_scale_zero_is_ideal_and_noise_lowers_purity() -> None:
    sim = simulator_for(DEVICES["rigetti_ankaa_3"])
    ghz = H2 + "qreg q[4]; creg c[4];\nh q[0];\n" + "".join(f"cx q[{k}],q[{k + 1}];\n" for k in range(3)) \
        + "measure q -> c;\n"
    r0 = sim.execute(ghz, ExecutionOptions(shots=2000, seed=1, noise={"scale": 0.0}))
    assert set(r0.counts) == {"0000", "1111"} and r0.noise["purity"] == pytest.approx(1.0)
    purities = [sim.execute(ghz, ExecutionOptions(shots=10, seed=1, noise={"scale": s})).noise["purity"]
                for s in (0.5, 1.0, 4.0)]
    assert 1.0 > purities[0] > purities[1] > purities[2]
    r = sim.execute(ghz, ExecutionOptions(shots=10, seed=1, noise={"model": "device"}))
    assert set(r.noise["per_gate"]) == {"cz", "rx"} and 0 < r.noise["per_gate"]["cz"]["depolarizing_p"] < 1
    assert 0 < r.noise["estimated_success_probability"] < 1


def test_relaxation_limited_gates_are_flagged() -> None:
    r = simulator_for(DEVICES["iqm_emerald"]).execute(
        H2 + "qreg q[2]; creg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n",
        ExecutionOptions(shots=10, seed=1, noise={"t1_us": 0.05, "t2_us": 0.05}))
    assert "cz" in r.noise["relaxation_limited_gates"] and r.noise["per_gate"]["cz"]["depolarizing_p"] == 0.0


def test_density_memory_guard_uses_4_to_the_n() -> None:
    sim = simulator_for(DEVICES["ionq_forte"], SimulatorConfig(memory_budget_bytes=8 << 20))
    src = H2 + "qreg q[10]; creg c[10];\nh q;\nmeasure q -> c;\n"
    sim.execute(src, ExecutionOptions(shots=10))                          # statevector fits easily
    with pytest.raises(ResourceLimitError) as ei:
        sim.execute(src, ExecutionOptions(shots=10, noise={"model": "device"}, engine="density_matrix"))
    d = ei.value.details
    assert d["engine"] == "density_matrix" and d["max_feasible_qubits"] < 10
    sim.execute(H2 + f"qreg q[{d['max_feasible_qubits']}];\nh q;\n",
                ExecutionOptions(shots=1, noise={}, engine="density_matrix"))
    # engine="auto" falls back to quantum trajectories when the density matrix does not fit
    r = sim.execute(src, ExecutionOptions(shots=10, noise={"model": "device"}))
    assert r.workload["engine"] == "trajectory" and r.noise["exact_sampling"]


@pytest.mark.parametrize("opts", [
    {"noise": {"model": "device"}, "engine": "statevector"},
    {"noise": {"model": "device"}, "return_statevector": True},
    {"return_density_matrix": True, "engine": "statevector"},
    {"noise": {"model": "bogus"}},
    {"noise": {"scale": -1}},
    {"noise": {"t1_us": 1.0, "t2_us": 5.0}},
    {"noise": {"colour": "pink"}},
    {"engine": "tensor_network"},
])
def test_noise_request_validation(opts: dict) -> None:
    with pytest.raises(InvalidRequestError):
        simulator_for(DEVICES["ionq_forte"]).execute(H2 + "qreg q[1];\nh q[0];\n", ExecutionOptions(**opts))


def test_noise_options_parsing() -> None:
    assert NoiseOptions.from_dict(None) is None and NoiseOptions.from_dict({"model": "ideal"}) is None
    o = NoiseOptions.from_dict({"scale": 2, "readout": False, "two_qubit_fidelity": 0.99})
    assert o is not None and o.scale == 2.0 and not o.readout and o.two_qubit_fidelity == 0.99
