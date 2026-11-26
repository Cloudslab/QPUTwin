"""Tests the extended noise terms: coherent errors, crosstalk, reset, readout and leakage.

Usage: uv run pytest tests/test_noise_extended.py

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qputwin_core.calibration import Calibration
from qputwin_core.density import DensityMatrixEngine
from qputwin_core.devices import DEVICES, IONQ_FORTE, IQM_EMERALD
from qputwin_core.errors import CalibrationProvenanceError, InvalidRequestError
from qputwin_core.noise import NoiseModel, NoiseOptions, superoperator
from qputwin_core.scheduling import asap_schedule
from qputwin_core.simulators import ExecutionOptions, SimulatorConfig, simulator_for
from qputwin_core.synthesis import NativeCompiler, NativeOp
from qputwin_core.qasm import load_circuit

H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
OFF = {"depolarizing": False, "relaxation": False, "idle": False, "readout": False, "coherent": False,
       "crosstalk": False, "reset_errors": False, "correlated_readout": False, "leakage": False,
       "quasi_static": False}


def _within(k: int, n: int, p: float, sigmas: float = 5.0) -> bool:
    return abs(k - n * p) <= sigmas * math.sqrt(max(n * p * (1 - p), 1e-12)) + 1


def _ones(counts: dict[str, int], bit: int) -> int:
    return sum(k for key, k in counts.items() if key[::-1][bit] == "1")


def _tv(a: dict[str, int], b: dict[str, float]) -> float:
    n = sum(a.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / n - b.get(k, 0.0)) for k in keys)


def _with_cal(spec, **extra):
    base = spec.calibration.to_dict()
    doc = {"device": spec.id, "gates": {g: {k: v for k, v in gv.items() if v is not None}
                                        for g, gv in base["gates"].items()},
           "readout": {"fidelity": 1 - base["readout"]["p01"], "duration_ns": base["readout"]["duration_ns"]},
           "reset": {"duration_ns": base["reset"]["duration_ns"]}, "coherence": base["coherence"]}
    for k, v in extra.items():
        doc[k] = {**doc.get(k, {}), **v} if isinstance(v, dict) and isinstance(doc.get(k), dict) else v
    return spec.with_overrides(calibration=Calibration.from_dict(doc, origin="test"))


# =================================================================== coherent
def test_overrotation_rotates_by_theta_times_one_plus_epsilon() -> None:
    eps, shots = 0.2, 100_000
    r = simulator_for(IQM_EMERALD).execute(H2 + "qreg q[1]; creg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n",
                                           ExecutionOptions(shots=shots, seed=1,
                                                            noise={**OFF, "coherent": True, "overrotation": eps}))
    assert _within(r.counts.get("0", 0), shots, math.sin(math.pi * eps / 2) ** 2)
    assert "coherent" in r.noise["features"] and r.noise["per_gate"]["prx"]["coherent"]


def test_composite_error_channel_still_matches_published_fidelity() -> None:
    """V (coherent) ∘ R (T1/T2) ∘ D_p together hit the calibrated average fidelity exactly."""
    spec = IQM_EMERALD
    model = NoiseModel(spec.calibration, NoiseOptions(overrotation=0.03, coherent_z_rad=0.02), spec.coupling)
    native = NativeCompiler(spec).compile(load_circuit(H2 + "qreg q[2];\ncz q[0],q[1];\n",
                                                       primitives={"cz", "U", "CX", "gphase"}))
    op: NativeOp = next(o for o in native.ops if o.name == "cz")
    g = model.gate_noise(op)
    assert g.coherent is not None and g.channels and g.p > 0
    phi = np.zeros(16, dtype=complex)                       # |Φ⟩ over (system qubits 0,1) ⊗ (reference 2,3)
    for i in range(4):
        phi[i + 4 * i] = 0.5
    eng = DensityMatrixEngine.from_statevector(phi)
    eng.apply_unitary((1, 0), g.coherent, None)             # op.qubits = (0, 1): first listed = MSB
    for q, k in g.channels:
        eng.apply_superop_1q(q, superoperator(k))
    eng.depolarize((0, 1), g.p)
    fe = float(np.vdot(phi, eng.matrix @ phi).real)
    assert abs((4 * fe + 1) / 5 - 0.995) < 1e-10


# =================================================================== crosstalk
def test_spectator_zz_crosstalk_phase() -> None:
    """q2 (neighbour of gate qubit 1, which is |0⟩) acquires Rz(θ): an X-basis read gives sin²(θ/2)."""
    theta, shots = 0.6, 100_000
    src = H2 + "qreg q[3]; creg c[1];\nh q[2];\nbarrier q;\ncz q[0],q[1];\nbarrier q;\nh q[2];\nmeasure q[2] -> c[0];\n"
    r = simulator_for(IQM_EMERALD).execute(src, ExecutionOptions(
        shots=shots, seed=2, noise={**OFF, "crosstalk": True, "crosstalk_zz_rad": theta}))
    assert _within(r.counts.get("1", 0), shots, math.sin(theta / 2) ** 2)
    assert "crosstalk" in r.noise["features"]


# ===================================================== reset + correlated readout
def test_reset_error() -> None:
    shots = 100_000
    r = simulator_for(IQM_EMERALD).execute(
        H2 + "qreg q[1]; creg c[1];\nx q[0];\nreset q[0];\nmeasure q[0] -> c[0];\n",
        ExecutionOptions(shots=shots, seed=3, noise={**OFF, "reset_errors": True, "reset_fidelity": 0.9}))
    assert _within(r.counts.get("1", 0), shots, 0.1)


def test_correlated_readout_uses_the_joint_matrix() -> None:
    m = [[0.90, 0.05, 0.05, 0.01], [0.04, 0.90, 0.01, 0.05], [0.04, 0.01, 0.90, 0.04], [0.02, 0.04, 0.04, 0.90]]
    spec = _with_cal(IQM_EMERALD, readout_correlations=[{"qubits": [0, 1], "matrix": m}])
    shots = 200_000
    r = simulator_for(spec).execute(H2 + "qreg q[2]; creg c[2];\nx q[1];\nmeasure q -> c;\n",
                                    ExecutionOptions(shots=shots, seed=4, noise={**OFF, "readout": True,
                                                                                 "correlated_readout": True}))
    # true state |q1 q0⟩ = |10⟩ → index 2 (bit k ↔ qubits[k]); recorded distribution = column 2 of M
    for idx in range(4):
        key = format(idx, "02b")
        assert _within(r.counts.get(key, 0), shots, m[idx][2]), key
    assert "correlated_readout" in r.noise["features"]


# ======================================================= trajectory ↔ density
@pytest.mark.parametrize("src", [
    H2 + "qreg q[3]; creg c[3];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\nrx(0.7) q[2];\nmeasure q -> c;\n",
    H2 + ("qreg q[3]; creg a[1]; creg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> a[0];\n"
          "if (a == 1) x q[2];\nreset q[0];\nh q[0];\nmeasure q[1] -> c[0];\nmeasure q[2] -> c[1];\n"),
])
def test_trajectories_agree_with_the_exact_density_matrix(src: str) -> None:
    m = [[0.95, 0.03, 0.03, 0.0], [0.02, 0.95, 0.0, 0.03], [0.03, 0.0, 0.95, 0.02], [0.0, 0.02, 0.02, 0.95]]
    spec = _with_cal(IQM_EMERALD, readout_correlations=[{"qubits": [1, 2], "matrix": m}],
                     reset={"fidelity": 0.97})
    noise = {"scale": 3.0, "overrotation": 0.05, "crosstalk_zz_rad": 0.2}
    sim = simulator_for(spec, SimulatorConfig(max_trajectories=6000))
    exact = sim.execute(src, ExecutionOptions(shots=1_000_000, seed=5, noise=noise, engine="density_matrix"))
    p = {k: v / 1_000_000 for k, v in exact.counts.items()}
    shots = 6000
    traj = sim.execute(src, ExecutionOptions(shots=shots, seed=6, noise=noise, engine="trajectory"))
    assert traj.workload["engine"] == "trajectory" and traj.noise["exact_sampling"]
    assert traj.noise["trajectories"] == shots and traj.noise["jumps"] > 0
    assert _tv(traj.counts, p) < 0.03, (traj.counts, p)


def test_auto_engine_choice_and_trajectory_budget() -> None:
    sim = simulator_for(IQM_EMERALD, SimulatorConfig(density_auto_max_qubits=3, max_trajectories=50))
    small = sim.execute(H2 + "qreg q[3];\nh q[0];\n", ExecutionOptions(shots=100, noise={}))
    wide = sim.execute(H2 + "qreg q[4];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\n",
                       ExecutionOptions(shots=100, noise={}))
    assert small.workload["engine"] == "density_matrix" and wide.workload["engine"] == "trajectory"
    assert wide.noise["trajectories"] == 50 and not wide.noise["exact_sampling"]
    assert wide.noise["shots_per_trajectory"] == [2, 2]
    with pytest.raises(InvalidRequestError, match="QPUTWIN_MAX_TRAJECTORIES"):
        sim.execute(H2 + "qreg q[1];\nh q[0];\n", ExecutionOptions(shots=100, noise={"trajectories": 51}))
    with pytest.raises(InvalidRequestError):
        sim.execute(H2 + "qreg q[1];\nh q[0];\n", ExecutionOptions(engine="trajectory"))


def test_wide_noisy_circuit_runs_on_trajectories() -> None:
    """16 noisy qubits: a 4^16 density matrix (64 GiB) is out of reach, trajectories are not."""
    n = 16
    src = H2 + f"qreg q[{n}]; creg c[{n}];\nh q[0];\n" + "".join(f"cx q[{k}],q[{k + 1}];\n" for k in range(n - 1)) \
        + "measure q -> c;\n"
    r = simulator_for(DEVICES["quantinuum_helios"]).execute(src, ExecutionOptions(shots=200, seed=1, noise={}))
    assert r.workload["engine"] == "trajectory" and r.compilation["simulated_qubits"] == n
    ghz = r.counts.get("0" * n, 0) + r.counts.get("1" * n, 0)
    assert 0.8 * 200 < ghz < 200                                  # noisy, but still mostly GHZ


# ===================================================================== leakage
def test_leakage_partner_depolarisation_and_readout() -> None:
    lrate, shots = 0.1, 20_000
    src = H2 + "qreg q[2]; creg c[2];\nx q[0];\ncz q[0],q[1];\nx q[0];\nmeasure q -> c;\n"
    r = simulator_for(IQM_EMERALD).execute(src, ExecutionOptions(
        shots=shots, seed=7, noise={**OFF, "leakage": True, "leakage_rate": lrate}))
    assert r.workload["engine"] == "trajectory" and r.noise["leak_events"] > 0
    assert _within(_ones(r.counts, 0), shots, 1 - (1 - lrate) ** 3)            # q0 leaked at some point
    assert _within(_ones(r.counts, 1), shots, 1 - (1 - lrate) * (1 - lrate / 2))   # own leak or partner Pauli
    with pytest.raises(InvalidRequestError, match="trajectory"):
        simulator_for(IQM_EMERALD).execute(src, ExecutionOptions(
            shots=10, noise={**OFF, "leakage": True, "leakage_rate": lrate}, engine="density_matrix"))


def test_seepage_returns_leaked_qubits() -> None:
    src = H2 + "qreg q[2]; creg c[2];\nx q[0];\n" + "cz q[0],q[1];\n" * 20 + "x q[0];\nmeasure q -> c;\n"
    sim = simulator_for(IQM_EMERALD)
    stuck = sim.execute(src, ExecutionOptions(shots=4000, seed=8, noise={**OFF, "leakage": True, "leakage_rate": 0.05}))
    back = sim.execute(src, ExecutionOptions(shots=4000, seed=8, noise={**OFF, "leakage": True, "leakage_rate": 0.05,
                                                                       "seepage_rate": 0.5}))
    assert _ones(back.counts, 0) < _ones(stuck.counts, 0)


# ====================================================== quasi-static detuning
def test_quasi_static_detuning_ramsey_decay() -> None:
    """Ramsey over t = 600 ns with δ ~ N(0, σ): P(1) = (1 − exp(−(2πσt)²/2)) / 2 (Gaussian, not exponential)."""
    sigma_khz, t_ns, shots = 300.0, 600.0, 10_000
    src = H2 + ("qreg q[3]; creg c[1];\nh q[0];\n" + "cz q[1],q[2];\n" * 10
                + "barrier q[0],q[1],q[2];\nh q[0];\nmeasure q[0] -> c[0];\n")
    r = simulator_for(IQM_EMERALD).execute(src, ExecutionOptions(
        shots=shots, seed=9, noise={**OFF, "quasi_static": True, "quasi_static_khz": sigma_khz}))
    x = 2 * math.pi * sigma_khz * 1e3 * t_ns * 1e-9
    assert _within(r.counts.get("1", 0), shots, (1 - math.exp(-x * x / 2)) / 2)
    assert "quasi_static" in r.noise["features"] and r.workload["engine"] == "trajectory"


# =================================================== fixed-timeline idle slots
def test_unfired_conditional_gate_still_occupies_its_slot() -> None:
    """q0 relaxes for 40 + 1460 + 40 ns: the conditional X slot counts even though it never fires."""
    t1_us, shots = 1.54, 200_000
    src = H2 + "qreg q[2]; creg c[1]; creg o[1];\nx q[0];\nmeasure q[1] -> c[0];\nif (c == 1) x q[0];\n" \
                "measure q[0] -> o[0];\n"
    r = simulator_for(IQM_EMERALD).execute(src, ExecutionOptions(
        shots=shots, seed=10, noise={**OFF, "relaxation": True, "idle": True, "t1_us": t1_us}))
    decayed = sum(k for key, k in r.counts.items() if key[0] == "0")
    assert _within(decayed, shots, 1 - math.exp(-1.54 / t1_us))


# ==================================================================== schedule
def test_parallelism_limit_and_qccd_transport() -> None:
    cal = IONQ_FORTE.calibration
    assert cal.max_parallel_two_qubit == 1
    ops = [NativeOp("ms", (0, 1), (), None, None), NativeOp("ms", (2, 3), (), None, None)]
    s = asap_schedule(ops, cal)
    assert s.starts_ns == [0.0, 600_000.0]                        # serialised on one motional bus
    qccd = Calibration.from_dict({"device": "t", "gates": {"zzmax": {"duration_ns": 100}},
                                  "qccd": {"transport_ns_per_two_qubit_gate": 300, "cooling_ns_per_two_qubit_gate": 50}})
    s = asap_schedule([NativeOp("zzmax", (0, 1), (), None, None)], qccd)
    assert s.starts_ns == [350.0] and s.makespan_ns == 450.0 and s.busy_ns["transport"] == 350.0


# =================================================================== provenance
def test_noise_report_lists_assumed_parameters() -> None:
    r = simulator_for(IQM_EMERALD).execute(H2 + "qreg q[2]; creg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n",
                                           ExecutionOptions(shots=10, seed=1, noise={}))
    used = r.noise["parameters_used"]
    assert "gates.cz.fidelity" in used["published"] and "coherence.t1_us" in used["published"]
    assert "gates.cz.duration_ns" in r.noise["assumed_parameters"]
    assert "readout.fidelity" in r.noise["assumed_parameters"]
    assert "gates.cz.duration_ns" in r.compilation["hardware_time_estimate"]["assumed_parameters"]


def test_strict_provenance_mode() -> None:
    sim = simulator_for(IQM_EMERALD, SimulatorConfig(require_published_calibration=True))
    bell = H2 + "qreg q[2]; creg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"
    with pytest.raises(CalibrationProvenanceError) as ei:
        sim.execute(bell, ExecutionOptions(shots=10, noise={}))
    assert "gates.cz.duration_ns" in ei.value.details["unpublished_parameters"]
    # only published numbers (or explicit request overrides) are used → allowed
    r = sim.execute(bell, ExecutionOptions(shots=10, noise={"relaxation": False, "idle": False,
                                                            "readout_fidelity": 0.97}))
    assert r.noise["assumed_parameters"] == []
    sim.execute(bell, ExecutionOptions(shots=10))                         # ideal runs are never blocked


def test_ionq_forte_provenance_is_honest() -> None:
    cal = IONQ_FORTE.calibration
    assert cal.provenance_class("gates.ms.fidelity") == "published"
    for path in ("gates.ms.duration_ns", "gates.gpi.duration_ns", "coherence.t1_us"):
        assert cal.provenance_class(path) == "assumed" and "Aria" in cal.provenance_of(path)
