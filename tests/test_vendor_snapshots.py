"""Tests the IBM and Google twins built from the vendor snapshots.

Usage: uv run pytest tests/test_vendor_snapshots.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qputwin_core import gates as G
from qputwin_core.devices import DEVICES
from qputwin_core.errors import NativeGateError, TopologicalCompilationError
from qputwin_core.simulators import ExecutionOptions, simulator_for

H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
NEW = ("ibm_heron", "ibm_nighthawk", "google_willow")


def _cirq_phased_xz(x: float, z: float, a: float) -> np.ndarray:
    """cirq.PhasedXZGate(x, z, a) = Z^z · Z^a · X^x · Z^-a, with Cirq's XPow / ZPow conventions."""
    def zpow(t: float) -> np.ndarray:
        return np.diag([1, np.exp(1j * math.pi * t)])

    def xpow(t: float) -> np.ndarray:
        g = np.exp(1j * math.pi * t / 2)
        c, s = math.cos(math.pi * t / 2), math.sin(math.pi * t / 2)
        return g * np.array([[c, -1j * s], [-1j * s, c]])

    return zpow(z) @ zpow(a) @ xpow(x) @ zpow(-a)


def test_phased_xz_matches_cirq_definition() -> None:
    rng = np.random.default_rng(5)
    for _ in range(200):
        theta, phi, zeta = rng.uniform(-math.pi, math.pi, 3)
        ours = G.phased_xz(theta, phi, zeta)
        cirq = _cirq_phased_xz(theta / math.pi, zeta / math.pi, phi / math.pi)
        assert G.allclose_up_to_phase(ours, cirq, atol=1e-12)


@pytest.mark.parametrize(("device", "qubits", "couplers", "system"), [
    ("ibm_heron", 156, 175, "ibm_boston"),
    ("ibm_nighthawk", 120, 211, "ibm_miami"),
    ("google_willow", 105, 182, "willow_pink"),
])
def test_published_topology(device: str, qubits: int, couplers: int, system: str) -> None:
    spec = DEVICES[device]
    cmap = spec.coupling
    assert spec.n_physical == cmap.n_qubits == qubits and cmap.n_edges == couplers
    assert len(cmap.components()) == 1 and len(cmap.components()[0]) == qubits
    assert system in cmap.provenance or system in str(cmap.layout)
    assert cmap.degree_stats()["max"] <= 4


def test_lattice_shapes() -> None:
    heron = DEVICES["ibm_heron"].coupling
    assert heron.degree_stats()["max"] == 3                       # heavy-hex: degree 2 or 3
    assert heron.layout["inactive_couplers"] == [[84, 85]]
    nighthawk = DEVICES["ibm_nighthawk"].coupling
    rows, cols = nighthawk.layout["rows"], nighthawk.layout["cols"]
    assert (rows, cols) == (12, 10)
    on_chip = {tuple(e) for e in nighthawk.edges} | {tuple(e) for e in nighthawk.layout["inactive_couplers"]}
    lattice = {(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)} | \
              {(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)}
    assert on_chip == lattice and len(lattice) == 218             # IBM: 218 tunable couplers
    willow = DEVICES["google_willow"].coupling
    coords = willow.layout["coordinates"]
    assert coords[0] == [0, 6] and coords[-1] == [12, 8]
    for a, b in willow.edges:                                      # grid neighbours only
        assert abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) == 1
    assert 2 * willow.n_edges / willow.n_qubits == pytest.approx(3.47, abs=0.005)   # spec sheet


def test_snapshot_values_are_per_qubit_and_per_coupler() -> None:
    heron = DEVICES["ibm_heron"].calibration
    assert heron.snapshot_date.startswith("2026-04-17") and heron.shot_overhead_us == 250.0
    assert len(heron.qubits) == 156 and len(heron.couplers) == 175
    assert heron.gate_duration_ns("cz") == 68.0 and heron.gate_duration_ns("sx") == 32.0
    assert heron.readout_duration_ns == 2200.0
    fids = {heron.gate_fidelity("cz", tuple(map(int, k.split("-")))) for k in heron.couplers}
    assert len(fids) > 100                                         # not one number copied everywhere
    nighthawk = DEVICES["ibm_nighthawk"].calibration
    assert nighthawk.shot_overhead_us == 4000.0                    # IBM: rep delay raised to 4 ms
    assert 300 < nighthawk.t1_us < 400                             # IBM: median T1 ~ 350 us
    willow = DEVICES["google_willow"].calibration
    assert willow.gate_duration_ns("cz") == 42.0 and willow.gate_duration_ns("phased_xz") == 25.0
    assert willow.readout_duration_ns == 600.0 and willow.reset_duration_ns == 160.0
    assert len(willow.qubits) == 105 and len(willow.couplers) == 182
    assert 60 < willow.t1_us < 80                                  # spec sheet: 68 +/- 13 us
    for name in NEW:
        cal = DEVICES[name].calibration
        assert cal.provenance_class("gates.cz.fidelity") == "published"
        assert cal.provenance_class("coherence.t1_us") == "published"


@pytest.mark.parametrize("device", NEW)
def test_ghz_compiles_to_natives_on_a_coupled_path(device: str) -> None:
    spec = DEVICES[device]
    path = [0]
    while len(path) < 6:
        path.append(next(q for q in spec.coupling.neighbors(path[-1]) if q not in path))
    lines = [f"h q[{path[0]}];"] + [f"cx q[{a}],q[{b}];" for a, b in zip(path, path[1:], strict=False)]
    src = H2 + f"qreg q[{max(path) + 1}];\ncreg c[{max(path) + 1}];\n" + "\n".join(lines) + "\nmeasure q -> c;\n"
    r = simulator_for(spec).execute(src, ExecutionOptions(shots=2000, seed=1))
    assert set(r.compilation["native_gate_histogram"]) <= set(spec.native_gates)
    assert r.compilation["native_gate_histogram"]["cz"] == 5
    ones = "".join("1" if q in path else "0" for q in reversed(range(max(path) + 1)))
    assert set(r.counts) == {"0" * len(ones), ones}


def test_uncoupled_gate_is_rejected_and_natives_pass_strict_mode() -> None:
    with pytest.raises(TopologicalCompilationError):
        simulator_for(DEVICES["ibm_heron"]).execute(H2 + "qreg q[3];\ncz q[0],q[2];\n", ExecutionOptions(shots=1))
    ok = H2 + "qreg q[2];\nsx q[0];\nx q[1];\nrz(0.3) q[0];\ncz q[0],q[1];\n"
    r = simulator_for(DEVICES["ibm_nighthawk"]).execute(ok, ExecutionOptions(shots=1, compilation="strict"))
    assert r.compilation["native_gate_histogram"] == {"cz": 1, "rz": 1, "sx": 1, "x": 1}
    with pytest.raises(NativeGateError):
        simulator_for(DEVICES["ibm_heron"]).execute(H2 + "qreg q[1];\nh q[0];\n",
                                                    ExecutionOptions(shots=1, compilation="strict"))
    w = H2 + "qreg q[2];\nphased_xz(0.4, 1.0, -0.2) q[0];\nrz(0.1) q[1];\n"
    r = simulator_for(DEVICES["google_willow"]).execute(w, ExecutionOptions(shots=1, compilation="strict"))
    assert r.compilation["native_gate_histogram"] == {"phased_xz": 1, "rz": 1}


def test_one_qubit_gate_costs_one_willow_pulse() -> None:
    r = simulator_for(DEVICES["google_willow"]).execute(H2 + "qreg q[1];\nu3(0.3, 0.7, -1.1) q[0];\n",
                                                        ExecutionOptions(shots=1))
    assert r.compilation["native_physical_op_count"] == 1
    r = simulator_for(DEVICES["ibm_heron"]).execute(H2 + "qreg q[1];\nu3(0.3, 0.7, -1.1) q[0];\n",
                                                    ExecutionOptions(shots=1))
    assert r.compilation["native_gate_histogram"]["sx"] == 2


def test_fsim_matches_cirq_definition() -> None:
    """cirq.FSimGate(θ, φ): |01⟩,|10⟩ block [[cos θ, −i sin θ], [−i sin θ, cos θ]], |11⟩ phase e^{−iφ}."""
    theta, phi = 0.31, -0.7
    m = G.fsim(theta, phi)
    expected = np.diag([1, 0, 0, np.exp(-1j * phi)]).astype(complex)
    expected[1:3, 1:3] = [[math.cos(theta), -1j * math.sin(theta)], [-1j * math.sin(theta), math.cos(theta)]]
    np.testing.assert_allclose(m, expected, atol=1e-15)
    np.testing.assert_allclose(m @ m.conj().T, np.eye(4), atol=1e-14)


def test_willow_cz_channel_reproduces_each_couplers_fidelity_with_fsim_errors() -> None:
    """V = PhasedFSim(θ, φ) ∘ R (T1/T2) ∘ D_p hits each coupler's published total fidelity exactly."""
    from qputwin_core.density import DensityMatrixEngine
    from qputwin_core.noise import NoiseModel, NoiseOptions, superoperator
    from qputwin_core.qasm import load_circuit
    from qputwin_core.synthesis import NativeCompiler

    spec = DEVICES["google_willow"]
    cal = spec.calibration
    checked = 0
    for key in ("0-1", "0-4", "50-51"):
        if key not in cal.couplers:
            continue
        a, b = (int(x) for x in key.split("-"))
        g_cal = cal.gate("cz", (a, b))
        assert g_cal.fsim_theta_rad is not None and g_cal.fsim_phi_rad is not None
        model = NoiseModel(cal, NoiseOptions(), spec.coupling)
        native = NativeCompiler(spec).compile(load_circuit(H2 + f"qreg q[{b + 1}];\ncz q[{a}],q[{b}];\n",
                                                           primitives={"cz", "U", "CX", "gphase"}))
        op = next(o for o in native.ops if o.name == "cz")
        g = model.gate_noise(op)
        assert g.coherent is not None
        phi = np.zeros(16, dtype=complex)
        for i in range(4):
            phi[i + 4 * i] = 0.5
        eng = DensityMatrixEngine.from_statevector(phi)
        eng.apply_unitary((1, 0), g.coherent, None)
        for q, k in g.channels:
            eng.apply_superop_1q({a: 1, b: 0}[q], superoperator(k))
        eng.depolarize((0, 1), g.p)
        fe = float(np.vdot(phi, eng.matrix @ phi).real)
        assert abs((4 * fe + 1) / 5 - cal.gate_fidelity("cz", (a, b))) < 1e-10
        checked += 1
    assert checked >= 2


def test_willow_angle_errors_are_part_of_the_published_total() -> None:
    """As in Google's GoogleNoiseProperties: the XEB error is the CZ total, the PhasedFSim angles a coherent part."""
    from qputwin_core.noise import NoiseModel, NoiseOptions
    from qputwin_core.qasm import load_circuit
    from qputwin_core.synthesis import NativeCompiler

    spec = DEVICES["google_willow"]
    cal = spec.calibration
    limited = 0
    for key in cal.couplers:
        a, b = (int(x) for x in key.split("-"))
        g = cal.gate("cz", (a, b))
        fe_v = abs(np.trace(G.fsim(g.fsim_theta_rad, g.fsim_phi_rad))) ** 2 / 16
        coherent_avg_error = 1 - (4 * fe_v + 1) / 5
        limited += coherent_avg_error > 1 - g.fidelity            # the angles alone exceed the total
    assert limited <= 0.1 * len(cal.couplers), limited
    native = NativeCompiler(spec).compile(load_circuit(H2 + "qreg q[2];\ncz q[0],q[1];\n",
                                                       primitives={"cz", "U", "CX", "gphase"}))
    g = NoiseModel(cal, NoiseOptions(), spec.coupling).gate_noise(next(o for o in native.ops if o.name == "cz"))
    assert g.coherent is not None and g.p > 0
