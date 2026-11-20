"""Tests the QPU emulators: topology checks, resource limits and the memory guard.

Usage: uv run pytest tests/test_simulators.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qputwin_core.devices import DEVICES
from qputwin_core.errors import InvalidRequestError, ResourceLimitError, TopologicalCompilationError
from qputwin_core.simulators import (
    ExecutionOptions, SimulatorConfig, SuperconductingSimulator, TrappedIonSimulator, simulator_for,
)
from qputwin_core.timing import SpanTimer
from qputwin_core.topology import CouplingMap

H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def star_ghz(n: int) -> str:
    return H2 + f"qreg q[{n}];\ncreg c[{n}];\nh q[0];\n" + "".join(
        f"cx q[0],q[{k}];\n" for k in range(1, n)) + "measure q -> c;\n"


def chain_ghz(n: int) -> str:
    return H2 + f"qreg q[{n}];\ncreg c[{n}];\nh q[0];\n" + "".join(
        f"cx q[{k}],q[{k + 1}];\n" for k in range(n - 1)) + "measure q -> c;\n"


@pytest.mark.parametrize("vendor", ["rigetti_ankaa_3", "iqm_emerald"])
def test_grid_adjacency_properties(vendor: str) -> None:
    cmap = DEVICES[vendor].coupling
    a = cmap.adjacency
    rows, cols = cmap.layout["rows"], cmap.layout["cols"]
    assert a.shape == (rows * cols, rows * cols) == (DEVICES[vendor].n_physical,) * 2
    assert (a == a.T).all() and not a.diagonal().any()
    assert a.sum(axis=1).max() == 4 and a.sum(axis=1).min() == 2
    assert cmap.n_edges == rows * (cols - 1) + cols * (rows - 1)
    for p in range(rows * cols):
        for q in np.flatnonzero(a[p]):
            (r1, c1), (r2, c2) = divmod(p, cols), divmod(int(q), cols)
            assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_simulator_classes() -> None:
    assert isinstance(simulator_for(DEVICES["quantinuum_helios"]), TrappedIonSimulator)
    assert isinstance(simulator_for(DEVICES["ionq_forte"]), TrappedIonSimulator)
    assert isinstance(simulator_for(DEVICES["rigetti_ankaa_3"]), SuperconductingSimulator)
    assert isinstance(simulator_for(DEVICES["iqm_emerald"]), SuperconductingSimulator)
    assert CouplingMap.complete(98).coupled(0, 97) and not CouplingMap.complete(98).coupled(3, 3)


@pytest.mark.parametrize("vendor", ["quantinuum_helios", "ionq_forte"])
def test_trapped_ion_accepts_star_ghz(vendor: str) -> None:
    r = simulator_for(DEVICES[vendor]).execute(star_ghz(12), ExecutionOptions(shots=4000, seed=5,
                                                                            return_statevector=True))
    assert set(r.counts) == {"0" * 12, "1" * 12}
    sv = np.array(r.statevector)
    amps = sv[:, 0] + 1j * sv[:, 1]
    assert abs(amps[0] - 1 / math.sqrt(2)) < 1e-9 and abs(amps[-1] - 1 / math.sqrt(2)) < 1e-9
    assert abs(np.linalg.norm(amps) - 1) < 1e-12


@pytest.mark.parametrize("vendor", ["rigetti_ankaa_3", "iqm_emerald"])
def test_superconducting_rejects_star_ghz(vendor: str) -> None:
    sim = simulator_for(DEVICES[vendor])
    timer = SpanTimer()
    with pytest.raises(TopologicalCompilationError) as ei:
        sim.execute(star_ghz(12), ExecutionOptions(shots=10), timer)
    d = ei.value.details
    p, q = d["qubits"]
    assert sim.adjacency[p, q] == 0 and d["adjacency_entry"] == 0
    assert d["gate"] == "cx" and d["line"] == 7 and d["statement"] == "cx q[0],q[2];"
    assert q not in d["neighbors"][str(p)]
    assert "simulation" not in timer.spans_ns        # engine never invoked


@pytest.mark.parametrize("vendor", ["rigetti_ankaa_3", "iqm_emerald"])
def test_superconducting_accepts_nearest_neighbour_chain(vendor: str) -> None:
    r = simulator_for(DEVICES[vendor]).execute(chain_ghz(8), ExecutionOptions(shots=2000, seed=2))
    assert set(r.counts) == {"0" * 8, "1" * 8}


def test_superconducting_three_qubit_gate_needs_triangle() -> None:
    with pytest.raises(TopologicalCompilationError) as ei:
        simulator_for(DEVICES["iqm_emerald"]).execute(H2 + "qreg q[3];\nccx q[0],q[1],q[2];\n")
    assert ei.value.details["gate"] == "ccx"


def test_physical_index_and_declared_size_limits() -> None:
    sim = simulator_for(DEVICES["ionq_forte"])
    with pytest.raises(ResourceLimitError, match="36 physical qubits"):
        sim.execute(H2 + "qreg q[40];\nh q[0];\n")
    # a large register is fine as long as it fits the device: only touched qubits are simulated
    r = simulator_for(DEVICES["quantinuum_helios"]).execute(
        H2 + "qreg q[98];\ncreg c[2];\nh q[90];\ncx q[90],q[3];\nmeasure q[90] -> c[0];\nmeasure q[3] -> c[1];\n",
        ExecutionOptions(shots=500, seed=1))
    assert r.compilation["simulated_qubits"] == 2 and set(r.counts) <= {"00", "11"}


def test_max_sim_qubits_cap() -> None:
    sim = simulator_for(DEVICES["quantinuum_helios"], SimulatorConfig(max_sim_qubits=5))
    assert sim.max_sim_qubits == 5
    with pytest.raises(ResourceLimitError) as ei:
        sim.execute(star_ghz(6))
    assert ei.value.details["max_sim_qubits"] == 5
    assert simulator_for(DEVICES["ionq_forte"]).max_sim_qubits == 36     # default = vendor size


def test_memory_guard_rejects_infeasible_width() -> None:
    sim = simulator_for(DEVICES["ionq_forte"], SimulatorConfig(memory_budget_bytes=64 * 1024))
    with pytest.raises(ResourceLimitError) as ei:
        sim.execute(star_ghz(14))
    d = ei.value.details
    assert d["required_bytes"] > d["available_bytes"] and d["max_feasible_qubits"] < 14
    sim.execute(star_ghz(d["max_feasible_qubits"]))                      # the advertised width fits


def test_tight_memory_budget_switches_the_executor_in_place() -> None:
    """Between 1.5× and 2× the state the run still fits, in place, with the same counts as the default."""
    n = 12
    qasm = star_ghz(n)
    state = (1 << n) * 16
    roomy = simulator_for(DEVICES["ionq_forte"]).execute(qasm, ExecutionOptions(shots=2000, seed=9))
    tight_sim = simulator_for(DEVICES["ionq_forte"], SimulatorConfig(memory_budget_bytes=int(1.75 * state),
                                                                      memory_reserve_bytes=0))
    tight = tight_sim.execute(qasm, ExecutionOptions(shots=2000, seed=9))
    assert tight.workload["in_place"] is True
    assert roomy.workload["in_place"] is False
    assert tight.counts == roomy.counts
    with pytest.raises(ResourceLimitError):                              # below 1.5× nothing fits
        simulator_for(DEVICES["ionq_forte"], SimulatorConfig(memory_budget_bytes=int(1.4 * state),
                                                             memory_reserve_bytes=0)).execute(qasm)


def test_request_validation() -> None:
    sim = simulator_for(DEVICES["ionq_forte"])
    with pytest.raises(InvalidRequestError):
        sim.execute(star_ghz(2), ExecutionOptions(shots=0))
    with pytest.raises(InvalidRequestError):
        sim.execute(star_ghz(2), ExecutionOptions(precision="float16"))


def test_memory_option_per_shot() -> None:
    r = simulator_for(DEVICES["ionq_forte"]).execute(star_ghz(3), ExecutionOptions(shots=100, seed=4, memory=True))
    assert len(r.memory) == 100 and {m for m in r.memory} == set(r.counts)
    assert all(r.memory.count(k) == v for k, v in r.counts.items())


def test_describe_contains_emulator_limits() -> None:
    d = simulator_for(DEVICES["rigetti_ankaa_3"]).describe()
    assert d["n_physical"] == 84 and d["connectivity"] == "nearest_neighbour"
    assert d["emulator"]["max_sim_qubits"] == 84 and len(d["coupling_map"]["edges"]) == 149
