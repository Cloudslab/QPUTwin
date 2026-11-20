"""Tests the calibration-aware re-layout after routing.

Usage: uv run pytest tests/test_calibrated_layout.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from qputwin_core.devices import DEVICES
from qputwin_core.simulators import ExecutionOptions, simulator_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import circuits as C  # noqa: E402

SNAPSHOT_TWINS = ("ibm_heron", "ibm_nighthawk", "google_willow")


def _run(device: str, qasm: str, **routing):
    return simulator_for(DEVICES[device]).execute(
        qasm, ExecutionOptions(shots=4000, seed=3, routing={"method": "sabre", "seed": 0, **routing}))


@pytest.mark.parametrize("device", SNAPSHOT_TWINS)
@pytest.mark.parametrize("fam,n", [("ghz", 7), ("qft", 6), ("qv", 5), ("bv", 6)])
def test_relayout_is_cheaper_executable_and_keeps_the_ideal_output(device: str, fam: str, n: int) -> None:
    qasm = C.FAMILIES[fam](n, seed=1)
    plain = _run(device, qasm, noise_aware=False)
    aware = _run(device, qasm)
    rl = aware.compilation["routing"]["calibrated_relayout"]
    assert rl["cost_after"] <= rl["cost_before"] + 1e-12
    # executable: the pipeline's topology check rejects any two-qubit gate off a coupler, so a result exists;
    # and the circuit moved without new SWAPs
    assert aware.compilation["routing"]["swaps_inserted"] == plain.compilation["routing"]["swaps_inserted"]
    assert set(aware.compilation["physical_qubits_used"]) <= set(DEVICES[device].coupling.usable_qubits)
    # ideal (noise-free) output is a property of the logical circuit, not of the placement
    assert plain.counts == aware.counts or _close(plain.counts, aware.counts)
    assert aware.compilation["native_two_qubit_count"] == plain.compilation["native_two_qubit_count"]


def _close(a: dict[str, int], b: dict[str, int]) -> bool:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / (2 * sum(a.values())) < 0.05


def test_relayout_avoids_herons_poor_qubit_and_raises_the_predicted_fidelity() -> None:
    """On ibm_boston, qubit 46 (T1 8 µs, read-out error 8 %) is where plain routing put GHZ-7."""
    qasm = C.FAMILIES["ghz"](7, seed=1)
    sim = simulator_for(DEVICES["ibm_heron"])
    opts = {"shots": 20000, "seed": 1, "noise": {}, "engine": "density_matrix"}
    plain = sim.execute(qasm, ExecutionOptions(routing={"method": "sabre", "seed": 0, "noise_aware": False}, **opts))
    aware = sim.execute(qasm, ExecutionOptions(routing={"method": "sabre", "seed": 0}, **opts))
    used = set(aware.compilation["physical_qubits_used"])
    assert 46 not in used
    good = lambda r: (r.counts.get("0" * 7, 0) + r.counts.get("1" * 7, 0)) / 20000  # noqa: E731
    assert good(aware) > good(plain) + 0.05


@pytest.mark.parametrize("device", ["iqm_emerald", "rigetti_ankaa_3"])
def test_median_only_calibrations_are_left_alone(device: str) -> None:
    r = _run(device, C.FAMILIES["qft"](6, seed=1))
    assert "calibrated_relayout" not in r.compilation["routing"]


def test_pinned_layouts_are_respected() -> None:
    qasm = C.FAMILIES["ghz"](4, seed=1)
    trivial = _run("ibm_heron", qasm, initial_layout="trivial")
    assert "calibrated_relayout" not in trivial.compilation["routing"]
    explicit = _run("ibm_heron", qasm, initial_layout=[0, 1, 2, 3])
    assert sorted(explicit.compilation["physical_qubits_used"]) == [0, 1, 2, 3]
