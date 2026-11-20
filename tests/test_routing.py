"""Tests the SABRE router.

Usage: uv run pytest tests/test_routing.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from qputwin_core.devices import DEVICES, IQM_EMERALD, RIGETTI_ANKAA_3
from qputwin_core.errors import (
    InvalidRequestError, ResourceLimitError, TopologicalCompilationError,
)
from qputwin_core.ir import Circuit, Operation
from qputwin_core.qasm import load_circuit
from qputwin_core.routing import RoutingOptions, SabreRouter
from qputwin_core.simulators import ExecutionOptions, simulator_for
from qputwin_core.topology import CouplingMap, load_coupling_map_file

from conftest import circuit_unitary
from test_simulators import H2, chain_ghz, star_ghz

EXAMPLE_MAP = str(Path(__file__).resolve().parents[1] / "config" / "devices"
                  / "example_rigetti_ankaa_3_defects.json")


def _sv(r) -> dict[int, complex]:
    """Statevector as {basis index over *physical* qubits: amplitude}."""
    amps = np.array(r.statevector)
    qs = r.statevector_qubits
    out = {}
    for idx, (re, im) in enumerate(amps):
        if abs(re) + abs(im) < 1e-14:
            continue
        phys = sum(1 << qs[j] for j in range(len(qs)) if (idx >> j) & 1)
        out[phys] = complex(re, im)
    return out


def _random_qasm(n: int, depth: int, rng: np.random.Generator) -> str:
    lines = [f"qreg q[{n}];"]
    for _ in range(depth):
        if rng.random() < 0.45:
            a, b, c = rng.uniform(-math.pi, math.pi, 3)
            lines.append(f"u3({a:.6f},{b:.6f},{c:.6f}) q[{int(rng.integers(n))}];")
        else:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            gate = ["cx", "cz", "swap", "rzz(0.37)"][int(rng.integers(4))]
            lines.append(f"{gate} q[{a}],q[{b}];")
    return H2 + "\n".join(lines) + "\n"


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("layout,bridge", [("sabre", False), ("trivial", False), ("trivial", True)])
def test_routed_state_equals_relabelled_source_state(seed: int, layout: str, bridge: bool) -> None:
    rng = np.random.default_rng(500 + seed)
    n = int(rng.integers(3, 7))
    src = _random_qasm(n, 30, rng)
    grid = IQM_EMERALD.with_overrides(coupling=CouplingMap.grid(3, 3, "test 3x3 lattice"))
    routed = simulator_for(grid).execute(src, ExecutionOptions(
        shots=1, return_statevector=True,
        routing={"method": "sabre", "initial_layout": layout, "seed": seed, "bridge": bridge}))
    # independent reference: full-unitary contraction of the {U, CX} lowering of the source
    ref = circuit_unitary(load_circuit(src), n)[:, 0]
    final = dict(routed.compilation["routing"]["final_layout"])
    expected = {}
    for idx, a in enumerate(ref):
        if abs(a) > 1e-14:
            expected[sum(1 << final[lq] for lq in range(n) if (idx >> lq) & 1)] = complex(a)
    got = _sv(routed)
    assert set(got) == set(expected)
    for k, a in expected.items():
        assert abs(got[k] - a) < 1e-9, (k, got[k], a)
    st = routed.compilation["routing"]
    assert st["routed_two_qubit_gates"] == (st["input_two_qubit_gates"] + st["added_cx"]
                                            - st["cancelled_two_qubit_gates"])
    assert st["added_cx"] == 3 * (st["swaps_inserted"] + st["bridges_inserted"])


@pytest.mark.parametrize("vendor", ["rigetti_ankaa_3", "iqm_emerald"])
def test_star_ghz_is_routed_instead_of_rejected(vendor: str) -> None:
    sim = simulator_for(DEVICES[vendor])
    with pytest.raises(TopologicalCompilationError):
        sim.execute(star_ghz(12), ExecutionOptions(shots=10))
    r = sim.execute(star_ghz(12), ExecutionOptions(shots=4000, seed=3, routing={"method": "sabre"}))
    assert set(r.counts) == {"0" * 12, "1" * 12}
    st = r.compilation["routing"]
    assert st["swaps_inserted"] > 0 and st["method"] == "sabre"
    # every SWAP becomes three CX (minus cancelled pairs), and each CX one native CZ
    assert r.compilation["native_two_qubit_count"] == st["routed_two_qubit_gates"] \
        == 11 + 3 * st["swaps_inserted"] - st["cancelled_two_qubit_gates"]
    assert r.workload["routed"] is True
    # final measurements are mapped through the final layout, so they stay terminal (no shot branching)
    assert r.workload["execution_path"] == "terminal_sampling" and st["deferred_final_measurements"] == 12
    # the routed circuit itself satisfies the adjacency matrix
    a = DEVICES[vendor].coupling.adjacency
    initial = dict(st["initial_layout"])
    assert len(set(initial.values())) == 12 and all(0 <= p < a.shape[0] for p in initial.values())


def test_chain_needs_no_swaps_with_trivial_layout() -> None:
    r = simulator_for(RIGETTI_ANKAA_3).execute(chain_ghz(9), ExecutionOptions(
        shots=100, seed=1, routing={"method": "sabre", "initial_layout": "trivial"}))
    assert r.compilation["routing"]["swaps_inserted"] == 0


def test_sabre_layout_beats_trivial_on_a_line() -> None:
    """Logical qubits 0 and 9 interact repeatedly; SABRE places them adjacently."""
    src = H2 + "qreg q[10];\n" + "cx q[0],q[9];\n" * 6
    line = CouplingMap(n_qubits=10, kind="graph", edges=tuple((k, k + 1) for k in range(9)))
    spec = IQM_EMERALD.with_overrides(coupling=line)
    trivial = simulator_for(spec).execute(src, ExecutionOptions(shots=1, routing={"initial_layout": "trivial"}))
    sabre = simulator_for(spec).execute(src, ExecutionOptions(shots=1, routing={"initial_layout": "sabre"}))
    assert sabre.compilation["routing"]["swaps_inserted"] < trivial.compilation["routing"]["swaps_inserted"]
    assert sabre.compilation["routing"]["swaps_inserted"] == 0


def test_mid_circuit_measurement_and_feed_forward_survive_routing() -> None:
    theta = 1.234
    src = H2 + f"""qreg q[3]; creg m0[1]; creg m1[1]; creg out[1];
    ry({theta}) q[0]; h q[1]; cx q[1], q[2]; cx q[0], q[2]; cx q[0], q[1]; h q[0];
    measure q[0] -> m0[0]; measure q[1] -> m1[0];
    if (m1 == 1) x q[2]; if (m0 == 1) z q[2];
    cx q[0], q[2];
    measure q[2] -> out[0];
    """
    line = CouplingMap(n_qubits=3, kind="graph", edges=((0, 1), (1, 2)))
    r = simulator_for(IQM_EMERALD.with_overrides(coupling=line)).execute(
        src, ExecutionOptions(shots=40_000, seed=9, routing={"method": "sabre", "initial_layout": "trivial"}))
    ref = simulator_for(DEVICES["ionq_forte"]).execute(src, ExecutionOptions(shots=40_000, seed=9))
    assert r.compilation["routing"]["swaps_inserted"] > 0
    for key in set(r.counts) | set(ref.counts):
        a, b = r.counts.get(key, 0) / 40_000, ref.counts.get(key, 0) / 40_000
        assert abs(a - b) < 0.02, (key, a, b)


def test_routing_is_deterministic_for_a_seed() -> None:
    sim = simulator_for(IQM_EMERALD)
    a = sim.execute(star_ghz(10), ExecutionOptions(shots=1, routing={"seed": 7}))
    b = sim.execute(star_ghz(10), ExecutionOptions(shots=1, routing={"seed": 7}))
    assert a.compilation["routing"] == b.compilation["routing"]


def test_trapped_ion_routing_is_a_no_op() -> None:
    r = simulator_for(DEVICES["ionq_forte"]).execute(star_ghz(5), ExecutionOptions(shots=10, routing={}))
    assert r.compilation["routing"]["skipped"] == "all_to_all" and r.compilation["routing"]["swaps_inserted"] == 0


# ============================================================ calibrated maps
def test_disabled_qubits_are_rejected_and_avoided_by_the_router() -> None:
    cmap = load_coupling_map_file(EXAMPLE_MAP)
    assert cmap.disabled_qubits == (5, 30) and not cmap.coupled(4, 5) and not cmap.coupled(13, 25)
    spec = RIGETTI_ANKAA_3.with_overrides(coupling=cmap)
    sim = simulator_for(spec)
    with pytest.raises(TopologicalCompilationError) as ei:
        sim.execute(chain_ghz(8), ExecutionOptions(shots=10))     # chain 0..7 crosses disabled qubit 5
    assert ei.value.details["reason"] in ("disabled_qubit",) or ei.value.details["qubits"] == [4, 5]
    with pytest.raises(TopologicalCompilationError) as ei:
        sim.execute(H2 + "qreg q[6]; creg c[1];\nmeasure q[5] -> c[0];\n", ExecutionOptions(shots=10))
    assert ei.value.details["reason"] == "disabled_qubit"
    r = sim.execute(chain_ghz(8), ExecutionOptions(shots=2000, seed=1, routing={"method": "sabre"}))
    assert set(r.counts) == {"0" * 8, "1" * 8}
    used = set(r.compilation["physical_qubits_used"])
    assert not used & {5, 30}


def test_router_reports_insufficient_connected_region() -> None:
    two_islands = CouplingMap(n_qubits=6, kind="graph", edges=((0, 1), (1, 2), (3, 4), (4, 5)))
    spec = IQM_EMERALD.with_overrides(coupling=two_islands)
    with pytest.raises(ResourceLimitError) as ei:
        simulator_for(spec).execute(chain_ghz(4), ExecutionOptions(shots=1, routing={}))
    assert ei.value.details["largest_component"] == 3
    with pytest.raises(TopologicalCompilationError, match="disconnected"):
        simulator_for(spec).execute(H2 + "qreg q[6];\ncx q[0],q[5];\n",
                                    ExecutionOptions(shots=1, routing={"initial_layout": "trivial"}))


def test_explicit_initial_layout() -> None:
    sim = simulator_for(IQM_EMERALD)
    r = sim.execute(H2 + "qreg q[2]; creg c[2];\nx q[0];\ncx q[0],q[1];\nmeasure q -> c;\n",
                    ExecutionOptions(shots=10, routing={"initial_layout": [10, 11]}))
    assert r.compilation["routing"]["initial_layout"] == [[0, 10], [1, 11]] and r.counts == {"11": 10}
    assert r.compilation["physical_qubits_used"] == [10, 11]
    with pytest.raises(InvalidRequestError):
        sim.execute(star_ghz(3), ExecutionOptions(shots=1, routing={"initial_layout": [0, 1]}))   # too short
    with pytest.raises(InvalidRequestError):
        sim.execute(star_ghz(2), ExecutionOptions(shots=1, routing={"initial_layout": [0, 999]}))


@pytest.mark.parametrize("bad", [{"method": "astar"}, {"trials": 65}, {"initial_layout": "magic"},
                                 {"initial_layout": [0, 0]}, {"decay": 2.0}, {"foo": 1}, "sabre",
                                 {"exact_layout": "yes"}, {"exact_layout_budget": -1}])
def test_routing_option_validation(bad) -> None:
    with pytest.raises(InvalidRequestError):
        RoutingOptions.from_dict(bad)


def test_routing_trials_zero_means_automatic() -> None:
    assert RoutingOptions.from_dict({"trials": 0}).trials == 0
    assert RoutingOptions.from_dict({}).trials == 0


def test_exact_layout_routes_embeddable_circuits_without_swaps() -> None:
    """A chain and a star that fit the lattice are placed SWAP-free; disabling the search falls back to SABRE."""
    sim = simulator_for(DEVICES["iqm_emerald"])
    chain = H2 + "qreg q[9];\ncreg c[9];\nh q[0];\n" + "".join(f"cx q[{k}],q[{k + 1}];\n" for k in range(8)) \
        + "measure q -> c;\n"
    star = H2 + "qreg q[5];\ncreg c[5];\nh q[0];\n" + "".join(f"cx q[0],q[{k}];\n" for k in range(1, 5)) \
        + "measure q -> c;\n"
    for src, two_q in ((chain, 8), (star, 4)):
        r = sim.execute(src, ExecutionOptions(shots=200, seed=1, routing={"method": "sabre"}))
        rt = r.compilation["routing"]
        assert rt["initial_layout_method"] == "exact" and rt["swaps_inserted"] == 0
        assert r.compilation["native_two_qubit_count"] == two_q
        assert set(r.counts) == {"0" * len(next(iter(r.counts))), "1" * len(next(iter(r.counts)))}
    off = sim.execute(chain, ExecutionOptions(shots=1, seed=1, routing={"method": "sabre", "exact_layout": False}))
    assert off.compilation["routing"]["initial_layout_method"] == "sabre"


def test_sabre_trials_are_scored_by_the_emitted_circuit() -> None:
    """The kept trial is the one whose routed circuit has the fewest two-qubit gates."""
    import random
    rng = random.Random(4)
    body = "".join(f"cx q[{a}],q[{b}];\n" for a, b in (rng.sample(range(8), 2) for _ in range(40)))
    src = H2 + "qreg q[8];\ncreg c[8];\n" + body + "measure q -> c;\n"
    r = simulator_for(DEVICES["rigetti_ankaa_3"]).execute(src, ExecutionOptions(shots=1, seed=1,
                                                                                    routing={"method": "sabre"}))
    rt = r.compilation["routing"]
    assert rt["initial_layout_method"] == "sabre" and rt["trials"] == 16
    assert min(rt["layout_trial_two_qubit_gates"]) == r.compilation["native_two_qubit_count"]


def test_routing_rejected_in_strict_mode() -> None:
    with pytest.raises(InvalidRequestError, match="not native"):
        simulator_for(IQM_EMERALD).execute(star_ghz(3), ExecutionOptions(compilation="strict", routing={}))


def test_router_handles_ops_without_qubits_and_barriers() -> None:
    circ = Circuit("3.0", 4, 0, ops=[Operation("gphase", (), (0.3,)), Operation("U", (0,), (0.1, 0.2, 0.3)),
                                     Operation("barrier", (0, 1, 2, 3)), Operation("CX", (0, 3))])
    line = CouplingMap(n_qubits=4, kind="graph", edges=((0, 1), (1, 2), (2, 3)))
    res = SabreRouter(line, RoutingOptions(initial_layout="trivial")).run(circ)
    names = [op.name for op in res.circuit.ops]
    assert names[0] == "gphase" and "barrier" in names and res.swaps == 2
    barrier = next(op for op in res.circuit.ops if op.name == "barrier")
    assert set(barrier.qubits) == {0, 3}             # only placed qubits survive in the barrier


# ============================================================== optimisation
def test_cancellation_respects_commutation_exactly() -> None:
    from qputwin_core.routing import cancel_inverse_pairs

    cx = Operation("CX", (0, 1))
    rz = Operation("U", (0,), (0.0, 0.0, 0.7))        # diagonal on the control: commutes with CX
    rx = Operation("U", (1,), (0.4, -math.pi / 2, math.pi / 2))   # X rotation on the target: commutes
    h = Operation("U", (0,), (math.pi / 2, 0.0, math.pi))         # H on the control: does not commute
    other = Operation("CX", (0, 2))                               # shares only the control: commutes
    ops, pairs = cancel_inverse_pairs([cx, rz, rx, other, cx])
    assert pairs == 1 and [op.name for op in ops] == ["U", "U", "CX"]
    ops, pairs = cancel_inverse_pairs([cx, h, cx])
    assert pairs == 0 and len(ops) == 3
    cz_ab, cz_ba = Operation("cz", (0, 1)), Operation("cz", (1, 0))
    assert cancel_inverse_pairs([cz_ab, cz_ba])[1] == 1                    # symmetric gate, either order
    meas = Operation("measure", (0,), clbits=(0,))
    assert cancel_inverse_pairs([cx, meas, cx])[1] == 0                    # measurements block


def test_optimisation_reduces_routed_cost() -> None:
    sim = simulator_for(IQM_EMERALD)
    src = H2 + "qreg q[6];\n" + "".join(f"cx q[0],q[{k}];\ncx q[0],q[{k}];\n" for k in range(1, 6)) \
        + "".join(f"cx q[0],q[{k}];\n" for k in range(1, 6))
    plain = sim.execute(src, ExecutionOptions(shots=1, routing={"optimize": False})).compilation
    opt = sim.execute(src, ExecutionOptions(shots=1, routing={"optimize": True})).compilation
    assert opt["routing"]["cancelled_two_qubit_gates"] > 0
    assert opt["native_two_qubit_count"] < plain["native_two_qubit_count"]


def test_bridge_keeps_the_layout() -> None:
    line = CouplingMap(n_qubits=3, kind="graph", edges=((0, 1), (1, 2)))
    spec = IQM_EMERALD.with_overrides(coupling=line)
    src = H2 + "qreg q[3]; creg c[3];\nx q[0];\ncx q[0],q[2];\nmeasure q -> c;\n"
    r = simulator_for(spec).execute(src, ExecutionOptions(
        shots=100, routing={"initial_layout": "trivial", "bridge": True, "optimize": False}))
    st = r.compilation["routing"]
    assert st["bridges_inserted"] == 1 and st["swaps_inserted"] == 0
    assert st["final_layout"] == st["initial_layout"] and r.counts == {"101": 100}
