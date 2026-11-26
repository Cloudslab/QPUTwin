"""Tests the calibration data, the gate-duration model and calibration overrides.

Usage: uv run pytest tests/test_calibration.py

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qputwin_core.calibration import Calibration, load_calibration_file
from qputwin_core.devices import DEVICES, RIGETTI_ANKAA_3, apply_overrides
from qputwin_core.ir import Condition
from qputwin_core.scheduling import asap_schedule, duration_report
from qputwin_core.simulators import ExecutionOptions, SuperconductingSimulator, simulator_for
from qputwin_core.synthesis import NativeOp
from qputwin_core.topology import CouplingMap, load_coupling_map_file
from qputwin_worker.app import create_worker_app

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "config" / "devices"
H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


# ================================================================ calibration
@pytest.mark.parametrize("vendor", sorted(DEVICES))
def test_packaged_calibration_covers_every_native_gate(vendor: str) -> None:
    spec = DEVICES[vendor]
    cal = spec.calibration
    assert cal is not None and cal.device == vendor and cal.sources
    for g in spec.native_gates:
        gc = cal.gate(g)
        assert gc is not None and gc.fidelity is not None and gc.duration_ns is not None, g
        if g in spec.virtual_gates:
            assert gc.fidelity == 1.0 and gc.duration_ns == 0.0
    assert cal.readout_duration_ns and cal.reset_duration_ns
    # research honesty: every quoted number carries a provenance tag
    tagged = " ".join(cal.provenance)
    for g in spec.native_gates:
        if g not in spec.virtual_gates:
            assert f"gates.{g}.fidelity" in tagged and f"gates.{g}.duration_ns" in tagged, g
    assert all(v.split(":")[0].split(" ")[0] in ("published", "assumed", "not", "unknown")
               for v in cal.provenance.values())


def test_calibration_overrides_and_validation() -> None:
    cal = load_calibration_file(EXAMPLES / "example_rigetti_ankaa_3_calibration.json")
    assert cal.t1(0) == 12.0 and cal.t1(7) == 30.0
    assert cal.readout_errors(0) == pytest.approx((0.10, 0.10)) and cal.readout_errors(3) == pytest.approx((0.03, 0.03))
    assert cal.gate_fidelity("cz", (1, 0)) == 0.95 and cal.gate_duration_ns("cz", (0, 1)) == 120.0
    assert cal.gate_fidelity("cz", (1, 2)) == 0.99
    assert cal.gate_fidelity("rx", (1,)) == 0.995 and cal.gate_duration_ns("rx", (1,)) == 40.0
    for bad in ({"device": "x", "coherence": {"t1_us": 10, "t2_us": 30}},
                {"device": "x", "gates": {"cz": {"fidelity": 1.5}}},
                {"device": "x", "readout": {"p01": 0.7}},
                {"device": "x", "fidelity_convention": "diamond_norm"},
                {"device": "x", "gates": {"cz": {"overrotation": 0.9}}},
                {"device": "x", "readout_correlations": [{"qubits": [0, 1], "matrix": [[1, 0], [0, 1]]}]},
                {"device": "x", "readout_correlations": [{"qubits": [0, 1], "matrix": [[0.9] * 4] * 4}]},
                {"device": "x", "parallelism": {"max_parallel_two_qubit": 0}},
                {"gates": {}}):
        with pytest.raises(ValueError):
            Calibration.from_dict(bad)


def test_process_fidelity_convention_is_converted_exactly() -> None:
    cal = Calibration.from_dict({"device": "x", "fidelity_convention": "process_fidelity",
                                 "gates": {"cz": {"fidelity": 0.99}, "prx": {"fidelity": 0.999}}})
    assert cal.gate_fidelity("cz", (0, 1)) == pytest.approx((4 * 0.99 + 1) / 5)
    assert cal.gate_fidelity("prx", (0,)) == pytest.approx((2 * 0.999 + 1) / 3)


def test_provenance_lookup() -> None:
    cal = DEVICES["iqm_emerald"].calibration
    assert cal.provenance_class("gates.cz.fidelity") == "published"
    assert cal.provenance_class("gates.cz.duration_ns") == "assumed"
    assert cal.provenance_class("crosstalk") == "untagged"
    snap = load_calibration_file(EXAMPLES / "example_rigetti_ankaa_3_calibration.json")
    assert snap.provenance_class("couplers.0-1.cz.fidelity") == "assumed"          # file-wide "*" tag


# ================================================================ scheduling
def _op(name: str, qubits: tuple[int, ...], *, virtual: bool = False, clbits: tuple[int, ...] = (),
        condition: Condition | None = None) -> NativeOp:
    return NativeOp(name, qubits, (), None, None, clbits, condition, None, virtual)


def test_asap_schedule_parallelism_virtual_gates_and_barriers() -> None:
    cal = DEVICES["iqm_emerald"].calibration                  # prx 40 ns, cz 60 ns, readout 1500 ns
    ops = [_op("prx", (0,)), _op("prx", (1,)), _op("rz", (0,), virtual=True), _op("cz", (0, 1)),
           _op("prx", (2,)), _op("barrier", (1, 2)), _op("prx", (2,)), _op("measure", (0,), clbits=(0,))]
    s = asap_schedule(ops, cal)
    assert s.starts_ns[:5] == [0.0, 0.0, 40.0, 40.0, 0.0]          # rz is free; prx(2) runs in parallel
    assert s.starts_ns[6] == 100.0                                # barrier syncs q2 with q1 (cz ends at 100)
    assert s.starts_ns[7] == 140.0                                # final read-out aligned after the last gate
    assert s.makespan_ns == 140.0 + 1500.0
    rep = duration_report(s, cal, shots=1000)
    assert rep["circuit_duration_us"] == pytest.approx(1.64)
    assert rep["estimated_job_duration_s"] == pytest.approx(1000 * 1.64e-6)
    assert rep["complete"] and not rep["includes_shot_overhead"] and rep["final_measurements"] == "aligned"


@pytest.mark.parametrize("vendor", ["ionq_forte", "quantinuum_helios", "iqm_emerald"])
def test_interval_pruning_keeps_the_schedule_identical(vendor: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pruning booked intervals (the O(N) fast path) must reproduce the exhaustive O(N²) schedule exactly."""
    import random

    import qputwin_core.scheduling as sch
    from qputwin_core.timing import SpanTimer

    rng = random.Random(3)
    body = "".join(f"cx q[{a}],q[{b}];\n" if rng.random() < 0.4 else f"h q[{rng.randrange(8)}];\n"
                   for a, b in (rng.sample(range(8), 2) for _ in range(1500)))
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[8];\ncreg c[8];\n' + body + "measure q -> c;\n"
    opts = {} if DEVICES[vendor].coupling.all_to_all else {"routing": {"method": "sabre", "seed": 0}}
    sim = simulator_for(DEVICES[vendor])
    ops = sim.plan(qasm, ExecutionOptions(shots=1, seed=1, **opts), SpanTimer()).native.ops
    fast = asap_schedule(ops, sim.spec.calibration)
    monkeypatch.setattr(sch, "_PRUNE_MIN", 10**9)
    slow = asap_schedule(ops, sim.spec.calibration)
    assert fast.starts_ns == slow.starts_ns and fast.makespan_ns == slow.makespan_ns


def test_feed_forward_waits_for_its_measurement() -> None:
    cal = Calibration.from_dict({"device": "t", "gates": {"x": {"duration_ns": 10}},
                                 "readout": {"duration_ns": 100}, "feedforward_latency_ns": 50,
                                 "shot_overhead_us": 2.0})
    ops = [_op("measure", (0,), clbits=(0,)), _op("x", (1,), condition=Condition((0,), 1)),
           _op("measure", (1,), clbits=(1,))]
    s = asap_schedule(ops, cal)
    assert s.starts_ns[1] == 150.0 and s.makespan_ns == 260.0
    rep = duration_report(s, cal, shots=10)
    assert rep["includes_shot_overhead"] and rep["estimated_job_duration_s"] == pytest.approx(10 * 2.26e-6)
    assert rep["conditional_ops_scheduled_as_executed"] == 1


def test_missing_durations_are_reported() -> None:
    cal = Calibration.from_dict({"device": "t", "gates": {"x": {"fidelity": 0.99}}})
    s = asap_schedule([_op("x", (0,)), _op("measure", (0,), clbits=(0,))], cal)
    assert s.missing == ["measure", "x"]
    assert not duration_report(s, cal, 1)["complete"]


def test_estimate_reported_for_every_job() -> None:
    for vendor in DEVICES:
        r = simulator_for(DEVICES[vendor]).execute(
            H2 + "qreg q[2]; creg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n", ExecutionOptions(shots=100))
        est = r.compilation["hardware_time_estimate"]
        assert est["complete"] and est["circuit_duration_us"] > 0
        overhead = est["shot_overhead_us"] if est["includes_shot_overhead"] else 0.0
        assert est["estimated_job_duration_s"] == pytest.approx(100 * (est["circuit_duration_us"] + overhead) * 1e-6)
        assert est["includes_shot_overhead"] == (DEVICES[vendor].calibration.shot_overhead_us is not None)
    ion = simulator_for(DEVICES["ionq_forte"]).execute(H2 + "qreg q[1];\nh q[0];\n", ExecutionOptions(shots=1))
    sc = simulator_for(DEVICES["iqm_emerald"]).execute(H2 + "qreg q[1];\nh q[0];\n", ExecutionOptions(shots=1))
    assert ion.compilation["hardware_time_estimate"]["circuit_duration_us"] > \
        1000 * sc.compilation["hardware_time_estimate"]["circuit_duration_us"]


# ============================================================ topology defects
def test_disabled_qubits_drop_couplers_and_split_components() -> None:
    g = CouplingMap(n_qubits=5, kind="graph", edges=((0, 1), (1, 2), (2, 3), (3, 4)), disabled_qubits=(2,))
    assert g.edges == ((0, 1), (3, 4)) and g.neighbors(2) == [] and g.usable_qubits == [0, 1, 3, 4]
    assert g.distance_matrix()[0][4] == float("inf") and g.components() == [[0, 1], [3, 4]]
    k = CouplingMap.complete(4, disabled_qubits=(1,))
    assert not k.coupled(0, 1) and k.coupled(0, 3) and k.n_edges == 3 and not k.adjacency[0, 1]
    grid = CouplingMap.grid(3, 4)
    d = grid.distance_matrix()
    assert all(d[p][q] == abs(p // 4 - q // 4) + abs(p % 4 - q % 4) for p in range(12) for q in range(12))
    with pytest.raises(ValueError):
        CouplingMap(n_qubits=3, kind="graph", edges=((0, 1),), disabled_qubits=(7,))


def test_drop_in_overrides_from_environment() -> None:
    env = {"QPUTWIN_COUPLING_MAP": str(EXAMPLES / "example_rigetti_ankaa_3_defects.json"),
           "QPUTWIN_CALIBRATION": str(EXAMPLES / "example_rigetti_ankaa_3_calibration.json")}
    spec = apply_overrides(RIGETTI_ANKAA_3, env)
    assert spec.coupling.disabled_qubits == (5, 30) and spec.n_physical == 84
    assert spec.calibration is not None and spec.calibration.snapshot_date == "2026-10-24"
    assert any("overridden" in n for n in spec.notes)
    assert apply_overrides(RIGETTI_ANKAA_3, {}) is RIGETTI_ANKAA_3
    with pytest.raises(ValueError, match="not 'iqm_emerald'"):
        apply_overrides(DEVICES["iqm_emerald"], {"QPUTWIN_CALIBRATION": env["QPUTWIN_CALIBRATION"]})
    with pytest.raises(ValueError, match="connectivity class"):
        DEVICES["ionq_forte"].with_overrides(coupling=CouplingMap.grid(6, 6))


def test_worker_serves_overridden_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QPUTWIN_COUPLING_MAP", str(EXAMPLES / "example_rigetti_ankaa_3_defects.json"))
    monkeypatch.setenv("QPUTWIN_CALIBRATION", str(EXAMPLES / "example_rigetti_ankaa_3_calibration.json"))
    app = create_worker_app(RIGETTI_ANKAA_3, SuperconductingSimulator)
    with TestClient(app) as c:
        dev = c.get("/v1/device").json()
        assert dev["coupling_map"]["disabled_qubits"] == [5, 30] and dev["coupling_map"]["kind"] == "graph"
        assert dev["calibration"]["snapshot_date"] == "2026-10-24" and dev["calibration"]["qubit_overrides"] == 2
        r = c.post("/v1/execute", json={"qasm": H2 + "qreg q[6]; creg c[1];\nx q[5];\nmeasure q[5] -> c[0];\n",
                                        "shots": 10})
        assert r.status_code == 422 and r.json()["error"]["details"]["reason"] == "disabled_qubit"
        r = c.post("/v1/execute", json={"qasm": H2 + "qreg q[1]; creg c[1];\nmeasure q[0] -> c[0];\n",
                                        "shots": 50_000, "seed": 1,
                                        "options": {"noise": {"depolarizing": False, "relaxation": False}}})
        ones = r.json()["result"]["counts"].get("1", 0)
        assert abs(ones / 50_000 - 0.10) < 0.01                  # qubit 0's 90 % readout from the snapshot
        assert r.json()["worker"]["calibration"].endswith("example_rigetti_ankaa_3_calibration.json")


# ================================================================== importer
def test_import_coupling_map_formats(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import import_coupling_map as imp                   # noqa: PLC0415

    iqm = tmp_path / "iqm.json"
    iqm.write_text(json.dumps({"qubits": ["QB1", "QB2", "QB10", "QB3"],
                               "connectivity": [["QB1", "QB2"], ["QB2", "QB3"], ["QB3", "QB10"]]}))
    out = tmp_path / "out.json"
    imp.main([str(iqm), "-o", str(out), "--provenance", "unit test", "--disabled", "QB10"])
    doc = json.loads(out.read_text())
    assert doc["layout"]["labels"] == {"0": "QB1", "1": "QB2", "2": "QB3", "3": "QB10"}
    assert doc["edges"] == [[0, 1], [1, 2], [2, 3]] and doc["disabled_qubits"] == [3]   # file keeps the hardware graph
    cmap = load_coupling_map_file(out)
    assert cmap.n_qubits == 4 and cmap.is_disabled(3) and cmap.edges == ((0, 1), (1, 2))   # defect applied on load

    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps([[0, 1], [1, 0], [1, 2], [4, 5]]))
    imp.main([str(pairs), "-o", str(out), "--provenance", "p", "--keep-indices", "--n-qubits", "7",
              "--disable-isolated"])
    doc = json.loads(out.read_text())
    assert doc["n_qubits"] == 7 and doc["edges"] == [[0, 1], [1, 2], [4, 5]] and doc["disabled_qubits"] == [3, 6]

    text = tmp_path / "edges.txt"
    text.write_text("# a, b\n0,1\n1 2\n")
    imp.main([str(text), "-o", str(out), "--provenance", "p", "--keep-indices"])
    assert json.loads(out.read_text())["edges"] == [[0, 1], [1, 2]]
