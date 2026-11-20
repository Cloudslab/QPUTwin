"""Tests the Amazon Braket device importer.

Usage: uv run pytest tests/test_braket_import.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from qputwin_core.calibration import load_calibration_file
from qputwin_core.devices import IONQ_FORTE, IQM_EMERALD, apply_overrides
from qputwin_core.simulators import ExecutionOptions, simulator_for
from qputwin_core.topology import load_coupling_map_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import import_braket_device as imp  # noqa: E402

ARN = "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald"


def _rb(f: float) -> dict:
    return {"fidelityType": {"name": "RANDOMIZED_BENCHMARKING"}, "fidelity": f}


def _iqm_payload() -> dict:
    """Four qubits labelled 1..4 (IQM style) on a line, one with T2 > 2·T1 to exercise clipping."""
    one = {str(q): {"T1": {"value": t1, "unit": "us"}, "T2": {"value": t2, "unit": "us"},
                    "oneQubitFidelity": [_rb(0.999 - 0.0002 * q),
                                         {"fidelityType": {"name": "READOUT"}, "fidelity": 0.96 + 0.005 * q}]}
           for q, t1, t2 in ((1, 50.0, 20.0), (2, 40.0, 90.0), (3, 60.0, 30.0), (4, 45.0, 25.0))}
    two = {"1-2": {"twoQubitGateFidelity": [{"gateName": "CZ", "fidelity": 0.995,
                                             "fidelityType": {"name": "RANDOMIZED_BENCHMARKING"}}]},
           "2-3": {"twoQubitGateFidelity": [{"gateName": "CZ", "fidelity": 0.990,
                                             "fidelityType": {"name": "RANDOMIZED_BENCHMARKING"}}]},
           "3-4": {"twoQubitGateFidelity": [{"gateName": "CZ", "fidelity": 0.985,
                                             "fidelityType": {"name": "RANDOMIZED_BENCHMARKING"}}]}}
    caps = {"service": {"updatedAt": "2026-10-20T08:00:00Z"},
            "paradigm": {"qubitCount": 4, "connectivity": {"fullyConnected": False,
                                                          "connectivityGraph": {"1": ["2"], "2": ["3"], "3": ["4"]}}},
            "standardized": {"braketSchemaHeader": {"name": "braket.device_schema.standardized_gate_model_qpu_device_properties",
                                                    "version": "1"},
                             "oneQubitProperties": one, "twoQubitProperties": two}}
    return {"deviceArn": ARN, "deviceName": "Emerald", "deviceCapabilities": json.dumps(caps)}


def test_superconducting_snapshot(tmp_path: Path) -> None:
    src = tmp_path / "emerald.json"
    src.write_text(json.dumps(_iqm_payload()))
    imp.main([str(src), "--device", "iqm_emerald", "-o", str(tmp_path)])
    cmap = load_coupling_map_file(tmp_path / "iqm_emerald_2026-10-20_coupling.json")
    assert cmap.n_qubits == 4 and cmap.edges == ((0, 1), (1, 2), (2, 3))
    assert cmap.layout["labels"] == {"0": "1", "1": "2", "2": "3", "3": "4"}
    cal = load_calibration_file(tmp_path / "iqm_emerald_2026-10-20_calibration.json")
    assert cal.snapshot_date == "2026-10-20"
    assert cal.gate_fidelity("cz", (2, 1)) == 0.990 and cal.gate_fidelity("cz") == 0.990     # median
    assert cal.t1(1) == 40.0 and cal.t2(1) == 80.0                                           # clipped to 2·T1
    assert "clipped" in cal.provenance_of("qubits.1.t2_us")
    assert cal.readout_errors(3) == pytest.approx((0.02, 0.02))
    assert cal.gate_fidelity("prx", (0,)) == pytest.approx(0.9988)
    for path in ("gates.cz.fidelity", "coherence.t1_us", "readout.fidelity", "couplers.1-2.cz.fidelity",
                 "qubits.0.gates.prx.fidelity", "qubits.2.t1_us"):
        assert cal.provenance_class(path) == "published" and ARN in cal.provenance_of(path), path
    assert cal.provenance_class("gates.cz.duration_ns") == "assumed"        # Braket publishes no durations
    # end to end: the snapshot drives a worker
    spec = apply_overrides(IQM_EMERALD, {"QPUTWIN_COUPLING_MAP": str(tmp_path / "iqm_emerald_2026-10-20_coupling.json"),
                                         "QPUTWIN_CALIBRATION": str(tmp_path / "iqm_emerald_2026-10-20_calibration.json")})
    assert spec.n_physical == 4
    r = simulator_for(spec).execute('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4]; creg c[4];\nh q[0];\n'
                                    "cx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\nmeasure q -> c;\n",
                                    ExecutionOptions(shots=200, seed=1, noise={}))
    assert "couplers.2-3.cz.fidelity" in r.noise["parameters_used"]["published"]


def test_ionq_provider_properties(tmp_path: Path) -> None:
    caps = {"service": {"updatedAt": "2026-10-21T00:00:00Z"},
            "paradigm": {"qubitCount": 36, "connectivity": {"fullyConnected": True, "connectivityGraph": {}}},
            "provider": {"fidelity": {"1Q": {"mean": 0.9998}, "2Q": {"mean": 0.9952}, "spam": {"mean": 0.9951}},
                         "timing": {"T1": 100.0, "T2": 1.0, "1Q": 6.3e-5, "2Q": 9.7e-4, "readout": 1.5e-4,
                                    "reset": 5e-5}}}
    src = tmp_path / "forte.json"
    src.write_text(json.dumps({"deviceArn": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
                               "deviceCapabilities": caps}))
    imp.main([str(src), "--device", "ionq_forte", "-o", str(tmp_path)])
    assert not (tmp_path / "ionq_forte_2026-10-21_coupling.json").exists()      # all-to-all: no map
    cal = load_calibration_file(tmp_path / "ionq_forte_2026-10-21_calibration.json")
    assert cal.gate_fidelity("ms") == 0.9952 and cal.gate_duration_ns("ms") == pytest.approx(970_000.0)
    assert cal.gate_duration_ns("gpi2") == pytest.approx(63_000.0) and cal.t1(0) == pytest.approx(1e8)
    assert cal.readout_errors(0) == pytest.approx((0.0049, 0.0049))
    assert cal.provenance_class("gates.ms.duration_ns") == "published"
    assert cal.provenance_class("parallelism") == "assumed"
    spec = apply_overrides(IONQ_FORTE, {"QPUTWIN_CALIBRATION": str(tmp_path / "ionq_forte_2026-10-21_calibration.json")})
    assert spec.calibration is not None and spec.calibration.gate_fidelity("gpi") == 0.9998
