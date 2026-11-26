"""Qiskit Aer noise models built from the same calibration as the QPUTwin twins.

Usage: from noise_models import device_model

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from qiskit.quantum_info import average_gate_fidelity
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error

# qiskit basis per device, and which calibration entry each noisy qiskit gate takes
NATIVE = {
    "quantinuum_helios": {"basis": ["r", "rz", "rzz"], "map": {"r": "u1q", "rzz": "zzmax"}},
    "ionq_forte": {"basis": ["r", "rz", "rxx"], "map": {"r": "gpi2", "rxx": "ms"}},
    "rigetti_ankaa_3": {"basis": ["rx", "rz", "cz"], "map": {"rx": "rx", "cz": "cz"}},
    "iqm_emerald": {"basis": ["r", "rz", "cz"], "map": {"r": "prx", "cz": "cz"}},
    "ibm_heron": {"basis": ["sx", "x", "rz", "cz"], "map": {"sx": "sx", "x": "x", "cz": "cz"}},
    "ibm_nighthawk": {"basis": ["sx", "x", "rz", "cz"], "map": {"sx": "sx", "x": "x", "cz": "cz"}},
    "google_willow": {"basis": ["r", "rz", "cz"], "map": {"r": "phased_xz", "cz": "cz"}},
}


@dataclass
class DeviceModel:
    backend: str
    basis: list[str]
    coupling: CouplingMap | None
    noise: NoiseModel
    depolarizing: dict[str, float]          # solved λ per noisy qiskit gate
    calibration: dict[str, Any]


def gate_error(fidelity: float, duration_ns: float, t1_us: float | None, t2_us: float | None, nq: int):
    """D_λ ∘ R with λ solved so the composite average gate fidelity equals ``fidelity``."""
    d = 2 ** nq
    relax = None
    if t1_us and duration_ns > 0:
        t2 = min(t2_us if t2_us else 2 * t1_us, 2 * t1_us)
        one = thermal_relaxation_error(t1_us * 1e3, t2 * 1e3, duration_ns)
        relax = one if nq == 1 else one.expand(one)
    f_avg_r = 1.0 if relax is None else float(average_gate_fidelity(relax.to_quantumchannel()))
    f_pro_r = ((d + 1) * f_avg_r - 1) / d
    f_pro_t = ((d + 1) * fidelity - 1) / d
    lam = max(0.0, (f_pro_r - f_pro_t) / (f_pro_r - 1 / d**2))
    dep = depolarizing_error(lam, nq)
    err = dep if relax is None else relax.compose(dep)
    check = float(average_gate_fidelity(err.to_quantumchannel()))
    assert abs(check - (fidelity if lam > 0 else f_avg_r)) < 1e-9, (check, fidelity)
    return err, lam


def device_model(backend: str, base_url: str = "http://localhost:8000") -> DeviceModel:
    dev = httpx.get(f"{base_url}/v1/backends/{backend}", timeout=30).raise_for_status().json()
    cal = dev["calibration"]
    spec = NATIVE[backend]
    t1, t2 = cal["coherence"]["t1_us"], cal["coherence"]["t2_us"]
    nm = NoiseModel(basis_gates=spec["basis"])
    lams = {}
    for qgate, native in spec["map"].items():
        g = cal["gates"][native]
        nq = 2 if qgate in ("rzz", "rxx", "cz") else 1
        if g.get("duration_ns") is None:
            g = {**g, "duration_ns": 0.0}
        err, lam = gate_error(g["fidelity"], g["duration_ns"], t1, t2, nq)
        nm.add_all_qubit_quantum_error(err, [qgate])
        lams[qgate] = lam
    p01, p10 = cal["readout"]["p01"], cal["readout"]["p10"]
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))
    cmap = dev["coupling_map"]
    coupling = None if cmap["kind"] == "all_to_all" else CouplingMap(
        [list(e) for e in cmap["edges"]] + [[b, a] for a, b in cmap["edges"]])
    return DeviceModel(backend, spec["basis"], coupling, nm, lams, cal)


# ======================================================================================
# Per-qubit cross-validation on QPUTwin's own compiled circuit (identical physical qubits)
# ======================================================================================
# The IBM and Google twins carry a calibration for every qubit and coupler, so a model
# built from device medians, or a circuit placed on other physical qubits by the Qiskit
# transpiler, would differ for reasons that have nothing to do with the noise physics.
# ``native_circuit`` therefore rebuilds QPUTwin's compiled native circuit (same gates, same
# physical qubits) with Qiskit's own gate definitions, and ``per_qubit_model`` attaches
# per-qubit / per-coupler errors with the same recipe as ``gate_error`` above.

import math  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import (  # noqa: E402
    CZGate, RGate, RXGate, RXXGate, RZGate, RZZGate, SXGate, XGate, iSwapGate,
)
from qiskit.quantum_info import Operator  # noqa: E402


TWO_QUBIT_NATIVES = {"cz", "iswap", "zzmax", "rzz", "zz", "ms"}


def _qiskit_ops(name: str, params: tuple[float, ...]) -> list[tuple[str, Any]]:
    """Qiskit gate(s) with the same action as a QPUTwin native (global phase is irrelevant here)."""
    p = list(params)
    table = {
        "u1q": lambda: [RGate(p[0], p[1])], "prx": lambda: [RGate(p[0], p[1])],
        "gpi": lambda: [RGate(math.pi, p[0])], "gpi2": lambda: [RGate(math.pi / 2, p[0])],
        "rx": lambda: [RXGate(p[0])], "sx": lambda: [SXGate()], "x": lambda: [XGate()],
        "phased_xz": lambda: [RGate(p[0], p[1]), RZGate(p[2])],
        "rz": lambda: [RZGate(p[0])], "cz": lambda: [CZGate()], "iswap": lambda: [iSwapGate()],
        "zzmax": lambda: [RZZGate(math.pi / 2)], "rzz": lambda: [RZZGate(p[0])], "zz": lambda: [RZZGate(p[0])],
    }
    if name == "ms":
        assert abs(p[0]) < 1e-12 and abs(p[1]) < 1e-12, "only MS(0, 0, θ) is produced by CX lowering"
        return [("ms", RXXGate(p[2]))]
    gates = table[name]()
    # the first gate carries the native's noise; trailing virtual Z (phased_xz) is noiseless
    return [(name, gates[0])] + [("rz", g) for g in gates[1:]]


def native_circuit(native: Any, virtual: set[str]) -> tuple[QuantumCircuit, list[int]]:
    """QPUTwin's compiled circuit on a compact register: (circuit, physical qubit of each wire)."""
    used = sorted({q for op in native.ops for q in op.qubits})
    wire = {q: k for k, q in enumerate(used)}
    qc = QuantumCircuit(len(used), native.num_clbits)
    for op in native.ops:
        qs = [wire[q] for q in op.qubits]
        if op.name == "measure":
            qc.measure(qs[0], op.clbits[0])
        elif op.name == "barrier":
            continue
        elif op.name == "reset":
            qc.reset(qs[0])
        else:
            assert op.condition is None, "conditional natives are not used in these families"
            for label, gate in _qiskit_ops(op.name, op.params):
                if label in virtual:
                    qc.append(gate, qs)
                else:      # a labelled unitary, so the noise model can address this native by name
                    qc.unitary(Operator(gate).data, qs, label=label)
    return qc, used


def per_qubit_model(cal: Any, used: list[int], natives: set[str], virtual: set[str]) -> NoiseModel:
    """One-qubit errors (D_λ ∘ R with each qubit's own T1/T2 and fidelity) and per-qubit readout.

    ``cal`` is QPUTwin's :class:`Calibration` (data only; the channels are built here with Qiskit).
    """
    nm = NoiseModel(basis_gates=["unitary", "rz", "measure"])
    for name in natives - virtual - TWO_QUBIT_NATIVES:
        for k, q in enumerate(used):
            f, d = cal.gate_fidelity(name, (q,)), cal.gate_duration_ns(name, (q,))
            if f is not None:
                err, _ = gate_error(f, d or 0.0, cal.t1(q), cal.t2(q), 1)
                nm.add_quantum_error(err, name, [k])
    for k, q in enumerate(used):
        p01, p10 = cal.readout_errors(q)
        nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [k])
    return nm


def add_pair_errors(nm: NoiseModel, cal: Any, used: list[int], name: str, pairs: set[tuple[int, int]]) -> None:
    """Two-qubit errors for the (ordered) wire pairs the circuit actually uses."""
    for a, b in pairs:
        qa, qb = used[a], used[b]
        f, d = cal.gate_fidelity(name, (qa, qb)), cal.gate_duration_ns(name, (qa, qb)) or 0.0
        relax = None
        if d > 0 and cal.t1(qa) and cal.t1(qb):
            ra = thermal_relaxation_error(cal.t1(qa) * 1e3, min(cal.t2(qa) or 2 * cal.t1(qa), 2 * cal.t1(qa)) * 1e3, d)
            rb = thermal_relaxation_error(cal.t1(qb) * 1e3, min(cal.t2(qb) or 2 * cal.t1(qb), 2 * cal.t1(qb)) * 1e3, d)
            relax = ra.expand(rb)          # ra on wire a (first), rb on wire b
        f_avg_r = 1.0 if relax is None else float(average_gate_fidelity(relax.to_quantumchannel()))
        dd = 4
        f_pro_r, f_pro_t = ((dd + 1) * f_avg_r - 1) / dd, ((dd + 1) * f - 1) / dd
        lam = max(0.0, (f_pro_r - f_pro_t) / (f_pro_r - 1 / dd**2))
        err = depolarizing_error(lam, 2) if relax is None else relax.compose(depolarizing_error(lam, 2))
        assert abs(float(average_gate_fidelity(err.to_quantumchannel())) - (f if lam > 0 else f_avg_r)) < 1e-9
        nm.add_quantum_error(err, name, [a, b])


def matched_model_for(qc: QuantumCircuit, cal: Any, used: list[int], natives: set[str],
                      virtual: set[str]) -> NoiseModel:
    """The complete per-qubit Aer model for one compiled circuit."""
    nm = per_qubit_model(cal, used, natives, virtual)
    by_gate: dict[str, set[tuple[int, int]]] = {}
    for inst in qc.data:
        label = getattr(inst.operation, "label", None)
        if inst.operation.name == "unitary" and len(inst.qubits) == 2:
            by_gate.setdefault(label, set()).add(tuple(qc.find_bit(q).index for q in inst.qubits))
    for name, pairs in by_gate.items():
        add_pair_errors(nm, cal, used, name, pairs)
    return nm
