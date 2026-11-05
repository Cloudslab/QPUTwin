"""A calibration-matched Qiskit Aer noise model, built independently of QPUTwin's code.

The calibration (gate fidelities and durations, T1/T2, readout errors, coupling map) is
read from the running QPUTwin orchestrator (``GET /v1/backends/{id}``), so both
simulators see identical numbers. The model is the conventional per-gate recipe:

    error(gate) = D_λ ∘ R(t_gate)        (thermal relaxation first, then depolarising)

where λ is solved, with Qiskit's own channel algebra, so that the composite channel's
average gate fidelity equals the published value:

    F_pro(D_λ ∘ R) = (1 − λ)·F_pro(R) + λ/d²   ⇒   λ = (F_pro(R) − F_pro*) / (F_pro(R) − 1/d²)

plus a symmetric readout confusion matrix. It deliberately has **no idle-window
decoherence**; QPUTwin adds that from its ASAP gate schedule, and the comparison runs
QPUTwin both with and without it.

Native gates are mapped to Qiskit gates of the same action:

| QPUTwin device | 1Q (noisy) | virtual | 2Q |
|---|---|---|---|
| quantinuum_helios | u1q → ``r`` | rz | zzmax → ``rzz`` |
| ionq_forte | gpi, gpi2 → ``r`` | rz | ms → ``rxx`` |
| rigetti_ankaa_3 | rx → ``rx`` | rz | cz → ``cz`` |
| iqm_emerald | prx → ``r`` | rz | cz → ``cz`` |

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
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
        err, lam = gate_error(g["fidelity"], g["duration_ns"], t1, t2, nq)
        nm.add_all_qubit_quantum_error(err, [qgate])
        lams[qgate] = lam
    p01, p10 = cal["readout"]["p01"], cal["readout"]["p10"]
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))
    cmap = dev["coupling_map"]
    coupling = None if cmap["kind"] == "all_to_all" else CouplingMap(
        [list(e) for e in cmap["edges"]] + [[b, a] for a, b in cmap["edges"]])
    return DeviceModel(backend, spec["basis"], coupling, nm, lams, cal)
