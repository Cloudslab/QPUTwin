"""Vendor device specifications for the four emulated QPUs.

Each :class:`DeviceSpec` fixes a device's physical size, its native gate set, the
native entangler used to realise ``CX``, the one-qubit synthesis basis, and its
coupling graph. Gate sets follow the vendors' public documentation. ``rz`` is
modelled as a *virtual* (frame-change, zero-duration) native on every platform,
mirroring how each vendor's compiler realises Z rotations.

Topology fidelity note: the superconducting coupling maps are synthetic square
lattices matching published qubit counts and connectivity class — see
``scripts/generate_coupling_maps.py`` and the README's methods section.

Each spec also carries a :class:`~qputwin_core.calibration.Calibration`
(fidelities, durations, coherence, readout) loaded from
``devices_data/calibration/<id>.json``. A worker container can replace the
coupling map and/or the calibration at start-up with ``QPUTWIN_COUPLING_MAP`` and
``QPUTWIN_CALIBRATION`` (see :func:`apply_overrides`) — calibrated vendor data is a
drop-in replacement, not a code change.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from qputwin_core.calibration import Calibration, load_calibration, load_calibration_file
from qputwin_core.topology import CouplingMap, load_coupling_map, load_coupling_map_file

__all__ = ["DeviceSpec", "DEVICES", "get_device", "apply_overrides", "QUANTINUUM_HELIOS", "IONQ_FORTE",
           "RIGETTI_ANKAA_3", "IQM_EMERALD"]


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    id: str
    vendor: str
    name: str
    technology: Literal["trapped_ion", "superconducting"]
    qubit_modality: str
    n_physical: int
    native_1q: tuple[str, ...]
    native_2q: tuple[str, ...]
    virtual_gates: tuple[str, ...]
    entangler: str                       # native 2Q gate used to realise CX
    basis: Literal["quantinuum", "ionq", "rigetti", "iqm"]
    coupling: CouplingMap
    description: str
    notes: tuple[str, ...] = ()
    # strict-mode parameter constraints: gate -> allowed angle grid (radians), checked mod 2π
    param_grid: dict[str, tuple[float, ...]] = field(default_factory=dict)
    calibration: Calibration | None = None

    @property
    def native_gates(self) -> tuple[str, ...]:
        return self.native_1q + self.native_2q

    def with_overrides(self, *, coupling: CouplingMap | None = None,
                       calibration: Calibration | None = None) -> DeviceSpec:
        """A copy with a replacement coupling map (which fixes ``n_physical``) and/or calibration."""
        changes: dict[str, Any] = {}
        notes = list(self.notes)
        if coupling is not None:
            if coupling.all_to_all != self.coupling.all_to_all:
                raise ValueError(f"{self.id}: replacement coupling map must keep the connectivity class "
                                 f"({self.coupling.kind!r} -> {coupling.kind!r})")
            changes["coupling"] = coupling
            changes["n_physical"] = coupling.n_qubits
            notes.append(f"Coupling map overridden at runtime: {coupling.provenance or coupling.kind}.")
        if calibration is not None:
            if calibration.device != self.id:
                raise ValueError(f"calibration file is for {calibration.device!r}, not {self.id!r}")
            changes["calibration"] = calibration
            notes.append(f"Calibration overridden at runtime: {calibration.label or calibration.origin}.")
        return dataclasses.replace(self, **changes, notes=tuple(notes)) if changes else self

    def to_dict(self, *, include_edges: bool = True) -> dict[str, Any]:
        return {
            "id": self.id, "vendor": self.vendor, "name": self.name,
            "technology": self.technology, "qubit_modality": self.qubit_modality,
            "n_physical": self.n_physical,
            "native_gates": {"one_qubit": list(self.native_1q), "two_qubit": list(self.native_2q),
                             "virtual": list(self.virtual_gates)},
            "entangler": self.entangler, "synthesis_basis": self.basis,
            "connectivity": "all_to_all" if self.coupling.all_to_all else "nearest_neighbour",
            "coupling_map": self.coupling.to_dict(include_edges=include_edges),
            "description": self.description, "notes": list(self.notes),
            "param_constraints": {k: list(v) for k, v in self.param_grid.items()},
            "calibration": self.calibration.to_dict() if self.calibration is not None else None,
        }


_PI = math.pi

QUANTINUUM_HELIOS = DeviceSpec(
    id="quantinuum_helios", vendor="Quantinuum", name="Helios",
    technology="trapped_ion", qubit_modality="137Ba+ hyperfine qubits, QCCD architecture",
    n_physical=98,
    native_1q=("u1q", "rz"), native_2q=("zzmax", "rzz"), virtual_gates=("rz",),
    entangler="zzmax", basis="quantinuum",
    coupling=CouplingMap.complete(98, "All-to-all via ion transport in a QCCD trap (modelled as K_98)."),
    description="Quantinuum Helios trapped-ion QPU: arbitrary-angle U1q/Rz single-qubit gates and "
                "native ZZ (ZZMax) / arbitrary-angle RZZ two-qubit gates between any pair of qubits.",
    notes=("Connectivity is all-to-all: any two-qubit gate between any pair is legal.",),
    calibration=load_calibration("quantinuum_helios"),
)

IONQ_FORTE = DeviceSpec(
    id="ionq_forte", vendor="IonQ", name="Forte",
    technology="trapped_ion", qubit_modality="171Yb+ hyperfine qubits, AOD-addressed linear chain",
    n_physical=36,
    native_1q=("gpi", "gpi2", "rz"), native_2q=("ms", "zz"), virtual_gates=("rz",),
    entangler="ms", basis="ionq",
    coupling=CouplingMap.complete(36, "All-to-all through the shared motional bus (modelled as K_36)."),
    description="IonQ Forte trapped-ion QPU: GPI/GPI2 single-qubit pulses with virtual Z, and "
                "Mølmer–Sørensen (MS) / ZZ two-qubit gates between any pair of qubits.",
    notes=("MS(φ0, φ1, θ) = exp(-iθ/2 · σ_φ0⊗σ_φ1), angles in radians (IonQ's API uses turns).",
           "Connectivity is all-to-all: any two-qubit gate between any pair is legal."),
    calibration=load_calibration("ionq_forte"),
)

RIGETTI_ANKAA_3 = DeviceSpec(
    id="rigetti_ankaa_3", vendor="Rigetti", name="Ankaa-3",
    technology="superconducting", qubit_modality="fixed-frequency transmons with tunable couplers",
    n_physical=84,
    native_1q=("rx", "rz"), native_2q=("cz", "iswap"), virtual_gates=("rz",),
    entangler="cz", basis="rigetti",
    coupling=load_coupling_map("rigetti_ankaa_3.json"),
    description="Rigetti Ankaa-3 superconducting QPU: RX(kπ/2) pulses with virtual RZ and CZ / iSWAP "
                "two-qubit gates restricted to nearest neighbours of a planar square lattice.",
    notes=("Coupling map: synthetic 7x12 square lattice (not a calibration snapshot).",
           "Two-qubit gates are legal only between physically coupled qubits."),
    param_grid={"rx": (-_PI, -_PI / 2, _PI / 2, _PI)},
    calibration=load_calibration("rigetti_ankaa_3"),
)

IQM_EMERALD = DeviceSpec(
    id="iqm_emerald", vendor="IQM", name="Emerald",
    technology="superconducting", qubit_modality="transmons with tunable couplers (square lattice)",
    n_physical=54,
    native_1q=("prx", "rz"), native_2q=("cz",), virtual_gates=("rz",),
    entangler="cz", basis="iqm",
    coupling=load_coupling_map("iqm_emerald.json"),
    description="IQM Emerald superconducting QPU: phased-RX (PRX) single-qubit gates with virtual Z "
                "and CZ two-qubit gates restricted to nearest neighbours of a planar square lattice.",
    notes=("Coupling map: synthetic 6x9 square lattice (not a calibration snapshot).",
           "Two-qubit gates are legal only between physically coupled qubits."),
    calibration=load_calibration("iqm_emerald"),
)

DEVICES: dict[str, DeviceSpec] = {d.id: d for d in (QUANTINUUM_HELIOS, IONQ_FORTE, RIGETTI_ANKAA_3, IQM_EMERALD)}


def apply_overrides(spec: DeviceSpec, environ: dict[str, str] | None = None) -> DeviceSpec:
    """Apply the worker's runtime overrides: ``QPUTWIN_COUPLING_MAP`` and ``QPUTWIN_CALIBRATION`` (JSON paths)."""
    env = os.environ if environ is None else environ
    cmap_path = (env.get("QPUTWIN_COUPLING_MAP") or "").strip()
    cal_path = (env.get("QPUTWIN_CALIBRATION") or "").strip()
    return spec.with_overrides(
        coupling=load_coupling_map_file(cmap_path) if cmap_path else None,
        calibration=load_calibration_file(cal_path) if cal_path else None)


def get_device(device_id: str) -> DeviceSpec:
    try:
        return DEVICES[device_id]
    except KeyError as exc:
        raise KeyError(f"unknown device {device_id!r}; known: {sorted(DEVICES)}") from exc
