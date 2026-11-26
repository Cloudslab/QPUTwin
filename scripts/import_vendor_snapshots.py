"""Builds the IBM Heron, IBM Nighthawk and Google Willow twins from the vendor snapshots.

Usage: uv run --no-project --with qiskit-ibm-runtime==0.51.0 --with cirq-google==1.7.0 python scripts/import_vendor_snapshots.py

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import statistics
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "qputwin_core" / "devices_data"
CONTACT = "Koustav Kumar Mondal <koustav@koustavmondal.com>"


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def _r(x: float, nd: int = 6) -> float:
    return float(f"{x:.{nd}g}")


def _clip_t2(t1: float | None, t2: float | None, q: int, clipped: list[int]) -> float | None:
    """Clip ``T2`` to ``2·T1`` (compared after rounding, so the stored pair is always physical)."""
    if t1 is not None and t2 is not None and _r(t2) > 2 * _r(t1):
        clipped.append(q)
        return 2 * _r(t1)
    return t2


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


# =============================================================================== IBM
def _ibm_coordinates(n: int, edges: list[tuple[int, int]], family: str, cols: int | None) -> dict[str, Any]:
    if family == "Nighthawk":
        rows = n // cols
        return {"kind": "square_lattice", "rows": rows, "cols": cols, "index": "r*cols + c"}
    # Heron heavy-hex: rows of 16 qubits (index 20r .. 20r+15) joined by 4 bridge qubits
    # (20r+16 .. 20r+19). A bridge sits below the row qubit it couples to.
    coords: dict[int, tuple[int, int]] = {}
    for q in range(n):
        r, k = divmod(q, 20)
        if k < 16:
            coords[q] = (2 * r, k)
    for q in range(n):
        if q in coords:
            continue
        above = min(p for e in edges for p in e if q in e and p != q)
        coords[q] = (coords[above][0] + 1, coords[above][1])
    return {"kind": "heavy_hex", "coordinates": [list(coords[q]) for q in range(n)],
            "note": "row/column of each qubit for drawing; rows of 16 joined by bridge qubits"}


def import_ibm(fake_name: str, device_id: str, family: str, out_dir: Path) -> None:
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    warnings.filterwarnings("ignore")
    backend = FakeProviderForBackendV2().backend(fake_name)
    conf, props, target = backend.configuration(), backend.properties(), backend.target
    system = fake_name.replace("fake_", "ibm_")
    proc = conf.processor_type or {}
    if proc.get("family") != family:
        raise SystemExit(f"{fake_name} is a {proc.get('family')} processor, not {family}")
    label_name = f"{family} r{proc.get('revision')}"
    n = backend.num_qubits
    stamp = props.last_update_date.isoformat()
    sdk = f"qiskit-ibm-runtime {md.version('qiskit-ibm-runtime')}"
    src_tag = f"published: IBM {system} calibration snapshot ({stamp}), shipped in {sdk} fake_provider"

    def qprop(q: int, name: str) -> float | None:
        for item in props.qubits[q]:
            if item.name == name:
                return float(item.value)
        return None

    # ---- couplers
    cz: dict[tuple[int, int], tuple[float, float]] = {}
    inactive: list[list[int]] = []
    for (a, b), inst in target["cz"].items():
        key = (min(a, b), max(a, b))
        if inst is None or inst.error is None:
            continue
        if inst.error >= 1.0:
            if [*key] not in inactive:
                inactive.append([*key])
            continue
        err, dur = float(inst.error), float(inst.duration) * 1e9
        if key in cz:   # both directions are listed; keep the worse one
            err = max(err, cz[key][0])
            dur = max(dur, cz[key][1])
        cz[key] = (err, dur)
    inactive.sort()
    edges = sorted(cz)
    all_edges = sorted(set(edges) | {tuple(e) for e in inactive})

    # ---- qubits
    clipped: list[int] = []
    qubits: dict[str, Any] = {}
    t1s, t2s, p01s, p10s, ro_len, sx_err, sx_len, x_err, x_len, reset_len = ([] for _ in range(10))
    for q in range(n):
        t1, t2 = qprop(q, "T1"), qprop(q, "T2")
        t2 = _clip_t2(t1, t2, q, clipped)
        p01, p10 = qprop(q, "prob_meas1_prep0"), qprop(q, "prob_meas0_prep1")
        entry: dict[str, Any] = {}
        if t1:
            entry["t1_us"] = _r(t1)
            t1s.append(t1)
        if t2:
            entry["t2_us"] = _r(t2, 8)
            t2s.append(t2)
        if p01 is not None and p10 is not None:
            entry["readout_p01"], entry["readout_p10"] = _r(min(p01, 0.5)), _r(min(p10, 0.5))
            p01s.append(p01)
            p10s.append(p10)
        gates: dict[str, Any] = {}
        for g, errs, lens in (("sx", sx_err, sx_len), ("x", x_err, x_len)):
            inst = target[g].get((q,))
            if inst is not None and inst.error is not None and inst.error < 1.0:
                gates[g] = {"fidelity": _r(1 - inst.error, 8), "duration_ns": _r(inst.duration * 1e9)}
                errs.append(inst.error)
                lens.append(inst.duration * 1e9)
        if gates:
            entry["gates"] = gates
        m = target["measure"].get((q,))
        if m is not None and m.duration:
            ro_len.append(m.duration * 1e9)
        if "reset" in target.operation_names:
            rs = target["reset"].get((q,))
            if rs is not None and rs.duration:
                reset_len.append(rs.duration * 1e9)
        qubits[str(q)] = entry

    cz_err = [e for e, _ in cz.values()]
    cz_len = [d for _, d in cz.values()]
    couplers = {f"{a}-{b}": {"cz": {"fidelity": _r(1 - e, 8), "duration_ns": _r(d)}} for (a, b), (e, d) in sorted(cz.items())}
    rep_delay_us = float(conf.default_rep_delay) * 1e6 if getattr(conf, "default_rep_delay", None) else None
    reset_ns = _median(reset_len) if reset_len else None

    layout = _ibm_coordinates(n, all_edges, family, 10 if family == "Nighthawk" else None)
    layout["system"] = system
    layout["inactive_couplers"] = inactive
    cmap = {
        "device": f"IBM {label_name} ({system})", "kind": "graph",
        "n_qubits": n, "layout": layout,
        "provenance": (f"Published: IBM's {system} coupling map from its calibration snapshot ({stamp}), shipped in "
                       f"{sdk}. {len(all_edges)} couplers on the chip; {len(inactive)} reported not working "
                       f"(cz gate_error = 1) in the snapshot are removed."),
        "contact": CONTACT,
        "edges": [list(e) for e in edges],
    }
    _write(out_dir / f"{device_id}.json", cmap)

    med = {"sx": _median(sx_err), "x": _median(x_err), "cz": _median(cz_err)}
    provenance = {
        "gates.sx.fidelity": f"{src_tag}; median sx error {med['sx']:.3e} over {len(sx_err)} qubits (RB, average gate error)",
        "gates.sx.duration_ns": f"{src_tag}; median sx length",
        "gates.x.fidelity": f"{src_tag}; median x error {med['x']:.3e} over {len(x_err)} qubits",
        "gates.x.duration_ns": f"{src_tag}; median x length",
        "gates.cz.fidelity": f"{src_tag}; median cz error {med['cz']:.3e} over {len(cz_err)} working couplers",
        "gates.cz.duration_ns": f"{src_tag}; median cz length",
        "readout": f"{src_tag}; median prob_meas1_prep0 / prob_meas0_prep1 and readout_length over {n} qubits",
        "reset.duration_ns": (f"{src_tag}; median reset length" if reset_ns else
                              "assumed: not in the snapshot (reset is not listed); readout length used"),
        "coherence.t1_us": f"{src_tag}; median T1 over {len(t1s)} qubits",
        "coherence.t2_us": f"{src_tag}; median T2 (Hahn echo) over {len(t2s)} qubits",
        "qubits": f"{src_tag}; per-qubit T1, T2, readout and sx / x error and length",
        "couplers": f"{src_tag}; per-coupler cz error and length (worse of the two directions)",
        "shot_overhead_us": (f"published: default_rep_delay = {rep_delay_us:g} us in the {system} backend configuration. "
                             "It is the wait between shots only; other per-shot overheads are not published"
                             if rep_delay_us else "unknown: not published"),
        "feedforward_latency_ns": "unknown: not published in the snapshot",
    }
    if clipped:
        provenance["coherence.t2_clipped"] = f"assumed: measured T2 > 2*T1 clipped to 2*T1 on qubits {clipped}"
    cal = {
        "device": device_id,
        "label": f"IBM {label_name} - {system} calibration snapshot",
        "snapshot_date": stamp,
        "fidelity_convention": "average_gate_fidelity",
        "gates": {
            "sx": {"fidelity": _r(1 - med["sx"], 8), "duration_ns": _r(_median(sx_len))},
            "x": {"fidelity": _r(1 - med["x"], 8), "duration_ns": _r(_median(x_len))},
            "rz": {"fidelity": 1.0, "duration_ns": 0.0},
            "cz": {"fidelity": _r(1 - med["cz"], 8), "duration_ns": _r(_median(cz_len))},
        },
        "readout": {"p01": _r(_median(p01s)), "p10": _r(_median(p10s)), "duration_ns": _r(_median(ro_len))},
        "reset": {"duration_ns": _r(reset_ns if reset_ns else _median(ro_len))},
        "coherence": {"t1_us": _r(_median(t1s)), "t2_us": _r(_median(t2s))},
        "feedforward_latency_ns": None,
        "shot_overhead_us": _r(rep_delay_us) if rep_delay_us else None,
        "qubits": qubits,
        "couplers": couplers,
        "provenance": provenance,
        "sources": [
            f"https://github.com/Qiskit/qiskit-ibm-runtime/tree/main/qiskit_ibm_runtime/fake_provider/backends/{system[4:]}",
            "https://quantum.cloud.ibm.com/computers",
            "https://newsroom.ibm.com/2025-11-12-ibm-delivers-new-quantum-processors,-software,-and-algorithm-breakthroughs-on-path-to-advantage-and-fault-tolerance",
        ],
        "snapshot_summary": {
            "system": system, "processor": f"{proc.get('family')} r{proc.get('revision')}", "qubits": n,
            "couplers_on_chip": len(all_edges), "couplers_working": len(edges),
            "median_cz_error": _r(med["cz"]), "couplers_below_1e-3": sum(e < 1e-3 for e in cz_err),
            "median_sx_error": _r(med["sx"]), "median_readout_error": _r(_median([(a + b) / 2 for a, b in zip(p01s, p10s, strict=True)])),
            "median_t1_us": _r(_median(t1s)), "median_t2_us": _r(_median(t2s)),
            "basis_gates": list(conf.basis_gates), "dt_ns": _r(conf.dt * 1e9),
        },
        "note": (f"Calibration of IBM {system} as IBM published it. Device-wide values are medians; per-qubit and "
                 "per-coupler values override them. Fractional gates (rx(theta), rzz(theta)) are opt-in on IBM "
                 "hardware, are not in this snapshot, and are not modelled. Leakage, crosstalk and drift are not modelled."),
        "contact": CONTACT,
    }
    _write(out_dir / "calibration" / f"{device_id}.json", cal)


# ============================================================================ Google
def import_willow(out_dir: Path) -> None:
    import cirq_google as cg
    from cirq_google.engine import virtual_engine_factory as vef

    warnings.filterwarnings("ignore")
    processor = "willow_pink"
    spec_file = vef.MOST_RECENT_TEMPLATES[processor]
    cal_file = vef.MEDIAN_CALIBRATIONS[processor]
    device = cg.engine.create_device_from_processor_id(processor)
    cal = vef.load_median_device_calibration(processor)
    sdk = f"cirq-google {md.version('cirq-google')}"
    import datetime as dt

    stamp = dt.datetime.fromtimestamp(cal.timestamp / 1000, tz=dt.UTC).isoformat()
    tag = f"published: Google Willow ({processor}) calibration snapshot {cal_file} ({stamp}), shipped in {sdk}"
    spec_tag = f"published: Google Willow ({processor}) device specification {spec_file}, shipped in {sdk}"

    grid = sorted(device.metadata.qubit_set)
    index = {q: i for i, q in enumerate(grid)}
    edges = sorted(tuple(sorted((index[a], index[b]))) for a, b in (tuple(p) for p in device.metadata.qubit_pairs))

    durations: dict[str, float] = {}
    for family, d in device.metadata.gate_durations.items():
        gate = getattr(family, "gate", None)
        name = gate.__name__ if isinstance(gate, type) else str(gate)
        tags = getattr(family, "tags_to_accept", frozenset())
        if tags:
            name += "+physical"
        durations[name] = d.total_nanos()
    t_mw = durations["PhasedXZGate"]
    t_cz = durations["CZ"]
    t_meas = durations["MeasurementGate"]
    t_reset = durations["ResetChannel"]

    def one(metric: str) -> dict[int, float]:
        return {index[k[0]]: float(v[0]) for k, v in cal[metric].items()}

    t1 = one("single_qubit_idle_t1_micros")
    p00 = one("single_qubit_p00_error")
    p11 = one("single_qubit_p11_error")
    rb_pauli = one("single_qubit_rb_pauli_error_per_gate")
    rb_inc = one("single_qubit_rb_incoherent_error_per_gate")
    def two(metric: str) -> dict[tuple[int, int], float]:
        return {tuple(sorted((index[k[0]], index[k[1]]))): float(v[0]) for k, v in cal[metric].items()}

    xeb = two("two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle")
    theta_err = two("two_qubit_parallel_cz_gate_xeb_entangler_theta_error_per_cycle")
    phi_err = two("two_qubit_parallel_cz_gate_xeb_entangler_phi_error_per_cycle")


    clipped: list[int] = []
    t2: dict[int, float] = {}
    for q, t1_us in t1.items():
        e_phi = rb_inc[q] - t_mw / (3 * t1_us * 1000)
        if e_phi <= 0:
            t2[q] = 2 * t1_us                         # pure T1 limit (Cirq uses Tphi = 1e10 ns)
            continue
        tphi_us = t_mw / (3 * e_phi) / 1000
        t2[q] = _clip_t2(t1_us, 1 / (1 / (2 * t1_us) + 1 / tphi_us), q, clipped)

    qubits = {}
    for q in range(len(grid)):
        qubits[str(q)] = {
            "t1_us": _r(t1[q]), "t2_us": _r(t2[q], 8),
            "readout_p01": _r(p00[q]), "readout_p10": _r(p11[q]),
            "gates": {"phased_xz": {"fidelity": _r(1 - rb_pauli[q] * 2 / 3, 8)}},
        }
    couplers = {}
    for (a, b), e in sorted(xeb.items()):
        # the XEB Pauli error is the CZ's total error; the angle errors are a coherent part of it, which
        # QPUTwin's solver (like Google's GoogleNoiseProperties) takes out of the depolarising remainder
        couplers[f"{a}-{b}"] = {"cz": {"fidelity": _r(1 - e * 4 / 5, 8),
                                       "fsim_theta_rad": _r(theta_err.get((a, b), 0.0)),
                                       "fsim_phi_rad": _r(phi_err.get((a, b), 0.0))}}
    med_1q = _median(list(rb_pauli.values())) * 2 / 3
    med_2q = _median(list(xeb.values())) * 4 / 5
    coords = [[q.row, q.col] for q in grid]

    cmap = {
        "device": f"Google Willow ({processor})", "kind": "graph", "n_qubits": len(grid),
        "layout": {"kind": "square_grid", "coordinates": coords, "processor": processor,
                   "note": "GridQubit (row, column) of each index, as on Google's Willow spec sheet"},
        "provenance": (f"Published: Google's Willow coupling graph from {spec_file}, shipped in {sdk}: "
                       f"{len(grid)} GridQubits, {len(edges)} CZ pairs (average connectivity "
                       f"{2 * len(edges) / len(grid):.2f}; spec sheet: 3.47). Index i is the i-th GridQubit in (row, column) order."),
        "contact": CONTACT,
        "edges": [list(e) for e in edges],
    }
    _write(out_dir / "google_willow.json", cmap)

    conv = "converted to average gate error with e_avg = d/(d+1) * e_Pauli"
    calib = {
        "device": "google_willow",
        "label": f"Google Willow - {processor} calibration snapshot",
        "snapshot_date": stamp,
        "fidelity_convention": "average_gate_fidelity",
        "gates": {
            "phased_xz": {"fidelity": _r(1 - med_1q, 8), "duration_ns": _r(t_mw)},
            "rz": {"fidelity": 1.0, "duration_ns": 0.0},
            "cz": {"fidelity": _r(1 - med_2q, 8), "duration_ns": _r(t_cz)},
        },
        "readout": {"p01": _r(_median(list(p00.values()))), "p10": _r(_median(list(p11.values()))),
                    "duration_ns": _r(t_meas)},
        "reset": {"duration_ns": _r(t_reset)},
        "coherence": {"t1_us": _r(_median(list(t1.values()))), "t2_us": _r(_median(list(t2.values())))},
        "feedforward_latency_ns": None,
        "shot_overhead_us": None,
        "qubits": qubits,
        "couplers": couplers,
        "provenance": {
            "gates.phased_xz.fidelity": f"{tag}; median single_qubit_rb_pauli_error_per_gate, {conv}",
            "gates.phased_xz.duration_ns": f"{spec_tag}; PhasedXZGate duration",
            "gates.cz.fidelity": (f"{tag}; median two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle (simultaneous), "
                                  f"{conv}. As in Google's GoogleNoiseProperties it is the CZ's total error: the "
                                  "per-coupler PhasedFSim angle errors are a coherent part of it. The XEB cycle "
                                  "includes a single-qubit layer"),
            "gates.cz.duration_ns": f"{spec_tag}; CZ duration",
            "readout": f"{tag}; median single_qubit_p00_error (p01) and single_qubit_p11_error (p10); duration from the device specification",
            "reset.duration_ns": f"{spec_tag}; ResetChannel duration (multi-level reset)",
            "coherence.t1_us": f"{tag}; median single_qubit_idle_t1_micros",
            "coherence.t2_us": ("published (derived): from single_qubit_rb_incoherent_error_per_gate and T1 with Cirq's "
                                "noise_properties_from_calibration formula; Google publishes no T2 for this snapshot"),
            "qubits": f"{tag}; per-qubit T1, readout and single-qubit RB error; T2 derived as above",
            "couplers": (f"{tag}; per coupler: CZ XEB Pauli error (total) and the signed entangler angle errors "
                         "two_qubit_parallel_cz_gate_xeb_entangler_theta/phi_error_per_cycle (fsim_theta_rad, "
                         "fsim_phi_rad), the coherent part of that total"),
            "shot_overhead_us": "unknown: not published for this chip (the spec sheet gives 63,000 circuit repetitions/s for the RCS chip only)",
            "feedforward_latency_ns": "unknown: not published",
        },
        "sources": [
            "https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf",
            f"https://github.com/quantumlib/Cirq/blob/main/cirq-google/cirq_google/devices/specifications/{spec_file}",
            "https://github.com/quantumlib/Cirq/tree/main/cirq-google/cirq_google/devices/calibrations",
            "https://doi.org/10.1038/s41586-024-08449-y",
        ],
        "spec_sheet_reference": {
            "published": "2024-12-09",
            "qubits": 105, "average_connectivity": 3.47,
            "qec_chip": {"one_qubit_error_mean": "0.035% +/- 0.029%", "two_qubit_cz_error_mean": "0.33% +/- 0.18%",
                         "measurement_error_mean": "0.77% +/- 0.21% (repetitive, measure qubits)",
                         "t1_mean_us": "68 +/- 13", "surface_code_cycle_us": 1.1, "lambda_3_5_7": "2.14 +/- 0.02"},
            "rcs_chip": {"one_qubit_error_mean": "0.036% +/- 0.013%", "two_qubit_error_mean": "0.14% +/- 0.052% (iSWAP-like)",
                         "measurement_error_mean": "0.67% +/- 0.51%", "t1_mean_us": "98 +/- 32",
                         "circuit_repetitions_per_s": 63000},
            "note": "Headline means from the spec sheet, for comparison. The twin itself uses the per-qubit snapshot above.",
        },
        "snapshot_summary": {
            "processor": processor, "qubits": len(grid), "couplers": len(edges),
            "median_one_qubit_pauli_error": _r(_median(list(rb_pauli.values()))),
            "median_cz_xeb_pauli_error_per_cycle": _r(_median(list(xeb.values()))),
            "median_abs_entangler_theta_error_rad": _r(_median([abs(x) for x in theta_err.values()])),
            "median_abs_entangler_phi_error_rad": _r(_median([abs(x) for x in phi_err.values()])),
            "median_p00_error": _r(_median(list(p00.values()))), "median_p11_error": _r(_median(list(p11.values()))),
            "median_t1_us": _r(_median(list(t1.values()))),
            "durations_ns": {k: _r(v) for k, v in sorted(durations.items())},
            "t2_clipped_to_2t1": clipped,
        },
        "note": ("Calibration of Google Willow as Google published it in Cirq. Native gates are PhasedXZ microwave "
                 "pulses (25 ns), virtual Z and CZ (42 ns). Leakage, leakage removal and crosstalk are not modelled."),
        "contact": CONTACT,
    }
    _write(out_dir / "calibration" / "google_willow.json", calib)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--heron", default="fake_boston", help="IBM Heron fake backend (default fake_boston = ibm_boston, r3)")
    ap.add_argument("--nighthawk", default="fake_miami", help="IBM Nighthawk fake backend (default fake_miami = ibm_miami)")
    ap.add_argument("-o", "--out", type=Path, default=DATA, help="output directory (default: packaged devices_data)")
    ap.add_argument("--only", choices=("ibm_heron", "ibm_nighthawk", "google_willow"), action="append")
    args = ap.parse_args()
    only = set(args.only or ("ibm_heron", "ibm_nighthawk", "google_willow"))
    if "ibm_heron" in only:
        import_ibm(args.heron, "ibm_heron", "Heron", args.out)
    if "ibm_nighthawk" in only:
        import_ibm(args.nighthawk, "ibm_nighthawk", "Nighthawk", args.out)
    if "google_willow" in only:
        import_willow(args.out)


if __name__ == "__main__":
    main()
