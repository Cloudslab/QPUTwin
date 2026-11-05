"""Turn Amazon Braket device properties into a dated QPUTwin coupling map + calibration snapshot.

Fetch the properties with the AWS CLI (credentials required; the data are free)::

    aws braket get-device --device-arn arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald > emerald.json
    aws braket get-device --device-arn arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3 > ankaa3.json
    aws braket get-device --device-arn arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1 > forte.json

then convert::

    uv run python scripts/import_braket_device.py emerald.json --device iqm_emerald -o config/devices

and select the files in ``.env`` (``QPUTWIN_EMERALD_COUPLING_MAP`` / ``QPUTWIN_EMERALD_CALIBRATION``).

What is read
------------
* ``paradigm.qubitCount`` and ``paradigm.connectivity`` (``fullyConnected`` or the
  ``connectivityGraph`` adjacency lists) → coupling map. Vendor qubit labels are
  mapped to dense indices in natural order and recorded in ``layout.labels``.
* ``standardized`` gate-model properties (schema v1/v2): per qubit ``T1``, ``T2``
  (any time unit) and ``oneQubitFidelity`` entries (``*RANDOMIZED_BENCHMARKING`` →
  every physical native one-qubit gate, ``READOUT`` → readout fidelity); per pair
  ``twoQubitGateFidelity`` entries whose ``gateName`` is a native of the device.
  Device-wide defaults are the medians.
* IonQ ``provider`` properties: ``fidelity`` (``1Q`` / ``2Q`` / ``spam`` means) and
  ``timing`` (``1Q``, ``2Q``, ``readout``, ``reset``, ``T1``, ``T2`` in seconds).

Everything imported is tagged ``published: Amazon Braket device properties
(<ARN>, updated <timestamp>)``; every value Braket does not provide (typically gate
durations of the superconducting devices) keeps the packaged value *and its
packaged provenance tag*, so the gap stays visible. A measured ``T2 > 2·T1`` is
clipped to ``2·T1`` and the clip is recorded.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from qputwin_core.calibration import Calibration
from qputwin_core.devices import DEVICES
from qputwin_core.topology import CouplingMap

_UNIT_TO_US = {"s": 1e6, "ms": 1e3, "us": 1.0, "µs": 1.0, "μs": 1.0, "ns": 1e-3}


def _natural(label: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(label))]


def _to_us(c: dict[str, Any] | None) -> float | None:
    if not c or c.get("value") is None:
        return None
    unit = str(c.get("unit", "s")).strip().lower()
    if unit not in _UNIT_TO_US:
        raise SystemExit(f"unknown time unit {unit!r} in Braket properties")
    return float(c["value"]) * _UNIT_TO_US[unit]


def load_capabilities(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(capabilities, metadata)`` from ``get-device`` output or a bare capabilities object."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    caps = doc.get("deviceCapabilities", doc)
    if isinstance(caps, str):
        caps = json.loads(caps)
    meta = {"arn": doc.get("deviceArn") or caps.get("deviceArn") or "unknown ARN",
            "name": doc.get("deviceName"), "updated": (caps.get("service") or {}).get("updatedAt")}
    return caps, meta


def convert(caps: dict[str, Any], meta: dict[str, Any], device: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    spec = DEVICES[device]
    base = json.loads((Path(__file__).resolve().parents[1] / "src" / "qputwin_core" / "devices_data"
                       / "calibration" / f"{device}.json").read_text(encoding="utf-8"))
    tag = f"published: Amazon Braket device properties ({meta['arn']}, updated {meta['updated'] or 'unknown'})"
    prov: dict[str, str] = dict(base.get("provenance", {}))
    cal: dict[str, Any] = {k: v for k, v in base.items() if k not in ("provenance", "note")}
    cal["label"] = f"{spec.vendor} {spec.name} - Amazon Braket snapshot {meta['updated'] or ''}".strip()
    cal["snapshot_date"] = (meta["updated"] or "")[:10] or None
    cal["sources"] = [f"Amazon Braket get-device {meta['arn']}"]
    physical_1q = [g for g in spec.native_1q if g not in spec.virtual_gates]

    # ------------------------------------------------------------ topology
    paradigm = caps.get("paradigm") or {}
    conn = paradigm.get("connectivity") or {}
    std = caps.get("standardized") or {}
    one = std.get("oneQubitProperties") or {}
    two = std.get("twoQubitProperties") or {}
    labels = set(one) | {x for k in two for x in str(k).split("-")}
    graph = conn.get("connectivityGraph") or {}
    labels |= set(graph) | {str(x) for v in graph.values() for x in v}
    ordered = sorted(labels, key=_natural)
    index = {lab: i for i, lab in enumerate(ordered)}
    cmap_doc = None
    if not conn.get("fullyConnected", spec.coupling.all_to_all):
        n = max(int(paradigm.get("qubitCount") or 0), len(ordered))
        edges = sorted({(min(index[a], index[str(b)]), max(index[a], index[str(b)]))
                        for a, nbrs in graph.items() for b in nbrs if a != str(b)})
        cmap_doc = {"device": device, "kind": "graph", "n_qubits": n,
                    "layout": {"labels": {str(i): lab for lab, i in index.items()}},
                    "provenance": tag.replace("published: ", ""), "disabled_qubits": [],
                    "edges": [list(e) for e in edges]}
        CouplingMap.from_dict(cmap_doc)

    # ------------------------------------------------------- standardized
    qubits: dict[str, Any] = {}
    t1s, t2s, ro, rb = [], [], [], []
    for lab, props in one.items():
        q: dict[str, Any] = {}
        i = str(index[lab])
        t1, t2 = _to_us(props.get("T1")), _to_us(props.get("T2"))
        if t1 is not None and t2 is not None and t2 > 2 * t1:
            prov[f"qubits.{i}.t2_us"] = f"{tag}; measured T2 = {t2:.4g} us clipped to 2*T1"
            t2 = 2 * t1
        if t1 is not None:
            q["t1_us"] = t1
            t1s.append(t1)
        if t2 is not None:
            q["t2_us"] = t2
            t2s.append(t2)
        for f in props.get("oneQubitFidelity") or ():
            name = str((f.get("fidelityType") or {}).get("name", "")).upper()
            if name == "READOUT":
                q["readout_fidelity"] = float(f["fidelity"])
                ro.append(float(f["fidelity"]))
            elif name.endswith("RANDOMIZED_BENCHMARKING"):
                q.setdefault("gates", {}).update({g: {"fidelity": float(f["fidelity"])} for g in physical_1q})
                rb.append(float(f["fidelity"]))
        if q:
            qubits[i] = q
    couplers: dict[str, Any] = {}
    medians_2q: dict[str, list[float]] = {}
    for key, props in two.items():
        a, b = str(key).split("-")
        ek = f"{min(index[a], index[b])}-{max(index[a], index[b])}"
        for f in props.get("twoQubitGateFidelity") or ():
            g = str(f.get("gateName", "")).lower()
            if g in spec.native_2q:
                couplers.setdefault(ek, {})[g] = {"fidelity": float(f["fidelity"])}
                medians_2q.setdefault(g, []).append(float(f["fidelity"]))
    if qubits:
        cal["qubits"] = qubits
    if couplers:
        cal["couplers"] = couplers
    gates = cal.setdefault("gates", {})
    coh = cal.setdefault("coherence", {})
    if t1s:
        coh["t1_us"] = statistics.median(t1s)
        prov["coherence.t1_us"] = f"{tag}; median over {len(t1s)} qubits"
    if t2s:
        coh["t2_us"] = min(statistics.median(t2s), 2 * coh.get("t1_us", float("inf")))
        prov["coherence.t2_us"] = f"{tag}; median over {len(t2s)} qubits"
    if ro:
        cal.setdefault("readout", {})["fidelity"] = statistics.median(ro)
        prov["readout.fidelity"] = f"{tag}; median over {len(ro)} qubits"
    if rb:
        for g in physical_1q:
            gates.setdefault(g, {})["fidelity"] = statistics.median(rb)
            prov[f"gates.{g}.fidelity"] = f"{tag}; median randomized-benchmarking fidelity"
    for g, xs in medians_2q.items():
        gates.setdefault(g, {})["fidelity"] = statistics.median(xs)
        prov[f"gates.{g}.fidelity"] = f"{tag}; median over {len(xs)} couplers"
    if qubits:
        prov["qubits"] = tag
    if couplers:
        prov["couplers"] = tag

    # ------------------------------------------------------- IonQ provider
    provider = caps.get("provider") or {}
    fid, timing = provider.get("fidelity") or {}, provider.get("timing") or {}
    if fid.get("1Q", {}).get("mean") is not None:
        for g in physical_1q:
            gates.setdefault(g, {})["fidelity"] = float(fid["1Q"]["mean"])
            prov[f"gates.{g}.fidelity"] = tag
    if fid.get("2Q", {}).get("mean") is not None:
        for g in spec.native_2q:
            gates.setdefault(g, {})["fidelity"] = float(fid["2Q"]["mean"])
            prov[f"gates.{g}.fidelity"] = tag
    if fid.get("spam", {}).get("mean") is not None:
        cal.setdefault("readout", {})["fidelity"] = float(fid["spam"]["mean"])
        prov["readout.fidelity"] = f"{tag}; SPAM fidelity applied as readout fidelity"
    for key, gs in (("1Q", physical_1q), ("2Q", list(spec.native_2q))):
        if timing.get(key) is not None:
            for g in gs:
                gates.setdefault(g, {})["duration_ns"] = float(timing[key]) * 1e9
                prov[f"gates.{g}.duration_ns"] = tag
    for key, block in (("readout", "readout"), ("reset", "reset")):
        if timing.get(key) is not None:
            cal.setdefault(block, {})["duration_ns"] = float(timing[key]) * 1e9
            prov[f"{block}.duration_ns"] = tag
    for key, field in (("T1", "t1_us"), ("T2", "t2_us")):
        if timing.get(key) is not None:
            coh[field] = float(timing[key]) * 1e6
            prov[f"coherence.{field}"] = tag
    if coh.get("t1_us") and coh.get("t2_us") and coh["t2_us"] > 2 * coh["t1_us"]:
        coh["t2_us"] = 2 * coh["t1_us"]
        prov["coherence.t2_us"] += "; clipped to 2*T1"

    cal["provenance"] = prov
    Calibration.from_dict(cal)                                   # validates the result
    return cmap_doc, cal


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="output of `aws braket get-device` (or its deviceCapabilities)")
    ap.add_argument("--device", required=True, choices=sorted(DEVICES))
    ap.add_argument("-o", "--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    caps, meta = load_capabilities(args.source)
    cmap, cal = convert(caps, meta, args.device)
    stamp = cal["snapshot_date"] or "undated"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_cal = args.output_dir / f"{args.device}_{stamp}_calibration.json"
    out_cal.write_text(json.dumps(cal, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_cal}")
    if cmap is not None:
        out_map = args.output_dir / f"{args.device}_{stamp}_coupling.json"
        out_map.write_text(json.dumps(cmap, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out_map}: {cmap['n_qubits']} qubits, {len(cmap['edges'])} couplers")
    assumed = sorted(k for k, v in cal["provenance"].items() if not str(v).startswith("published"))
    if assumed:
        print(f"note: {len(assumed)} field(s) are still not vendor-published: {', '.join(assumed)}", file=sys.stderr)


if __name__ == "__main__":
    main()
