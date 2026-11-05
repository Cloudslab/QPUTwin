"""Convert a vendor-published coupling map into QPUTwin's coupling-map JSON.

Accepted inputs (auto-detected):

* a plain list of pairs ``[[0, 1], [1, 2], ...]`` (Qiskit ``coupling_map`` style;
  directed duplicates are merged);
* an object with one of the keys ``edges`` / ``coupling_map`` / ``couplers`` /
  ``connectivity`` holding pairs, optionally with ``qubits`` listing every qubit
  label (IQM-style ``"QB1"`` labels and Rigetti-style integer ids both work);
* a QPUTwin coupling-map file (re-validated and normalised);
* a text/CSV edge list, one ``a,b`` or ``a b`` pair per line (``#`` comments).

Qubit labels are mapped to dense indices in natural order (``QB2`` < ``QB10``)
unless ``--keep-indices`` is given for integer labels. The label of each index is
recorded in ``layout.labels``. Defective qubits can be listed with
``--disabled`` (by label); any qubit with no couplers can be disabled
automatically with ``--disable-isolated``.

Usage::

    uv run python scripts/import_coupling_map.py vendor_map.json \\
        --device rigetti_ankaa_3 --provenance "Rigetti Ankaa-3 calibration 2026-10-01" \\
        --disabled 5 30 -o config/devices/ankaa3_2026-10-01.json

then start the worker with ``QPUTWIN_COUPLING_MAP=/app/config/devices/ankaa3_2026-10-01.json``
(``QPUTWIN_ANKAA3_COUPLING_MAP`` in ``.env``). No code change is needed.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from qputwin_core.devices import DEVICES
from qputwin_core.topology import CouplingMap

_EDGE_KEYS = ("edges", "coupling_map", "couplers", "connectivity")


def _natural_key(label: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(label))]


def read_pairs(path: Path) -> tuple[list[tuple[str, str]], list[str] | None, dict[str, Any]]:
    """Return ``(pairs, all_labels_or_None, passthrough_metadata)``."""
    text = path.read_text(encoding="utf-8")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        pairs = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                a, b = re.split(r"[,\s]+", line)[:2]
                pairs.append((a, b))
        return pairs, None, {}
    if isinstance(doc, list):
        return [(str(a), str(b)) for a, b in doc], None, {}
    key = next((k for k in _EDGE_KEYS if k in doc), None)
    if key is None:
        raise SystemExit(f"{path}: no edge list found (looked for {', '.join(_EDGE_KEYS)})")
    pairs = [(str(e[0]), str(e[1])) for e in doc[key]]
    labels = [str(q) for q in doc["qubits"]] if isinstance(doc.get("qubits"), list) else None
    meta = {k: doc[k] for k in ("provenance", "layout", "disabled_qubits", "n_qubits", "kind") if k in doc}
    return pairs, labels, meta


def build(pairs: list[tuple[str, str]], labels: list[str] | None, *, keep_indices: bool, n_qubits: int | None,
          disabled: list[str], disable_isolated: bool, provenance: str, device: str | None) -> dict[str, Any]:
    all_labels = set(labels or []) | {x for p in pairs for x in p} | set(disabled)
    if keep_indices:
        if not all(x.lstrip("-").isdigit() for x in all_labels):
            raise SystemExit("--keep-indices needs integer qubit labels")
        index = {x: int(x) for x in all_labels}
        n = max(n_qubits or 0, max(index.values(), default=-1) + 1)
    else:
        ordered = sorted(all_labels, key=_natural_key)
        index = {x: i for i, x in enumerate(ordered)}
        n = max(n_qubits or 0, len(ordered))
    edges = sorted({(min(index[a], index[b]), max(index[a], index[b])) for a, b in pairs if a != b})
    off = {index[x] for x in disabled}
    if disable_isolated:
        degree = {q: 0 for q in range(n)}
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        off |= {q for q, d in degree.items() if d == 0}
    if device is not None:
        spec = DEVICES[device]
        if n != spec.n_physical:
            print(f"warning: {n} qubits, but the packaged {device} spec has {spec.n_physical}; the worker "
                  "will adopt the map's size", file=sys.stderr)
    doc = {
        "device": device, "kind": "graph", "n_qubits": n,
        "layout": {"labels": {str(i): x for x, i in sorted(index.items(), key=lambda t: t[1])}},
        "provenance": provenance, "disabled_qubits": sorted(off), "edges": [list(e) for e in edges],
    }
    CouplingMap.from_dict(doc)                                    # validates indices, self-loops, ranges
    return doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--device", choices=sorted(DEVICES), help="check the size against a packaged device")
    ap.add_argument("--provenance", required=True, help="where the map comes from (quoted in error messages)")
    ap.add_argument("--n-qubits", type=int, help="total qubit count, if some qubits have no couplers")
    ap.add_argument("--disabled", nargs="*", default=[], help="labels of defective qubits")
    ap.add_argument("--disable-isolated", action="store_true", help="disable qubits without couplers")
    ap.add_argument("--keep-indices", action="store_true", help="use integer labels as indices unchanged")
    args = ap.parse_args(argv)
    pairs, labels, meta = read_pairs(args.source)
    disabled = [*args.disabled, *(str(q) for q in meta.get("disabled_qubits", []))]
    doc = build(pairs, labels, keep_indices=args.keep_indices or meta.get("kind") in ("grid", "graph"),
                n_qubits=args.n_qubits or meta.get("n_qubits"), disabled=disabled,
                disable_isolated=args.disable_isolated, provenance=args.provenance, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {doc['n_qubits']} qubits, {len(doc['edges'])} couplers, "
          f"{len(doc['disabled_qubits'])} disabled")


if __name__ == "__main__":
    main()
