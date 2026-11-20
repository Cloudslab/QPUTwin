"""Generates the packaged coupling-map JSON files of the superconducting twins.

Usage: uv run python scripts/generate_coupling_maps.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path

from qputwin_core.topology import grid_edges

OUT = Path(__file__).resolve().parents[1] / "src" / "qputwin_core" / "devices_data"

DEVICES = {
    "rigetti_ankaa_3.json": ("Rigetti Ankaa-3", 7, 12),
    "iqm_emerald.json": ("IQM Emerald", 6, 9),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, (label, rows, cols) in DEVICES.items():
        doc = {
            "device": label,
            "kind": "grid",
            "n_qubits": rows * cols,
            "layout": {"rows": rows, "cols": cols, "index": "r*cols + c"},
            "provenance": (
                f"Synthetic {rows}x{cols} nearest-neighbour square lattice matching the published "
                f"qubit count and connectivity class of {label}. Not a vendor calibration map."
            ),
            "contact": "Koustav Kumar Mondal <erkoustavkumar@gmail.com>",
            "edges": [list(e) for e in grid_edges(rows, cols)],
        }
        (OUT / filename).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {filename}: {rows * cols} qubits, {len(doc['edges'])} couplers")


if __name__ == "__main__":
    main()
