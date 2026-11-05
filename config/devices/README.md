# Calibrated device data (drop-in)

QPUTwin — Koustav Kumar Mondal <erkoustavkumar@gmail.com>

This directory is mounted read-only into every worker container at
`/app/config/devices`. Put vendor-published coupling maps and calibration
snapshots here and select them per QPU in `.env`:

```
QPUTWIN_ANKAA3_COUPLING_MAP=/app/config/devices/<map>.json
QPUTWIN_ANKAA3_CALIBRATION=/app/config/devices/<calibration>.json
```

then `docker compose up -d`. Empty values use the packaged defaults
(`src/qputwin_core/devices_data/`). No code change is needed.

* **Coupling maps** use the format of `src/qputwin_core/devices_data/*.json`
  (`kind` = `grid` | `graph`, `edges`, optional `disabled_qubits`). Convert a
  vendor file with `uv run python scripts/import_coupling_map.py --help`.
* **Calibration snapshots** use the format documented in
  `src/qputwin_core/calibration.py` (gate fidelities and durations, T1/T2,
  readout, per-qubit and per-coupler overrides, provenance).

The two `example_*` files are **synthetic** and exist only to demonstrate and
test the mechanism; they are not vendor data.
