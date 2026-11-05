"""Run a vendor worker: ``QPUTWIN_VENDOR=<backend_id> python -m qputwin_worker``.

Environment: ``QPUTWIN_VENDOR`` (required; one of qputwin_worker.vendors.VENDORS),
``QPUTWIN_HOST`` (0.0.0.0), ``QPUTWIN_PORT`` (8001), ``QPUTWIN_LOG_LEVEL`` (info), ``QPUTWIN_ACCESS_LOG`` (0),
plus the simulator settings read by ``SimulatorConfig.from_env`` (``QPUTWIN_MAX_SIM_QUBITS``,
``QPUTWIN_PRECISION``, ``QPUTWIN_MEMORY_GUARD``, ``QPUTWIN_MEMORY_BUDGET_BYTES``, ...).

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import os
import sys

import uvicorn

from qputwin_worker.vendors import VENDORS


def main() -> None:
    vendor = os.environ.get("QPUTWIN_VENDOR", "").strip()
    if vendor not in VENDORS:
        sys.exit(f"QPUTWIN_VENDOR must be one of {', '.join(VENDORS)} (got {vendor!r})")
    uvicorn.run(
        f"qputwin_worker.vendors.{vendor}:app",
        host=os.environ.get("QPUTWIN_HOST", "0.0.0.0"),
        port=int(os.environ.get("QPUTWIN_PORT", "8001")),
        log_level=os.environ.get("QPUTWIN_LOG_LEVEL", "info"),
        access_log=os.environ.get("QPUTWIN_ACCESS_LOG", "0") == "1",
    )


if __name__ == "__main__":
    main()
