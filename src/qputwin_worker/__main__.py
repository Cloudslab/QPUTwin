"""Starts a QPUTwin worker for the device named in QPUTWIN_VENDOR.

Usage: QPUTWIN_VENDOR=ionq_forte python -m qputwin_worker

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
        # keep idle connections open longer than any client reuses them (orchestrator: 30 s, nginx: 60 s);
        # uvicorn's 5 s default let a client reuse a connection the server was closing (rare HTTP 503s)
        timeout_keep_alive=int(os.environ.get("QPUTWIN_KEEPALIVE_S", "75")),
    )


if __name__ == "__main__":
    main()
