"""Run the QPUTwin orchestrator: ``python -m qputwin_orchestrator``.

Environment: ``QPUTWIN_HOST`` (0.0.0.0), ``QPUTWIN_PORT`` (8000), ``QPUTWIN_LOG_LEVEL`` (info),
``QPUTWIN_ACCESS_LOG`` (0 — access logging is off by default because it perturbs latency).

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "qputwin_orchestrator.app:app",
        host=os.environ.get("QPUTWIN_HOST", "0.0.0.0"),
        port=int(os.environ.get("QPUTWIN_PORT", "8000")),
        log_level=os.environ.get("QPUTWIN_LOG_LEVEL", "info"),
        access_log=os.environ.get("QPUTWIN_ACCESS_LOG", "0") == "1",
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
