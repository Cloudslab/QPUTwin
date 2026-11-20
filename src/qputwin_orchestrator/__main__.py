"""Starts the QPUTwin orchestrator server.

Usage: python -m qputwin_orchestrator

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
        # keep idle connections open longer than any client reuses them (orchestrator: 30 s, nginx: 60 s);
        # uvicorn's 5 s default let a client reuse a connection the server was closing (rare HTTP 503s)
        timeout_keep_alive=int(os.environ.get("QPUTWIN_KEEPALIVE_S", "75")),
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
