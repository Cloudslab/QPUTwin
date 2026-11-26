"""Worker for the Google Willow twin.

Usage: QPUTWIN_VENDOR=google_willow python -m qputwin_worker

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from qputwin_core.devices import GOOGLE_WILLOW
from qputwin_core.simulators import SuperconductingSimulator
from qputwin_worker.app import create_worker_app

SPEC = GOOGLE_WILLOW
SIMULATOR_CLASS = SuperconductingSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
