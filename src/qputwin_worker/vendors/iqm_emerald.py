"""Worker for the IQM Emerald twin.

Usage: QPUTWIN_VENDOR=iqm_emerald python -m qputwin_worker

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from qputwin_core.devices import IQM_EMERALD
from qputwin_core.simulators import SuperconductingSimulator
from qputwin_worker.app import create_worker_app

SPEC = IQM_EMERALD
SIMULATOR_CLASS = SuperconductingSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
