"""Worker for the IBM Heron r3 twin.

Usage: QPUTWIN_VENDOR=ibm_heron python -m qputwin_worker

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from qputwin_core.devices import IBM_HERON
from qputwin_core.simulators import SuperconductingSimulator
from qputwin_worker.app import create_worker_app

SPEC = IBM_HERON
SIMULATOR_CLASS = SuperconductingSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
