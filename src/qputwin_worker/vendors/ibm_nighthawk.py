"""Worker for the IBM Nighthawk twin.

Usage: QPUTWIN_VENDOR=ibm_nighthawk python -m qputwin_worker

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.devices import IBM_NIGHTHAWK
from qputwin_core.simulators import SuperconductingSimulator
from qputwin_worker.app import create_worker_app

SPEC = IBM_NIGHTHAWK
SIMULATOR_CLASS = SuperconductingSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
