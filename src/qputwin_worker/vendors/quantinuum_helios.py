"""Worker for the Quantinuum Helios twin.

Usage: QPUTWIN_VENDOR=quantinuum_helios python -m qputwin_worker

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from qputwin_core.devices import QUANTINUUM_HELIOS
from qputwin_core.simulators import TrappedIonSimulator
from qputwin_worker.app import create_worker_app

SPEC = QUANTINUUM_HELIOS
SIMULATOR_CLASS = TrappedIonSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
