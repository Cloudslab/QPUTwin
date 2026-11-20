"""Worker for the IonQ Forte twin.

Usage: QPUTWIN_VENDOR=ionq_forte python -m qputwin_worker

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.devices import IONQ_FORTE
from qputwin_core.simulators import TrappedIonSimulator
from qputwin_worker.app import create_worker_app

SPEC = IONQ_FORTE
SIMULATOR_CLASS = TrappedIonSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
