"""IonQ Forte (trapped ion, 36 qubits) emulator worker.

All-to-all connectivity (K_36). Natives: gpi(φ), gpi2(φ), virtual rz, ms(φ0,φ1,θ), zz(θ); CX is realised with one MS.

Run standalone:  ``QPUTWIN_VENDOR=ionq_forte uv run python -m qputwin_worker``

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.devices import IONQ_FORTE
from qputwin_core.simulators import TrappedIonSimulator
from qputwin_worker.app import create_worker_app

SPEC = IONQ_FORTE
SIMULATOR_CLASS = TrappedIonSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
