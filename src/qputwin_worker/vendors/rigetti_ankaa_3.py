"""Rigetti Ankaa-3 (superconducting, 84 qubits) emulator worker.

Nearest-neighbour connectivity on a 7x12 square lattice enforced via the adjacency matrix. Natives: rx(kπ/2), virtual rz, cz, iswap.

Run standalone:  ``QPUTWIN_VENDOR=rigetti_ankaa_3 uv run python -m qputwin_worker``

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.devices import RIGETTI_ANKAA_3
from qputwin_core.simulators import SuperconductingSimulator
from qputwin_worker.app import create_worker_app

SPEC = RIGETTI_ANKAA_3
SIMULATOR_CLASS = SuperconductingSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
