"""IBM Nighthawk r1 (superconducting, 120 qubits) emulator worker.

Nearest-neighbour connectivity on a 12x10 square lattice (IBM's ibm_miami snapshot) enforced via the adjacency matrix. Natives: sx, x, virtual rz, cz.

Run standalone:  ``QPUTWIN_VENDOR=ibm_nighthawk uv run python -m qputwin_worker``

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.devices import IBM_NIGHTHAWK
from qputwin_core.simulators import SuperconductingSimulator
from qputwin_worker.app import create_worker_app

SPEC = IBM_NIGHTHAWK
SIMULATOR_CLASS = SuperconductingSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
