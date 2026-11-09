"""IBM Heron r3 (superconducting, 156 qubits) emulator worker.

Nearest-neighbour connectivity on a heavy-hex lattice (IBM's ibm_boston snapshot) enforced via the adjacency matrix. Natives: sx, x, virtual rz, cz.

Run standalone:  ``QPUTWIN_VENDOR=ibm_heron uv run python -m qputwin_worker``

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.devices import IBM_HERON
from qputwin_core.simulators import SuperconductingSimulator
from qputwin_worker.app import create_worker_app

SPEC = IBM_HERON
SIMULATOR_CLASS = SuperconductingSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
