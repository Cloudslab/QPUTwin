"""IQM Emerald (superconducting, 54 qubits) emulator worker.

Nearest-neighbour connectivity on a 6x9 square lattice enforced via the adjacency matrix. Natives: prx(θ,φ), virtual rz, cz.

Run standalone:  ``QPUTWIN_VENDOR=iqm_emerald uv run python -m qputwin_worker``

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.devices import IQM_EMERALD
from qputwin_core.simulators import SuperconductingSimulator
from qputwin_worker.app import create_worker_app

SPEC = IQM_EMERALD
SIMULATOR_CLASS = SuperconductingSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
