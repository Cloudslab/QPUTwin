"""Google Willow (superconducting, 105 qubits) emulator worker.

Nearest-neighbour connectivity on a 105-qubit square grid (Google's willow_pink snapshot) enforced via the adjacency matrix. Natives: phased_xz(θ,φ,ζ), virtual rz, cz.

Run standalone:  ``QPUTWIN_VENDOR=google_willow uv run python -m qputwin_worker``

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.devices import GOOGLE_WILLOW
from qputwin_core.simulators import SuperconductingSimulator
from qputwin_worker.app import create_worker_app

SPEC = GOOGLE_WILLOW
SIMULATOR_CLASS = SuperconductingSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
