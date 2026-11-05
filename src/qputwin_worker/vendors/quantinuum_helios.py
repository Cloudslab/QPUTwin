"""Quantinuum Helios (trapped ion, 98 qubits) emulator worker.

All-to-all connectivity (K_98). Natives: u1q(θ,φ), virtual rz(λ), zzmax, rzz(θ); CX is realised with one ZZMax.

Run standalone:  ``QPUTWIN_VENDOR=quantinuum_helios uv run python -m qputwin_worker``

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.devices import QUANTINUUM_HELIOS
from qputwin_core.simulators import TrappedIonSimulator
from qputwin_worker.app import create_worker_app

SPEC = QUANTINUUM_HELIOS
SIMULATOR_CLASS = TrappedIonSimulator

app = create_worker_app(SPEC, SIMULATOR_CLASS)
