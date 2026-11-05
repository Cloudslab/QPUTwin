"""qputwin_core — the shared physics and language core of QPUTwin.

QPUTwin: Calibration-Faithful Digital Twins of Trapped-Ion and Superconducting
Quantum Processors.

Sub-modules
-----------
``qasm``        From-scratch OpenQASM 2.0 / 3.x front-end (pure Python).
``ir``          Flat primitive IR over physical qubits (pure Python).
``gates``       Closed-form unitary library (numpy).
``synthesis``   Phase-exact native-gate compiler (numpy).
``topology``    Coupling maps and adjacency matrices (numpy).
``devices``     Vendor device specifications.
``engine``      Statevector engine: kernels, measurement, shot-branching sampler (numpy).
``density``     Density-matrix engine (numpy).
``trajectory``  Monte-Carlo quantum-trajectory engine (numpy).
``noise``       Calibration-driven noise model.
``calibration`` Fidelities, durations, coherence, readout, provenance (pure Python).
``routing``     Opt-in SABRE router.
``scheduling``  ASAP gate-duration model.
``simulators``  ``TrappedIonSimulator`` and ``SuperconductingSimulator``.

Only the pure-Python modules are imported eagerly, so ``import qputwin_core`` works
in the orchestrator image, which does not ship numpy.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

__version__ = "0.1.0"
__author__ = "Koustav Kumar Mondal"
__email__ = "erkoustavkumar@gmail.com"

__all__ = ["__version__", "__author__", "__email__"]
