"""From-scratch OpenQASM 2.0 / 3.x front-end (lexer, parser, flattener).

Pure Python — importable without numpy so the orchestrator can validate circuits.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from qputwin_core.qasm.flatten import BASE_PRIMITIVES, DEFAULT_MAX_OPS, flatten, load_circuit
from qputwin_core.qasm.parser import parse

__all__ = ["parse", "flatten", "load_circuit", "BASE_PRIMITIVES", "DEFAULT_MAX_OPS"]
