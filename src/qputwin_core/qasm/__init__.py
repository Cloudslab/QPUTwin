"""OpenQASM 2.0 / 3.x front end: lexer, parser and flattener.

Usage: from qputwin_core.qasm import load_circuit

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from qputwin_core.qasm.flatten import BASE_PRIMITIVES, DEFAULT_MAX_OPS, flatten, load_circuit
from qputwin_core.qasm.parser import parse

__all__ = ["parse", "flatten", "load_circuit", "BASE_PRIMITIVES", "DEFAULT_MAX_OPS"]
