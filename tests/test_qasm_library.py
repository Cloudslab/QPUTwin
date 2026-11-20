"""Tests the built-in gate libraries and OpenQASM 3 gate modifiers.

Usage: uv run pytest tests/test_qasm_library.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qputwin_core import gates as G
from qputwin_core.qasm import load_circuit, parse
from qputwin_core.qasm.ast import GateDef
from qputwin_core.qasm.stdlib import BUILTIN_INC, QELIB1_INC, STDGATES_INC

from conftest import circuit_unitary

PARAMS = (0.37, -1.21, 2.05, 0.66)


def _defs(src: str) -> list[GateDef]:
    return [s for s in parse(src, permissive=True).statements
            if isinstance(s, GateDef) and not s.name.startswith("_qputwin_")]


def _program(header: str, gate: GateDef) -> str:
    n = len(gate.qargs)
    params = ",".join(str(p) for p in PARAMS[: len(gate.params)])
    call = f"{gate.name}({params})" if gate.params else gate.name
    qargs = ", ".join(f"q[{i}]" for i in range(n))
    return f"{header}\nqubit[{n}] q;\n{call} {qargs};\n" if "3" in header.split()[1] else \
        f"{header}\nqreg q[{n}];\n{call} {qargs};\n"


def _check(src: str, gate: GateDef, *, exact: bool) -> None:
    circ = load_circuit(src)                       # base primitives only: full inlining
    n = len(gate.qargs)
    u = circuit_unitary(circ, n)
    # gate matrices use first-qubit-MSB order; reorder to little-endian state order
    ref = G.matrix(gate.name, PARAMS[: len(gate.params)])
    perm = [int(format(i, f"0{n}b")[::-1], 2) for i in range(1 << n)]
    ref = ref[np.ix_(perm, perm)]
    if exact:
        np.testing.assert_allclose(u, ref, atol=1e-10, err_msg=f"{gate.name} not phase-exact")
    else:
        assert G.allclose_up_to_phase(u, ref, atol=1e-10), f"{gate.name} wrong even up to phase"


@pytest.mark.parametrize("gate", _defs(STDGATES_INC), ids=lambda g: g.name)
def test_stdgates_phase_exact(gate: GateDef) -> None:
    _check(_program('OPENQASM 3.0;\ninclude "stdgates.inc";', gate), gate, exact=True)


@pytest.mark.parametrize("gate", _defs(BUILTIN_INC), ids=lambda g: g.name)
def test_builtin_natives_phase_exact(gate: GateDef) -> None:
    _check(_program("OPENQASM 3.0;", gate), gate, exact=True)


@pytest.mark.parametrize("gate", [g for g in _defs(QELIB1_INC) if g.name in G.GATE_TABLE], ids=lambda g: g.name)
def test_qelib1_up_to_phase(gate: GateDef) -> None:
    _check(_program('OPENQASM 2.0;\ninclude "qelib1.inc";', gate), gate, exact=False)


def _u(src: str, n: int) -> np.ndarray:
    return circuit_unitary(load_circuit(f'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[{n}] q;\n{src}'), n)


def test_modifiers_are_exact() -> None:
    np.testing.assert_allclose(_u("ctrl @ ry(0.4) q[0], q[1];", 2), _u("cry(0.4) q[0], q[1];", 2), atol=1e-12)
    np.testing.assert_allclose(_u("ctrl @ x q[1], q[0];", 2), _u("cx q[1], q[0];", 2), atol=1e-12)
    np.testing.assert_allclose(_u("ctrl(2) @ x q[0], q[1], q[2];", 3), _u("ccx q[0], q[1], q[2];", 3), atol=1e-12)
    np.testing.assert_allclose(_u("ctrl @ ctrl @ x q[0], q[1], q[2];", 3), _u("ccx q[0], q[1], q[2];", 3),
                               atol=1e-12)
    np.testing.assert_allclose(_u("inv @ s q[0];", 1), _u("sdg q[0];", 1), atol=1e-12)
    np.testing.assert_allclose(_u("inv @ rz(0.9) q[0];", 1), _u("rz(-0.9) q[0];", 1), atol=1e-12)
    np.testing.assert_allclose(_u("pow(2) @ t q[0];", 1), _u("s q[0];", 1), atol=1e-12)
    np.testing.assert_allclose(_u("pow(-1) @ sx q[0];", 1), _u("sxdg q[0];", 1), atol=1e-12)
    np.testing.assert_allclose(_u("ctrl @ gphase(0.8) q[0];", 1), _u("p(0.8) q[0];", 1), atol=1e-12)
    np.testing.assert_allclose(_u("negctrl @ x q[0], q[1];", 2),
                               _u("x q[0]; cx q[0], q[1]; x q[0];", 2), atol=1e-12)
    # controlled-rz exposes rz's global phase: must equal crz, not cp
    np.testing.assert_allclose(_u("ctrl @ rz(1.3) q[0], q[1];", 2), _u("crz(1.3) q[0], q[1];", 2), atol=1e-12)


def test_three_controls_on_arbitrary_unitary() -> None:
    u = _u("ctrl(3) @ U(0.7, -0.3, 1.9) q[0], q[1], q[2], q[3];", 4)
    ref = np.eye(16, dtype=complex)
    block = [i for i in range(16) if i & 0b0111 == 0b0111]       # q0,q1,q2 = 1
    ref[np.ix_(block, block)] = G.u_gate(0.7, -0.3, 1.9)          # target q3 is the high bit
    np.testing.assert_allclose(u, ref, atol=1e-10)


def test_user_gate_lexical_binding_and_override() -> None:
    src = ('OPENQASM 2.0;\ninclude "qelib1.inc";\n'
           'gate h a { x a; }\n'                       # user override of a library gate
           'qreg q[3];\nccx q[0],q[1],q[2];\nh q[0];\n')
    u = circuit_unitary(load_circuit(src), 3)
    x0 = np.kron(np.eye(4), G.X)                       # X on q0 (little-endian: lowest bit)
    ccx = np.eye(8, dtype=complex)
    ccx[[3, 7]] = ccx[[7, 3]]                          # flips q2 when q0=q1=1
    assert G.allclose_up_to_phase(u, x0 @ ccx), "library ccx must keep the library h"
    assert math.isclose(abs(np.trace(u.conj().T @ x0 @ ccx)), 8.0, rel_tol=1e-12)
