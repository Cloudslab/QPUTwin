"""OpenQASM front-end: lexer, parser, semantic analysis and flattening.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math

import pytest

from qputwin_core.errors import QASMSemanticError, QASMSyntaxError, ResourceLimitError, UnsupportedFeatureError
from qputwin_core.qasm import load_circuit, parse
from qputwin_core.qasm.expr import evaluate
from qputwin_core.qasm.parser import Parser

H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
H3 = 'OPENQASM 3.0;\ninclude "stdgates.inc";\n'


def expr(text: str) -> float:
    return evaluate(Parser(text, permissive=True).parse_expr())


def test_expression_evaluator() -> None:
    assert math.isclose(expr("pi/2"), math.pi / 2)
    assert math.isclose(expr("-2^2"), -4.0)              # unary minus binds looser than ^
    assert math.isclose(expr("2^3^2"), 2.0 ** 9)          # right-associative power
    assert math.isclose(expr("2**3"), 8.0)
    assert math.isclose(expr("sin(pi/6) + cos(0)*ln(euler)"), 1.5)
    assert math.isclose(expr("π + τ"), 3 * math.pi)
    assert math.isclose(expr("1.5e-1 * 2"), 0.3)
    with pytest.raises(QASMSemanticError):
        expr("foo + 1")
    with pytest.raises(QASMSemanticError):
        expr("1/0")


def test_qasm2_spec_example_teleportation_parses() -> None:
    src = H2 + """
    qreg q[3]; creg c0[1]; creg c1[1]; creg c2[1];
    // optional state preparation
    u3(0.3,0.2,0.1) q[0];
    h q[1]; cx q[1],q[2];
    barrier q;
    cx q[0],q[1]; h q[0];
    measure q[0] -> c0[0]; measure q[1] -> c1[0];
    if(c0==1) z q[2];
    if(c1==1) x q[2];
    measure q[2] -> c2[0];
    """
    c = load_circuit(src)
    assert c.num_qubits == 3 and c.num_clbits == 3
    assert [op.name for op in c.ops].count("measure") == 3
    conditioned = [op for op in c.ops if op.condition is not None]
    assert conditioned and all(op.condition.clbits in ((0,), (1,)) for op in conditioned)


def test_qft_with_user_gates_and_broadcast() -> None:
    src = H2 + """
    gate cphase_(lam) a, b { cu1(lam) a, b; }
    qreg q[4]; creg c[4];
    h q;
    cphase_(pi/2) q[1], q[0];
    cphase_(pi/4) q[2], q[0];
    measure q -> c;
    """
    c = load_circuit(src)
    assert sum(1 for op in c.ops if op.name == "measure") == 4
    assert {op.origin.gate for op in c.ops if op.name == "CX"} == {"cphase_"}


def test_register_broadcast_two_registers() -> None:
    c = load_circuit(H2 + "qreg a[3]; qreg b[3];\ncx a, b;\n")
    assert [op.qubits for op in c.ops] == [(0, 3), (1, 4), (2, 5)]


def test_mixed_scalar_register_broadcast() -> None:
    c = load_circuit(H2 + "qreg a[1]; qreg b[3];\ncx a[0], b;\n")
    assert [op.qubits for op in c.ops] == [(0, 1), (0, 2), (0, 3)]


def test_hardware_qubits_qasm3() -> None:
    c = load_circuit(H3 + "h $5;\ncx $5, $70;\n")
    assert c.hardware_qubits and c.num_qubits == 71
    assert c.touched_qubits() == [5, 70]


def test_qasm3_measure_forms_and_if_block() -> None:
    c = load_circuit(H3 + "qubit[2] q; bit[2] c;\nh q[0];\nc[0] = measure q[0];\n"
                          "if (c[0] == 1) { x q[1]; z q[1]; }\nc = measure q;\nmeasure q[0];\n")
    assert c.num_clbits == 2
    cond = [op for op in c.ops if op.condition is not None]
    assert len(cond) >= 2 and all(op.condition.clbits == (0,) for op in cond)


@pytest.mark.parametrize("src,needle,line", [
    ("OPENQASM 2.0;\nqreg q[2]\nh q[0];", "expected ';'", 3),
    ("qreg q[1];", "OPENQASM", 1),
    ("OPENQASM 2.0;\nqreg q[2];\ncx q[0] q[1];", "';' after gate call", 3),
    ("OPENQASM 2.0;\nqreg q[2];\nU(0.1, 0.2 q[0];", "expected ')'", 3),
    ("OPENQASM 2.0;\nqreg q[2];\n@ q;", "unexpected", 3),
])
def test_syntax_errors_have_positions(src: str, needle: str, line: int) -> None:
    with pytest.raises(QASMSyntaxError) as ei:
        load_circuit(src)
    assert needle in str(ei.value)
    assert ei.value.details["line"] == line


@pytest.mark.parametrize("src,needle", [
    ("OPENQASM 2.0;\nqreg q[2];\nh q[0];", "did you forget include"),
    (H2 + "qreg q[2];\ncx q[0], q[2];", "out of range"),
    (H2 + "qreg q[2];\ncx q[0], q[0];", "repeated qubit"),
    (H2 + "qreg a[2]; qreg b[3];\ncx a, b;", "mismatched sizes"),
    (H2 + "qreg q[1];\nrx q[0];", "expects 1 parameter"),
    (H2 + "qreg q[1]; creg c[2];\nmeasure q -> c;", "mismatched sizes"),
    ('OPENQASM 2.0;\ninclude "/etc/passwd";', "never access the filesystem"),
    (H2 + "qreg q[1]; qreg q[2];", "already declared"),
    (H2 + "gate g a { g a; }", "cannot call itself"),
    (H2 + "gate g a { foo a; }", "unknown gate"),
    (H2 + "gate g a { h b; }", "formal qubit"),
])
def test_semantic_errors(src: str, needle: str) -> None:
    with pytest.raises(QASMSemanticError) as ei:
        load_circuit(src)
    assert needle in str(ei.value)


@pytest.mark.parametrize("src", [
    H2 + "opaque magic a;",
    H3 + "qubit[2] q;\nfor int i in [0:1] { h q[i]; }",
    H3 + "def f() {}",
    H3 + "qubit[4] q;\nh q[0:2];",
    H3 + "qubit q;\npow(0.5) @ x q;",
])
def test_unsupported_features(src: str) -> None:
    with pytest.raises(UnsupportedFeatureError):
        load_circuit(src)


def test_qasm2_rejects_qasm3_syntax() -> None:
    with pytest.raises(QASMSyntaxError):
        load_circuit(H2 + "qubit[2] q;")
    with pytest.raises(QASMSyntaxError):
        load_circuit(H2 + "qreg q[2];\ninv @ h q[0];")


def test_qasm2_soft_keywords_as_identifiers() -> None:
    c = load_circuit(H2 + "gate g(angle, pow) a { rz(angle*pow) a; }\nqreg ctrl[1];\ng(0.5, 2) ctrl[0];\n")
    assert len(c.ops) == 1 and math.isclose(c.ops[0].params[2], 1.0)


def test_comments_pragmas_and_unicode() -> None:
    c = load_circuit("/* header\n comment */ OPENQASM 3;\n#pragma anything\n"
                     'include "stdgates.inc"; // trailing\nqubit θ;\nrz(π/2) θ;\n')
    assert c.num_qubits == 1


def test_op_limit_guard() -> None:
    src = H2 + "qreg q[3];\n" + "ccx q[0],q[1],q[2];\n" * 50
    with pytest.raises(ResourceLimitError):
        load_circuit(src, max_ops=100)


def test_primitives_are_kept_opaque() -> None:
    src = H2 + "qreg q[2];\nrzz(0.3) q[0], q[1];\nms(0,0,pi/2) q[0], q[1];\n"
    assert [op.name for op in load_circuit(src, primitives={"rzz", "ms"}).ops] == ["rzz", "ms"]
    assert "rzz" not in [op.name for op in load_circuit(src).ops]


def test_user_defined_gate_named_like_native_is_inlined() -> None:
    src = H2 + "gate gpi2(phi) a { rz(phi) a; }\nqreg q[1];\ngpi2(0.2) q[0];\n"
    assert [op.name for op in load_circuit(src, primitives={"gpi2"}).ops] == ["U"]


def test_program_parse_keeps_version() -> None:
    assert parse("OPENQASM 3;").version == "3.0"
    assert parse("OPENQASM 2.0;").version == "2.0"
    with pytest.raises(UnsupportedFeatureError):
        parse("OPENQASM 4.0;")
