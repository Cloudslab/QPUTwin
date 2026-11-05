"""Safe evaluation of OpenQASM classical parameter expressions.

Expressions are evaluated by walking the AST: Python's ``eval`` is never used, so
untrusted QASM cannot execute code. Supported: real/integer literals, the constants
``pi``/``π``, ``tau``/``τ`` and ``euler``/``ℇ``, the operators ``+ - * / ^ **``
(``^`` is exponentiation, as in OpenQASM 2), unary ``-``/``+``, and the functions
``sin cos tan exp ln sqrt arcsin arccos arctan`` (``asin``/``acos``/``atan`` also
accepted).

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from qputwin_core.errors import QASMSemanticError
from qputwin_core.qasm.ast import BinOp, Call, Expr, Ident, Num, UnaryOp

__all__ = ["evaluate", "CONSTANTS", "FUNCTIONS"]

CONSTANTS: dict[str, float] = {
    "pi": math.pi, "π": math.pi,
    "tau": math.tau, "τ": math.tau,
    "euler": math.e, "ℇ": math.e,
}


def _ln(x: float) -> float:
    if x <= 0:
        raise ValueError("ln of non-positive value")
    return math.log(x)


FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "ln": _ln, "log": _ln, "sqrt": math.sqrt,
    "arcsin": math.asin, "arccos": math.acos, "arctan": math.atan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
}


def evaluate(expr: Expr, env: Mapping[str, float] | None = None, *, line: int | None = None) -> float:
    """Evaluate ``expr`` to a float using gate-parameter bindings in ``env``."""
    env = env or {}
    try:
        return float(_eval(expr, env, line))
    except QASMSemanticError:
        raise
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise QASMSemanticError(f"cannot evaluate parameter expression: {exc}", line=line) from exc


def _eval(e: Expr, env: Mapping[str, float], line: int | None) -> float:
    match e:
        case Num(value=v):
            return v
        case Ident(name=n):
            if n in env:
                return env[n]
            if n in CONSTANTS:
                return CONSTANTS[n]
            raise QASMSemanticError(f"undefined identifier {n!r} in expression", line=e.line or line)
        case UnaryOp(op=op, operand=x):
            v = _eval(x, env, line)
            return -v if op == "-" else v
        case BinOp(op=op, left=a, right=b):
            x, y = _eval(a, env, line), _eval(b, env, line)
            if op == "+":
                return x + y
            if op == "-":
                return x - y
            if op == "*":
                return x * y
            if op == "/":
                return x / y
            if op in ("^", "**"):
                return math.pow(x, y)
            raise QASMSemanticError(f"unsupported operator {op!r}", line=line)
        case Call(func=f, args=args):
            fn = FUNCTIONS.get(f)
            if fn is None:
                raise QASMSemanticError(f"unknown function {f!r}", line=e.line or line)
            if len(args) != 1:
                raise QASMSemanticError(f"function {f!r} takes exactly one argument", line=e.line or line)
            return fn(_eval(args[0], env, line))
    raise QASMSemanticError(f"invalid expression node {e!r}", line=line)  # pragma: no cover
