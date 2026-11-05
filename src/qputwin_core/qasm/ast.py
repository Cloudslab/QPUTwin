"""Abstract syntax tree for the supported OpenQASM 2.0 / 3.x language.

Every statement node carries ``line``/``col`` and ``text`` (the exact source slice)
so that compiler diagnostics can quote the user's original statement.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- expressions


@dataclass(frozen=True, slots=True)
class Num:
    value: float


@dataclass(frozen=True, slots=True)
class Ident:
    name: str
    line: int = 0
    col: int = 0


@dataclass(frozen=True, slots=True)
class UnaryOp:
    op: str
    operand: Expr


@dataclass(frozen=True, slots=True)
class BinOp:
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class Call:
    func: str
    args: tuple[Expr, ...]
    line: int = 0
    col: int = 0


Expr = Num | Ident | UnaryOp | BinOp | Call

# ---------------------------------------------------------------- operands


@dataclass(frozen=True, slots=True)
class QArg:
    """``name`` (whole register), ``name[index]`` or hardware qubit ``$n``."""

    name: str
    index: Expr | None = None
    hardware: int | None = None
    line: int = 0
    col: int = 0


@dataclass(frozen=True, slots=True)
class Modifier:
    kind: str                 # "inv" | "ctrl" | "negctrl" | "pow"
    arg: Expr | None = None   # number of controls for ctrl/negctrl, exponent for pow


# ---------------------------------------------------------------- statements


@dataclass(slots=True, kw_only=True)
class Stmt:
    line: int
    col: int
    text: str = ""


@dataclass(slots=True, kw_only=True)
class Include(Stmt):
    path: str


@dataclass(slots=True, kw_only=True)
class QRegDecl(Stmt):
    name: str
    size: int
    scalar: bool = False


@dataclass(slots=True, kw_only=True)
class CRegDecl(Stmt):
    name: str
    size: int
    scalar: bool = False


@dataclass(slots=True, kw_only=True)
class GateCall(Stmt):
    name: str
    params: tuple[Expr, ...] = ()
    qargs: tuple[QArg, ...] = ()
    modifiers: tuple[Modifier, ...] = ()


@dataclass(slots=True, kw_only=True)
class GateDef(Stmt):
    name: str
    params: tuple[str, ...]
    qargs: tuple[str, ...]
    body: list[Stmt] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class Measure(Stmt):
    qarg: QArg
    carg: QArg | None


@dataclass(slots=True, kw_only=True)
class Reset(Stmt):
    qarg: QArg


@dataclass(slots=True, kw_only=True)
class Barrier(Stmt):
    qargs: tuple[QArg, ...]   # empty => all qubits (OpenQASM 3)


@dataclass(slots=True, kw_only=True)
class Delay(Stmt):
    qargs: tuple[QArg, ...]


@dataclass(slots=True, kw_only=True)
class If(Stmt):
    target: QArg              # whole creg or a single bit
    value: int
    equal: bool               # True for ``==``, False for ``!=``
    body: list[Stmt] = field(default_factory=list)


@dataclass(slots=True)
class Program:
    version: str
    statements: list[Stmt]
    source: str = ""
