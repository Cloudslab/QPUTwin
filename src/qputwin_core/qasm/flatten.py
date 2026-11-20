"""Checks an OpenQASM syntax tree and flattens it into the circuit representation.

Usage: from qputwin_core.qasm.flatten import load_circuit

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

from qputwin_core.errors import QASMSemanticError, ResourceLimitError, UnsupportedFeatureError
from qputwin_core.ir import Circuit, Condition, Operation, SourceRef
from qputwin_core.qasm import su2
from qputwin_core.qasm.ast import (
    Barrier, CRegDecl, Delay, Expr, GateCall, GateDef, If, Include, Measure, Modifier, Program,
    QArg, QRegDecl, Reset, Stmt,
)
from qputwin_core.qasm.expr import evaluate
from qputwin_core.qasm.parser import parse
from qputwin_core.qasm.stdlib import BUILTIN_INC, LIBRARIES

__all__ = ["flatten", "load_circuit", "BASE_PRIMITIVES", "DEFAULT_MAX_OPS"]

BASE_PRIMITIVES = frozenset({"U", "CX", "gphase"})
DEFAULT_MAX_OPS = 5_000_000
_BUILTIN_SIGS: dict[str, tuple[int, int]] = {"U": (3, 1), "CX": (0, 2), "gphase": (1, 0)}
_PI = math.pi


@dataclass(slots=True)
class GateInfo:
    name: str
    params: tuple[str, ...]
    qargs: tuple[str, ...]
    body: list[Stmt]
    library: bool
    bindings: dict[str, GateInfo | None] = field(default_factory=dict)   # None => builtin

    @property
    def num_params(self) -> int:
        return len(self.params)

    @property
    def num_qubits(self) -> int:
        return len(self.qargs)


@lru_cache(maxsize=None)
def _library_program(name: str) -> Program:
    src = BUILTIN_INC if name == "<builtin>" else LIBRARIES[name]
    return parse(src, permissive=True)


def load_circuit(source: str, *, primitives: Iterable[str] = BASE_PRIMITIVES,
                 max_ops: int = DEFAULT_MAX_OPS) -> Circuit:
    """Parse and flatten OpenQASM ``source`` in one step."""
    return flatten(parse(source), primitives=primitives, max_ops=max_ops)


def flatten(program: Program, *, primitives: Iterable[str] = BASE_PRIMITIVES,
            max_ops: int = DEFAULT_MAX_OPS) -> Circuit:
    return _Flattener(program, frozenset(primitives) | BASE_PRIMITIVES, max_ops).run()


class _Flattener:
    def __init__(self, program: Program, primitives: frozenset[str], max_ops: int) -> None:
        self.program = program
        self.primitives = primitives
        self.max_ops = max_ops
        self.gates: dict[str, GateInfo] = {}
        self.qregs: dict[str, tuple[int, int, bool]] = {}   # name -> (offset, size, scalar)
        self.cregs: dict[str, tuple[int, int, bool]] = {}
        self.nq = 0
        self.nc = 0
        self.max_hw = -1
        self.ops: list[Operation] = []
        self.included: set[str] = set()
        for stmt in _library_program("<builtin>").statements:
            if isinstance(stmt, GateDef):
                self.define_gate(stmt, library=True)

    # ------------------------------------------------------------------ driver
    def run(self) -> Circuit:
        for stmt in self.program.statements:
            self.top_statement(stmt)
        hw = self.max_hw >= 0
        num_qubits = self.max_hw + 1 if hw else self.nq
        return Circuit(
            version=self.program.version,
            num_qubits=num_qubits,
            num_clbits=self.nc,
            qregs={k: (o, s) for k, (o, s, _) in self.qregs.items()},
            cregs={k: (o, s) for k, (o, s, _) in self.cregs.items()},
            ops=self.ops,
            hardware_qubits=hw,
        )

    def emit(self, op: Operation) -> None:
        self.ops.append(op)
        if len(self.ops) > self.max_ops:
            raise ResourceLimitError(
                f"circuit expands to more than {self.max_ops} primitive operations",
                details={"max_ops": self.max_ops})

    def top_statement(self, stmt: Stmt, condition: Condition | None = None) -> None:
        match stmt:
            case Include():
                self.include(stmt)
            case QRegDecl():
                self.declare(stmt, quantum=True)
            case CRegDecl():
                self.declare(stmt, quantum=False)
            case GateDef():
                self.define_gate(stmt, library=False)
            case GateCall():
                self.gate_statement(stmt, condition)
            case Measure():
                self.measure_statement(stmt, condition)
            case Reset():
                origin = self.origin(stmt, "reset")
                for q in self.resolve_qarg(stmt.qarg):
                    self.emit(Operation("reset", (q,), condition=condition, origin=origin))
            case Barrier():
                if condition is not None:
                    return
                origin = self.origin(stmt, "barrier")
                if stmt.qargs:
                    qs: list[int] = []
                    for qa in stmt.qargs:
                        qs.extend(self.resolve_qarg(qa))
                    qubits = tuple(dict.fromkeys(qs))
                else:
                    qubits = tuple(range(self.nq))
                if qubits:
                    self.emit(Operation("barrier", qubits, origin=origin))
            case Delay():
                for qa in stmt.qargs:      # validate operands; an ideal emulator idles exactly
                    self.resolve_qarg(qa)
            case If():
                self.if_statement(stmt)
            case _:  # pragma: no cover
                raise QASMSemanticError(f"unsupported statement {type(stmt).__name__}", line=stmt.line)

    # ---------------------------------------------------------- declarations
    def include(self, stmt: Include) -> None:
        if stmt.path not in LIBRARIES:
            raise QASMSemanticError(
                f"cannot include {stmt.path!r}: only {sorted(LIBRARIES)} are available "
                "(includes never access the filesystem)", line=stmt.line,
                details={"include": stmt.path})
        if stmt.path in self.included:
            return
        self.included.add(stmt.path)
        for s in _library_program(stmt.path).statements:
            if isinstance(s, GateDef):
                self.define_gate(s, library=True)

    def declare(self, stmt: QRegDecl | CRegDecl, *, quantum: bool) -> None:
        if stmt.name in self.qregs or stmt.name in self.cregs:
            raise QASMSemanticError(f"register {stmt.name!r} already declared", line=stmt.line)
        if quantum:
            if self.max_hw >= 0:
                raise QASMSemanticError("cannot declare qubit registers in a program that uses "
                                        "hardware qubits ($n)", line=stmt.line)
            self.qregs[stmt.name] = (self.nq, stmt.size, stmt.scalar)
            self.nq += stmt.size
        else:
            self.cregs[stmt.name] = (self.nc, stmt.size, stmt.scalar)
            self.nc += stmt.size

    def define_gate(self, stmt: GateDef, *, library: bool) -> None:
        if stmt.name in _BUILTIN_SIGS:
            raise QASMSemanticError(f"cannot redefine builtin gate {stmt.name!r}", line=stmt.line)
        prev = self.gates.get(stmt.name)
        if prev is not None and not prev.library:
            raise QASMSemanticError(f"gate {stmt.name!r} is already defined", line=stmt.line)
        formals_q = set(stmt.qargs)
        bindings: dict[str, GateInfo | None] = {}
        for b in stmt.body:
            if isinstance(b, Barrier):
                for qa in b.qargs:
                    if qa.name not in formals_q or qa.index is not None:
                        raise QASMSemanticError(f"barrier in gate {stmt.name!r} must use its formal "
                                                "qubit arguments", line=b.line)
                continue
            if not isinstance(b, GateCall):
                raise QASMSemanticError(f"only gate calls and barriers are allowed in the body of "
                                        f"gate {stmt.name!r}", line=b.line)
            if b.name == stmt.name:
                raise QASMSemanticError(f"gate {stmt.name!r} cannot call itself", line=b.line)
            target = self.lookup(b.name, b.line)
            bindings[b.name] = target
            n_params, n_qubits = self.signature(b.name, target)
            self.check_call_arity(b, n_params, n_qubits)
            for qa in b.qargs:
                if qa.hardware is not None or qa.index is not None or qa.name not in formals_q:
                    raise QASMSemanticError(
                        f"gate {stmt.name!r} body must address only its formal qubit arguments "
                        f"{sorted(formals_q)}; found {qa.name!r}", line=b.line)
        self.gates[stmt.name] = GateInfo(stmt.name, stmt.params, stmt.qargs, stmt.body, library, bindings)

    def lookup(self, name: str, line: int) -> GateInfo | None:
        if name in _BUILTIN_SIGS:
            return None
        info = self.gates.get(name)
        if info is None:
            hint = ""
            if not self.included and name in ("h", "x", "cx", "rz", "ry", "s", "t", "ccx", "swap"):
                hint = ' (did you forget include "qelib1.inc" / "stdgates.inc"?)'
            raise QASMSemanticError(f"unknown gate {name!r}{hint}", line=line, details={"gate": name})
        return info

    @staticmethod
    def signature(name: str, info: GateInfo | None) -> tuple[int, int]:
        if info is None:
            return _BUILTIN_SIGS[name]
        return info.num_params, info.num_qubits

    def check_call_arity(self, call: GateCall, n_params: int, n_qubits: int) -> None:
        if len(call.params) != n_params:
            raise QASMSemanticError(
                f"gate {call.name!r} expects {n_params} parameter(s), got {len(call.params)}", line=call.line)
        extra = sum(self.modifier_count(m, call.line) for m in call.modifiers if m.kind in ("ctrl", "negctrl"))
        expected = n_qubits + extra
        if len(call.qargs) != expected:
            raise QASMSemanticError(
                f"gate {call.name!r} expects {expected} qubit argument(s), got {len(call.qargs)}",
                line=call.line)

    def modifier_count(self, m: Modifier, line: int) -> int:
        if m.arg is None:
            return 1
        v = evaluate(m.arg, line=line)
        if v < 1 or abs(v - round(v)) > 1e-9:
            raise QASMSemanticError(f"{m.kind}(n) requires a positive integer, got {v}", line=line)
        return int(round(v))

    # ----------------------------------------------------------- resolution
    def origin(self, stmt: Stmt, gate: str) -> SourceRef:
        return SourceRef(gate=gate, line=stmt.line, col=stmt.col, text=stmt.text)

    def index_value(self, e: Expr, line: int) -> int:
        v = evaluate(e, line=line)
        if abs(v - round(v)) > 1e-9:
            raise QASMSemanticError(f"register index must be an integer, got {v}", line=line)
        return int(round(v))

    def resolve_qarg(self, qa: QArg) -> list[int]:
        if qa.hardware is not None:
            if self.qregs:
                raise QASMSemanticError("cannot mix hardware qubits ($n) with declared qubit registers",
                                        line=qa.line)
            self.max_hw = max(self.max_hw, qa.hardware)
            return [qa.hardware]
        reg = self.qregs.get(qa.name)
        if reg is None:
            what = "classical register" if qa.name in self.cregs else "undeclared qubit register"
            raise QASMSemanticError(f"{what} {qa.name!r} used as a qubit operand", line=qa.line)
        return self._resolve(qa, reg)

    def resolve_carg(self, ca: QArg) -> list[int]:
        reg = self.cregs.get(ca.name)
        if reg is None:
            what = "qubit register" if ca.name in self.qregs else "undeclared classical register"
            raise QASMSemanticError(f"{what} {ca.name!r} used as a classical operand", line=ca.line)
        return self._resolve(ca, reg)

    def _resolve(self, a: QArg, reg: tuple[int, int, bool]) -> list[int]:
        offset, size, scalar = reg
        if a.index is None:
            return [offset] if scalar else list(range(offset, offset + size))
        if scalar:
            raise QASMSemanticError(f"{a.name!r} is not a register and cannot be indexed", line=a.line)
        i = self.index_value(a.index, a.line)
        if not 0 <= i < size:
            raise QASMSemanticError(f"index {i} out of range for register {a.name!r} of size {size}",
                                    line=a.line, details={"register": a.name, "index": i, "size": size})
        return [offset + i]

    def is_register(self, a: QArg) -> bool:
        if a.hardware is not None or a.index is not None:
            return False
        reg = self.qregs.get(a.name) or self.cregs.get(a.name)
        return reg is not None and not reg[2]

    # ----------------------------------------------------------- statements
    def gate_statement(self, call: GateCall, condition: Condition | None) -> None:
        info = self.lookup(call.name, call.line)
        n_params, n_qubits = self.signature(call.name, info)
        self.check_call_arity(call, n_params, n_qubits)
        params = tuple(evaluate(p, line=call.line) for p in call.params)
        operands = [self.resolve_qarg(qa) for qa in call.qargs]
        regs = [len(o) for o, qa in zip(operands, call.qargs, strict=True) if self.is_register(qa)]
        width = regs[0] if regs else 1
        if any(r != width for r in regs):
            raise QASMSemanticError(f"register operands of {call.name!r} have mismatched sizes {regs}",
                                    line=call.line)
        origin = self.origin(call, call.name)
        for k in range(width):
            qubits = tuple(o[k] if len(o) > 1 or self.is_register(qa) else o[0]
                           for o, qa in zip(operands, call.qargs, strict=True))
            if len(set(qubits)) != len(qubits):
                raise QASMSemanticError(f"gate {call.name!r} applied to repeated qubit operands {list(qubits)}",
                                        line=call.line)
            for op in self.expand(call.name, info, params, qubits, call.modifiers, call.line, origin):
                self.emit(op if condition is None else _with_condition(op, condition))

    def measure_statement(self, stmt: Measure, condition: Condition | None) -> None:
        origin = self.origin(stmt, "measure")
        qs = self.resolve_qarg(stmt.qarg)
        if stmt.carg is None:
            for q in qs:
                self.emit(Operation("measure", (q,), condition=condition, origin=origin))
            return
        cs = self.resolve_carg(stmt.carg)
        if len(qs) != len(cs):
            raise QASMSemanticError(f"measure operands have mismatched sizes ({len(qs)} qubits -> "
                                    f"{len(cs)} bits)", line=stmt.line)
        for q, c in zip(qs, cs, strict=True):
            self.emit(Operation("measure", (q,), clbits=(c,), condition=condition, origin=origin))

    def if_statement(self, stmt: If) -> None:
        clbits = tuple(self.resolve_carg(stmt.target))
        if stmt.value < 0:
            raise QASMSemanticError("condition value must be non-negative", line=stmt.line)
        cond = Condition(clbits=clbits, value=stmt.value, equal=stmt.equal)
        for s in stmt.body:
            self.top_statement(s, cond)

    # --------------------------------------------------------------- expansion
    def expand(self, name: str, info: GateInfo | None, params: tuple[float, ...], qubits: tuple[int, ...],
               modifiers: tuple[Modifier, ...], line: int, origin: SourceRef,
               primitives: frozenset[str] | None = None) -> list[Operation]:
        prims = self.primitives if primitives is None else primitives
        if modifiers:
            return self.apply_modifiers(name, info, params, qubits, modifiers, line, origin)
        if info is None:   # builtin U / CX / gphase
            return [Operation(name, qubits, params, origin=origin)] if name != "gphase" or params[0] else []
        if name in prims and info.library:
            return [Operation(name, qubits, params, origin=origin)]
        env = dict(zip(info.params, params, strict=True))
        qmap = dict(zip(info.qargs, qubits, strict=True))
        out: list[Operation] = []
        for b in info.body:
            if isinstance(b, Barrier):
                continue
            assert isinstance(b, GateCall)
            target = info.bindings[b.name]
            bparams = tuple(evaluate(p, env, line=line) for p in b.params)
            bqubits = tuple(qmap[qa.name] for qa in b.qargs)
            out.extend(self.expand(b.name, target, bparams, bqubits, b.modifiers, line, origin, prims))
        return out

    def apply_modifiers(self, name: str, info: GateInfo | None, params: tuple[float, ...],
                        qubits: tuple[int, ...], modifiers: tuple[Modifier, ...], line: int,
                        origin: SourceRef) -> list[Operation]:
        m, rest = modifiers[0], modifiers[1:]
        if m.kind == "inv":
            inner = self.apply_or_expand(name, info, params, qubits, rest, line, origin)
            return [_inverse(op) for op in reversed(inner)]
        if m.kind == "pow":
            assert m.arg is not None
            k = evaluate(m.arg, line=line)
            if abs(k - round(k)) > 1e-9:
                raise UnsupportedFeatureError(f"pow({k}) @ with a non-integer exponent is outside the "
                                              "supported subset", line=line, details={"feature": "pow"})
            k = int(round(k))
            inner = self.apply_or_expand(name, info, params, qubits, rest, line, origin)
            if k < 0:
                inner = [_inverse(op) for op in reversed(inner)]
            return inner * abs(k)
        n = self.modifier_count(m, line)
        controls, targets = qubits[:n], qubits[n:]
        inner = self.apply_or_expand(name, info, params, targets, rest, line, origin)
        body = _control(inner, list(controls), origin)
        if m.kind == "negctrl":
            flips = [Operation("U", (c,), (_PI, 0.0, _PI), origin=origin) for c in controls]
            body = flips + body + flips
        return body

    def apply_or_expand(self, name: str, info: GateInfo | None, params: tuple[float, ...],
                        qubits: tuple[int, ...], modifiers: tuple[Modifier, ...], line: int,
                        origin: SourceRef) -> list[Operation]:
        # Modified gates are always lowered to the exact {U, CX, gphase} basis.
        return self.expand(name, info, params, qubits, modifiers, line, origin, BASE_PRIMITIVES)


# ----------------------------------------------------------------------- helpers
def _with_condition(op: Operation, cond: Condition) -> Operation:
    return Operation(op.name, op.qubits, op.params, op.clbits, cond, op.origin)


def _inverse(op: Operation) -> Operation:
    if op.name == "U":
        theta, phi, lam = op.params
        return Operation("U", op.qubits, (-theta, -lam, -phi), origin=op.origin)
    if op.name == "CX":
        return op
    if op.name == "gphase":
        return Operation("gphase", (), (-op.params[0],), origin=op.origin)
    raise AssertionError(f"cannot invert non-base primitive {op.name}")  # pragma: no cover


def _control(ops: list[Operation], controls: list[int], origin: SourceRef) -> list[Operation]:
    out: list[Operation] = []
    for op in ops:
        if set(op.qubits) & set(controls):
            raise QASMSemanticError("control qubit is also a target of the controlled gate", line=origin.line)
        if op.name == "gphase":
            out.extend(_mc_phase(controls, op.params[0], origin))
        elif op.name == "U":
            out.extend(_mcu(controls, op.qubits[0], su2.u_matrix(*op.params), origin))
        elif op.name == "CX":
            out.extend(_mcu(controls + [op.qubits[0]], op.qubits[1], su2.PAULI_X, origin))
        else:  # pragma: no cover
            raise AssertionError(op.name)
    return out


def _mc_phase(controls: list[int], gamma: float, origin: SourceRef) -> list[Operation]:
    """C^k(e^{iγ}) == C^{k-1}(P(γ)) acting on the last control."""
    if not controls:
        return [Operation("gphase", (), (gamma,), origin=origin)] if gamma else []
    return _mcu(controls[:-1], controls[-1], (1 + 0j, 0j, 0j, complex(math.cos(gamma), math.sin(gamma))), origin)


def _mcu(controls: list[int], target: int, m: su2.M2, origin: SourceRef) -> list[Operation]:
    """Exact multi-controlled single-qubit unitary over {U, CX, gphase}."""
    if su2.is_identity(m):
        return []
    gamma, theta, phi, lam = su2.to_u_params(m)
    if not controls:
        out = [Operation("U", (target,), (theta, phi, lam), origin=origin)]
        if abs(gamma) > 1e-15:
            out.append(Operation("gphase", (), (gamma,), origin=origin))
        return out
    if len(controls) == 1:
        c = controls[0]
        out = _mc_phase([c], gamma, origin) if abs(gamma) > 1e-15 else []
        out += [
            Operation("U", (c,), (0.0, 0.0, (lam + phi) / 2), origin=origin),
            Operation("U", (target,), (0.0, 0.0, (lam - phi) / 2), origin=origin),
            Operation("CX", (c, target), origin=origin),
            Operation("U", (target,), (-theta / 2, 0.0, -(phi + lam) / 2), origin=origin),
            Operation("CX", (c, target), origin=origin),
            Operation("U", (target,), (theta / 2, phi, 0.0), origin=origin),
        ]
        return out
    v = su2.sqrt_unitary(m)
    vd = su2.dagger(v)
    last, rest = controls[-1], controls[:-1]
    return (_mcu([last], target, v, origin) + _mcu(rest, last, su2.PAULI_X, origin)
            + _mcu([last], target, vd, origin) + _mcu(rest, last, su2.PAULI_X, origin)
            + _mcu(rest, target, v, origin))
