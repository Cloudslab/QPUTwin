"""Parser for OpenQASM 2.0 and a subset of OpenQASM 3.x.

Usage: from qputwin_core.qasm.parser import parse

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

from qputwin_core.errors import QASMSyntaxError, UnsupportedFeatureError
from qputwin_core.qasm.ast import (
    Barrier, BinOp, Call, CRegDecl, Delay, Expr, GateCall, GateDef, Ident, If, Include,
    Measure, Modifier, Num, Program, QArg, QRegDecl, Reset, Stmt, UnaryOp,
)
from qputwin_core.qasm.lexer import UNSUPPORTED_STATEMENT_KEYWORDS, Token, tokenize

__all__ = ["parse", "Parser"]

# Keywords that OpenQASM 2 programs may legitimately use as identifiers.
_SOFT_KEYWORDS = frozenset({"inv", "ctrl", "negctrl", "pow", "delay", "gphase"})


def parse(source: str, *, permissive: bool = False) -> Program:
    """Parse OpenQASM source text into a :class:`Program`.

    ``permissive`` relaxes the version header requirement and enables the
    OpenQASM 3 feature set; it is used for QPUTwin's embedded gate libraries.
    """
    return Parser(source, permissive=permissive).parse_program()


class Parser:
    def __init__(self, source: str, *, permissive: bool = False) -> None:
        self.source = source
        self.tokens: list[Token] = tokenize(source)
        self.i = 0
        self.permissive = permissive
        self.version = "3.0" if permissive else "2.0"

    # ------------------------------------------------------------- token helpers
    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def peek(self, k: int = 1) -> Token:
        return self.tokens[min(self.i + k, len(self.tokens) - 1)]

    def advance(self) -> Token:
        t = self.tokens[self.i]
        if t.kind != "EOF":
            self.i += 1
        return t

    def at(self, value: str, kind: str | None = None) -> bool:
        t = self.tok
        return t.value == value and (kind is None or t.kind == kind) and t.kind not in ("STRING", "EOF")

    def accept(self, value: str) -> Token | None:
        if self.at(value):
            return self.advance()
        return None

    def expect(self, value: str, what: str | None = None) -> Token:
        if not self.at(value):
            self.error(f"expected {what or repr(value)}, found {self.describe(self.tok)}")
        return self.advance()

    def expect_ident(self, what: str = "identifier") -> Token:
        t = self.tok
        if t.kind == "ID" or (t.kind == "KW" and t.value in _SOFT_KEYWORDS and not self.is_v3):
            return self.advance()
        self.error(f"expected {what}, found {self.describe(t)}")

    def expect_int(self, what: str = "integer") -> int:
        t = self.tok
        if t.kind != "INT":
            self.error(f"expected {what}, found {self.describe(t)}")
        self.advance()
        return int(t.value)

    @staticmethod
    def describe(t: Token) -> str:
        return "end of input" if t.kind == "EOF" else repr(t.value)

    def error(self, msg: str, tok: Token | None = None) -> None:
        t = tok or self.tok
        raise QASMSyntaxError(msg, line=t.line, col=t.col, details={"found": t.value})

    @property
    def is_v3(self) -> bool:
        return self.version.startswith("3")

    def require_v3(self, feature: str, tok: Token) -> None:
        if not self.is_v3:
            raise QASMSyntaxError(f"{feature} requires OPENQASM 3", line=tok.line, col=tok.col)

    def text_from(self, start: Token) -> str:
        prev = self.tokens[self.i - 1] if self.i > 0 else start
        return " ".join(self.source[start.pos:prev.end].split())

    # ------------------------------------------------------------------ program
    def parse_program(self) -> Program:
        if self.at("OPENQASM"):
            start = self.advance()
            t = self.tok
            if t.kind not in ("REAL", "INT"):
                self.error("expected version number after OPENQASM")
            self.advance()
            ver = t.value
            major = ver.split(".")[0]
            if major not in ("2", "3"):
                raise UnsupportedFeatureError(f"unsupported OpenQASM version {ver}", line=start.line)
            if major == "2" and ver not in ("2", "2.0"):
                raise UnsupportedFeatureError(f"unsupported OpenQASM version {ver}", line=start.line)
            self.version = "2.0" if major == "2" else ("3.0" if ver == "3" else ver)
            self.expect(";")
        elif not self.permissive:
            self.error("program must begin with an 'OPENQASM <version>;' header")
        stmts: list[Stmt] = []
        while self.tok.kind != "EOF":
            stmts.append(self.parse_statement(top_level=True))
        return Program(version=self.version, statements=stmts, source=self.source)

    # --------------------------------------------------------------- statements
    def parse_statement(self, *, top_level: bool = False, in_gate: bool = False) -> Stmt:
        t = self.tok
        if t.kind == "ID" and t.value in UNSUPPORTED_STATEMENT_KEYWORDS and self.is_v3:
            raise UnsupportedFeatureError(
                f"OpenQASM 3 '{t.value}' statements are outside the supported subset",
                line=t.line, details={"feature": t.value})
        if t.kind == "KW":
            v = t.value
            if v == "OPENQASM":
                self.error("OPENQASM header may only appear once, at the start of the program")
            if in_gate and v not in ("U", "CX", "gphase", "inv", "ctrl", "negctrl", "pow", "barrier"):
                self.error(f"'{v}' is not allowed inside a gate body")
            if v == "include":
                return self.parse_include(top_level)
            if v in ("qreg", "creg"):
                return self.parse_legacy_decl()
            if v in ("qubit", "bit"):
                if v == "bit" and self.peek().kind != "OP" and self.peek().kind != "ID":
                    self.error("malformed bit declaration")
                return self.parse_v3_decl()
            if v == "gate":
                return self.parse_gate_def(top_level)
            if v == "opaque":
                raise UnsupportedFeatureError(
                    "'opaque' gates have no defined unitary and cannot be emulated",
                    line=t.line, details={"feature": "opaque"})
            if v == "measure":
                return self.parse_measure_arrow()
            if v == "reset":
                return self.parse_reset()
            if v == "barrier":
                return self.parse_barrier()
            if v == "if":
                return self.parse_if()
            if v == "else":
                raise UnsupportedFeatureError("'else' branches are outside the supported subset",
                                              line=t.line, details={"feature": "else"})
            if v == "delay":
                if self.is_v3:
                    return self.parse_delay()
                return self.parse_gate_call()
            if v in ("U", "CX", "gphase", "inv", "ctrl", "negctrl", "pow"):
                return self.parse_gate_call()
        if t.kind in ("ID", "HWQUBIT"):
            # QASM3 assignment form ``c = measure q;`` / ``c[0] = measure q[0];``
            if self.is_v3 and self.looks_like_measure_assignment():
                return self.parse_measure_assign()
            if t.kind == "ID":
                return self.parse_gate_call()
        self.error(f"unexpected {self.describe(t)} at start of statement")
        raise AssertionError  # unreachable

    def looks_like_measure_assignment(self) -> bool:
        j = self.i + 1
        if self.tokens[j].value == "[":
            depth = 0
            while j < len(self.tokens):
                v = self.tokens[j].value
                if v == "[":
                    depth += 1
                elif v == "]":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            j += 1
        return (j + 1 < len(self.tokens) and self.tokens[j].value == "="
                and self.tokens[j].kind == "OP" and self.tokens[j + 1].value == "measure")

    def parse_include(self, top_level: bool) -> Include:
        start = self.advance()
        if not top_level:
            self.error("include is only allowed at top level", start)
        t = self.tok
        if t.kind != "STRING":
            self.error("expected file name string after include")
        self.advance()
        self.expect(";")
        return Include(path=t.value, line=start.line, col=start.col, text=self.text_from(start))

    def parse_legacy_decl(self) -> Stmt:
        start = self.advance()
        name = self.expect_ident("register name").value
        self.expect("[")
        size = self.expect_int("register size")
        self.expect("]")
        self.expect(";")
        if size <= 0:
            self.error("register size must be positive", start)
        cls = QRegDecl if start.value == "qreg" else CRegDecl
        return cls(name=name, size=size, line=start.line, col=start.col, text=self.text_from(start))

    def parse_v3_decl(self) -> Stmt:
        start = self.advance()
        self.require_v3(f"'{start.value}' declarations", start)
        size, scalar = 1, True
        if self.accept("["):
            size = self.expect_int("register size")
            self.expect("]")
            scalar = False
        name = self.expect_ident("register name").value
        if self.at("="):
            raise UnsupportedFeatureError("initialised declarations are outside the supported subset",
                                          line=start.line, details={"feature": "declaration-initialiser"})
        self.expect(";")
        if size <= 0:
            self.error("register size must be positive", start)
        cls = QRegDecl if start.value == "qubit" else CRegDecl
        return cls(name=name, size=size, scalar=scalar, line=start.line, col=start.col,
                   text=self.text_from(start))

    def parse_gate_def(self, top_level: bool) -> GateDef:
        start = self.advance()
        if not top_level:
            self.error("gate definitions are only allowed at top level", start)
        name_tok = self.tok
        if name_tok.kind == "KW" and name_tok.value in ("U", "CX", "gphase"):
            self.error(f"cannot redefine builtin '{name_tok.value}'")
        name = self.expect_ident("gate name").value
        params: list[str] = []
        if self.accept("("):
            if not self.at(")"):
                params.append(self.expect_ident("parameter name").value)
                while self.accept(","):
                    params.append(self.expect_ident("parameter name").value)
            self.expect(")")
        qargs = [self.expect_ident("qubit argument").value]
        while self.accept(","):
            qargs.append(self.expect_ident("qubit argument").value)
        if len(set(params)) != len(params) or len(set(qargs)) != len(qargs):
            self.error(f"duplicate argument names in definition of gate '{name}'", name_tok)
        self.expect("{")
        body: list[Stmt] = []
        while not self.at("}"):
            if self.tok.kind == "EOF":
                self.error(f"unterminated body of gate '{name}'")
            body.append(self.parse_statement(in_gate=True))
        self.expect("}")
        return GateDef(name=name, params=tuple(params), qargs=tuple(qargs), body=body,
                       line=start.line, col=start.col, text=f"gate {name}")

    def parse_measure_arrow(self) -> Measure:
        start = self.advance()
        q = self.parse_qarg()
        carg = None
        if self.accept("->"):
            carg = self.parse_qarg(classical=True)
        elif not self.is_v3:
            self.error("expected '->' in measure statement")
        self.expect(";")
        return Measure(qarg=q, carg=carg, line=start.line, col=start.col, text=self.text_from(start))

    def parse_measure_assign(self) -> Measure:
        start = self.tok
        c = self.parse_qarg(classical=True)
        self.expect("=")
        self.expect("measure")
        q = self.parse_qarg()
        self.expect(";")
        return Measure(qarg=q, carg=c, line=start.line, col=start.col, text=self.text_from(start))

    def parse_reset(self) -> Reset:
        start = self.advance()
        q = self.parse_qarg()
        self.expect(";")
        return Reset(qarg=q, line=start.line, col=start.col, text=self.text_from(start))

    def parse_barrier(self) -> Barrier:
        start = self.advance()
        qargs: list[QArg] = []
        if not self.at(";"):
            qargs.append(self.parse_qarg())
            while self.accept(","):
                qargs.append(self.parse_qarg())
        elif not self.is_v3:
            self.error("barrier requires at least one operand in OpenQASM 2")
        self.expect(";")
        return Barrier(qargs=tuple(qargs), line=start.line, col=start.col, text=self.text_from(start))

    def parse_delay(self) -> Delay:
        start = self.advance()
        self.expect("[")
        depth = 1
        while depth:
            t = self.advance()
            if t.kind == "EOF":
                self.error("unterminated delay duration")
            if t.value == "[":
                depth += 1
            elif t.value == "]":
                depth -= 1
        qargs: list[QArg] = []
        if not self.at(";"):
            qargs.append(self.parse_qarg())
            while self.accept(","):
                qargs.append(self.parse_qarg())
        self.expect(";")
        return Delay(qargs=tuple(qargs), line=start.line, col=start.col, text=self.text_from(start))

    def parse_if(self) -> If:
        start = self.advance()
        self.expect("(")
        negate_bare = bool(self.accept("!"))
        target = self.parse_qarg(classical=True)
        equal, value = True, 1
        if self.at("==") or self.at("!="):
            op = self.advance().value
            if negate_bare:
                self.error("cannot combine '!' with a comparison")
            equal = op == "=="
            t = self.tok
            if t.kind != "INT":
                if t.value in ("true", "false"):
                    self.advance()
                    value = 1 if t.value == "true" else 0
                else:
                    self.error("condition must compare against a non-negative integer literal")
            else:
                self.advance()
                value = int(t.value)
        else:
            if not self.is_v3:
                self.error("expected '==' in if condition")
            equal = not negate_bare
        if not self.is_v3 and not equal:
            self.error("'!=' conditions require OPENQASM 3")
        self.expect(")")
        body: list[Stmt] = []
        if self.at("{"):
            self.require_v3("block 'if' bodies", self.tok)
            self.advance()
            while not self.at("}"):
                if self.tok.kind == "EOF":
                    self.error("unterminated if block")
                body.append(self.parse_quantum_statement())
            self.expect("}")
        else:
            body.append(self.parse_quantum_statement())
        if self.at("else"):
            raise UnsupportedFeatureError("'else' branches are outside the supported subset",
                                          line=self.tok.line, details={"feature": "else"})
        return If(target=target, value=value, equal=equal, body=body,
                  line=start.line, col=start.col, text=self.text_from(start))

    def parse_quantum_statement(self) -> Stmt:
        t = self.tok
        if t.kind == "KW" and t.value in ("gate", "include", "qreg", "creg", "qubit", "bit", "if"):
            self.error(f"'{t.value}' is not allowed in a conditional body")
        return self.parse_statement()

    # ---------------------------------------------------------------- gate call
    def parse_gate_call(self) -> GateCall:
        start = self.tok
        modifiers: list[Modifier] = []
        while self.tok.kind == "KW" and self.tok.value in ("inv", "ctrl", "negctrl", "pow") and (
                self.is_v3 or self.peek().value in ("@", "(")):
            mt = self.advance()
            self.require_v3("gate modifiers", mt)
            arg: Expr | None = None
            if mt.value in ("ctrl", "negctrl", "pow") and self.accept("("):
                arg = self.parse_expr()
                self.expect(")")
            elif mt.value == "pow":
                self.error("pow modifier requires an exponent: pow(k) @")
            self.expect("@", "'@' after gate modifier")
            modifiers.append(Modifier(kind=mt.value, arg=arg))
        name_tok = self.tok
        if name_tok.kind == "KW" and name_tok.value in ("U", "CX", "gphase"):
            self.advance()
            name = name_tok.value
        else:
            name = self.expect_ident("gate name").value
        params: list[Expr] = []
        if self.accept("("):
            if not self.at(")"):
                params.append(self.parse_expr())
                while self.accept(","):
                    params.append(self.parse_expr())
            self.expect(")")
        qargs: list[QArg] = []
        if not self.at(";"):
            qargs.append(self.parse_qarg())
            while self.accept(","):
                qargs.append(self.parse_qarg())
        self.expect(";", "';' after gate call")
        if not qargs and name != "gphase":
            self.error(f"gate call '{name}' has no qubit operands", name_tok)
        return GateCall(name=name, params=tuple(params), qargs=tuple(qargs), modifiers=tuple(modifiers),
                        line=start.line, col=start.col, text=self.text_from(start))

    def parse_qarg(self, *, classical: bool = False) -> QArg:
        t = self.tok
        if t.kind == "HWQUBIT":
            if classical:
                self.error("hardware qubit used where a classical bit was expected")
            self.require_v3("hardware qubits", t)
            self.advance()
            return QArg(name=t.value, hardware=int(t.value[1:]), line=t.line, col=t.col)
        name = self.expect_ident("bit" if classical else "qubit").value
        index: Expr | None = None
        if self.accept("["):
            index = self.parse_expr()
            if self.at(":") or self.at(","):
                raise UnsupportedFeatureError("register slicing is outside the supported subset",
                                              line=t.line, details={"feature": "slice"})
            self.expect("]")
        return QArg(name=name, index=index, line=t.line, col=t.col)

    # -------------------------------------------------------------- expressions
    def parse_expr(self) -> Expr:
        node = self.parse_term()
        while self.at("+") or self.at("-"):
            op = self.advance().value
            node = BinOp(op, node, self.parse_term())
        return node

    def parse_term(self) -> Expr:
        node = self.parse_unary()
        while self.at("*") or self.at("/"):
            op = self.advance().value
            node = BinOp(op, node, self.parse_unary())
        return node

    def parse_unary(self) -> Expr:
        if self.at("-") or self.at("+"):
            op = self.advance().value
            return UnaryOp(op, self.parse_unary())
        return self.parse_power()

    def parse_power(self) -> Expr:
        base = self.parse_atom()
        if self.at("^") or self.at("**"):
            op = self.advance().value
            return BinOp(op, base, self.parse_unary())   # right-associative
        return base

    def parse_atom(self) -> Expr:
        t = self.tok
        if t.kind in ("REAL", "INT"):
            self.advance()
            return Num(float(t.value))
        if t.kind == "ID" or (t.kind == "KW" and t.value in _SOFT_KEYWORDS):
            self.advance()
            if self.accept("("):
                args = [self.parse_expr()]
                while self.accept(","):
                    args.append(self.parse_expr())
                self.expect(")")
                return Call(t.value, tuple(args), t.line, t.col)
            return Ident(t.value, t.line, t.col)
        if self.accept("("):
            e = self.parse_expr()
            self.expect(")")
            return e
        self.error(f"expected expression, found {self.describe(t)}")
        raise AssertionError  # unreachable
