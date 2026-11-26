"""Splits OpenQASM source text into tokens.

Usage: from qputwin_core.qasm.lexer import tokenize

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from qputwin_core.errors import QASMSyntaxError

__all__ = ["Token", "tokenize", "KEYWORDS", "UNSUPPORTED_STATEMENT_KEYWORDS"]

KEYWORDS = frozenset({
    "OPENQASM", "include", "qreg", "creg", "qubit", "bit", "gate", "opaque", "measure",
    "reset", "barrier", "if", "else", "U", "CX", "gphase", "inv", "ctrl", "negctrl", "pow",
    "delay",
})

# OpenQASM 3 statements outside the supported subset. They are lexed as plain
# identifiers (so that OpenQASM 2 programs may still use e.g. ``angle`` as a gate
# parameter name) and rejected by the parser with a precise UnsupportedFeatureError
# when they appear at the start of a statement.
UNSUPPORTED_STATEMENT_KEYWORDS = frozenset({
    "for", "while", "def", "extern", "defcal", "defcalgrammar", "cal", "const", "input",
    "output", "int", "uint", "float", "angle", "bool", "duration", "stretch", "box", "let",
    "return", "break", "continue", "switch", "array", "complex",
})

_TOKEN_SPEC: list[tuple[str, str]] = [
    ("BLOCK_COMMENT", r"/\*.*?\*/"),
    ("LINE_COMMENT", r"//[^\n]*"),
    ("PRAGMA", r"\#pragma[^\n]*"),
    ("NEWLINE", r"\n"),
    ("WS", r"[ \t\r\f\v]+"),
    ("REAL", r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+"),
    ("INT", r"\d+"),
    ("STRING", r'"[^"\n]*"|\'[^\'\n]*\''),
    ("HWQUBIT", r"\$\d+"),
    ("ID", r"[^\W\d]\w*"),          # Unicode letters/underscore (OpenQASM 3 allows θ, λ, π, ...)
    ("OP", r"->|==|!=|<=|>=|\*\*|\+\+|&&|\|\||[@()\[\]{};,:=+\-*/^<>!~%&|]"),
    ("MISMATCH", r"."),
]
_MASTER = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC), re.DOTALL)


@dataclass(frozen=True, slots=True)
class Token:
    kind: str      # REAL | INT | STRING | HWQUBIT | ID | KW | OP | EOF
    value: str
    line: int
    col: int
    pos: int       # absolute offset of the first character
    end: int       # absolute offset one past the last character

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.kind},{self.value!r}@{self.line}:{self.col})"


def tokenize(source: str) -> list[Token]:
    """Tokenize ``source`` into a list terminated by an ``EOF`` token."""
    tokens: list[Token] = []
    line = 1
    line_start = 0
    for m in _MASTER.finditer(source):
        kind = m.lastgroup
        text = m.group()
        start = m.start()
        col = start - line_start + 1
        if kind == "NEWLINE":
            line += 1
            line_start = m.end()
            continue
        if kind in ("WS", "LINE_COMMENT", "PRAGMA"):
            continue
        if kind == "BLOCK_COMMENT":
            newlines = text.count("\n")
            if newlines:
                line += newlines
                line_start = start + text.rfind("\n") + 1
            continue
        if kind == "MISMATCH":
            if text == "/" and source.startswith("/*", start):
                raise QASMSyntaxError("unterminated block comment", line=line, col=col)
            raise QASMSyntaxError(f"unexpected character {text!r}", line=line, col=col)
        if kind == "ID" and text in KEYWORDS:
            kind = "KW"
        if kind == "STRING":
            text = text[1:-1]
        tokens.append(Token(kind, text, line, col, start, m.end()))  # type: ignore[arg-type]
    tokens.append(Token("EOF", "", line, len(source) - line_start + 1, len(source), len(source)))
    return tokens
