"""Flat circuit representation produced by the OpenQASM front end.

Usage: from qputwin_core.ir import Circuit, Operation

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

from dataclasses import dataclass, field

NON_UNITARY = frozenset({"measure", "reset", "barrier"})


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Provenance of an operation: the top-level user statement it came from."""

    gate: str
    line: int
    col: int
    text: str

    def to_dict(self) -> dict[str, object]:
        return {"gate": self.gate, "line": self.line, "col": self.col, "text": self.text}


@dataclass(frozen=True, slots=True)
class Condition:
    """Classical control: compare the little-endian value of ``clbits`` to ``value``."""

    clbits: tuple[int, ...]
    value: int
    equal: bool = True

    def holds(self, bits: dict[int, int] | list[int]) -> bool:
        v = 0
        for k, cb in enumerate(self.clbits):
            v |= (bits[cb] & 1) << k
        return (v == self.value) == self.equal


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    clbits: tuple[int, ...] = ()
    condition: Condition | None = None
    origin: SourceRef | None = None

    @property
    def is_gate(self) -> bool:
        return self.name not in NON_UNITARY and self.name != "gphase"


@dataclass(slots=True)
class Circuit:
    version: str
    num_qubits: int
    num_clbits: int
    qregs: dict[str, tuple[int, int]] = field(default_factory=dict)   # name -> (offset, size)
    cregs: dict[str, tuple[int, int]] = field(default_factory=dict)
    ops: list[Operation] = field(default_factory=list)
    hardware_qubits: bool = False

    @property
    def gate_ops(self) -> list[Operation]:
        return [op for op in self.ops if op.is_gate]

    @property
    def gate_count(self) -> int:
        return sum(1 for op in self.ops if op.is_gate)

    @property
    def multi_qubit_gate_count(self) -> int:
        return sum(1 for op in self.ops if op.is_gate and len(op.qubits) >= 2)

    def touched_qubits(self) -> list[int]:
        """Physical qubits addressed by any gate, measurement or reset (not barriers)."""
        s: set[int] = set()
        for op in self.ops:
            if op.name != "barrier":
                s.update(op.qubits)
        return sorted(s)

    def max_qubit_index(self) -> int:
        idx = self.touched_qubits()
        return idx[-1] if idx else -1
