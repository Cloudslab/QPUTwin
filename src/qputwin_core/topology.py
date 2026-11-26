"""Qubit connectivity: coupling maps and adjacency matrices.

Usage: from qputwin_core.topology import CouplingMap

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np

__all__ = ["CouplingMap", "grid_edges", "load_coupling_map", "load_coupling_map_file"]


def grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    """Nearest-neighbour edges of a ``rows × cols`` square lattice (index = r·cols + c)."""
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return edges


@dataclass(slots=True)
class CouplingMap:
    n_qubits: int
    kind: str                                   # "all_to_all" | "grid" | "graph"
    edges: tuple[tuple[int, int], ...] = ()
    layout: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    disabled_qubits: tuple[int, ...] = ()
    _adj: np.ndarray | None = field(default=None, repr=False)
    _dist: list[list[float]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        bad = [q for q in self.disabled_qubits if not 0 <= q < self.n_qubits]
        if bad:
            raise ValueError(f"disabled qubits {bad} out of range for a {self.n_qubits}-qubit device")
        self.disabled_qubits = tuple(sorted(set(self.disabled_qubits)))
        if self.kind == "all_to_all":
            return
        off = set(self.disabled_qubits)
        a = np.zeros((self.n_qubits, self.n_qubits), dtype=bool)
        kept: set[tuple[int, int]] = set()
        for p, q in self.edges:
            if not (0 <= p < self.n_qubits and 0 <= q < self.n_qubits) or p == q:
                raise ValueError(f"invalid coupler ({p}, {q}) for a {self.n_qubits}-qubit device")
            if p in off or q in off:        # a defective qubit takes its couplers down with it
                continue
            a[p, q] = a[q, p] = True
            kept.add((min(p, q), max(p, q)))
        a.setflags(write=False)
        self._adj = a
        self.edges = tuple(sorted(kept))

    # ------------------------------------------------------------------ queries
    @property
    def all_to_all(self) -> bool:
        return self.kind == "all_to_all"

    @property
    def adjacency(self) -> np.ndarray:
        """Dense symmetric boolean adjacency matrix (materialised on demand for K_n)."""
        if self._adj is None:
            a = ~np.eye(self.n_qubits, dtype=bool)
            if self.disabled_qubits:
                off = list(self.disabled_qubits)
                a[off, :] = False
                a[:, off] = False
            a.setflags(write=False)
            return a
        return self._adj

    def is_disabled(self, q: int) -> bool:
        return q in self.disabled_qubits

    @property
    def usable_qubits(self) -> list[int]:
        off = set(self.disabled_qubits)
        return [q for q in range(self.n_qubits) if q not in off]

    def coupled(self, p: int, q: int) -> bool:
        if p == q:
            return False
        if self._adj is None:
            return (0 <= p < self.n_qubits and 0 <= q < self.n_qubits
                    and p not in self.disabled_qubits and q not in self.disabled_qubits)
        return bool(self._adj[p, q])

    def neighbors(self, q: int) -> list[int]:
        if self._adj is None:
            if q in self.disabled_qubits:
                return []
            off = set(self.disabled_qubits)
            return [k for k in range(self.n_qubits) if k != q and k not in off]
        return np.flatnonzero(self._adj[q]).tolist()

    @property
    def n_edges(self) -> int:
        if self._adj is None:
            m = self.n_qubits - len(self.disabled_qubits)
            return m * (m - 1) // 2
        return len(self.edges)

    def degree_stats(self) -> dict[str, float]:
        if self._adj is None:
            d = self.n_qubits - len(self.disabled_qubits) - 1
            return {"min": d, "max": d, "mean": float(d)}
        deg = self._adj.sum(axis=1)
        return {"min": int(deg.min()), "max": int(deg.max()), "mean": float(deg.mean())}

    # --------------------------------------------------------- graph metrics
    def distance_matrix(self) -> list[list[float]]:
        """All-pairs shortest-path lengths (BFS; ``inf`` between disconnected or disabled qubits)."""
        if self._dist is None:
            n = self.n_qubits
            inf = float("inf")
            if self._adj is None:
                off = set(self.disabled_qubits)
                dist = [[inf if (p in off or q in off) and p != q else (0.0 if p == q else 1.0)
                         for q in range(n)] for p in range(n)]
            else:
                nbrs = [self.neighbors(q) for q in range(n)]
                dist = []
                for src in range(n):
                    row = [inf] * n
                    row[src] = 0.0
                    frontier = [src]
                    d = 0.0
                    while frontier:
                        d += 1.0
                        nxt = []
                        for u in frontier:
                            for v in nbrs[u]:
                                if row[v] == inf:
                                    row[v] = d
                                    nxt.append(v)
                        frontier = nxt
                    dist.append(row)
            self._dist = dist
        return self._dist

    def components(self) -> list[list[int]]:
        """Connected components of the usable qubits, largest first (ties: lowest index first)."""
        dist = self.distance_matrix()
        seen: set[int] = set()
        comps: list[list[int]] = []
        for q in self.usable_qubits:
            if q in seen:
                continue
            comp = [k for k in self.usable_qubits if dist[q][k] != float("inf")]
            seen.update(comp)
            comps.append(comp)
        comps.sort(key=lambda c: (-len(c), c[0]))
        return comps

    def to_dict(self, *, include_edges: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind, "n_qubits": self.n_qubits, "n_edges": self.n_edges,
            "degree": self.degree_stats(), "layout": self.layout, "provenance": self.provenance,
            "disabled_qubits": list(self.disabled_qubits),
        }
        if include_edges and not self.all_to_all:
            d["edges"] = [list(e) for e in self.edges]
        return d

    # ------------------------------------------------------------ constructors
    @classmethod
    def complete(cls, n: int, provenance: str = "", disabled_qubits: tuple[int, ...] = ()) -> CouplingMap:
        return cls(n_qubits=n, kind="all_to_all", provenance=provenance, disabled_qubits=tuple(disabled_qubits))

    @classmethod
    def grid(cls, rows: int, cols: int, provenance: str = "") -> CouplingMap:
        return cls(n_qubits=rows * cols, kind="grid", edges=tuple(grid_edges(rows, cols)),
                   layout={"rows": rows, "cols": cols}, provenance=provenance)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CouplingMap:
        kind = d["kind"]
        n = int(d["n_qubits"])
        disabled = tuple(int(q) for q in d.get("disabled_qubits", ()))
        if kind == "all_to_all":
            return cls.complete(n, d.get("provenance", ""), disabled)
        if kind not in ("grid", "graph"):
            raise ValueError(f"unknown coupling-map kind {kind!r} (expected all_to_all | grid | graph)")
        edges = tuple((int(p), int(q)) for p, q in d["edges"])
        return cls(n_qubits=n, kind=kind, edges=edges, layout=dict(d.get("layout", {})),
                   provenance=d.get("provenance", ""), disabled_qubits=disabled)


def load_coupling_map(filename: str) -> CouplingMap:
    """Load a coupling map JSON from the packaged ``devices_data`` directory."""
    text = resources.files("qputwin_core.devices_data").joinpath(filename).read_text(encoding="utf-8")
    return CouplingMap.from_dict(json.loads(text))


def load_coupling_map_file(path: str) -> CouplingMap:
    """Load a coupling map JSON from an arbitrary path (runtime override, ``QPUTWIN_COUPLING_MAP``)."""
    from pathlib import Path

    return CouplingMap.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
