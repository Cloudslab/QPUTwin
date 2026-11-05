"""Optional SABRE qubit router (Li, Ding & Xie, ASPLOS 2019), written from scratch.

QPUTwin's default is still **no routing**: a circuit that violates the coupling
graph is rejected with ``TopologicalCompilationError``. Passing
``options.routing = {"method": "sabre"}`` instead treats the circuit's qubit
indices as *logical* qubits, places them on physical qubits and inserts SWAPs
until every two-qubit operation acts on a coupled pair. The cost of routing
(SWAPs, extra entanglers, depth, and — with a noise model — fidelity) can then
be compared directly against outright rejection.

Algorithm
---------
* **Dependency DAG** over qubit wires *and* classical-bit wires, so
  measurements, resets and classically-controlled operations keep their
  semantics.
* **Front layer** ``F``: operations whose predecessors have all executed. Every
  executable operation (one-qubit, measurement, barrier, or a two-qubit gate on
  a coupled pair) is emitted immediately.
* When ``F`` is blocked, every SWAP on a coupler touching a qubit of ``F`` is
  scored with the SABRE heuristic::

      H = max(decay[p], decay[q]) · ( Σ_{g∈F} D[π(g)] / |F|  +  W · Σ_{g∈E} D[π(g)] / |E| )

  where ``D`` is the all-pairs shortest-path matrix of the coupling graph, ``E``
  the extended (look-ahead) set of the next two-qubit gates, ``W`` the
  look-ahead weight and ``decay`` a per-qubit penalty that discourages
  serialising SWAPs on the same qubits. Ties are broken by a seeded RNG.
* A **release valve** routes the closest blocked gate along a shortest path if
  the heuristic stalls, so routing always terminates.
* **Initial layout** ``"sabre"``: several trials (a compact BFS placement plus
  seeded random placements), each refined by forward/backward passes over the
  two-qubit gate sequence (the reverse-traversal trick of the SABRE paper); the
  layout needing the fewest SWAPs wins. ``"trivial"`` (logical *i* → physical
  *i*) and an explicit list are also accepted.

**Final measurements** (the last operation on their qubit, writing bits that
nothing later reads or overwrites) are held back and appended after routing,
mapped through the *final* layout. Otherwise a later SWAP could route through an
already-measured physical qubit, turning a terminal measurement into a
mid-circuit one, which is wasteful on hardware and forces shot branching in the
emulator.

**Bridges** (``bridge: true``): a blocked CX at distance 2 may be executed in
place through the middle qubit ``m`` as ``CX(a,m) CX(m,b) CX(a,m) CX(m,b)``. It
costs the same four CX as SWAP + CX but leaves the layout untouched, so it is
chosen only when the best SWAP would not bring the look-ahead set closer.

**Cancellation** (``optimize: true``, the default): after routing, pairs of
identical self-inverse two-qubit gates (CX, CZ, …) are removed when everything
between them commutes with them. Commutation is decided exactly on the joint
unitary of the two operations (≤ 4 qubits), not by pattern rules, so the
optimised circuit is the same unitary, global phase included. This typically
removes a CX of a SWAP that meets the user's own CX on the same pair.

Logical qubits are placed only if some operation other than a barrier touches
them, and only on *usable* qubits (defective qubits of a calibrated map are
avoided) inside one connected component. A SWAP is emitted as three ``CX``
operations, which the native compiler then lowers to the device entangler.

Pure Python apart from the numpy commutation check of the cancellation pass.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import heapq
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qputwin_core import gates as G
from qputwin_core.errors import (
    CompilationError, InvalidRequestError, ResourceLimitError, TopologicalCompilationError,
)
from qputwin_core.ir import Circuit, Operation, SourceRef
from qputwin_core.topology import CouplingMap

__all__ = ["RoutingOptions", "RoutingResult", "SabreRouter", "route_circuit"]

_INF = float("inf")


# ===================================================================== options
@dataclass(frozen=True, slots=True)
class RoutingOptions:
    method: str = "sabre"
    initial_layout: str | tuple[int, ...] = "sabre"     # "sabre" | "trivial" | explicit logical->physical
    trials: int = 0                                     # 0 = automatic: 16 / 8 / 4 / 2 by circuit size
    layout_iterations: int = 2                          # forward+backward refinement rounds per trial
    seed: int = 0
    lookahead: int = 20
    lookahead_weight: float = 0.5
    decay: float = 0.001
    decay_reset: int = 5
    bridge: bool = False
    optimize: bool = True
    exact_layout: bool = True                           # place SWAP-free when the interaction graph embeds
    exact_layout_budget: int = 200_000                  # search steps before falling back to SABRE

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RoutingOptions | None:
        """Validate a request's ``options.routing`` (``None`` or ``{"method": "none"}`` = no routing)."""
        if d is None:
            return None
        if not isinstance(d, dict):
            raise InvalidRequestError("options.routing must be an object or null")
        d = dict(d)
        method = d.pop("method", "sabre")
        if method == "none":
            return None
        if method != "sabre":
            raise InvalidRequestError(f"unknown routing method {method!r} (supported: 'none', 'sabre')")
        layout = d.pop("initial_layout", "sabre")
        if isinstance(layout, list):
            if not all(isinstance(x, int) and x >= 0 for x in layout) or len(set(layout)) != len(layout):
                raise InvalidRequestError("options.routing.initial_layout must be distinct non-negative integers")
            layout = tuple(layout)
        elif layout not in ("sabre", "trivial"):
            raise InvalidRequestError("options.routing.initial_layout must be 'sabre', 'trivial' or a list")
        kw: dict[str, Any] = {}
        limits = {"trials": (0, 64), "layout_iterations": (0, 16), "seed": (0, 2 ** 63 - 1),
                  "lookahead": (0, 1000), "decay_reset": (1, 1000)}
        for k, (lo, hi) in limits.items():
            if k in d:
                v = d.pop(k)
                if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
                    raise InvalidRequestError(f"options.routing.{k} must be an integer in [{lo}, {hi}]")
                kw[k] = v
        for k, (lo, hi) in {"lookahead_weight": (0.0, 10.0), "decay": (0.0, 1.0)}.items():
            if k in d:
                v = d.pop(k)
                if not isinstance(v, int | float) or isinstance(v, bool) or not lo <= float(v) <= hi:
                    raise InvalidRequestError(f"options.routing.{k} must be a number in [{lo}, {hi}]")
                kw[k] = float(v)
        if "exact_layout_budget" in d:
            v = d.pop("exact_layout_budget")
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 10_000_000:
                raise InvalidRequestError("options.routing.exact_layout_budget must be an integer in [0, 10000000]")
            kw["exact_layout_budget"] = v
        for k in ("bridge", "optimize", "exact_layout"):
            if k in d:
                v = d.pop(k)
                if not isinstance(v, bool):
                    raise InvalidRequestError(f"options.routing.{k} must be a boolean")
                kw[k] = v
        if d:
            raise InvalidRequestError(f"unknown options.routing field(s): {sorted(d)}")
        return cls(method=method, initial_layout=layout, **kw)


@dataclass(slots=True)
class RoutingResult:
    circuit: Circuit
    initial_layout: dict[int, int]
    final_layout: dict[int, int]
    swaps: int
    stats: dict[str, Any] = field(default_factory=dict)


# ===================================================================== helpers
def _wires(op: Operation) -> list[int]:
    """Qubit wires are ``q``; classical-bit wires are ``-(c + 1)``."""
    w = list(op.qubits) if op.name != "barrier" or op.qubits else []
    if op.condition is not None:
        w += [-(c + 1) for c in op.condition.clbits]
    w += [-(c + 1) for c in op.clbits]
    return w


def _dag(ops: Sequence[Operation]) -> tuple[list[list[int]], list[int]]:
    succ: list[list[int]] = [[] for _ in ops]
    indeg = [0] * len(ops)
    last: dict[int, int] = {}
    for i, op in enumerate(ops):
        preds: set[int] = set()
        for w in _wires(op):
            j = last.get(w)
            if j is not None:
                preds.add(j)
            last[w] = i
        indeg[i] = len(preds)
        for j in preds:
            succ[j].append(i)
    return succ, indeg


def _is_two_qubit_gate(op: Operation) -> bool:
    return op.is_gate and len(op.qubits) == 2


def _final_measurements(ops: Sequence[Operation]) -> set[int]:
    """Measurements that are the last op on their qubit and whose bits nothing later reads or writes."""
    seen_q: set[int] = set()
    seen_c: set[int] = set()
    final: set[int] = set()
    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]
        if (op.name == "measure" and op.condition is None and not set(op.qubits) & seen_q
                and not set(op.clbits) & seen_c):
            final.add(i)
        if op.name != "barrier":
            seen_q.update(op.qubits)
        if op.condition is not None:
            seen_c.update(op.condition.clbits)
        seen_c.update(op.clbits)
    return final


_SWAP_TEXT = "swap ${a}, ${b}  // inserted by the SABRE router"


@dataclass(slots=True)
class _PassResult:
    ops: list[Operation]
    l2p: dict[int, int]
    swaps: int
    valve: int
    bridges: int = 0


# ------------------------------------------------------------- cancellation
_MAT: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}
_COMMUTE: dict[tuple[Any, ...], bool] = {}
_SELF_INV: dict[tuple[str, tuple[float, ...]], bool] = {}
_SYMMETRIC: dict[tuple[str, tuple[float, ...]], bool] = {}


def _unitary(op: Operation) -> np.ndarray | None:
    if op.condition is not None or not op.is_gate:
        return None
    key = (op.name, op.params)
    m = _MAT.get(key)
    if m is None:
        try:
            m = G.matrix(op.name, op.params)
        except Exception:                        # noqa: BLE001 - unknown primitive: treat as non-commuting
            return None
        if len(_MAT) < 4096:
            _MAT[key] = m
    return m


def _embed(m: np.ndarray, pos: tuple[int, ...], k: int) -> np.ndarray:
    """``m`` (first listed qubit most significant) acting on positions ``pos`` of a k-qubit register."""
    u = np.eye(1 << k, dtype=np.complex128).reshape((2,) * k + (-1,))
    axes = [k - 1 - q for q in pos]
    g = m.reshape((2,) * (2 * len(pos)))
    u = np.tensordot(g, u, axes=(list(range(len(pos), 2 * len(pos))), axes))
    return np.moveaxis(u, list(range(len(pos))), axes).reshape(1 << k, -1)


def _commute(a: Operation, b: Operation) -> bool:
    ma, mb = _unitary(a), _unitary(b)
    if ma is None or mb is None:
        return False
    qs = sorted(set(a.qubits) | set(b.qubits))
    if len(qs) > 4:
        return False
    idx = {q: i for i, q in enumerate(qs)}
    pa, pb = tuple(idx[q] for q in a.qubits), tuple(idx[q] for q in b.qubits)
    key = (a.name, a.params, pa, b.name, b.params, pb, len(qs))
    hit = _COMMUTE.get(key)
    if hit is None:
        ea, eb = _embed(ma, pa, len(qs)), _embed(mb, pb, len(qs))
        hit = bool(np.allclose(ea @ eb, eb @ ea, atol=1e-12))
        if len(_COMMUTE) < 65536:
            _COMMUTE[key] = hit
    return hit


def _self_inverse_2q(op: Operation) -> bool:
    if len(op.qubits) != 2 or op.condition is not None or not op.is_gate:
        return False
    key = (op.name, op.params)
    hit = _SELF_INV.get(key)
    if hit is None:
        m = _unitary(op)
        hit = m is not None and bool(np.allclose(m @ m, np.eye(4), atol=1e-12))
        if len(_SELF_INV) < 4096:
            _SELF_INV[key] = hit
    return hit


def _same(a: Operation, b: Operation) -> bool:
    if a.name != b.name or a.params != b.params or a.condition is not None or b.condition is not None:
        return False
    if a.qubits == b.qubits:
        return True
    if len(a.qubits) != 2 or set(a.qubits) != set(b.qubits):
        return False
    key = (a.name, a.params)                     # symmetric gates (CZ, ZZ…) match in either order
    hit = _SYMMETRIC.get(key)
    if hit is None:
        m = _unitary(a)
        hit = m is not None and bool(np.allclose(m, G.swap() @ m @ G.swap(), atol=1e-12))
        if len(_SYMMETRIC) < 4096:
            _SYMMETRIC[key] = hit
    return hit


def cancel_inverse_pairs(ops: list[Operation], *, window: int = 64) -> tuple[list[Operation], int]:
    """Remove pairs of identical self-inverse 2Q gates separated only by commuting operations."""
    ops = list(ops)
    pairs = 0
    for _ in range(4):                           # cancellations can expose further cancellations
        removed = [False] * len(ops)
        found = 0
        for i, op in enumerate(ops):
            if removed[i] or not _self_inverse_2q(op):
                continue
            qset = set(op.qubits)
            j, steps = i - 1, 0
            while j >= 0 and steps < window:
                oj = ops[j]
                if removed[j] or not (qset & set(oj.qubits)):
                    j -= 1
                    continue
                steps += 1
                if _same(oj, op):
                    removed[i] = removed[j] = True
                    found += 1
                    break
                if not _commute(oj, op):
                    break
                j -= 1
        if not found:
            break
        pairs += found
        ops = [op for k, op in enumerate(ops) if not removed[k]]
    return ops, pairs


# ====================================================================== router
class SabreRouter:
    def __init__(self, coupling: CouplingMap, options: RoutingOptions | None = None) -> None:
        self.coupling = coupling
        self.options = options or RoutingOptions()
        self.dist = coupling.distance_matrix()
        self.nbrs = [coupling.neighbors(q) for q in range(coupling.n_qubits)]

    # ------------------------------------------------------------ one pass
    def _pass(self, ops: Sequence[Operation], l2p_init: dict[int, int], rng: random.Random, *,
              emit: bool) -> _PassResult:
        o = self.options
        dist, nbrs = self.dist, self.nbrs
        succ, indeg = _dag(ops)
        l2p = dict(l2p_init)
        p2l = {p: q for q, p in l2p.items()}
        decay = [1.0] * self.coupling.n_qubits
        out: list[Operation] = []
        ready = [i for i, d in enumerate(indeg) if d == 0]
        heapq.heapify(ready)
        front: list[int] = []
        swaps = valve = bridges = 0
        swaps_since_progress = 0
        stall_limit = 10 * max(1, len(l2p)) + 10

        def remap(op: Operation) -> Operation:
            qs = tuple(l2p[q] for q in op.qubits if q in l2p) if op.name == "barrier" \
                else tuple(l2p[q] for q in op.qubits)
            return Operation(op.name, qs, op.params, op.clbits, op.condition, op.origin)

        def executable(op: Operation) -> bool:
            if not op.is_gate or len(op.qubits) < 2:
                return True
            if len(op.qubits) > 2:
                gate = op.origin.gate if op.origin else op.name
                raise CompilationError(f"the router supports at most two-qubit operations; '{gate}' lowers "
                                       f"to the {len(op.qubits)}-qubit primitive '{op.name}'",
                                       details={"gate": gate, "primitive": op.name})
            return dist[l2p[op.qubits[0]]][l2p[op.qubits[1]]] == 1.0

        def do_swap(p: int, q: int) -> None:
            nonlocal swaps
            lp, lq = p2l.get(p), p2l.get(q)
            if lp is not None:
                l2p[lp] = q
            if lq is not None:
                l2p[lq] = p
            p2l.pop(p, None)
            p2l.pop(q, None)
            if lp is not None:
                p2l[q] = lp
            if lq is not None:
                p2l[p] = lq
            swaps += 1
            if emit:
                origin = SourceRef("swap", 0, 0, _SWAP_TEXT.replace("{a}", str(p)).replace("{b}", str(q)))
                out.extend((Operation("CX", (p, q), origin=origin), Operation("CX", (q, p), origin=origin),
                            Operation("CX", (p, q), origin=origin)))

        def pair(i: int) -> tuple[int, int]:
            a, b = ops[i].qubits
            return l2p[a], l2p[b]

        while ready or front:
            # ---- emit everything executable (lowest index first: keeps source order)
            progressed = False
            blocked: list[int] = []
            for i in front:
                heapq.heappush(ready, i)
            while ready:
                i = heapq.heappop(ready)
                op = ops[i]
                if executable(op):
                    if emit:
                        out.append(remap(op))
                    progressed = True
                    for s in succ[i]:
                        indeg[s] -= 1
                        if indeg[s] == 0:
                            heapq.heappush(ready, s)
                else:
                    blocked.append(i)
            front = blocked
            if not front:
                break
            if progressed:
                decay = [1.0] * len(decay)
                swaps_since_progress = 0
            for i in front:
                pa, pb = pair(i)
                if dist[pa][pb] == _INF:
                    op = ops[i]
                    raise TopologicalCompilationError(
                        f"cannot route '{op.origin.text if op.origin else op.name}': physical qubits {pa} and {pb} "
                        "lie in disconnected parts of the coupling graph",
                        details={"qubits": [pa, pb], "line": op.origin.line if op.origin else None,
                                 "routing": "sabre"})

            # ---- release valve: walk the closest blocked gate together along a shortest path
            if swaps_since_progress > stall_limit:
                valve += 1
                i = min(front, key=lambda k: (dist[pair(k)[0]][pair(k)[1]], k))
                pa, pb = pair(i)
                while dist[pa][pb] > 1.0:
                    step = min((v for v in nbrs[pa] if dist[v][pb] == dist[pa][pb] - 1.0))
                    do_swap(pa, step)
                    pa = step
                swaps_since_progress = 0
                continue

            # ---- extended set: upcoming two-qubit gates (breadth-first over successors)
            ext: list[int] = []
            if o.lookahead:
                seen = set(front)
                queue = list(front)
                k = 0
                while k < len(queue) and len(ext) < o.lookahead:
                    for s in succ[queue[k]]:
                        if s not in seen:
                            seen.add(s)
                            queue.append(s)
                            if _is_two_qubit_gate(ops[s]) and all(q in l2p for q in ops[s].qubits):
                                ext.append(s)
                                if len(ext) >= o.lookahead:
                                    break
                    k += 1
            fpairs = [ops[i].qubits for i in front]
            epairs = [ops[i].qubits for i in ext]

            # ---- score every candidate SWAP touching the front layer
            cand: set[tuple[int, int]] = set()
            for i in front:
                for p in pair(i):
                    for v in nbrs[p]:
                        cand.add((min(p, v), max(p, v)))
            best: list[tuple[int, int]] = []
            best_h = _INF
            for p, q in sorted(cand):
                def loc(lq: int, p: int = p, q: int = q) -> int:     # position of lq after SWAP(p, q)
                    x = l2p[lq]
                    return q if x == p else (p if x == q else x)
                h = sum(dist[loc(a)][loc(b)] for a, b in fpairs) / len(fpairs)
                if epairs:
                    h += o.lookahead_weight * sum(dist[loc(a)][loc(b)] for a, b in epairs) / len(epairs)
                h *= max(decay[p], decay[q])
                if h < best_h - 1e-12:
                    best_h, best = h, [(p, q)]
                elif abs(h - best_h) <= 1e-12:
                    best.append((p, q))
            p, q = best[rng.randrange(len(best))] if len(best) > 1 else best[0]
            if o.bridge:
                near = [i for i in front if ops[i].name == "CX" and ops[i].condition is None
                        and dist[pair(i)[0]][pair(i)[1]] == 2.0]
                if near:
                    def moved(lq: int, p: int = p, q: int = q) -> int:
                        x = l2p[lq]
                        return q if x == p else (p if x == q else x)
                    before = sum(dist[l2p[a]][l2p[b]] for a, b in epairs)
                    after = sum(dist[moved(a)][moved(b)] for a, b in epairs)
                    if after >= before:                  # the SWAP does not help the future: bridge instead
                        i = near[0]
                        pa, pb = pair(i)
                        mid = min(v for v in nbrs[pa] if dist[v][pb] == 1.0)
                        if emit:
                            src = ops[i].origin
                            origin = SourceRef("bridge", src.line if src else 0, src.col if src else 0,
                                               f"bridge CX ${pa} -> ${pb} via ${mid}  // SABRE router")
                            out.extend(Operation("CX", qs, origin=origin)
                                       for qs in ((pa, mid), (mid, pb), (pa, mid), (mid, pb)))
                        bridges += 1
                        front.remove(i)
                        for s_ in succ[i]:
                            indeg[s_] -= 1
                            if indeg[s_] == 0:
                                heapq.heappush(ready, s_)
                        swaps_since_progress = 0
                        continue
            do_swap(p, q)
            swaps_since_progress += 1
            decay[p] += o.decay
            decay[q] += o.decay
            if swaps % o.decay_reset == 0:
                decay = [1.0] * len(decay)
        return _PassResult(out, l2p, swaps, valve, bridges)

    # ------------------------------------------------------------ layouts
    def _exact_layout(self, logical: list[int], two_q: Sequence[Operation], budget: int) -> dict[int, int] | None:
        """A SWAP-free layout: an embedding of the interaction graph into the coupling graph.

        Backtracking subgraph-monomorphism search (VF2-style): logical qubits are placed
        in BFS order from the busiest qubit; each candidate must be coupled to the images
        of all already-placed neighbours and have enough free degree. Roots are tried from
        the centre of the chip outwards, keeping layouts compact. Returns ``None`` when no
        embedding exists or the step ``budget`` runs out (SABRE then routes as usual).
        """
        ladj: dict[int, set[int]] = {q: set() for q in logical}
        for op in two_q:
            a, b = op.qubits[0], op.qubits[1]
            if a != b:
                ladj[a].add(b)
                ladj[b].add(a)
        if budget <= 0 or not any(ladj.values()):
            return None
        usable = self.coupling.usable_qubits
        padj = {p: set(self.nbrs[p]) for p in usable}
        if max(len(v) for v in ladj.values()) > max((len(v) for v in padj.values()), default=0):
            return None
        dist = self.dist
        center = min(usable, key=lambda c: (sum(dist[c][k] for k in usable if dist[c][k] < float("inf")), c))
        rank = {p: (dist[center][p], p) for p in usable}
        order: list[int] = []
        seen: set[int] = set()
        for root in sorted(logical, key=lambda q: (-len(ladj[q]), q)):
            if root in seen or not ladj[root]:
                continue
            seen.add(root)
            frontier = [root]
            while frontier:
                u = frontier.pop(0)
                order.append(u)
                for v in sorted(ladj[u] - seen, key=lambda q: (-len(ladj[q]), q)):
                    seen.add(v)
                    frontier.append(v)
        l2p: dict[int, int] = {}
        used: set[int] = set()
        steps = 0

        def place(i: int) -> bool:
            nonlocal steps
            if i == len(order):
                return True
            u = order[i]
            mapped = [l2p[v] for v in ladj[u] if v in l2p]
            cands = set(padj[mapped[0]]).intersection(*(padj[p] for p in mapped[1:])) if mapped else set(usable)
            need = len(ladj[u])
            for p in sorted(cands - used, key=rank.__getitem__):
                if len(padj[p]) < need:
                    continue
                steps += 1
                if steps > budget:
                    return False
                l2p[u] = p
                used.add(p)
                if place(i + 1):
                    return True
                del l2p[u]
                used.discard(p)
            return False

        if not place(0):
            return None
        free = sorted((p for p in usable if p not in used),
                      key=lambda p: (min(dist[p][q] for q in used), rank[p]))
        for q, p in zip((q for q in logical if q not in l2p), free, strict=False):
            l2p[q] = p
        return l2p if len(l2p) == len(logical) else None

    def _component_for(self, n_logical: int) -> list[int]:
        comps = self.coupling.components()
        if not comps or len(comps[0]) < n_logical:
            usable = len(self.coupling.usable_qubits)
            raise ResourceLimitError(
                f"routing needs {n_logical} connected usable physical qubits, but the largest connected "
                f"region of the coupling graph has {len(comps[0]) if comps else 0} ({usable} usable in total)",
                details={"logical_qubits": n_logical, "largest_component": len(comps[0]) if comps else 0,
                         "usable_qubits": usable, "disabled_qubits": list(self.coupling.disabled_qubits)})
        return comps[0]

    def _compact_layout(self, logical: list[int], comp: list[int]) -> dict[int, int]:
        dist = self.dist
        center = min(comp, key=lambda c: (sum(dist[c][k] for k in comp), c))
        order = sorted(comp, key=lambda k: (dist[center][k], k))
        return dict(zip(logical, order, strict=False))

    def _explicit_layout(self, logical: list[int], layout: Sequence[int] | None) -> dict[int, int]:
        n = self.coupling.n_qubits
        if layout is None:
            l2p = {lq: lq for lq in logical}
        else:
            missing = [lq for lq in logical if lq >= len(layout)]
            if missing:
                raise InvalidRequestError(f"initial_layout has no entry for logical qubit(s) {missing}",
                                          details={"layout_length": len(layout), "missing": missing})
            l2p = {lq: int(layout[lq]) for lq in logical}
        bad = [p for p in l2p.values() if not 0 <= p < n or self.coupling.is_disabled(p)]
        if bad:
            raise InvalidRequestError(f"initial layout uses physical qubits {sorted(bad)} that do not exist or "
                                      "are disabled", details={"qubits": sorted(bad),
                                                               "disabled_qubits": list(self.coupling.disabled_qubits)})
        return l2p

    # ---------------------------------------------------------------- run
    def run(self, circuit: Circuit) -> RoutingResult:
        o = self.options
        ops = circuit.ops
        order: dict[int, int] = {}
        for op in ops:
            if op.name != "barrier":
                for q in op.qubits:
                    order.setdefault(q, len(order))
        logical = sorted(order, key=order.__getitem__)            # first-appearance order
        rng = random.Random(o.seed)
        two_q = [op for op in ops if _is_two_qubit_gate(op)]
        trials: list[int] = []
        method = ""
        l2p: dict[int, int] = {}
        if isinstance(o.initial_layout, tuple):
            l2p = self._explicit_layout(logical, o.initial_layout)
            method = "explicit"
        elif o.initial_layout == "trivial":
            l2p = self._explicit_layout(logical, None)
            method = "trivial"
        elif o.exact_layout and (exact := self._exact_layout(logical, two_q, o.exact_layout_budget)) is not None:
            l2p = exact
            method = "exact"
        final = _final_measurements(ops)
        body_in = [op for i, op in enumerate(ops) if i not in final]
        trial_cost: list[int] = []
        if method == "":                                     # SABRE layout trials
            method = "sabre"
            comp = self._component_for(len(logical))
            # each trial routes the whole circuit, so large circuits get fewer (quality gains shrink there)
            nq2 = len(two_q)
            n_trials = o.trials or (16 if nq2 <= 250 else 8 if nq2 <= 1_000 else 4 if nq2 <= 10_000 else 2)
            starts = [self._compact_layout(logical, comp)]
            for _ in range(n_trials - 1):
                starts.append(dict(zip(logical, rng.sample(comp, len(logical)), strict=True)))
            reverse = two_q[::-1]
            best: tuple[int, int, _PassResult, list[Operation], int, dict[int, int]] | None = None
            for t, start in enumerate(starts):
                trng = random.Random(f"{o.seed}:{t}")
                lay = start
                for _ in range(o.layout_iterations):
                    lay = self._pass(two_q, lay, trng, emit=False).l2p
                    lay = self._pass(reverse, lay, trng, emit=False).l2p
                # score each trial by the circuit it actually produces (after cancellation)
                res_t = self._pass(body_in, lay, trng, emit=True)
                body_t, canc_t = cancel_inverse_pairs(res_t.ops) if o.optimize else (res_t.ops, 0)
                cost = sum(1 for op in body_t if _is_two_qubit_gate(op))
                trials.append(res_t.swaps)
                trial_cost.append(cost)
                if best is None or cost < best[0]:
                    best = (cost, t, res_t, body_t, canc_t, lay)
            assert best is not None
            _, _, res, body, cancelled, l2p = best
        else:
            res = self._pass(body_in, l2p, rng, emit=True)
            body, cancelled = cancel_inverse_pairs(res.ops) if o.optimize else (res.ops, 0)
        out_ops = body + [Operation(op.name, tuple(res.l2p[q] for q in op.qubits), op.params, op.clbits,
                                       op.condition, op.origin) for i, op in enumerate(ops) if i in final]
        routed = Circuit(version=circuit.version, num_qubits=self.coupling.n_qubits,
                         num_clbits=circuit.num_clbits, qregs=dict(circuit.qregs), cregs=dict(circuit.cregs),
                         ops=out_ops, hardware_qubits=True)
        stats = {
            "method": "sabre", "initial_layout_method": method, "seed": o.seed,
            "trials": len(trials) or 1, "layout_trial_swaps": trials,
            "layout_trial_two_qubit_gates": trial_cost,
            "swaps_inserted": res.swaps, "bridges_inserted": res.bridges,
            "added_cx": 3 * res.swaps + 3 * res.bridges,
            "cancelled_two_qubit_gates": 2 * cancelled,
            "input_two_qubit_gates": len(two_q),
            "routed_two_qubit_gates": len(two_q) + 3 * res.swaps + 3 * res.bridges - 2 * cancelled,
            "release_valve_activations": res.valve,
            "deferred_final_measurements": len(final),
            "initial_layout": sorted([lq, p] for lq, p in l2p.items()),
            "final_layout": sorted([lq, p] for lq, p in res.l2p.items()),
        }
        return RoutingResult(routed, dict(l2p), dict(res.l2p), res.swaps, stats)


def route_circuit(circuit: Circuit, coupling: CouplingMap, options: RoutingOptions | None = None) -> RoutingResult:
    return SabreRouter(coupling, options).run(circuit)
