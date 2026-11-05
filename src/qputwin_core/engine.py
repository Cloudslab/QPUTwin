"""From-scratch numpy statevector engine.

State layout
------------
A dense vector ``ψ ∈ C^{2^n}`` (``complex128`` or ``complex64``) in little-endian
order: bit ``q`` of the flat index is simulated qubit ``q``. A one-qubit kernel on
qubit ``q`` views ψ as ``(2^{n-q-1}, 2, 2^q)``; a two-qubit kernel on ``(a, b)``
views it as ``(2^{n-h-1}, 2, 2^{h-l-1}, 2, 2^l)`` with ``h = max(a, b)``,
``l = min(a, b)``. Both kernels update ψ **in place** (peak working set ≈ 2× the
state), and diagonal gates (``rz``, ``cz``, ``rzz``, ``zzmax`` …) reduce to
element-wise phase multiplications with no extra memory.

Sampling
--------
*Terminal fast path* — when every measurement is terminal (its qubit is never
touched again and its bit never read by a condition) the circuit is simulated
once and all shots are drawn from the marginal with one multinomial draw:
``O(2^n + shots)``.

*Shot branching* — mid-circuit measurements, resets and classically-controlled
operations are handled by splitting the shot budget at each projective event:
``n1 ~ Binomial(n, p1)``, ``n0 = n − n1``, and continuing each non-empty branch
with its collapsed state. This samples exactly the same distribution as
independent per-shot trajectories, but costs one simulation per *distinct
reachable classical branch* rather than one per shot.

*Fused execution* — when no branching is needed (the common case) the gate list
is fused into dense blocks of up to ``FUSION_MAX_QUBITS`` qubits and diagonal
blocks of up to ``FUSION_MAX_DIAGONAL_QUBITS``, and executed by
:class:`~qputwin_core.fused.FusedState`: one pass per block over a lazily
activated, lazily permuted tensor, multithreaded. Terminal measurements are then
sampled from the stored order without transposing the state back. The per-gate
kernels below stay the reference implementation and serve the branching path.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qputwin_core.fused import FusedState, plan_blocks
from qputwin_core.ir import Condition
from qputwin_core.synthesis import NativeCircuit, NativeOp
from qputwin_core.timing import SpanTimer

__all__ = ["FUSION_MAX_DIAGONAL_QUBITS", "FUSION_MAX_QUBITS", "Program", "RunOutput", "StatevectorEngine",
           "fused_state", "fused_statevector", "prepare", "run"]

FUSION_MAX_QUBITS = 5            # widest fused statevector block (measured optimum on 20–24 qubits)
FUSION_MAX_DIAGONAL_QUBITS = 12  # widest fused phase block (a 4096-entry table, applied per cache chunk)


# ===================================================================== kernels
class StatevectorEngine:
    def __init__(self, n: int, dtype: np.dtype | type = np.complex128, psi: np.ndarray | None = None) -> None:
        self.n = n
        self.dtype = np.dtype(dtype)
        if psi is None:
            psi = np.zeros(1 << n, dtype=self.dtype)
            psi[0] = 1.0
        self.psi = psi

    def copy(self) -> StatevectorEngine:
        return StatevectorEngine(self.n, self.dtype, self.psi.copy())

    # --------------------------------------------------------------- gates
    def apply_1q(self, q: int, m: np.ndarray) -> None:
        v = self.psi.reshape(-1, 2, 1 << q)
        a0, a1 = v[:, 0, :], v[:, 1, :]
        m00, m01, m10, m11 = (self.dtype.type(x) for x in m.ravel())
        t = a0.copy()
        np.multiply(a0, m00, out=a0)
        if m01 != 0:
            a0 += m01 * a1
        np.multiply(a1, m11, out=a1)
        if m10 != 0:
            a1 += m10 * t

    def apply_diag_1q(self, q: int, d: np.ndarray) -> None:
        v = self.psi.reshape(-1, 2, 1 << q)
        if d[0] != 1:
            v[:, 0, :] *= self.dtype.type(d[0])
        if d[1] != 1:
            v[:, 1, :] *= self.dtype.type(d[1])

    def _view2(self, a: int, b: int) -> tuple[np.ndarray, list[tuple[Any, ...]]]:
        """5-D view and the four index tuples for |b_a b_b⟩ = 00, 01, 10, 11 (``a`` MSB)."""
        hi, lo = (a, b) if a > b else (b, a)
        v = self.psi.reshape(-1, 2, 1 << (hi - lo - 1), 2, 1 << lo)
        idx: list[tuple[Any, ...]] = []
        for k in range(4):
            ba, bb = k >> 1, k & 1
            bh, bl = (ba, bb) if a > b else (bb, ba)
            idx.append((slice(None), bh, slice(None), bl, slice(None)))
        return v, idx

    def apply_2q(self, a: int, b: int, m: np.ndarray) -> None:
        v, idx = self._view2(a, b)
        mt = m.astype(self.dtype, copy=False)
        cols_used = [c for c in range(4) if np.any(mt[:, c] != 0)]
        xs = {c: v[idx[c]].copy() for c in cols_used}
        for k in range(4):
            out = v[idx[k]]
            terms = [(mt[k, c], xs[c]) for c in cols_used if mt[k, c] != 0]
            if not terms:
                out[...] = 0
                continue
            c0, x0 = terms[0]
            np.multiply(x0, c0, out=out)
            for c, x in terms[1:]:
                out += c * x

    def apply_diag_2q(self, a: int, b: int, d: np.ndarray) -> None:
        v, idx = self._view2(a, b)
        for k in range(4):
            if d[k] != 1:
                v[idx[k]] *= self.dtype.type(d[k])

    def apply(self, op: NativeOp, qmap: dict[int, int]) -> None:
        qs = [qmap[q] for q in op.qubits]
        if len(qs) == 1:
            if op.diagonal is not None:
                self.apply_diag_1q(qs[0], op.diagonal)
            else:
                self.apply_1q(qs[0], op.matrix)  # type: ignore[arg-type]
        elif len(qs) == 2:
            if op.diagonal is not None:
                self.apply_diag_2q(qs[0], qs[1], op.diagonal)
            else:
                self.apply_2q(qs[0], qs[1], op.matrix)  # type: ignore[arg-type]
        else:  # pragma: no cover - the compiler never emits >2-qubit natives
            raise ValueError(f"engine supports 1- and 2-qubit kernels, got {len(qs)}")

    def apply_x(self, q: int) -> None:
        v = self.psi.reshape(-1, 2, 1 << q)
        t = v[:, 0, :].copy()
        v[:, 0, :] = v[:, 1, :]
        v[:, 1, :] = t

    # ---------------------------------------------------------- measurement
    def prob_one(self, q: int) -> float:
        a1 = self.psi.reshape(-1, 2, 1 << q)[:, 1, :]
        return float(min(1.0, max(0.0, np.vdot(a1, a1).real)))

    def collapse(self, q: int, outcome: int, prob: float) -> None:
        v = self.psi.reshape(-1, 2, 1 << q)
        v[:, 1 - outcome, :] = 0
        v[:, outcome, :] *= self.dtype.type(1.0 / np.sqrt(prob))

    def probabilities(self) -> np.ndarray:
        p = np.square(self.psi.real, dtype=np.float64)
        p += np.square(self.psi.imag, dtype=np.float64)
        return p

    def marginal(self, qubits: list[int]) -> np.ndarray:
        """Marginal over ``qubits`` (ascending); bit j of the result index ↔ ``qubits[j]``."""
        p = self.probabilities()
        if len(qubits) == self.n:
            pm = p
        else:
            keep = set(qubits)
            axes = tuple(self.n - 1 - q for q in range(self.n) if q not in keep)
            pm = p.reshape((2,) * self.n).sum(axis=axes).ravel() if axes else p
        return _normalised(pm)


def _normalised(pm: np.ndarray) -> np.ndarray:
    total = pm.sum()
    if not np.isfinite(total) or abs(total - 1.0) > 1e-6:
        raise FloatingPointError(f"statevector norm drifted: Σp = {total!r}")
    pm /= total                                    # fresh array: normalise in place
    return pm


def _marginal(eng: StatevectorEngine | FusedState, mq: list[int]) -> tuple[np.ndarray, list[int] | None]:
    """Marginal over ``mq`` plus the bit relabelling of its index (``None`` = bit j ↔ ``mq[j]``)."""
    if isinstance(eng, StatevectorEngine):
        return eng.marginal(mq), None
    p, order = eng.marginal(mq, consume=True)      # stored order: no transpose; the state is spent
    total = p.sum()                                # the sampler scales by the total: check, don't divide
    if not np.isfinite(total) or abs(total - 1.0) > 1e-6:
        raise FloatingPointError(f"statevector norm drifted: Σp = {total!r}")
    pos = {q: j for j, q in enumerate(mq)}
    bits = [pos[order[len(order) - 1 - i]] for i in range(len(order))]
    return p, None if bits == list(range(len(mq))) else bits


# ======================================================================= plan
@dataclass(slots=True)
class Program:
    """A native circuit prepared for execution on ``n_sim`` compacted qubits."""

    native: NativeCircuit
    active: list[int]                          # physical qubits simulated (ascending)
    qmap: dict[int, int]                       # physical -> simulated index
    terminal: dict[int, str]                   # op index -> "terminal" | "dead"
    branching: bool                            # needs the shot-branching path
    branch_points: int                         # mid-circuit measure/reset count
    idle_measured: set[int] = field(default_factory=set)

    @property
    def n_sim(self) -> int:
        return len(self.active)


def prepare(native: NativeCircuit) -> Program:
    """Compact active qubits and classify every measurement (terminal / dead / branching)."""
    gate_qubits: set[int] = set()
    for op in native.ops:
        if op.is_gate or op.name == "reset":
            gate_qubits.update(op.qubits)
    active = sorted(gate_qubits)
    qmap = {q: i for i, q in enumerate(active)}
    touched_later: set[int] = set()
    read_later: set[int] = set()
    written_later: set[int] = set()
    terminal: dict[int, str] = {}
    branching = False
    branch_points = 0
    for i in range(len(native.ops) - 1, -1, -1):
        op = native.ops[i]
        if op.name == "barrier":                  # scheduling directive only: no effect on the state
            continue
        if op.condition is not None:
            branching = True
        if op.name == "measure":
            q = op.qubits[0]
            if q not in gate_qubits:
                terminal[i] = "idle"
            elif q in touched_later or any(c in read_later for c in op.clbits) or op.condition is not None:
                branching = True
                branch_points += 1
            elif op.clbits and all(c in written_later for c in op.clbits):
                terminal[i] = "dead"
            else:
                terminal[i] = "terminal"
            touched_later.add(q)
            written_later.update(op.clbits)
        elif op.name == "reset":
            branching = True
            branch_points += 1
            touched_later.update(op.qubits)
        else:
            touched_later.update(op.qubits)
        if op.condition is not None:
            read_later.update(op.condition.clbits)
    idle = {native.ops[i].qubits[0] for i, k in terminal.items() if k == "idle"}
    return Program(native, active, qmap, terminal, branching, branch_points, idle)


# ======================================================================== run
@dataclass(slots=True)
class RunOutput:
    counts: dict[str, int]
    memory: list[str] | None
    statevector: np.ndarray | None
    execution_path: str
    branches: int


def _clbit_value(bits: dict[int, int]) -> int:
    v = 0
    for c, b in bits.items():
        if b:
            v |= 1 << c
    return v


def run(prog: Program, *, shots: int, rng: np.random.Generator, dtype: np.dtype | type = np.complex128,
        timer: SpanTimer | None = None, want_statevector: bool = False, want_memory: bool = False) -> RunOutput:
    timer = timer or SpanTimer()
    native = prog.native
    nc = native.num_clbits
    ops = native.ops
    tallies: dict[int, int] = {}

    # terminal measurements (in program order: later writes overwrite earlier ones)
    term: list[tuple[int, int]] = []           # (sim qubit, clbit)
    idle_bits: list[int] = []
    for i, kind in sorted(prog.terminal.items()):
        op = ops[i]
        if op.condition is not None:
            continue
        for c in op.clbits:
            if kind == "terminal":
                term = [(q, cb) for q, cb in term if cb != c]
                term.append((prog.qmap[op.qubits[0]], c))
            elif kind == "idle":
                term = [(q, cb) for q, cb in term if cb != c]
                idle_bits.append(c)

    def sample_terminal(eng: StatevectorEngine | FusedState, base: dict[int, int], n: int) -> None:
        bits = dict(base)
        for c in idle_bits:
            bits[c] = 0
        if not term:
            key = _clbit_value(bits)
            tallies[key] = tallies.get(key, 0) + n
            return
        with timer.span("sampling"):
            mq = sorted({q for q, _ in term})
            pm, relabel = _marginal(eng, mq)
            outcomes, freq = _sample(pm, n, rng)
            if relabel is not None:
                o = np.zeros_like(outcomes)
                for i, j in enumerate(relabel):
                    o |= ((outcomes >> i) & 1) << j
                outcomes = o
            pos = {q: j for j, q in enumerate(mq)}
            for c in [cb for _, cb in term]:
                bits.pop(c, None)
            base_val = _clbit_value(bits)
            if nc <= 62:
                vals = np.full(outcomes.shape, base_val, dtype=np.int64)
                for q, c in term:
                    vals |= ((outcomes >> pos[q]) & 1).astype(np.int64) << c
                for v, k in zip(vals.tolist(), freq.tolist(), strict=True):
                    tallies[v] = tallies.get(v, 0) + k
            else:
                for o, k in zip(outcomes.tolist(), freq.tolist(), strict=True):
                    v = base_val
                    for q, c in term:
                        if (o >> pos[q]) & 1:
                            v |= 1 << c
                    tallies[v] = tallies.get(v, 0) + k

    statevector = None
    branches = 1
    if not prog.branching:
        with timer.span("simulation"):
            st = fused_state(prog, dtype, exact_phases=want_statevector)
            if want_statevector:
                statevector = st.standard() * np.exp(1j * native.global_phase)
            st.release()
        sample_terminal(st, {}, shots)
        path = "terminal_sampling"
    else:
        path = "shot_branching"
        stack: list[tuple[int, StatevectorEngine, dict[int, int], int]] = [
            (0, StatevectorEngine(prog.n_sim, dtype), {}, shots)]
        while stack:
            start, eng, bits, n = stack.pop()
            i = start
            while i < len(ops):
                op = ops[i]
                i += 1
                if op.condition is not None and not op.condition.holds(_BitView(bits)):
                    continue
                if op.is_gate:
                    with timer.span("simulation"):
                        eng.apply(op, prog.qmap)
                    continue
                if op.name == "barrier":
                    continue
                kind = prog.terminal.get(i - 1)
                if kind in ("terminal", "dead"):
                    continue
                if kind == "idle":
                    for c in op.clbits:
                        bits[c] = 0
                    continue
                with timer.span("simulation"):
                    q = prog.qmap[op.qubits[0]]
                    p1 = eng.prob_one(q)
                    n1 = int(rng.binomial(n, p1)) if 0.0 < p1 < 1.0 else (n if p1 >= 1.0 else 0)
                    n0 = n - n1
                    if n0 and n1:
                        branches += 1
                        other = eng.copy()
                        other.collapse(q, 1, p1)
                        obits = dict(bits)
                        if op.name == "measure":
                            for c in op.clbits:
                                obits[c] = 1
                        else:
                            other.apply_x(q)
                        stack.append((i, other, obits, n1))
                        eng.collapse(q, 0, 1.0 - p1)
                        outcome, n = 0, n0
                    else:
                        outcome = 1 if n1 else 0
                        eng.collapse(q, outcome, p1 if outcome else 1.0 - p1)
                    if op.name == "measure":
                        for c in op.clbits:
                            bits[c] = outcome
                    elif outcome == 1:
                        eng.apply_x(q)
            sample_terminal(eng, bits, n)

    with timer.span("sampling"):
        counts = {format(v, f"0{nc}b") if nc else "": k for v, k in sorted(tallies.items())}
        memory = None
        if want_memory:
            keys = list(counts)
            reps = np.repeat(np.arange(len(keys)), [counts[k] for k in keys])
            rng.shuffle(reps)
            memory = [keys[j] for j in reps.tolist()]
    return RunOutput(counts, memory, statevector, path, branches)


def _gate_matrix(op: NativeOp) -> np.ndarray:
    """The gate's matrix, or its diagonal (a 1-D array) for diagonal gates."""
    return np.asarray(op.diagonal) if op.diagonal is not None else op.matrix  # type: ignore[return-value]


def _drop_final_diagonals(ops: list[NativeOp]) -> list[NativeOp]:
    """Remove diagonal gates with no later non-diagonal gate on any of their qubits.

    They only multiply amplitudes by phases, so measurement probabilities are unchanged.
    """
    later_dense: set[int] = set()
    keep: list[NativeOp] = []
    for op in reversed(ops):
        if op.diagonal is not None and not (set(op.qubits) & later_dense):
            continue
        if op.diagonal is None:
            later_dense.update(op.qubits)
        keep.append(op)
    return keep[::-1]


def fused_state(prog: Program, dtype: np.dtype | type = np.complex128, max_qubits: int | None = None,
                *, exact_phases: bool = True) -> FusedState:
    """Final state of a branch-free program: fused blocks on a lazily permuted tensor.

    ``exact_phases=False`` drops trailing diagonal gates, which leaves every measurement
    probability unchanged but not the relative phases of the state.
    """
    ops = [op for op in prog.native.ops if op.is_gate]
    if not exact_phases:
        ops = _drop_final_diagonals(ops)
    gates = [(tuple(prog.qmap[q] for q in op.qubits), _gate_matrix(op)) for op in ops]
    st = FusedState(prog.n_sim, dtype)
    for block in plan_blocks(gates, max_qubits or FUSION_MAX_QUBITS, FUSION_MAX_DIAGONAL_QUBITS):
        st.apply(block)
    return st


def fused_statevector(prog: Program, dtype: np.dtype | type = np.complex128, max_qubits: int | None = None,
                      *, exact_phases: bool = True) -> np.ndarray:
    """The :func:`fused_state` as a standard-order vector."""
    st = fused_state(prog, dtype, max_qubits, exact_phases=exact_phases)
    vec = st.standard()
    st.release()
    return vec


def _sample(p: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Exact inverse-CDF sampling of ``n`` outcomes from ``p`` (consumed in place).

    Memory is O(n) beyond ``p`` itself (a multinomial over 2^m categories would
    allocate an int64 array as large as the statevector). Zero-probability
    outcomes have empty CDF bins and can never be drawn.

    Large distributions use a two-level CDF: block totals first, then the cumulative
    sum of only the blocks a draw lands in — the same inverse-CDF map without a pass
    that rewrites all of ``p``.
    """
    if len(p) >= 1 << 17 and n * _CDF_BLOCK <= len(p):
        return _sample_blocked(p, n, rng)
    cdf = np.cumsum(p, out=p)
    u = rng.random(n) * cdf[-1]
    idx = np.searchsorted(cdf, u, side="right")
    np.minimum(idx, len(cdf) - 1, out=idx)
    outcomes, freq = np.unique(idx, return_counts=True)
    return outcomes.astype(np.int64), freq


_CDF_BLOCK = 1 << 12


def _sample_blocked(p: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    blocks = p.reshape(-1, _CDF_BLOCK)
    coarse = blocks.sum(axis=1)
    ccdf = np.cumsum(coarse)
    u = np.sort(rng.random(n) * ccdf[-1])
    blk = np.minimum(np.searchsorted(ccdf, u, side="right"), len(ccdf) - 1)
    while coarse[blk[-1]] == 0:                    # a draw rounded past the last non-empty block
        blk[blk == blk[-1]] -= 1
    out = np.empty(n, dtype=np.int64)
    ub, first = np.unique(blk, return_index=True)
    bounds = [*first.tolist(), n]
    for t, b in enumerate(ub.tolist()):
        s, e = bounds[t], bounds[t + 1]
        seg = np.cumsum(blocks[b])
        last = int(np.flatnonzero(blocks[b])[-1])  # never land on an empty bin through rounding
        off = ccdf[b - 1] if b else 0.0
        out[s:e] = b * _CDF_BLOCK + np.minimum(np.searchsorted(seg, u[s:e] - off, side="right"), last)
    outcomes, freq = np.unique(out, return_counts=True)
    return outcomes, freq


class _BitView:
    """Read-only bit mapping that defaults unwritten classical bits to 0."""

    __slots__ = ("_bits",)

    def __init__(self, bits: dict[int, int]) -> None:
        self._bits = bits

    def __getitem__(self, c: int) -> int:
        return self._bits.get(c, 0)


def condition_holds(cond: Condition, bits: dict[int, int]) -> bool:
    return cond.holds(_BitView(bits))
