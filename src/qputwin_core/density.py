"""Density-matrix engine for noisy circuits.

Usage: from qputwin_core.density import DensityMatrixEngine

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

from qputwin_core.engine import Program, RunOutput, StatevectorEngine, _BitView, _sample
from qputwin_core.fused import FusedState, plan_blocks
from qputwin_core.noise import NoisyInstr
from qputwin_core.timing import SpanTimer

__all__ = ["FUSION_MAX_AXES", "FUSION_MAX_DIAGONAL_AXES", "DensityMatrixEngine", "needs_branching", "run_density",
           "superoperators"]

FUSION_MAX_AXES = 4               # fused superoperator blocks span at most two qubits (4 row/column axes)
FUSION_MAX_DIAGONAL_AXES = 12     # diagonal superoperators (phases, dephasing) fuse across up to six qubits

_RESET_SUPEROP = (np.kron(np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]]))
                  + np.kron(np.array([[0, 1], [0, 0]]), np.array([[0, 1], [0, 0]]))).astype(np.complex128)


class DensityMatrixEngine:
    def __init__(self, n: int, dtype: np.dtype | type = np.complex128, vec: np.ndarray | None = None) -> None:
        self.n = n
        self.dtype = np.dtype(dtype)
        self.vec = StatevectorEngine(2 * n, self.dtype, vec)       # |0…0⟩ ⇔ ρ = |0⟩⟨0|

    @classmethod
    def from_statevector(cls, psi: np.ndarray, dtype: np.dtype | type = np.complex128) -> DensityMatrixEngine:
        n = int(psi.size).bit_length() - 1
        rho = np.outer(psi, psi.conj()).astype(dtype).reshape(-1)
        return cls(n, dtype, rho)

    def copy(self) -> DensityMatrixEngine:
        return DensityMatrixEngine(self.n, self.dtype, self.vec.psi.copy())

    @property
    def matrix(self) -> np.ndarray:
        d = 1 << self.n
        return self.vec.psi.reshape(d, d)

    # ------------------------------------------------------------- unitaries
    def apply_unitary(self, qs: tuple[int, ...], m: np.ndarray | None, diag: np.ndarray | None) -> None:
        n, v = self.n, self.vec
        if len(qs) == 1:
            (q,) = qs
            if diag is not None:
                v.apply_diag_1q(n + q, diag)
                v.apply_diag_1q(q, diag.conj())
            else:
                v.apply_1q(n + q, m)                                 # type: ignore[arg-type]
                v.apply_1q(q, m.conj())                              # type: ignore[union-attr]
        elif len(qs) == 2:
            a, b = qs
            if diag is not None:
                v.apply_diag_2q(n + a, n + b, diag)
                v.apply_diag_2q(a, b, diag.conj())
            else:
                v.apply_2q(n + a, n + b, m)                          # type: ignore[arg-type]
                v.apply_2q(a, b, m.conj())                           # type: ignore[union-attr]
        else:  # pragma: no cover - the compiler never emits >2-qubit natives
            raise ValueError(f"density engine supports 1- and 2-qubit gates, got {len(qs)}")

    # -------------------------------------------------------------- channels
    def apply_superop_1q(self, q: int, s: np.ndarray, diag: np.ndarray | None = None) -> None:
        if diag is not None:
            self.vec.apply_diag_2q(self.n + q, q, diag)
        else:
            self.vec.apply_2q(self.n + q, q, s)

    def depolarize(self, qs: tuple[int, ...], p: float) -> None:
        """``ρ → (1−p)ρ + p · Tr_qs(ρ) ⊗ I/2^k``."""
        if p <= 0:
            return
        n2 = 2 * self.n
        t = self.vec.psi.reshape((2,) * n2)
        k = len(qs)
        index: list[tuple[slice | int, ...]] = []
        for bits in itertools.product((0, 1), repeat=k):
            ix: list[slice | int] = [slice(None)] * n2
            for q, b in zip(qs, bits, strict=True):
                ix[self.n - 1 - q] = b                               # row axis of qubit q
                ix[n2 - 1 - q] = b                                   # column axis of qubit q
            index.append(tuple(ix))
        trace = t[index[0]].copy()
        for ix in index[1:]:
            trace += t[ix]
        t *= self.dtype.type(1.0 - p)
        trace *= self.dtype.type(p / (1 << k))
        for ix in index:
            t[ix] += trace

    def reset(self, q: int) -> None:
        self.vec.apply_2q(self.n + q, q, _RESET_SUPEROP)

    # ----------------------------------------------------------- measurement
    def diagonal(self) -> np.ndarray:
        d = 1 << self.n
        p = self.vec.psi[:: d + 1].real.astype(np.float64)
        np.maximum(p, 0.0, out=p)
        return p

    def trace(self) -> float:
        return float(self.diagonal().sum())

    def purity(self) -> float:
        psi = self.vec.psi
        return float(np.vdot(psi, psi).real)

    def prob_one(self, q: int) -> float:
        d = self.diagonal().reshape(-1, 2, 1 << q)
        tot = float(d.sum())
        return float(min(1.0, max(0.0, d[:, 1, :].sum() / tot))) if tot > 0 else 0.0

    def project(self, q: int, w0: float, w1: float) -> None:
        """``ρ → w0·Π0ρΠ0 + w1·Π1ρΠ1`` (coherences between the two outcomes vanish)."""
        self.vec.apply_diag_2q(self.n + q, q, np.array([w0, 0.0, 0.0, w1], dtype=self.dtype))

    def marginal(self, qubits: list[int]) -> np.ndarray:
        p = self.diagonal()
        n = self.n
        if len(qubits) != n:
            keep = set(qubits)
            axes = tuple(n - 1 - q for q in range(n) if q not in keep)
            p = p.reshape((2,) * n).sum(axis=axes).ravel() if axes else p
        total = p.sum()
        if not np.isfinite(total) or abs(total - 1.0) > 1e-6:
            raise FloatingPointError(f"density-matrix trace drifted: Tr ρ = {total!r}")
        return p / total


def apply_readout_confusion(p: np.ndarray, errors: list[tuple[float, float]]) -> np.ndarray:
    """Apply ``A_j = [[1−p01, p10], [p01, 1−p10]]`` to bit ``j`` of the outcome distribution ``p``."""
    m = len(errors)
    if m == 0 or not any(a or b for a, b in errors):
        return p
    t = p.reshape((2,) * m)
    for j, (p01, p10) in enumerate(errors):
        if not (p01 or p10):
            continue
        ax = m - 1 - j
        a = np.array([[1.0 - p01, p10], [p01, 1.0 - p10]])
        t = np.moveaxis(np.tensordot(a, t, axes=([1], [ax])), 0, ax)
    out = np.ascontiguousarray(t).reshape(-1)
    return out / out.sum()


ReadoutFn = Callable[[int], tuple[float, float]]
GroupsFn = Callable[[list[int]], list[tuple[tuple[int, ...], np.ndarray]]]


def apply_readout_model(p: np.ndarray, phys: list[int], readout: ReadoutFn, groups: GroupsFn | None) -> np.ndarray:
    """Apply readout errors to the distribution ``p`` over ``phys`` (bit j ↔ ``phys[j]``).

    Qubits in a correlated group use its joint matrix ``M[recorded, true]`` (bit k ↔
    ``group[k]``); every other qubit its independent confusion matrix.
    """
    m = len(phys)
    pos = {q: j for j, q in enumerate(phys)}
    joint = [([pos[q] for q in qs], mat) for qs, mat in (groups(phys) if groups is not None else [])]
    covered = {j for js, _ in joint for j in js}
    p = apply_readout_confusion(p, [(0.0, 0.0) if j in covered else readout(q) for j, q in enumerate(phys)])
    if not joint:
        return p
    t = p.reshape((2,) * m)
    for js, mat in joint:
        axes = [m - 1 - j for j in reversed(js)]          # leading axis = most significant group bit
        front = list(range(len(js)))
        t = np.moveaxis(t, axes, front)
        shape = t.shape
        t = np.moveaxis((mat @ t.reshape(1 << len(js), -1)).reshape(shape), front, axes)
    out = np.ascontiguousarray(t).reshape(-1)
    return out / out.sum()


class TerminalSampler:
    """Terminal measurements of a noisy run, sampled exactly with (correlated) readout errors.

    Shared by the density-matrix and trajectory engines. Idle-measured qubits
    (never gated, hence exactly ``|0⟩``) whose bits no condition reads are deferred
    to the end: inside a correlated-readout group they are sampled jointly with
    their partners, otherwise with independent flips. Idle measurements whose
    bits feed a condition are resolved during the walk (``walks(index)``).
    """

    def __init__(self, prog: Program, readout: ReadoutFn, groups: GroupsFn | None, rng: np.random.Generator,
                 timer: SpanTimer) -> None:
        self.readout, self.groups, self.rng, self.timer = readout, groups, rng, timer
        ops = prog.native.ops
        self.inv_qmap = {s: p for p, s in prog.qmap.items()}
        read = {c for op in ops if op.condition is not None for c in op.condition.clbits}
        self.term: list[tuple[int, int]] = []          # (simulated qubit, clbit); later writes win
        self.idle: list[tuple[int, int]] = []          # (physical qubit, clbit) deferred to the end
        self._walk: set[int] = set()
        for i, kind in sorted(prog.terminal.items()):
            op = ops[i]
            if op.condition is not None:
                continue
            for c in op.clbits:
                self.term = [(q, cb) for q, cb in self.term if cb != c]
                self.idle = [(q, cb) for q, cb in self.idle if cb != c]
            if kind == "idle" and read & set(op.clbits):
                self._walk.add(i)
            elif kind == "terminal":
                self.term += [(prog.qmap[op.qubits[0]], c) for c in op.clbits]
            elif kind == "idle":
                self.idle += [(op.qubits[0], c) for c in op.clbits]

    def walks(self, index: int) -> bool:
        return index in self._walk

    def _tally(self, buckets: list[tuple[int, int]], flips: list[tuple[int, int]], tallies: dict[int, int]) -> None:
        for q, c in flips:                                  # independent readout flips of |0⟩ qubits
            p01, _ = self.readout(q)
            nxt: list[tuple[int, int]] = []
            for v, k in buckets:
                k1 = int(self.rng.binomial(k, p01)) if p01 > 0 else 0
                if k - k1:
                    nxt.append((v & ~(1 << c), k - k1))
                if k1:
                    nxt.append((v | (1 << c), k1))
            buckets = nxt
        for v, k in buckets:
            tallies[v] = tallies.get(v, 0) + k

    def sample(self, marginal: Callable[[list[int]], np.ndarray], bits: dict[int, int], n: int,
               tallies: dict[int, int]) -> None:
        base = 0
        for c, b in bits.items():
            if b:
                base |= 1 << c
        for _, c in [*self.term, *self.idle]:
            base &= ~(1 << c)
        mq = sorted({q for q, _ in self.term})
        phys = [self.inv_qmap[q] for q in mq]
        idle_q = [q for q, _ in self.idle]
        covered: set[int] = set()
        if self.groups is not None and self.idle:
            for qs, _ in self.groups(phys + [q for q in dict.fromkeys(idle_q) if idle_q.count(q) == 1]):
                covered |= set(qs)
        joint = [(q, c) for q, c in self.idle if q in covered]
        flips = [(q, c) for q, c in self.idle if q not in covered]
        if not mq and not joint:
            self._tally([(base, n)], flips, tallies)
            return
        with self.timer.span("sampling"):
            p = marginal(mq) if mq else np.ones(1)
            if joint:                                      # idle group members are exactly |0⟩
                ext = np.zeros(len(p) << len(joint))
                ext[: len(p)] = p
                p = ext
            p = apply_readout_model(p, phys + [q for q, _ in joint], self.readout, self.groups)
            outcomes, freq = _sample(p, n, self.rng)
            pos = {q: j for j, q in enumerate(mq)}
            buckets = []
            for o, k in zip(outcomes.tolist(), freq.tolist(), strict=True):
                v = base
                for q, c in self.term:
                    if (o >> pos[q]) & 1:
                        v |= 1 << c
                for j, (_, c) in enumerate(joint):
                    if (o >> (len(mq) + j)) & 1:
                        v |= 1 << c
                buckets.append((v, k))
            self._tally(buckets, flips, tallies)


def _depolarizing_superop(k: int, p: float) -> np.ndarray:
    """``(1−p)ρ + p·Tr(ρ)·I/2^k`` on a ``k``-qubit block, index = row·2^k + column."""
    d = 1 << k
    vec_i = np.eye(d).reshape(-1)
    return (1.0 - p) * np.eye(d * d) + (p / d) * np.outer(vec_i, vec_i)


def superoperators(n: int, instrs: list[NoisyInstr]) -> list[tuple[tuple[int, ...], np.ndarray]]:
    """Each non-measurement instruction as ``(axes, superoperator)`` on the 2n-axis vector of ρ.

    Qubit ``q`` owns row axis ``n + q`` and column axis ``q``; a block's index is
    (row bits of its qubits, then column bits), matching :class:`DensityMatrixEngine`.
    """
    out: list[tuple[tuple[int, ...], np.ndarray]] = []
    for ins in instrs:
        k = ins.kind
        qs = tuple(ins.qubits)
        if k == "gate":
            m = np.diag(ins.diagonal) if ins.diagonal is not None else ins.matrix
            out.append((tuple(n + q for q in qs) + qs, np.kron(m, np.conj(m))))
        elif k == "channel":
            s = np.diag(ins.diagonal) if ins.diagonal is not None else ins.matrix
            out.append(((n + qs[0], qs[0]), s))
        elif k == "depolarize":
            if ins.p > 0:
                out.append((tuple(n + q for q in qs) + qs, _depolarizing_superop(len(qs), ins.p)))
        elif k == "reset":
            if qs[0] >= 0:
                out.append(((n + qs[0], qs[0]), _RESET_SUPEROP))
        elif k != "measure":  # pragma: no cover - the simulator routes these programs to the trajectory engine
            raise ValueError(f"the density-matrix engine cannot apply {k!r} (trajectory-only)")
    return out


def fused_density(n: int, instrs: list[NoisyInstr], dtype: np.dtype | type = np.complex128) -> DensityMatrixEngine:
    """Final ρ of a branch-free noisy program, through fused superoperator blocks."""
    st = FusedState(2 * n, dtype)
    for block in plan_blocks(superoperators(n, instrs), FUSION_MAX_AXES, FUSION_MAX_DIAGONAL_AXES):
        st.apply(block)
    vec = st.standard()
    st.release()
    return DensityMatrixEngine(n, dtype, vec)


def needs_branching(prog: Program, instrs: list[NoisyInstr]) -> bool:
    """Shot branching is needed only for mid-circuit measurement and classical feed-forward."""
    return any(ins.condition is not None or (ins.kind == "measure" and ins.index not in prog.terminal)
               for ins in instrs)


# ========================================================================= run
def run_density(prog: Program, instrs: list[NoisyInstr], *, shots: int, rng: np.random.Generator,
                readout: ReadoutFn, groups: GroupsFn | None = None, dtype: np.dtype | type = np.complex128,
                timer: SpanTimer | None = None, want_density_matrix: bool = False,
                want_memory: bool = False) -> tuple[RunOutput, dict[str, float]]:
    """Execute a noisy program; returns the run output and ``{"purity": …}`` diagnostics.

    ``readout(physical_qubit) -> (p01, p10)`` supplies the assignment errors.
    """
    timer = timer or SpanTimer()
    native = prog.native
    ops = native.ops
    nc = native.num_clbits
    tallies: dict[int, int] = {}
    sampler = TerminalSampler(prog, readout, groups, rng, timer)

    def step(eng: DensityMatrixEngine, ins: NoisyInstr) -> None:
        k = ins.kind
        if k == "gate":
            eng.apply_unitary(ins.qubits, ins.matrix, ins.diagonal)
        elif k == "channel":
            eng.apply_superop_1q(ins.qubits[0], ins.matrix, ins.diagonal)   # type: ignore[arg-type]
        elif k == "depolarize":
            eng.depolarize(ins.qubits, ins.p)
        elif k == "reset":
            if ins.qubits[0] >= 0:
                eng.reset(ins.qubits[0])
        else:  # pragma: no cover - the simulator routes these programs to the trajectory engine
            raise ValueError(f"the density-matrix engine cannot apply {k!r} (trajectory-only)")

    branches = 1
    diagnostics: dict[str, float] = {}
    dm_out = None
    if not needs_branching(prog, instrs):
        path = "density_matrix"
        with timer.span("simulation"):
            eng = fused_density(prog.n_sim, instrs, dtype)
            diagnostics["purity"] = eng.purity()
            diagnostics["trace"] = eng.trace()
        if want_density_matrix:
            dm_out = eng.matrix.copy()
        sampler.sample(eng.marginal, {}, shots, tallies)
    else:
        path = "density_matrix_branching"
        stack: list[tuple[int, DensityMatrixEngine, dict[int, int], int]] = [
            (0, DensityMatrixEngine(prog.n_sim, dtype), {}, shots)]
        while stack:
            start, eng, bits, n = stack.pop()
            i = start
            while i < len(instrs):
                ins = instrs[i]
                i += 1
                if ins.condition is not None and not ins.condition.holds(_BitView(bits)):
                    continue
                if ins.kind != "measure":
                    with timer.span("simulation"):
                        step(eng, ins)
                    continue
                kind = prog.terminal.get(ins.index)
                if kind in ("terminal", "dead") or (kind == "idle" and not sampler.walks(ins.index)):
                    continue
                phys = ops[ins.index].qubits[0]
                p01, p10 = readout(phys)
                if kind == "idle":                  # |0⟩ qubit: only the readout error is random
                    for c in ins.clbits:
                        bits[c] = 0
                    if p01 > 0:
                        n1 = int(rng.binomial(n, p01))
                        if n1 == n:
                            for c in ins.clbits:
                                bits[c] = 1
                        elif n1:
                            branches += 1
                            stack.append((i, eng.copy(), {**bits, **{c: 1 for c in ins.clbits}}, n1))
                            n -= n1
                    continue
                with timer.span("simulation"):
                    q = ins.qubits[0]
                    pm1 = eng.prob_one(q)
                    pm0 = 1.0 - pm1
                    pr1 = p01 * pm0 + (1.0 - p10) * pm1          # P(recorded bit = 1)
                    n1 = int(rng.binomial(n, pr1)) if 0.0 < pr1 < 1.0 else (n if pr1 >= 1.0 else 0)
                    n0 = n - n1
                    if n0 and n1:
                        branches += 1
                        other = eng.copy()
                        other.project(q, p01 / pr1, (1.0 - p10) / pr1)
                        stack.append((i, other, {**bits, **{c: 1 for c in ins.clbits}}, n1))
                    r = 0 if n0 else 1
                    pr = (1.0 - pr1) if r == 0 else pr1
                    w0 = ((1.0 - p01) if r == 0 else p01) / pr
                    w1 = (p10 if r == 0 else (1.0 - p10)) / pr
                    eng.project(q, w0, w1)
                    n = n0 if n0 else n1
                    for c in ins.clbits:
                        bits[c] = r
            sampler.sample(eng.marginal, bits, n, tallies)

    with timer.span("sampling"):
        counts = {format(v, f"0{nc}b") if nc else "": k for v, k in sorted(tallies.items())}
        memory = None
        if want_memory:
            keys = list(counts)
            reps = np.repeat(np.arange(len(keys)), [counts[k] for k in keys])
            rng.shuffle(reps)
            memory = [keys[j] for j in reps.tolist()]
    return RunOutput(counts, memory, dm_out, path, branches), diagnostics
