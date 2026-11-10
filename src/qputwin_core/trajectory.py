"""Monte-Carlo quantum-trajectory engine (noisy execution beyond density-matrix widths).

Each trajectory evolves a *statevector* through the same noisy program the
density-matrix engine uses (:func:`qputwin_core.noise.build_noisy_program`),
unravelling every channel stochastically:

* a one-qubit Kraus channel ``{K_k}`` picks ``k`` with the Born weight
  ``p_k = Tr(K_k ρ_q K_k†)`` from the qubit's reduced density matrix ``ρ_q`` and
  applies ``K_k / √p_k``;
* depolarising ``D_p`` applies a uniformly random non-identity Pauli with
  probability ``p·(d²−1)/d²`` (``D_p(ρ) = (1 − p + p/d²)ρ + p/d² Σ_{P≠I} PρP``);
* **quasi-static detuning** draws ``δ_q ~ N(0, σ_q)`` once per trajectory and
  applies ``Rz(2π·δ_q·t)`` over every scheduled window, which makes the noise
  correlated in time (non-Markovian) and is why it is trajectory-only;
* **leakage** (phenomenological): a leak event collapses the qubit and parks it
  in ``|1⟩`` outside the computational dynamics, so gates and channels on it are
  skipped, a 2Q partner receives a uniformly random Pauli, it reads out as ``1``,
  and each later gate on it returns it with probability ``seepage``; a reset
  returns it to ``|0⟩``.

Measurements and resets keep the exact binomial shot-branching of the
statevector engine, and readout errors (independent or correlated) are applied
exactly to terminal distributions. The jumps of one trajectory are shared by the
shots assigned to it, so sampling is **exact when every shot has its own
trajectory** (``trajectories = shots``). With fewer trajectories the result
samples the empirical mixture of the drawn trajectories (stratified, so every
trajectory gets ``⌊shots/T⌋`` or ``⌈shots/T⌉`` shots), which converges to the
noisy distribution as ``T`` grows. The run reports which regime applied.

Cost: ``T × (circuit length) × O(2^n)``, with the memory of a statevector run.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from qputwin_core.density import GroupsFn, ReadoutFn, TerminalSampler
from qputwin_core.engine import Program, RunOutput, StatevectorEngine, _BitView
from qputwin_core.noise import NoisyInstr
from qputwin_core.timing import SpanTimer

__all__ = ["run_trajectories"]

_PAULI = (np.eye(2, dtype=np.complex128), np.array([[0, 1], [1, 0]], dtype=np.complex128),
          np.array([[0, -1j], [1j, 0]], dtype=np.complex128), np.array([[1, 0], [0, -1]], dtype=np.complex128))


def _apply(eng: StatevectorEngine, qs: tuple[int, ...], m: np.ndarray | None, diag: np.ndarray | None) -> None:
    if len(qs) == 1:
        if diag is not None:
            eng.apply_diag_1q(qs[0], diag)
        else:
            eng.apply_1q(qs[0], m)                                   # type: ignore[arg-type]
    elif diag is not None:
        eng.apply_diag_2q(qs[0], qs[1], diag)
    else:
        eng.apply_2q(qs[0], qs[1], m)                                # type: ignore[arg-type]


def _apply_pauli(eng: StatevectorEngine, q: int, k: int) -> None:
    if k == 1:
        eng.apply_x(q)
    elif k == 2:
        eng.apply_1q(q, _PAULI[2])
    elif k == 3:
        eng.apply_diag_1q(q, np.array([1.0, -1.0]))


def _reduced(eng: StatevectorEngine, q: int) -> np.ndarray:
    v = eng.psi.reshape(-1, 2, 1 << q)
    a0, a1 = v[:, 0, :], v[:, 1, :]
    r10 = complex(np.vdot(a0, a1))
    return np.array([[np.vdot(a0, a0).real, r10.conjugate()], [r10, np.vdot(a1, a1).real]])


class _Branch:
    __slots__ = ("pos", "eng", "bits", "n", "leaked")

    def __init__(self, pos: int, eng: StatevectorEngine, bits: dict[int, int], n: int, leaked: set[int]) -> None:
        self.pos, self.eng, self.bits, self.n, self.leaked = pos, eng, bits, n, leaked

    def fork(self, n: int, bits: dict[int, int] | None = None) -> _Branch:
        return _Branch(self.pos, self.eng.copy(), dict(bits if bits is not None else self.bits), n, set(self.leaked))


def run_trajectories(prog: Program, instrs: list[NoisyInstr], *, shots: int, trajectories: int,
                     rng: np.random.Generator, readout: ReadoutFn, groups: GroupsFn | None = None,
                     seepage: float = 0.0, detuning_khz: dict[int, float] | None = None,
                     dtype: np.dtype | type = np.complex128, timer: SpanTimer | None = None,
                     want_memory: bool = False, memory_budget: int | None = None,
                     batched: bool | None = None) -> tuple[RunOutput, dict[str, Any]]:
    timer = timer or SpanTimer()
    native = prog.native
    ops = native.ops
    nc = native.num_clbits
    inv_qmap = {s: p for p, s in prog.qmap.items()}
    detuning_khz = {q: s for q, s in (detuning_khz or {}).items() if s > 0}
    tallies: dict[int, int] = {}
    sampler = TerminalSampler(prog, readout, groups, rng, timer)

    leak_events = 0
    jumps = 0
    t_count = max(1, min(trajectories, shots))
    base_n, extra = divmod(shots, t_count)
    branches_total = 0

    if batched is None:
        batched = _BATCHED and batchable(prog, instrs, detuning_khz)
    if batched:
        with timer.span("simulation"):
            tallies, jumps, rows = _run_batched(prog, instrs, t_count=t_count, shots=shots, rng=rng,
                                                readout=readout, groups=groups, dtype=np.dtype(dtype),
                                                memory_budget=memory_budget)
        branches_total = t_count

    for t in range(t_count if not batched else 0):
        n_t = base_n + (1 if t < extra else 0)
        delta = {q: float(rng.normal(0.0, s)) for q, s in detuning_khz.items()}
        stack = [_Branch(0, StatevectorEngine(prog.n_sim, dtype), {}, n_t, set())]
        branches_total += 1
        while stack:
            br = stack.pop()
            eng = br.eng
            while br.pos < len(instrs):
                ins = instrs[br.pos]
                br.pos += 1
                if ins.condition is not None and not ins.condition.holds(_BitView(br.bits)):
                    continue
                k = ins.kind
                with timer.span("simulation"):
                    if k == "gate":
                        hit = [q for q in ins.qubits if q in br.leaked]
                        if not hit:
                            _apply(eng, ins.qubits, ins.matrix, ins.diagonal)
                        elif ins.physical:
                            if len(ins.qubits) == 2 and len(hit) == 1:
                                other = ins.qubits[1] if hit[0] == ins.qubits[0] else ins.qubits[0]
                                _apply_pauli(eng, other, int(rng.integers(4)))
                            for q in hit:                         # seepage back to |1⟩
                                if seepage > 0 and rng.random() < seepage:
                                    br.leaked.discard(q)
                        continue
                    if k == "channel":
                        q = ins.qubits[0]
                        if q in br.leaked:
                            continue
                        kraus = ins.kraus or ()
                        if len(kraus) > 1:
                            rho = _reduced(eng, q)
                            probs = [max(0.0, float(np.real(np.trace(kk @ rho @ kk.conj().T)))) for kk in kraus]
                            total = sum(probs)
                            u = rng.random() * total
                            j = 0
                            while j < len(probs) - 1 and u >= probs[j]:
                                u -= probs[j]
                                j += 1
                            kk = kraus[j] / math.sqrt(probs[j] / total)
                            jumps += j != 0
                            if abs(kk[0, 1]) == 0 and abs(kk[1, 0]) == 0:
                                eng.apply_diag_1q(q, np.diag(kk).copy())
                            else:
                                eng.apply_1q(q, kk)
                        elif kraus:
                            _apply(eng, (q,), kraus[0], None)
                        continue
                    if k == "depolarize":
                        if any(q in br.leaked for q in ins.qubits):
                            continue
                        d2 = 1 << (2 * len(ins.qubits))
                        if rng.random() < ins.p * (d2 - 1) / d2:
                            jumps += 1
                            code = int(rng.integers(1, d2))
                            for q in ins.qubits:
                                _apply_pauli(eng, q, code & 3)
                                code >>= 2
                        continue
                    if k == "detune":
                        q = ins.qubits[0]
                        if q in delta and q not in br.leaked:
                            phi = 2.0 * math.pi * delta[q] * ins.duration_ns * 1e-6
                            eng.apply_diag_1q(q, np.array([np.exp(-0.5j * phi), np.exp(0.5j * phi)]))
                        continue
                    if k == "leak":
                        q = ins.qubits[0]
                        if q not in br.leaked and rng.random() < ins.p:
                            leak_events += 1
                            p1 = eng.prob_one(q)
                            outcome = 1 if rng.random() < p1 else 0
                            eng.collapse(q, outcome, p1 if outcome else 1.0 - p1)
                            if outcome == 0:
                                eng.apply_x(q)                     # park the leaked qubit in |1⟩
                            br.leaked.add(q)
                        continue
                # -------- measure / reset: exact binomial shot branching
                kind = prog.terminal.get(ins.index) if k == "measure" else None
                if kind in ("terminal", "dead") or (kind == "idle" and not sampler.walks(ins.index)):
                    continue
                if kind == "idle":
                    p01, _ = readout(ops[ins.index].qubits[0])
                    for c in ins.clbits:
                        br.bits[c] = 0
                    if p01 > 0:
                        n1 = int(rng.binomial(br.n, p01))
                        if n1 == br.n:
                            for c in ins.clbits:
                                br.bits[c] = 1
                        elif n1:
                            stack.append(br.fork(n1, {**br.bits, **{c: 1 for c in ins.clbits}}))
                            branches_total += 1
                            br.n -= n1
                    continue
                q = ins.qubits[0]
                if q < 0:
                    continue
                with timer.span("simulation"):
                    p1 = eng.prob_one(q)
                    n1 = int(rng.binomial(br.n, p1)) if 0.0 < p1 < 1.0 else (br.n if p1 >= 1.0 else 0)
                    n0 = br.n - n1
                    if n0 and n1:
                        other = br.fork(n1)
                        other.eng.collapse(q, 1, p1)
                        stack.append(other)
                        branches_total += 1
                        eng.collapse(q, 0, 1.0 - p1)
                        outcome, br.n = 0, n0
                        pending_other: _Branch | None = other
                    else:
                        outcome = 1 if n1 else 0
                        eng.collapse(q, outcome, p1 if outcome else 1.0 - p1)
                        pending_other = None
                if k == "reset":
                    br.leaked.discard(q)
                    if outcome == 1:
                        eng.apply_x(q)
                    if pending_other is not None:
                        pending_other.leaked.discard(q)
                        pending_other.eng.apply_x(q)
                    continue
                # measurement: true outcome → recorded bit (readout error) for this branch and the forked one
                p01, p10 = readout(inv_qmap[q])
                for b, m in ((br, outcome), *(((pending_other, 1),) if pending_other is not None else ())):
                    flip_p = p01 if m == 0 else p10
                    nf = int(rng.binomial(b.n, flip_p)) if flip_p > 0 else 0
                    rec = m ^ 1
                    if nf == b.n:
                        for c in ins.clbits:
                            b.bits[c] = rec
                        continue
                    if nf:
                        stack.append(b.fork(nf, {**b.bits, **{c: rec for c in ins.clbits}}))
                        branches_total += 1
                        b.n -= nf
                    for c in ins.clbits:
                        b.bits[c] = m
            sampler.sample(br.eng.marginal, br.bits, br.n, tallies)

    with timer.span("sampling"):
        counts = {format(v, f"0{nc}b") if nc else "": k for v, k in sorted(tallies.items())}
        memory = None
        if want_memory:
            keys = list(counts)
            reps = np.repeat(np.arange(len(keys)), [counts[k] for k in keys])
            rng.shuffle(reps)
            memory = [keys[j] for j in reps.tolist()]
    diag = {"trajectories": t_count, "shots_per_trajectory": [base_n, base_n + (1 if extra else 0)],
            "exact_sampling": t_count == shots, "jumps": jumps, "leak_events": leak_events,
            "batched": bool(batched), **({"batch_rows": rows} if batched else {})}
    return RunOutput(counts, memory, None, "trajectories", branches_total), diag


# =========================================================== batched executor
_BATCHED = os.environ.get("QPUTWIN_TRAJECTORY_BATCHED", "1").strip().lower() not in ("0", "false", "no")
_BATCH_AMP_BITS = 18            # amplitudes per sub-batch, rows × 2^n ≈ 2^18 (4 MiB) (measured optimum, 4 and 16 vCPUs)
_WIDE_BITS = 20                 # from 2^20 amplitudes per trajectory, one trajectory at a time, threaded inside
_LEVELS = (96, 24, 6)           # statically fused window sizes, coarse to fine (each divides the previous)
_OUTER_MAX = int(os.environ.get("QPUTWIN_TRAJECTORY_WORKERS", "4"))  # concurrent sub-batches: the Python
# between NumPy calls holds the GIL, and on a 16-core host more than 4 was measured slower, not faster
_RENORM_FLOOR = 1e-30           # renormalise all rows once their norm² may have fallen below this
_I2 = np.eye(2, dtype=np.complex128)


def batchable(prog: Program, instrs: list[NoisyInstr], detuning_khz: dict[int, float] | None = None) -> bool:
    """The batched executor covers branch-free programs without leakage, resets or quasi-static detuning.

    Those three need a per-trajectory control flow (or a per-trajectory frequency) and keep the
    sequential walk; every other noise term is either a fixed operator or a state-independent jump.
    """
    for ins in instrs:
        if ins.condition is not None or ins.kind in ("reset", "leak"):
            return False
        if ins.kind == "measure" and ins.index not in prog.terminal:
            return False
        if ins.kind == "detune" and detuning_khz and ins.qubits[0] in detuning_khz:
            return False
    return True


def _compile(instrs: list[NoisyInstr]) -> list[tuple[Any, ...]]:
    """Noisy program → ``("u1", q, M)``, ``("u2", a, b, M)``, ``("k", q, Ks, Es, k*, α)``, ``("d", qs, p_jump, d²)``.

    For a Kraus channel the operator ``K*`` with the largest ``α = λ_min(K*†K*)`` is the no-jump
    branch: its Born weight ``Tr(K*ρK*†)/Tr ρ`` is at least ``α`` for *every* state, so a uniform draw
    ``u < α`` selects it without looking at the state. Only the rare ``u ≥ α`` trajectories need their
    reduced density matrix; all others take ``K*``, a fixed operator that fuses with the gates around it.
    """
    out: list[tuple[Any, ...]] = []
    for ins in instrs:
        k = ins.kind
        if k == "gate":
            m = np.diag(ins.diagonal) if ins.diagonal is not None else ins.matrix
            m = np.asarray(m, dtype=np.complex128)
            if len(ins.qubits) == 1:
                out.append(("u1", ins.qubits[0], m))
            else:
                out.append(("u2", ins.qubits[0], ins.qubits[1], m))
        elif k == "channel":
            ks = [np.asarray(x, dtype=np.complex128) for x in (ins.kraus or ())]
            if not ks:
                continue
            if len(ks) == 1:
                out.append(("u1", ins.qubits[0], ks[0]))
                continue
            es = [x.conj().T @ x for x in ks]
            lam = [float(np.linalg.eigvalsh(e)[0]) for e in es]
            star = int(np.argmax(lam))
            out.append(("k", ins.qubits[0], ks, es, star, max(0.0, lam[star])))
        elif k == "depolarize":
            d2 = 1 << (2 * len(ins.qubits))
            pj = ins.p * (d2 - 1) / d2
            if pj > 0:
                out.append(("d", ins.qubits, pj, d2))
        # terminal measurements and zero-spread detuning leave the state alone
    return out


def _windows(ops: list[tuple[Any, ...]]) -> list[dict[int, tuple[list[Any], float]]]:
    """Fused blocks for aligned windows at every level of ``_LEVELS`` (every channel taken as its ``K*``).

    ``levels[l][s]`` = (blocks, Π α) of the window ``[s, s + _LEVELS[l])``. A window that holds a jump
    is replayed from its sub-windows one level down, so a jump costs op-by-op work only inside the
    smallest window around it.
    """
    from qputwin_core.engine import FUSION_MAX_DIAGONAL_QUBITS, FUSION_MAX_QUBITS
    from qputwin_core.fused import plan_blocks
    out = []
    for size in _LEVELS:
        level: dict[int, tuple[list[Any], float]] = {}
        for s in range(0, len(ops), size):
            gates: list[tuple[tuple[int, ...], np.ndarray]] = []
            alpha = 1.0
            for op in ops[s:s + size]:
                if op[0] == "u1":
                    gates.append(((op[1],), op[2]))
                elif op[0] == "u2":
                    gates.append(((op[1], op[2]), op[3]))
                elif op[0] == "k":
                    gates.append(((op[1],), op[2][op[4]]))
                    alpha *= op[5]
            level[s] = (plan_blocks(gates, FUSION_MAX_QUBITS, FUSION_MAX_DIAGONAL_QUBITS) if gates else [], alpha)
        out.append(level)
    return out


def _is_diag(m: np.ndarray) -> bool:
    return not np.any(m - np.diag(np.diagonal(m)))


def _axis_apply(x: np.ndarray, pos: int, m: np.ndarray) -> None:
    """``m`` (2×2) on axis ``pos`` of the tensor (or tensor view) ``x``, in place."""
    a0 = x[(slice(None),) * pos + (0,)]
    a1 = x[(slice(None),) * pos + (1,)]
    m00, m01, m10, m11 = (complex(v) for v in m.ravel())
    if m01 == 0 and m10 == 0:
        if m00 != 1:
            a0 *= m00
        if m11 != 1:
            a1 *= m11
        return
    t = a0.copy()
    a0 *= m00
    if m01 != 0:
        a0 += m01 * a1
    a1 *= m11
    if m10 != 0:
        a1 += m10 * t


class _Batch:
    """``rows`` trajectories as one ``(n + r)``-axis state: axes ``n … n+r−1`` index the trajectory.

    Uniform work (gates, and the no-jump branch of every channel) runs on the fused executor, as
    precomputed blocks for whole windows. A window that holds a jump runs op by op: one-qubit operators
    wait as a pending ``2×2`` per qubit until a two-qubit gate absorbs them (``M·(U_a ⊗ U_b)``) or a jump
    on that qubit needs them. A jump touches only its own rows, through strided views in the executor's
    current axis order (no reordering of the batch). Rows are kept unnormalised: each Born weight is
    taken relative to the row's own norm, and rows are rescaled only when their norm could underflow.
    """

    def __init__(self, n: int, r: int, dtype: np.dtype) -> None:
        from qputwin_core.fused import FusedState
        self.n, self.r, self.rows, self.dtype = n, r, 1 << r, dtype
        self.st = FusedState(n + r, dtype)
        if r:                                                      # every row starts in |0…0⟩
            init = np.zeros((1 << r, 1 << r), dtype=np.complex128)
            init[:, 0] = 1.0
            self.st.apply_dense(list(range(n + r - 1, n - 1, -1)), init)
        self.pend: dict[int, np.ndarray] = {}
        self.floor = 1.0

    # ------------------------------------------------------------- fused path
    def flush(self, q: int) -> None:
        m = self.pend.pop(q, None)
        if m is not None:
            if _is_diag(m):
                self.st.apply_diagonal([q], np.diagonal(m).copy())
            else:
                self.st.apply_dense([q], m)

    def flush_all(self) -> None:
        for q in list(self.pend):
            self.flush(q)

    def gate2(self, a: int, b: int, m: np.ndarray) -> None:
        ua, ub = self.pend.pop(a, None), self.pend.pop(b, None)
        if ua is not None or ub is not None:
            m = m @ np.kron(_I2 if ua is None else ua, _I2 if ub is None else ub)
        if _is_diag(m):
            self.st.apply_diagonal([a, b], np.diagonal(m).copy())
        else:
            self.st.apply_dense([a, b], m)

    # ----------------------------------------------------- per-row (jump) path
    def activate(self, qs: Any) -> None:
        missing = [q for q in dict.fromkeys(qs) if q not in self.st.perm]
        if missing:                                               # an untouched qubit is |0⟩: store it
            e0 = np.zeros((1 << len(missing), 1), dtype=self.dtype)
            e0[0, 0] = 1.0
            self.st._apply([], missing, e0)

    def tensor(self) -> tuple[np.ndarray, list[int]]:
        """The stored tensor (a view) and the stored order of the data axes (batch axes left out)."""
        perm = self.st.perm
        return self.st.cur.reshape((2,) * len(perm)), [a for a in perm if a < self.n]

    def row(self, t: np.ndarray, rows: Any) -> tuple[Any, ...]:
        """Index of row ``rows`` (an int) in the stored tensor ``t``."""
        idx: list[Any] = [slice(None)] * t.ndim
        for pos, a in enumerate(self.st.perm):
            if a >= self.n:
                idx[pos] = (rows >> (a - self.n)) & 1
        return tuple(idx)

    def apply_all(self, q: int, m: np.ndarray) -> None:
        """``m`` on qubit ``q`` of every row, in place (the stored order is kept)."""
        t, _ = self.tensor()
        _axis_apply(t, self.st.perm.index(q), m)

    def renormalise(self) -> None:
        self.flush_all()
        P = self.st.standard().reshape(self.rows, 1 << self.n)
        norm = np.sqrt(np.einsum("ij,ij->i", P.real, P.real) + np.einsum("ij,ij->i", P.imag, P.imag))
        P /= norm[:, None].astype(P.real.dtype)
        self.floor = 1.0


def _channel_weights(y: np.ndarray, pos: int, es: list[np.ndarray]) -> np.ndarray:
    """Born weights ``Tr(E_k ρ_q)/Tr ρ_q`` of each row of ``y`` (rows first; qubit ``q`` on axis ``pos``)."""
    a0 = np.moveaxis(y.take(0, axis=pos), 0, 0).reshape(y.shape[0], -1)
    a1 = np.moveaxis(y.take(1, axis=pos), 0, 0).reshape(y.shape[0], -1)
    r00 = np.einsum("ij,ij->i", a0, a0.conj()).real
    r11 = np.einsum("ij,ij->i", a1, a1.conj()).real
    r10 = np.einsum("ij,ij->i", a1, a0.conj())
    tot = np.maximum(r00 + r11, 1e-300)
    w = np.stack([(e[0, 0].real * r00 + e[1, 1].real * r11 + 2.0 * (e[0, 1] * r10).real) / tot for e in es], axis=1)
    return np.maximum(w, 0.0)


def _op_by_op(b: _Batch, op: tuple[Any, ...], u: np.ndarray | None, rng: np.random.Generator, live: int) -> int:
    """One op of a window that holds a jump; ``u`` = this op's uniform draws.

    Returns the jumps taken by the first ``live`` rows (the rest pad the last batch to a power of two).
    """
    kind = op[0]
    if kind == "u1":
        q, m = op[1], op[2]
        p = b.pend.get(q)
        b.pend[q] = m if p is None else m @ p
        return 0
    if kind == "u2":
        b.gate2(op[1], op[2], op[3])
        return 0
    if kind == "k":
        _, q, ks, es, star, alpha = op
        cand = np.flatnonzero(u >= alpha) if u is not None else np.empty(0, dtype=np.int64)   # None: no candidate
        uq = b.pend.get(q, _I2)
        jumps = 0
        if cand.size:
            b.activate([q, *b.pend])
            t, dax = b.tensor()
            y = np.stack([t[b.row(t, int(c))] for c in cand])        # copies of the candidate rows
            for p, m in b.pend.items():                              # their exact state: pending ops applied
                _axis_apply(y, 1 + dax.index(p), m)
            pos = 1 + dax.index(q)
            w = _channel_weights(y, pos, es)
            choice = np.full(cand.size, star)
            rest = u[cand] - w[:, star]
            for k in range(len(ks)):
                if k == star:
                    continue
                take = (choice == star) & (rest >= 0) & (rest < w[:, k])
                choice[take] = k
                rest = np.where(choice == star, rest - w[:, k], rest)
            # rounding can leave Σw a hair below 1: such a draw keeps the no-jump branch
            jumped = np.flatnonzero(choice != star)
            if jumped.size:
                old = [t[b.row(t, int(cand[j]))].copy() for j in jumped]    # raw rows (q's pending not applied)
                b.pend.pop(q, None)
                b.apply_all(q, ks[star] @ uq)                       # every row: K*·U_q, in place
                for j, x in zip(jumped.tolist(), old, strict=True):
                    k = int(choice[j])
                    _axis_apply(x, pos - 1, ks[k] @ uq)
                    x *= 1.0 / math.sqrt(max(float(w[j, k]), 1e-300))
                    t[b.row(t, int(cand[j]))] = x
                jumps = int((cand[jumped] < live).sum()) or -1     # -1: rows jumped, none of them live
        if not jumps:
            b.pend[q] = ks[star] @ uq
        b.floor *= alpha
        if b.floor < _RENORM_FLOOR:
            b.renormalise()
        return max(jumps, 0)
    # "d": state-independent Pauli jumps
    _, qs, pj, d2 = op
    if u is None:                                                # no row of this batch jumps here
        return 0
    hit = np.flatnonzero(u < pj)
    if not hit.size:
        return 0
    codes = rng.integers(1, d2, size=hit.size)
    b.activate(qs)
    for q in qs:
        m = b.pend.pop(q, None)
        if m is not None:
            b.apply_all(q, m)
    t, dax = b.tensor()
    for rw, code in zip(hit.tolist(), codes.tolist(), strict=True):
        x = t[b.row(t, rw)]                                          # a view: written in place
        for q in qs:
            if code & 3:
                _axis_apply(x, dax.index(q), _PAULI[code & 3])
            code >>= 2
    return int((hit < live).sum())


def _stochastic(ops: list[tuple[Any, ...]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices of the stochastic ops, whether each is a Kraus channel, and its threshold (α or p_jump)."""
    idx = [i for i, op in enumerate(ops) if op[0] in ("k", "d")]
    return (np.array(idx, dtype=np.int64), np.array([ops[i][0] == "k" for i in idx], dtype=bool),
            np.array([ops[i][5] if ops[i][0] == "k" else ops[i][2] for i in idx], dtype=np.float64))


def _walk(ops: list[tuple[Any, ...]], levels: list[dict[int, tuple[list[Any], float]]],
          stochastic: tuple[np.ndarray, np.ndarray, np.ndarray], n: int, r: int,
          rng: np.random.Generator, dtype: np.dtype, live: int) -> tuple[np.ndarray, int]:
    import bisect
    rows = 1 << r
    # every random decision of this sub-batch, drawn up front: which ops jump is known before any work
    stoch, is_k, thr = stochastic
    draws = rng.random((len(stoch), rows))
    hit = np.where(is_k[:, None], draws >= thr[:, None], draws < thr[:, None]).any(axis=1)
    sel = np.flatnonzero(hit)
    events = stoch[sel].tolist()                                 # ascending op indices
    u_of = dict(zip(events, draws[sel], strict=True))
    b = _Batch(n, r, dtype)
    jumps = 0

    def run(level: int, s: int, e: int) -> None:
        nonlocal jumps
        size = _LEVELS[level]
        for w in range(s, e, size):
            we = min(e, w + size)
            k = bisect.bisect_left(events, w)
            if k == len(events) or events[k] >= we:            # no jump here: the precomputed blocks
                if b.pend:
                    b.flush_all()
                blocks, alpha = levels[level][w]
                for blk in blocks:
                    b.st.apply(blk)
                b.floor *= alpha
                if b.floor < _RENORM_FLOOR:
                    b.renormalise()
            elif level + 1 < len(_LEVELS):
                run(level + 1, w, we)
            else:
                for i in range(w, we):
                    jumps += _op_by_op(b, ops[i], u_of.get(i), rng, live)

    try:
        run(0, 0, len(ops))
        b.flush_all()
        P = b.st.standard().reshape(rows, 1 << n)
        p = np.square(P.real, dtype=np.float64)
        p += np.square(P.imag, dtype=np.float64)
        p /= p.sum(axis=1, keepdims=True)
        return p, jumps
    finally:
        # ``run`` refers to itself through its closure, a reference cycle that would keep this sub-batch's
        # state buffers alive until the cyclic garbage collector happens to run (gigabytes over a long job).
        # Clearing the cell breaks the cycle, so the buffers are freed as soon as the sub-batch ends.
        run = None  # type: ignore[assignment]


def _run_batched(prog: Program, instrs: list[NoisyInstr], *, t_count: int, shots: int, rng: np.random.Generator,
                 readout: ReadoutFn, groups: GroupsFn | None, dtype: np.dtype,
                 memory_budget: int | None) -> tuple[dict[int, int], int, int]:
    from concurrent.futures import ThreadPoolExecutor

    from qputwin_core.fused import engine_threads, thread_limit
    n = prog.n_sim
    # rows per sub-batch: a power of two (batch axes), ≤ T/8 so the padded last batch wastes little
    r = max(0, min(_BATCH_AMP_BITS - n, max(1, t_count // 8).bit_length() - 1))
    rows = 1 << r
    starts = list(range(0, t_count, rows))
    seeds = np.random.SeedSequence(int(rng.integers(1 << 63))).spawn(len(starts))
    ops = _compile(instrs)
    levels = _windows(ops)
    stochastic = _stochastic(ops)
    base_n, extra = divmod(shots, t_count)
    # one sub-batch holds two state buffers of rows × 2^n amplitudes and float64 probabilities
    per_batch = rows * (1 << n) * (2 * dtype.itemsize + 8)
    wide = n >= _WIDE_BITS
    threads = engine_threads()
    workers = 1 if wide else max(1, min(threads, _OUTER_MAX, len(starts),
                                         (memory_budget // per_batch) if memory_budget else len(starts)))
    inner = None if wide else 1                  # the fused pool inside many workers only adds contention

    def one(j: int) -> tuple[dict[int, int], int]:
        s = starts[j]
        g = np.random.default_rng(seeds[j])
        with thread_limit(inner):
            p, jumps = _walk(ops, levels, stochastic, n, r, g, dtype, min(rows, t_count - s))
        sampler = TerminalSampler(prog, readout, groups, g, SpanTimer())
        tallies: dict[int, int] = {}
        for i in range(min(rows, t_count - s)):                  # padding rows of the last batch are dropped
            t = s + i
            n_t = base_n + (1 if t < extra else 0)
            if n_t:
                row = p[i]
                sampler.sample(lambda qs, row=row: _row_marginal(row, qs, n), {}, n_t, tallies)
        return tallies, jumps

    if workers == 1:
        results = [one(j) for j in range(len(starts))]
    else:
        with ThreadPoolExecutor(workers, thread_name_prefix="qputwin-traj") as ex:
            results = list(ex.map(one, range(len(starts))))
    tallies: dict[int, int] = {}
    jumps = 0
    for tl, jp in results:
        jumps += jp
        for v, k in tl.items():
            tallies[v] = tallies.get(v, 0) + k
    return tallies, jumps, rows


def _row_marginal(p: np.ndarray, qubits: list[int], n: int) -> np.ndarray:
    if len(qubits) == n:
        return p.copy()
    keep = set(qubits)
    axes = tuple(n - 1 - q for q in range(n) if q not in keep)
    pm = p.reshape((2,) * n).sum(axis=axes).ravel() if axes else p.copy()
    return pm / pm.sum()
