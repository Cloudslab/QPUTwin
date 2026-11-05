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
                     want_memory: bool = False) -> tuple[RunOutput, dict[str, Any]]:
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

    for t in range(t_count):
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
            "exact_sampling": t_count == shots, "jumps": jumps, "leak_events": leak_events}
    return RunOutput(counts, memory, None, "trajectories", branches_total), diag
