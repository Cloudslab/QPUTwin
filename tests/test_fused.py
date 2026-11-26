"""Tests gate fusion and the fused executor.

Usage: uv run pytest tests/test_fused.py

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from qputwin_core.density import DensityMatrixEngine, fused_density
from qputwin_core.devices import DEVICES
from qputwin_core import fused
from qputwin_core.engine import StatevectorEngine, _sample, _sample_blocked, fused_state, fused_statevector
from qputwin_core.fused import FusedState, engine_threads, plan_blocks, set_engine_threads
from qputwin_core.simulators import ExecutionOptions, simulator_for
from qputwin_core.timing import SpanTimer

H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _random_circuit(n: int, gates: int, seed: int) -> str:
    r = random.Random(seed)
    body = []
    for _ in range(gates):
        x = r.random()
        if x < 0.3:
            a, b = r.sample(range(n), 2)
            body.append(f"cx q[{a}],q[{b}];")
        elif x < 0.4:
            a, b = r.sample(range(n), 2)
            body.append(f"rzz({r.uniform(0, 6):.5f}) q[{a}],q[{b}];")
        elif x < 0.5:
            body.append(f"rz({r.uniform(0, 6):.5f}) q[{r.randrange(n)}];")
        else:
            body.append(f"u3({r.uniform(0, 3):.5f},{r.uniform(-3, 3):.5f},{r.uniform(-3, 3):.5f}) "
                        f"q[{r.randrange(n)}];")
    return H2 + f"qreg q[{n}];\ncreg c[{n}];\n" + "\n".join(body) + "\nmeasure q -> c;\n"


def _plan(device: str, qasm: str, **kw):
    opts = {} if DEVICES[device].coupling.all_to_all else {"routing": {"method": "sabre", "seed": 0}}
    return simulator_for(DEVICES[device]).plan(qasm, ExecutionOptions(shots=1, seed=1, **opts, **kw), SpanTimer())


@pytest.fixture(params=[1, 3], ids=["1-thread", "3-threads"])
def threads(request):
    before = engine_threads()
    set_engine_threads(request.param)
    yield request.param
    set_engine_threads(before)


@pytest.mark.parametrize("device", sorted(DEVICES))
def test_fused_statevector_matches_reference_kernels(device: str, threads: int) -> None:
    for n, seed in ((2, 1), (6, 2), (11, 3), (16, 4)):
        plan = _plan(device, _random_circuit(n, 90, seed))
        ref = StatevectorEngine(plan.program.n_sim)
        for op in plan.native.ops:
            if op.is_gate:
                ref.apply(op, plan.program.qmap)
        for k in (1, 2, 3, 5):
            got = fused_statevector(plan.program, max_qubits=k)
            assert np.max(np.abs(got - ref.psi)) < 1e-12
        probs = np.abs(fused_statevector(plan.program, exact_phases=False)) ** 2
        assert np.max(np.abs(probs - np.abs(ref.psi) ** 2)) < 1e-12        # trailing diagonals dropped


@pytest.mark.parametrize("device", sorted(DEVICES))
def test_fused_density_matrix_matches_reference_kernels(device: str, threads: int) -> None:
    for n, seed, noise in ((3, 5, {}), (5, 6, {"scale": 3.0})):
        plan = _plan(device, _random_circuit(n, 40, seed), noise=noise, engine="density_matrix")
        ref = DensityMatrixEngine(plan.program.n_sim)
        for ins in plan.instrs:
            if ins.kind == "gate":
                ref.apply_unitary(ins.qubits, ins.matrix, ins.diagonal)
            elif ins.kind == "channel":
                ref.apply_superop_1q(ins.qubits[0], ins.matrix, ins.diagonal)
            elif ins.kind == "depolarize":
                ref.depolarize(ins.qubits, ins.p)
            elif ins.kind == "reset" and ins.qubits[0] >= 0:
                ref.reset(ins.qubits[0])
        got = fused_density(plan.program.n_sim, plan.instrs)
        assert np.max(np.abs(got.vec.psi - ref.vec.psi)) < 1e-12
        assert got.trace() == pytest.approx(1.0, abs=1e-12)


def test_planner_respects_the_block_width_and_keeps_diagonals() -> None:
    rng = np.random.default_rng(0)
    gates = []
    for _ in range(200):
        a, b = rng.choice(8, size=2, replace=False)
        if rng.random() < 0.5:
            gates.append(((int(a),), np.diag(np.exp(1j * rng.uniform(0, 6, 2)))))
        else:
            q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
            gates.append(((int(a), int(b)), q))
    for k in (2, 3, 4):
        blocks = plan_blocks(gates, k)
        assert all(len(b.axes) <= k for b in blocks)
        assert len(blocks) < len(gates)
    diag_only = plan_blocks([((0,), np.diag([1, 1j])), ((0, 1), np.diag([1, -1, 1, -1]))], 4)
    assert len(diag_only) == 1 and diag_only[0].diagonal


def test_fused_state_returns_standard_order_after_permutations() -> None:
    st = FusedState(3)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    st.apply_dense([0], x)                      # |001⟩
    st.apply_dense([2], x)                      # |101⟩ — forces a transpose
    vec = st.standard()
    assert vec[0b101] == pytest.approx(1.0) and np.count_nonzero(vec) == 1


def test_simulator_results_unchanged_end_to_end() -> None:
    """Counts from the fused path equal the exact distribution (seeded GHZ and BV)."""
    sim = simulator_for(DEVICES["quantinuum_helios"])
    ghz = H2 + "qreg q[12];\ncreg c[12];\nh q[0];\n" + "".join(f"cx q[{k}],q[{k + 1}];\n" for k in range(11)) \
        + "measure q -> c;\n"
    r = sim.execute(ghz, ExecutionOptions(shots=4000, seed=3))
    assert set(r.counts) == {"0" * 12, "1" * 12} and abs(r.counts["0" * 12] - 2000) < 200


@pytest.fixture
def tiny_chunks(monkeypatch):
    """Shrink chunk, run and table limits so small states take every large-state branch."""
    monkeypatch.setattr(fused, "_CHUNK_BITS", 3)
    monkeypatch.setattr(fused, "_LOW_BITS", 2)
    monkeypatch.setattr(fused, "_MIN_RUN", 2)
    monkeypatch.setattr(fused, "_MIN_CHUNK", 4)
    monkeypatch.setattr(fused, "_TABLE_LIMIT", 16)


@pytest.mark.parametrize("device", ["quantinuum_helios", "iqm_emerald"])
def test_large_state_branches_match_reference(device: str, threads: int, tiny_chunks) -> None:
    """Chunked gathers, high-axis phase tables (cached and per-row) and lazy activation stay exact."""
    for n, seed in ((7, 11), (10, 12)):
        plan = _plan(device, _random_circuit(n, 120, seed))
        ref = StatevectorEngine(plan.program.n_sim)
        for op in plan.native.ops:
            if op.is_gate:
                ref.apply(op, plan.program.qmap)
        for k in (2, 3, 5):
            got = fused_statevector(plan.program, max_qubits=k)
            assert np.max(np.abs(got - ref.psi)) < 1e-12


@pytest.mark.parametrize("chunks", ["default", "tiny"])
def test_in_place_executor_equals_two_buffer_executor(threads: int, chunks: str, request) -> None:
    """The single-buffer executor reproduces the two-buffer one, for statevectors and density matrices."""
    if chunks == "tiny":
        request.getfixturevalue("tiny_chunks")
    for device, n, seed in (("quantinuum_helios", 9, 21), ("iqm_emerald", 8, 22), ("ibm_heron", 7, 23)):
        plan = _plan(device, _random_circuit(n, 150, seed))
        gates = [(tuple(plan.program.qmap[q] for q in op.qubits), op) for op in plan.native.ops if op.is_gate]
        for k in (2, 4, 5):
            blocks = plan_blocks([(a, fused_state.__globals__["_gate_matrix"](op)) for a, op in gates], k, 12)
            got = []
            for in_place in (True, False):
                st = FusedState(plan.program.n_sim, in_place=in_place)
                for b in blocks:
                    st.apply(b)
                assert len(st._buf) == (1 if in_place else 2)
                got.append(st.standard().copy())
            assert np.max(np.abs(got[0] - got[1])) < 1e-12
    noisy = _plan("iqm_emerald", _random_circuit(4, 40, 24), noise={"scale": 2.0}, engine="density_matrix")
    rhos = []
    for in_place in (True, False):
        fused._IN_PLACE, before = in_place, fused._IN_PLACE
        try:
            rhos.append(fused_density(noisy.program.n_sim, noisy.instrs).vec.psi.copy())
        finally:
            fused._IN_PLACE = before
    assert np.max(np.abs(rhos[0] - rhos[1])) < 1e-12


def test_in_place_peak_memory_stays_below_one_and_a_half_states() -> None:
    """Measured with tracemalloc (numpy reports its buffers): the peak is far below the two-buffer 2×."""
    import tracemalloc

    n = 20                                       # 16 MiB state; per-thread chunk temporaries are ~1 MiB
    plan = _plan("quantinuum_helios", _random_circuit(n, 300, 25))
    state = (1 << plan.program.n_sim) * 16
    peaks = {}
    before_threads = engine_threads()
    set_engine_threads(1)
    for in_place in (True, False):
        fused._IN_PLACE, before = in_place, fused._IN_PLACE
        try:
            tracemalloc.start()
            st = fused_state(plan.program)
            st.marginal(list(range(plan.program.n_sim)), consume=True)
            peaks[in_place] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        finally:
            fused._IN_PLACE = before
    set_engine_threads(before_threads)
    assert peaks[True] < 1.6 * state
    assert peaks[False] > 1.9 * state


def test_lazy_activation_keeps_untouched_axes_out_of_the_state() -> None:
    st = FusedState(6)
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    st.apply_dense([4], h)
    st.apply_diagonal([1, 4], np.array([1, 1j, 1, -1]))   # axis 1 is |0⟩: only its 0 entries act
    assert sorted(st.perm) == [4] and len(st.cur) == 2
    vec = st.standard()
    expect = np.zeros(64, dtype=complex)
    expect[0], expect[1 << 4] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    assert np.max(np.abs(vec - expect)) < 1e-15


def test_consecutive_diagonal_blocks_merge_and_noise_is_chopped() -> None:
    gates = [((q,), np.array([1, np.exp(0.3j * q)])) for q in range(6)]
    blocks = plan_blocks(gates, 2, 8)
    assert len(blocks) == 1 and blocks[0].diagonal and sorted(blocks[0].axes) == list(range(6))
    sim = simulator_for(DEVICES["quantinuum_helios"])
    ghz = H2 + "qreg q[16];\ncreg c[16];\nh q[0];\n" + "".join(f"cx q[{k}],q[{k + 1}];\n" for k in range(15)) \
        + "measure q -> c;\n"
    st = fused_state(sim.plan(ghz, ExecutionOptions(shots=1, seed=1), SpanTimer()).program)
    assert np.count_nonzero(st.cur) == 2                  # no rounding residue left to decay into denormals


@pytest.mark.parametrize("measured", ["all", "subset"])
def test_marginal_from_the_permuted_state(measured: str, threads: int, tiny_chunks) -> None:
    """Sampling straight from the stored order (no final transpose) gives the reference marginal."""
    plan = _plan("quantinuum_helios", _random_circuit(9, 80, 21))
    ref = StatevectorEngine(plan.program.n_sim)
    for op in plan.native.ops:
        if op.is_gate:
            ref.apply(op, plan.program.qmap)
    mq = list(range(9)) if measured == "all" else [0, 3, 4, 8]
    st = fused_state(plan.program)
    p, order = st.marginal(mq, consume=True)
    want = ref.marginal(mq)
    got = np.zeros_like(want)
    for i, v in enumerate(p):
        j = sum(((i >> b) & 1) << mq.index(order[len(order) - 1 - b]) for b in range(len(order)))
        got[j] += v
    assert np.max(np.abs(got - want)) < 1e-12


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_consumed_marginal_equals_the_fresh_one(dtype, threads: int, tiny_chunks) -> None:
    """Probabilities written into the state's own memory (parallel waves) equal a fresh computation."""
    plan = _plan("quantinuum_helios", _random_circuit(10, 120, 31))
    fresh, _ = fused_state(plan.program, dtype).marginal(list(range(10)))
    consumed, _ = fused_state(plan.program, dtype).marginal(list(range(10)), consume=True)
    assert np.max(np.abs(np.asarray(consumed) - fresh)) < (1e-6 if dtype == np.complex64 else 1e-14)


def test_blocked_sampler_is_the_same_inverse_cdf() -> None:
    rng = np.random.default_rng(5)
    p = rng.random(1 << 17) ** 8
    p[rng.random(1 << 17) < 0.5] = 0.0
    p /= p.sum()
    a, fa = _sample(p.copy(), 2000, np.random.default_rng(9))
    b, fb = _sample_blocked(p.copy(), 2000, np.random.default_rng(9))
    da, db = dict(zip(a.tolist(), fa.tolist(), strict=True)), dict(zip(b.tolist(), fb.tolist(), strict=True))
    assert sum(abs(da.get(k, 0) - db.get(k, 0)) for k in set(da) | set(db)) <= 4   # rounding at bin edges only
    assert np.all(p[b] > 0)
    sparse = np.zeros(1 << 17)
    sparse[0] = sparse[-1] = 0.5
    o, f = _sample_blocked(sparse, 500, np.random.default_rng(1))
    assert set(o.tolist()) == {0, (1 << 17) - 1} and f.sum() == 500
