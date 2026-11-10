"""Batched, window-fused trajectory executor (``qputwin_core.trajectory._run_batched``).

Every trajectory is still an exact unravelling of the calibrated noise: the batched executor must
reproduce the exact density-matrix distribution, take the same number of jumps as the sequential
walk, give results that do not depend on the thread count, and step aside for leakage and
mid-circuit measurement.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import qputwin_core.trajectory as T
from qputwin_core.devices import DEVICES
from qputwin_core.fused import engine_threads, set_engine_threads
from qputwin_core.simulators import ExecutionOptions, SimulatorConfig, simulator_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import circuits as C  # noqa: E402

ROUTE = {"method": "sabre", "seed": 0}


def _tv(a: dict[str, int], b: dict[str, int]) -> float:
    sa, sb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in set(a) | set(b))


def _run(device: str, qasm: str, engine: str, shots: int, seed: int, **noise):
    sim = simulator_for(DEVICES[device], SimulatorConfig(max_trajectories=50_000))
    return sim.execute(qasm, ExecutionOptions(shots=shots, seed=seed, noise=noise, engine=engine, routing=ROUTE))


@pytest.mark.parametrize("device,fam,n", [("ibm_nighthawk", "qft", 5), ("google_willow", "qaoa", 5),
                                          ("rigetti_ankaa_3", "ghz", 6), ("iqm_emerald", "qv", 4)])
def test_batched_trajectories_reproduce_the_exact_density_matrix(device: str, fam: str, n: int) -> None:
    qasm = C.FAMILIES[fam](n, seed=1)
    exact = _run(device, qasm, "density_matrix", 1_000_000, 1, scale=5.0).counts
    traj = _run(device, qasm, "trajectory", 30_000, 2, scale=5.0)
    assert traj.noise["batched"] and traj.noise["exact_sampling"] and traj.noise["jumps"] > 0
    k = len(exact)
    # finite-sample TV of a correct sampler is ≈ √(K / 2πN); allow 2.5× that
    assert _tv(traj.counts, exact) < 2.5 * np.sqrt(k / (2 * np.pi * 30_000))


def test_batched_and_sequential_walks_take_the_same_jumps() -> None:
    qasm = C.FAMILIES["ghz"](5, seed=1)
    shots = 6000                                       # not a power of two: the last batch is padded
    old = T._BATCHED
    try:
        T._BATCHED = False
        seq = _run("ibm_heron", qasm, "trajectory", shots, 4, scale=5.0)
        T._BATCHED = True
        bat = _run("ibm_heron", qasm, "trajectory", shots, 4, scale=5.0)
    finally:
        T._BATCHED = old
    assert not seq.noise["batched"] and bat.noise["batched"]
    assert sum(bat.counts.values()) == shots
    j_seq, j_bat = seq.noise["jumps"], bat.noise["jumps"]
    assert abs(j_seq - j_bat) < 5 * np.sqrt(j_seq), (j_seq, j_bat)    # padding rows are not counted
    assert _tv(seq.counts, bat.counts) < 0.05


def test_batched_results_do_not_depend_on_the_thread_count() -> None:
    qasm = C.FAMILIES["qaoa"](6, seed=1)
    before = engine_threads()
    try:
        set_engine_threads(1)
        one = _run("ibm_heron", qasm, "trajectory", 2000, 9)
        set_engine_threads(4)
        four = _run("ibm_heron", qasm, "trajectory", 2000, 9)
    finally:
        set_engine_threads(before)
    assert one.counts == four.counts and one.noise["jumps"] == four.noise["jumps"]


def test_wide_mode_one_trajectory_at_a_time_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    qasm = C.FAMILIES["qft"](4, seed=1)
    exact = _run("ibm_nighthawk", qasm, "density_matrix", 1_000_000, 1, scale=5.0).counts
    monkeypatch.setattr(T, "_WIDE_BITS", 2)            # take the n ≥ _WIDE_BITS path on a small circuit
    monkeypatch.setattr(T, "_BATCH_AMP_BITS", 2)       # … with one row per sub-batch
    r = _run("ibm_nighthawk", qasm, "trajectory", 4000, 3, scale=5.0)
    assert r.noise["batched"] and r.noise["batch_rows"] == 1
    assert _tv(r.counts, exact) < 2.5 * np.sqrt(len(exact) / (2 * np.pi * 4000))


def test_sub_batch_state_is_freed_without_the_cycle_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finished sub-batch must free its state at once, not wait for the cyclic garbage collector.

    A reference cycle once kept every finished sub-batch's buffers alive until a collection: a
    16-qubit noisy job peaked at 1.6 GiB instead of about 0.1 GiB, and seven workers could exhaust a host.
    """
    import gc
    import weakref

    finished: list[weakref.ref] = []
    original = T._Batch.__init__

    def tracked(self, *args, **kwargs) -> None:
        original(self, *args, **kwargs)
        finished.append(weakref.ref(self))

    monkeypatch.setattr(T._Batch, "__init__", tracked)
    gc.collect()
    gc.disable()
    try:
        r = _run("quantinuum_helios", C.FAMILIES["ghz"](8, seed=1), "trajectory", 2000, 1)
        assert r.noise["batched"] and len(finished) > 1
        assert all(ref() is None for ref in finished), "a sub-batch state outlived its sub-batch"
    finally:
        gc.enable()


def test_leakage_and_mid_circuit_measurement_keep_the_sequential_walk() -> None:
    sim = simulator_for(DEVICES["iqm_emerald"])
    src = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2]; creg c[2];\nh q[0];\nmeasure q[0] -> c[0];\n'
           "cx q[0],q[1];\nmeasure q[1] -> c[1];\n")
    mid = sim.execute(src, ExecutionOptions(shots=200, seed=1, noise={}, engine="trajectory"))
    assert mid.noise["batched"] is False
    leak = sim.execute(C.FAMILIES["ghz"](3, seed=1), ExecutionOptions(
        shots=200, seed=1, noise={"leakage": True, "leakage_rate": 0.01}, engine="trajectory"))
    assert leak.noise["batched"] is False


def test_auto_engine_follows_the_cost_model() -> None:
    sim = simulator_for(DEVICES["ibm_heron"])
    qasm = C.FAMILIES["ghz"](12, seed=1)
    many = sim.execute(qasm, ExecutionOptions(shots=1024, seed=1, noise={}, routing=ROUTE))
    few = sim.execute(qasm, ExecutionOptions(shots=64, seed=1, noise={}, routing=ROUTE))
    assert many.workload["engine"] == "density_matrix"          # 2^12 ≤ 4 · 1024
    assert few.workload["engine"] == "trajectory"               # 2^12 > 4 · 64
