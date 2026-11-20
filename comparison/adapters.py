"""Common interface to the simulators compared with QPUTwin.

Usage: import adapters as A  (then A.QPUTwin, A.Aer, A.Braket, A.Cirq, A.Qsim, A.Qulacs)

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))       # the shared circuit library: identical circuits everywhere
import circuits as C  # noqa: E402

from qiskit import QuantumCircuit, qasm2, transpile  # noqa: E402
from qiskit.circuit.library import UGate  # noqa: E402

BASE_URL = os.environ.get("QPUTWIN_BASE_URL", "http://localhost:8000")


# ============================================================ common gate list
@dataclass
class Flat:
    """A circuit flattened to U/CX with its measured-qubit map."""

    n: int
    ops: list[tuple[str, tuple[int, ...], tuple[float, ...]]]
    measured: list[int]                       # measured[k] = qubit feeding clbit k
    qiskit: QuantumCircuit                    # parsed source (with measurements)
    convert_s: float = 0.0
    mats: dict[tuple[float, ...], np.ndarray] = field(default_factory=dict)

    def u_matrix(self, params: tuple[float, ...]) -> np.ndarray:
        m = self.mats.get(params)
        if m is None:
            m = self.mats[params] = UGate(*params).to_matrix()
        return m


def flatten(qasm: str) -> Flat:
    t0 = time.perf_counter()
    qc = qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)   # swap, rzz, … as in QPUTwin
    measured = [0] * qc.num_clbits
    for inst in qc.data:
        if inst.operation.name == "measure":
            measured[qc.find_bit(inst.clbits[0]).index] = qc.find_bit(inst.qubits[0]).index
    body = qc.remove_final_measurements(inplace=False)
    flat = transpile(body, basis_gates=["u", "cx"], optimization_level=0)
    ops = []
    for inst in flat.data:
        name = inst.operation.name
        qs = tuple(flat.find_bit(q).index for q in inst.qubits)
        if name == "u":
            ops.append(("u", qs, tuple(float(p) for p in inst.operation.params)))
        elif name == "cx":
            ops.append(("cx", qs, ()))
        elif name in ("barrier", "id", "delay"):
            continue
        else:                                                      # pragma: no cover - guarded by the basis
            raise ValueError(f"unexpected gate after flattening: {name}")
    return Flat(qc.num_qubits, ops, measured, qc, time.perf_counter() - t0)


def key_from_index(idx: int, measured: list[int], bit_of_qubit) -> str:
    """Bitstring c[n-1]…c[0] for basis-state ``idx``; ``bit_of_qubit(idx, q)`` reads qubit q."""
    return "".join(str(bit_of_qubit(idx, measured[k])) for k in reversed(range(len(measured))))


def marginal(probs: np.ndarray, flat: Flat, big_endian: bool) -> dict[str, float]:
    """Distribution over the measured clbits from a full probability vector."""
    n = flat.n
    out: dict[str, float] = {}
    if big_endian:
        def bit(i: int, q: int) -> int:
            return (i >> (n - 1 - q)) & 1
    else:
        def bit(i: int, q: int) -> int:
            return (i >> q) & 1
    for idx in np.nonzero(probs > 1e-15)[0]:
        k = key_from_index(int(idx), flat.measured, bit)
        out[k] = out.get(k, 0.0) + float(probs[idx])
    return out


# ================================================================== results
@dataclass
class Run:
    counts: dict[str, int] | None = None
    probs: dict[str, float] | None = None
    run_s: float = 0.0                       # simulate + sample (in-process) or client round trip (QPUTwin)
    engine_s: float | None = None            # QPUTwin only: engine time from telemetry
    extra: dict[str, Any] = field(default_factory=dict)


# ================================================================ QPUTwin
class QPUTwin:
    """The running QPUTwin stack, over HTTP."""

    name = "QPUTwin"

    def __init__(self, backend: str = "quantinuum_helios", timeout: float = 1800.0) -> None:
        self.backend = backend
        self.client = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(timeout, connect=5.0))
        self.client.get("/health").raise_for_status()

    def _post(self, qasm: str, shots: int, seed: int, options: dict[str, Any], backend: str | None = None
              ) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        r = self.client.post("/v1/jobs", json={"backend": backend or self.backend, "qasm": qasm, "shots": shots,
                                               "seed": seed, "options": options})
        dt = time.perf_counter() - t0
        body = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"QPUTwin {r.status_code}: {body.get('error', body)}")
        return body, dt

    def sample(self, qasm: str, shots: int, seed: int = 1, options: dict[str, Any] | None = None,
               backend: str | None = None) -> Run:
        body, dt = self._post(qasm, shots, seed, options or {}, backend)
        tel = body["telemetry"]
        return Run(counts=body["result"]["counts"], run_s=dt, engine_s=tel["qpu_compute_time_ms"] / 1e3,
                   extra={"total_roundtrip_ms": tel["total_roundtrip_ms"],
                          "engine": ((body["result"].get("noise") or {}).get("engine")
                                     or (body.get("compilation", {}).get("workload") or {}).get("engine")
                                     or "statevector"),
                          "worker_service_ms": tel["breakdown"]["worker_service_ms"],
                          "compilation": {k: body["compilation"].get(k) for k in
                                          ("native_two_qubit_count", "native_one_qubit_count", "native_depth",
                                           "simulated_qubits", "native_gate_histogram", "physical_qubits_used")}})

    def exact(self, qasm: str, flat: Flat) -> Run:
        body, dt = self._post(qasm, 1, 1, {"return_statevector": True})
        amps = np.array(body["result"]["statevector"])
        p = amps[:, 0] ** 2 + amps[:, 1] ** 2
        pos = {q: j for j, q in enumerate(body["result"]["statevector_qubits"])}
        out: dict[str, float] = {}
        for idx in np.nonzero(p > 1e-15)[0]:            # qubits compacted away were never touched: |0>
            k = key_from_index(int(idx), flat.measured, lambda i, q: (i >> pos[q]) & 1 if q in pos else 0)
            out[k] = out.get(k, 0.0) + float(p[idx])
        return Run(probs=out, run_s=dt)


class QPUTwinLocal:
    """QPUTwin's own engine in-process (same code as the containers), for single-thread timings."""

    name = "QPUTwin (in-process)"

    def __init__(self, threads: int = 1, backend: str = "quantinuum_helios", memory_guard: bool = False) -> None:
        sys.path.insert(0, str(ROOT / "src"))
        from qputwin_core.devices import DEVICES
        from qputwin_core.fused import set_engine_threads
        from qputwin_core.simulators import ExecutionOptions, SimulatorConfig, simulator_for
        set_engine_threads(threads)
        self._opts = ExecutionOptions
        # memory_guard=True: the worker's own policy (reject runs that cannot fit; run the fused executor
        # in place when two state buffers would not fit the available memory)
        self.sim = simulator_for(DEVICES[backend], SimulatorConfig(memory_guard=memory_guard,
                                                                   max_statevector_qubits=40))

    def sample(self, qasm: str, shots: int, seed: int = 1, options: dict[str, Any] | None = None) -> Run:
        from qputwin_core.timing import SpanTimer
        timer = SpanTimer()
        t0 = time.perf_counter()
        r = self.sim.execute(qasm, self._opts(shots=shots, seed=seed, **(options or {})), timer)
        dt = time.perf_counter() - t0
        return Run(counts=r.counts, run_s=dt, engine_s=(timer.ms("simulation") + timer.ms("sampling")) / 1e3)


# ================================================================ Qiskit Aer
class Aer:
    name = "Qiskit Aer"

    def __init__(self, threads: int = 1, method: str = "statevector", precision: str = "double") -> None:
        from qiskit_aer import AerSimulator
        self.sim = AerSimulator(method=method, max_parallel_threads=threads, max_parallel_experiments=1,
                                max_parallel_shots=1, precision=precision)
        self.method = method

    def _circuit(self, flat: Flat, noise: tuple[float, float] | None = None, measure: bool = True) -> QuantumCircuit:
        qc = QuantumCircuit(flat.n, len(flat.measured))
        for name, qs, ps in flat.ops:
            if name == "u":
                qc.u(*ps, qs[0])
            else:
                qc.cx(*qs)
        if measure:
            for k, q in enumerate(flat.measured):
                qc.measure(q, k)
        return qc

    def sample(self, flat: Flat, shots: int, seed: int = 1, noise_model=None) -> Run:
        qc = self._circuit(flat)
        t0 = time.perf_counter()
        res = self.sim.run(qc, shots=shots, seed_simulator=seed, noise_model=noise_model).result()
        return Run(counts=dict(res.get_counts()), run_s=time.perf_counter() - t0)

    def exact(self, flat: Flat) -> Run:
        qc = self._circuit(flat, measure=False)
        qc.save_statevector()
        t0 = time.perf_counter()
        sv = np.asarray(self.sim.run(qc).result().get_statevector())
        return Run(probs=marginal(np.abs(sv) ** 2, flat, big_endian=False), run_s=time.perf_counter() - t0)


# ========================================================== Amazon Braket
class Braket:
    name = "Braket local"

    def __init__(self, method: str = "braket_sv") -> None:
        from braket.devices import LocalSimulator
        self.dev = LocalSimulator(method)

    @staticmethod
    def circuit(flat: Flat, noise: tuple[float, float] | None = None):
        from braket.circuits import Circuit
        c = Circuit()
        for q in range(flat.n):                     # Braket drops untouched qubits; keep every one
            c.i(q)
        for name, qs, ps in flat.ops:
            if name == "u":
                c.u(qs[0], *ps)
                if noise:
                    c.depolarizing(qs[0], probability=0.75 * noise[0])          # Qiskit λ → Braket p
            else:
                c.cnot(*qs)
                if noise:
                    c.two_qubit_depolarizing(qs[0], qs[1], probability=15 / 16 * noise[1])
        return c

    def sample(self, flat: Flat, shots: int, seed: int = 1, noise: tuple[float, float] | None = None) -> Run:
        c = self.circuit(flat, noise)
        t0 = time.perf_counter()
        res = self.dev.run(c, shots=shots).result()
        dt = time.perf_counter() - t0
        counts: Counter[str] = Counter()
        for full, v in res.measurement_counts.items():           # big-endian over qubits 0..n-1
            counts["".join(full[flat.measured[k]] for k in reversed(range(len(flat.measured))))] += v
        return Run(counts=dict(counts), run_s=dt)

    def exact(self, flat: Flat) -> Run:
        c = self.circuit(flat)
        c.probability()
        t0 = time.perf_counter()
        p = np.asarray(self.dev.run(c, shots=0).result().values[0])
        return Run(probs=marginal(p, flat, big_endian=True), run_s=time.perf_counter() - t0)


# ===================================================================== Cirq
class Cirq:
    name = "Cirq"

    def __init__(self, density_matrix: bool = False) -> None:
        import cirq
        self.cirq = cirq
        self.density_matrix = density_matrix
        self.sim = self._make(seed=None)

    def _make(self, seed: int | None):
        cls = self.cirq.DensityMatrixSimulator if self.density_matrix else self.cirq.Simulator
        return cls(dtype=np.complex128, seed=seed)

    def circuit(self, flat: Flat, noise: tuple[float, float] | None = None, measure: bool = True):
        cirq = self.cirq
        qs = cirq.LineQubit.range(flat.n)
        ops = []
        for name, q, ps in flat.ops:
            if name == "u":
                ops.append(cirq.MatrixGate(flat.u_matrix(ps)).on(qs[q[0]]))
                if noise:
                    ops.append(cirq.depolarize(0.75 * noise[0]).on(qs[q[0]]))
            else:
                ops.append(cirq.CNOT(qs[q[0]], qs[q[1]]))
                if noise:
                    ops.append(cirq.depolarize(15 / 16 * noise[1], n_qubits=2).on(qs[q[0]], qs[q[1]]))
        if measure:
            ops.append(cirq.measure(*[qs[q] for q in flat.measured], key="m"))
        return cirq.Circuit(ops), qs

    def sample(self, flat: Flat, shots: int, seed: int = 1, noise: tuple[float, float] | None = None) -> Run:
        c, qs = self.circuit(flat, noise)
        self.sim = self._make(seed)
        t0 = time.perf_counter()
        bits = self.sim.run(c, repetitions=shots).measurements["m"]              # columns follow clbit order
        dt = time.perf_counter() - t0
        counts = Counter("".join(map(str, row[::-1])) for row in bits)
        return Run(counts=dict(counts), run_s=dt)

    def exact(self, flat: Flat) -> Run:
        c, qs = self.circuit(flat, measure=False)
        t0 = time.perf_counter()
        sv = self.sim.simulate(c, qubit_order=qs).final_state_vector
        return Run(probs=marginal(np.abs(sv) ** 2, flat, big_endian=True), run_s=time.perf_counter() - t0)


# ===================================================================== qsim
class Qsim(Cirq):
    name = "qsim"

    def __init__(self, threads: int = 1) -> None:
        import cirq
        import qsimcirq
        self.cirq = cirq
        self.options = {"t": threads, "f": 4}             # threads, gate-fusion width (qsim default 2)
        self.sim = qsimcirq.QSimSimulator(qsim_options=self.options)

    def sample(self, flat: Flat, shots: int, seed: int = 1, noise=None) -> Run:
        import qsimcirq
        c, qs = self.circuit(flat)
        self.sim = qsimcirq.QSimSimulator(qsim_options=self.options, seed=seed)
        t0 = time.perf_counter()
        bits = self.sim.run(c, repetitions=shots).measurements["m"]
        dt = time.perf_counter() - t0
        return Run(counts=dict(Counter("".join(map(str, row[::-1])) for row in bits)), run_s=dt)


# =================================================================== Qulacs
class Qulacs:
    """Thread count is fixed by ``OMP_NUM_THREADS`` at import time (set by the runner)."""

    name = "Qulacs"

    def __init__(self) -> None:
        import qulacs
        self.q = qulacs

    def circuit(self, flat: Flat):
        from qulacs.gate import DenseMatrix
        qc = self.q.QuantumCircuit(flat.n)
        for name, qs, ps in flat.ops:
            if name == "u":
                qc.add_gate(DenseMatrix(qs[0], flat.u_matrix(ps)))
            else:
                qc.add_CNOT_gate(*qs)
        return qc

    def _state(self, flat: Flat):
        st = self.q.QuantumState(flat.n)
        self.circuit(flat).update_quantum_state(st)
        return st

    def sample(self, flat: Flat, shots: int, seed: int = 1) -> Run:
        qc = self.circuit(flat)
        t0 = time.perf_counter()
        st = self.q.QuantumState(flat.n)
        qc.update_quantum_state(st)
        samples = st.sampling(shots, seed)
        dt = time.perf_counter() - t0
        counts = Counter(key_from_index(s, flat.measured, lambda i, q: (i >> q) & 1) for s in samples)
        return Run(counts=dict(counts), run_s=dt)

    def exact(self, flat: Flat) -> Run:
        t0 = time.perf_counter()
        sv = self._state(flat).get_vector()
        return Run(probs=marginal(np.abs(sv) ** 2, flat, big_endian=False), run_s=time.perf_counter() - t0)


def versions() -> dict[str, str]:
    import importlib.metadata as md
    out = {}
    for p in ("qiskit", "qiskit-aer", "amazon-braket-sdk", "amazon-braket-default-simulator", "cirq-core",
              "qsimcirq", "qulacs", "numpy"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:                    # pragma: no cover
            out[p] = "missing"
    return out


__all__ = ["C", "Flat", "flatten", "Run", "QPUTwin", "QPUTwinLocal", "Aer", "Braket", "Cirq", "Qsim", "Qulacs", "versions",
           "marginal", "BASE_URL"]
