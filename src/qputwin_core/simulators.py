"""Trapped-ion and superconducting QPU emulators that run OpenQASM jobs.

Usage: from qputwin_core.simulators import simulator_for, ExecutionOptions

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import itertools
import math
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qputwin_core.density import needs_branching, run_density
from qputwin_core.trajectory import batchable, run_trajectories
from qputwin_core.devices import DeviceSpec
from qputwin_core.engine import Program, prepare, run
from qputwin_core.fused import in_place_mode
from qputwin_core.errors import (
    CalibrationProvenanceError, InvalidRequestError, ResourceLimitError, TopologicalCompilationError,
    UnsupportedFeatureError,
)
from qputwin_core.ir import Circuit, Operation
from qputwin_core.noise import TRAJECTORY_ONLY, NoiseModel, NoiseOptions, NoisyInstr, build_noisy_program
from qputwin_core.qasm import BASE_PRIMITIVES, DEFAULT_MAX_OPS, load_circuit
from qputwin_core.routing import RoutingOptions, RoutingResult, SabreRouter, calibrated_relayout
from qputwin_core.scheduling import Schedule, asap_schedule, duration_report
from qputwin_core.synthesis import NativeCircuit, NativeCompiler
from qputwin_core.timing import SpanTimer

__all__ = ["SimulatorConfig", "ExecutionOptions", "ExecutionResult", "QPUSimulator",
           "TrappedIonSimulator", "SuperconductingSimulator", "available_memory_bytes",
           "simulator_for"]

_DTYPES = {"complex128": np.complex128, "complex64": np.complex64}
_ENGINES = ("auto", "statevector", "density_matrix", "trajectory")
_MiB = 1 << 20


# ============================================================ configuration
def _env_int(name: str, default: int | None) -> int | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def _in_place_setting(value: str) -> str:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "always"):
        return "always"
    if v in ("0", "false", "no", "never"):
        return "never"
    if v in ("", "auto"):
        return "auto"
    raise ValueError(f"QPUTWIN_ENGINE_IN_PLACE must be auto, always or never, got {value!r}")


@dataclass(slots=True)
class SimulatorConfig:
    max_sim_qubits: int | None = None          # None => device's physical qubit count
    precision: str = "complex128"
    memory_guard: bool = True
    memory_budget_bytes: int | None = None     # explicit override of the detected budget
    memory_reserve_bytes: int = 256 * _MiB
    max_shots: int = 1_000_000
    max_memory_shots: int = 100_000
    max_statevector_qubits: int = 16
    max_density_matrix_qubits: int = 8
    density_auto_max_qubits: int = 12          # engine="auto": density matrix at most this wide (see plan())
    max_trajectories: int = 2000
    require_published_calibration: bool = False
    max_ops: int = DEFAULT_MAX_OPS
    engine_in_place: str = "auto"              # fused executor: "auto" (in place only when 2 buffers won't fit),
                                               # "always" or "never"

    @classmethod
    def from_env(cls) -> SimulatorConfig:
        precision = os.environ.get("QPUTWIN_PRECISION", "complex128").strip() or "complex128"
        if precision not in _DTYPES:
            raise ValueError(f"QPUTWIN_PRECISION must be one of {sorted(_DTYPES)}, got {precision!r}")
        return cls(
            max_sim_qubits=_env_int("QPUTWIN_MAX_SIM_QUBITS", None),
            precision=precision,
            memory_guard=os.environ.get("QPUTWIN_MEMORY_GUARD", "1").strip().lower() not in ("0", "false", "no"),
            memory_budget_bytes=_env_int("QPUTWIN_MEMORY_BUDGET_BYTES", None),
            memory_reserve_bytes=_env_int("QPUTWIN_MEMORY_RESERVE_BYTES", 256 * _MiB) or 0,
            max_shots=_env_int("QPUTWIN_MAX_SHOTS", 1_000_000) or 1_000_000,
            max_memory_shots=_env_int("QPUTWIN_MAX_MEMORY_SHOTS", 100_000) or 100_000,
            max_statevector_qubits=_env_int("QPUTWIN_MAX_STATEVECTOR_QUBITS", 16) or 16,
            max_density_matrix_qubits=_env_int("QPUTWIN_MAX_DENSITY_MATRIX_QUBITS", 8) or 8,
            density_auto_max_qubits=_env_int("QPUTWIN_DENSITY_AUTO_MAX_QUBITS", 12) or 12,
            max_trajectories=_env_int("QPUTWIN_MAX_TRAJECTORIES", 2000) or 2000,
            require_published_calibration=os.environ.get("QPUTWIN_REQUIRE_PUBLISHED_CALIBRATION", "0").strip().lower()
            in ("1", "true", "yes"),
            max_ops=_env_int("QPUTWIN_MAX_OPS", DEFAULT_MAX_OPS) or DEFAULT_MAX_OPS,
            engine_in_place=_in_place_setting(os.environ.get("QPUTWIN_ENGINE_IN_PLACE", "auto")),
        )


def available_memory_bytes() -> int | None:
    """Memory currently available to this process: min(cgroup v2 headroom, MemAvailable)."""
    candidates: list[int] = []
    cg = Path("/sys/fs/cgroup")
    try:
        limit = (cg / "memory.max").read_text().strip()
        if limit != "max":
            current = int((cg / "memory.current").read_text().strip())
            candidates.append(max(0, int(limit) - current))
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    return min(candidates) if candidates else None


@dataclass(slots=True)
class ExecutionOptions:
    shots: int = 1024
    seed: int | None = None
    compilation: str = "native"
    return_statevector: bool = False
    memory: bool = False
    precision: str | None = None
    routing: RoutingOptions | dict[str, Any] | None = None     # None = reject topology violations
    noise: NoiseOptions | dict[str, Any] | None = None         # None = ideal emulator
    engine: str = "auto"                                       # auto | statevector | density_matrix
    return_density_matrix: bool = False


@dataclass(slots=True)
class ExecutionResult:
    shots: int
    seed: int
    counts: dict[str, int]
    memory: list[str] | None
    statevector: list[list[float]] | None
    statevector_qubits: list[int] | None
    num_clbits: int
    cregs: dict[str, list[int]]
    compilation: dict[str, Any]
    workload: dict[str, Any]
    engine_invoked: bool = True
    density_matrix: list[Any] | None = None
    noise: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": {
                "shots": self.shots, "seed": self.seed, "counts": self.counts,
                "num_clbits": self.num_clbits, "cregs": self.cregs, "memory": self.memory,
                "statevector": self.statevector, "statevector_qubits": self.statevector_qubits,
                "density_matrix": self.density_matrix, "noise": self.noise,
            },
            "compilation": self.compilation,
            "workload": self.workload,
            "engine_invoked": self.engine_invoked,
        }


# ================================================================ simulators
@dataclass(slots=True)
class _Plan:
    circuit: Circuit
    native: NativeCircuit
    program: Program
    dtype: Any
    required_bytes: int
    budget_bytes: int | None
    max_sim_qubits: int
    routing: RoutingResult | None = None
    routing_stats: dict[str, Any] | None = None
    schedule: Schedule | None = None
    engine: str = "statevector"
    noise_model: NoiseModel | None = None
    instrs: list[NoisyInstr] | None = None
    trajectories: int = 0
    in_place: bool = False                     # fused executor in single-buffer mode (memory-bound run)

    @property
    def density(self) -> bool:
        return self.engine == "density_matrix"


class QPUSimulator(ABC):
    """Base class: device-agnostic execution pipeline."""

    technology: str = ""

    def __init__(self, spec: DeviceSpec, config: SimulatorConfig | None = None) -> None:
        self.spec = spec
        self.config = config or SimulatorConfig()
        cap = self.config.max_sim_qubits
        self.max_sim_qubits = spec.n_physical if cap is None else max(0, min(cap, spec.n_physical))
        self.compiler = NativeCompiler(spec)
        self.primitives = BASE_PRIMITIVES | frozenset(spec.native_gates)

    # --------------------------------------------------------------- topology
    @abstractmethod
    def check_connectivity(self, op: Operation) -> None:
        """Raise TopologicalCompilationError if ``op`` addresses uncoupled qubits."""

    def check_topology(self, circuit: Circuit) -> None:
        disabled = set(self.spec.coupling.disabled_qubits)
        for op in circuit.ops:
            if disabled and op.name != "barrier":
                bad = [q for q in op.qubits if q in disabled]
                if bad:
                    self._reject_disabled(op, bad[0])
            if op.is_gate and len(op.qubits) >= 2:
                self.check_connectivity(op)

    def _reject_disabled(self, op: Operation, q: int) -> None:
        origin = op.origin
        line = origin.line if origin else None
        text = origin.text if origin else f"{op.name} {q}"
        raise TopologicalCompilationError(
            f"'{text}' (line {line}) addresses physical qubit {q}, which is disabled in the coupling map "
            f"of {self.spec.id} ({self.spec.coupling.provenance or 'calibration snapshot'})",
            details={"device": self.spec.id, "gate": origin.gate if origin else op.name, "primitive": op.name,
                     "qubits": [q], "line": line, "statement": text, "reason": "disabled_qubit",
                     "disabled_qubits": list(self.spec.coupling.disabled_qubits)})

    # ---------------------------------------------------------------- routing
    def route(self, circuit: Circuit, options: RoutingOptions) -> tuple[RoutingResult | None, dict[str, Any]]:
        """Run the SABRE router (skipped on an all-to-all device with no disabled qubits)."""
        cmap = self.spec.coupling
        if cmap.all_to_all and not cmap.disabled_qubits:
            touched = circuit.touched_qubits()
            ident = [[q, q] for q in touched]
            return None, {"method": options.method, "skipped": "all_to_all", "swaps_inserted": 0,
                          "added_cx": 0, "initial_layout": ident, "final_layout": ident}
        res = SabreRouter(cmap, options).run(circuit)
        cal = self.spec.calibration
        pinned = isinstance(options.initial_layout, tuple) or options.initial_layout == "trivial"
        if options.noise_aware and not pinned and cal is not None and (cal.qubits or cal.couplers):
            res = calibrated_relayout(res, cmap, budget=options.noise_aware_budget, **self._layout_costs())
        return res, res.stats

    def _layout_costs(self) -> dict[str, Any]:
        """Negative log-fidelities that score a placement: entangler per coupler, pulse per qubit, read-out."""
        cal = self.spec.calibration
        assert cal is not None
        pulse = next((g for g in self.spec.native_1q if g not in self.spec.virtual_gates), None)
        two = self.spec.entangler

        def nlog(f: float | None) -> float:
            return 50.0 if f is None or f <= 0.0 else -math.log(min(1.0, f))

        def edge_cost(p: int, q: int) -> float:
            return nlog(cal.gate_fidelity(two, (p, q)))

        def qubit_cost(p: int) -> float:
            return nlog(cal.gate_fidelity(pulse, (p,))) if pulse else 0.0

        def readout_cost(p: int) -> float:
            p01, p10 = cal.readout_errors(p)
            return nlog(1.0 - 0.5 * (p01 + p10))

        return {"edge_cost": edge_cost, "qubit_cost": qubit_cost, "readout_cost": readout_cost}

    # -------------------------------------------------------------- pipeline
    def parse(self, qasm: str) -> Circuit:
        return load_circuit(qasm, primitives=self.primitives, max_ops=self.config.max_ops)

    def check_resources(self, circuit: Circuit) -> None:
        n_phys = self.spec.n_physical
        if circuit.num_qubits > n_phys:
            raise ResourceLimitError(
                f"circuit declares {circuit.num_qubits} qubits but {self.spec.id} has {n_phys} physical qubits",
                details={"declared_qubits": circuit.num_qubits, "physical_qubits": n_phys, "device": self.spec.id})

    def compile(self, circuit: Circuit, mode: str = "native") -> NativeCircuit:
        self.check_resources(circuit)
        self.check_topology(circuit)
        return self.compiler.compile(circuit, mode)

    def memory_plan(self, program: Program, dtype: Any, *, density: bool = False, fused: bool = False,
                    want_vector: bool = False, probs_in_state: bool = False) -> tuple[int, int | None, bool]:
        """Estimated peak bytes, the detected budget (None = unknown) and whether to run in place.

        The trajectory engine needs what the statevector engine needs (``density=False``).
        ``fused`` says the run uses the fused executor, which can switch to its single-buffer
        (in-place) mode when the faster two-buffer mode does not fit the budget. ``probs_in_state``
        says the run frees its spare buffer and writes the probabilities into the state's own memory
        (the fused statevector path), so they cost no extra 16 B per basis state.
        """
        n = program.n_sim
        itemsize = np.dtype(dtype).itemsize
        state = (1 << (2 * n if density else n)) * itemsize
        # gate kernels ≈ 2×state (the density engine's kernels and partial traces stay within
        # that too); probability vector + temporary = 16 B per basis state; shot branching
        # keeps up to one extra state per pending branch.
        if probs_in_state:
            # two buffers, a returned copy of the statevector, a quarter state of slack for the marginal
            # over a subset of the measured qubits, and the per-thread chunk buffers (measured peak:
            # 2.07× the state at 25 qubits, 2.5× at 22 where the fixed part weighs more)
            required = 2 * state + (state if want_vector else 0) + state // 4 + (128 << 20)
        else:
            required = 2 * state + (1 << n) * 16 + min(program.branch_points, 64) * state
        budget = None
        if self.config.memory_guard:
            budget = self.config.memory_budget_bytes
            if budget is None:
                avail = available_memory_bytes()
                budget = None if avail is None else max(0, avail - self.config.memory_reserve_bytes)
        mode = self.config.engine_in_place
        if fused and mode != "never" and (mode == "always" or (budget is not None and required > budget)):
            # in place: one buffer, plus a copy of the (at most half-size) state while it grows, or a
            # second buffer for the final transpose when the statevector itself is returned; the
            # probabilities are written into the state's own memory
            lean = state + (state if want_vector else state // 2) + ((1 << n) * 16 if density else 0)
            if mode == "always" or lean < required:
                return lean, budget, True
        return required, budget, False

    def plan(self, qasm: str, options: ExecutionOptions, timer: SpanTimer) -> _Plan:
        if not 1 <= options.shots <= self.config.max_shots:
            raise InvalidRequestError(f"shots must be in [1, {self.config.max_shots}]",
                                      details={"shots": options.shots, "max_shots": self.config.max_shots})
        if options.memory and options.shots > self.config.max_memory_shots:
            raise InvalidRequestError(f"per-shot memory is limited to {self.config.max_memory_shots} shots",
                                      details={"max_memory_shots": self.config.max_memory_shots})
        precision = options.precision or self.config.precision
        if precision not in _DTYPES:
            raise InvalidRequestError(f"precision must be one of {sorted(_DTYPES)}")
        dtype = _DTYPES[precision]
        routing = options.routing
        if isinstance(routing, dict):
            routing = RoutingOptions.from_dict(routing)
        noise = options.noise
        if isinstance(noise, dict):
            noise = NoiseOptions.from_dict(noise)
        engine = options.engine
        if engine not in _ENGINES:
            raise InvalidRequestError(f"engine must be one of {list(_ENGINES)}", details={"engine": engine})
        if noise is not None and engine == "statevector":
            raise InvalidRequestError("noisy execution needs the density-matrix or trajectory engine; use "
                                      "engine 'auto', 'density_matrix' or 'trajectory'")
        if noise is None and engine == "trajectory":
            raise InvalidRequestError("the trajectory engine samples a noise model; add options.noise")
        if options.return_density_matrix and engine in ("statevector", "trajectory"):
            raise InvalidRequestError("return_density_matrix needs the density-matrix engine")
        if options.return_statevector and (noise is not None or engine == "density_matrix"):
            raise InvalidRequestError("return_statevector is unavailable on noisy / density-matrix runs; "
                                      "use return_density_matrix")
        cal = self.spec.calibration
        if noise is not None and cal is None:
            raise InvalidRequestError(f"{self.spec.id} has no calibration data, so it has no noise model")
        if noise is not None and noise.trajectories is not None and noise.trajectories > self.config.max_trajectories:
            raise InvalidRequestError(f"options.noise.trajectories is limited to {self.config.max_trajectories} "
                                      "(QPUTWIN_MAX_TRAJECTORIES)", details={"max_trajectories": self.config.max_trajectories})
        if routing is not None and options.compilation == "strict":
            raise InvalidRequestError("routing inserts SWAPs, which are not native gates; use "
                                      "compilation='native' with routing")
        with timer.span("parse"):
            circuit = self.parse(qasm)
        self.check_resources(circuit)
        routed: RoutingResult | None = None
        routing_stats: dict[str, Any] | None = None
        compiled_from = circuit
        if routing is not None:
            with timer.span("route"):
                routed, routing_stats = self.route(circuit, routing)
            if routed is not None:
                compiled_from = routed.circuit
        with timer.span("compile"):
            native = self.compile(compiled_from, options.compilation)
            program = prepare(native)
            schedule = asap_schedule(native.ops, cal) if cal is not None else None
        n = program.n_sim
        if n > self.max_sim_qubits:
            raise ResourceLimitError(
                f"circuit needs {n} simulated qubits; this {self.spec.id} emulator is capped at "
                f"{self.max_sim_qubits} (QPUTWIN_MAX_SIM_QUBITS)",
                details={"simulated_qubits": n, "max_sim_qubits": self.max_sim_qubits,
                         "physical_qubits": self.spec.n_physical, "device": self.spec.id})
        model = instrs = None
        if noise is not None or engine == "density_matrix" or options.return_density_matrix:
            with timer.span("compile"):
                model = NoiseModel(cal, noise, self.spec.coupling) if noise is not None and cal is not None else None
                instrs = build_noisy_program(native, program.qmap, model,
                                             schedule if noise is not None and (noise.idle or noise.quasi_static)
                                             else None)
                if model is not None:                     # resolve readout parameters (and their provenance) now
                    measured = sorted({op.qubits[0] for op in native.ops if op.name == "measure"})
                    for q in measured:
                        model.readout_errors(q)
                    model.readout_groups(measured)
        trajectory_only = sorted(model.features & TRAJECTORY_ONLY) if model is not None else []
        # ---- resolve the engine
        if engine == "auto":
            if options.return_density_matrix or (noise is None and instrs is not None):
                engine = "density_matrix"
            elif noise is not None:
                dm_required, dm_budget, _ = self.memory_plan(
                    program, dtype, density=True, fused=not needs_branching(program, instrs))
                dm_fits = dm_budget is None or dm_required <= dm_budget
                # cost model (measured): a fused density-matrix step costs ~4^n, a batched trajectory
                # step ~T·2^n, so the density matrix wins while 2^n ≤ 4T. It is also exact for any shot
                # count, so it is kept whenever the trajectories would not be one per shot.
                t = noise.trajectories or min(options.shots, self.config.max_trajectories)
                cheaper = (1 << n) <= 4 * t or t < options.shots
                engine = ("density_matrix" if not trajectory_only and n <= self.config.density_auto_max_qubits
                          and cheaper and dm_fits else "trajectory")
            else:
                engine = "statevector"
        if engine == "density_matrix" and trajectory_only:
            raise InvalidRequestError(
                f"noise term(s) {trajectory_only} are time-correlated or leave the qubit space and need the "
                "trajectory engine (engine 'auto' or 'trajectory'), or disable them",
                details={"trajectory_only_features": trajectory_only})
        if model is not None and self.config.require_published_calibration:
            unpublished = sorted(p for p, c in model.used.items() if c != "published")
            if unpublished:
                raise CalibrationProvenanceError(
                    f"strict provenance mode: this run would use {len(unpublished)} calibration value(s) that are "
                    f"not tagged as published ({', '.join(unpublished[:6])}{'…' if len(unpublished) > 6 else ''}); "
                    "load a vendor calibration snapshot (QPUTWIN_CALIBRATION) or override them in options.noise",
                    details={"unpublished_parameters": unpublished, "device": self.spec.id})
        density = engine == "density_matrix"
        fusable = (not program.branching) if engine == "statevector" else (
            instrs is not None and (not needs_branching(program, instrs) if density else batchable(program, instrs)))
        required, budget, in_place = self.memory_plan(
            program, dtype, density=density, fused=fusable,
            want_vector=options.return_statevector or options.return_density_matrix,
            probs_in_state=engine == "statevector" and fusable)
        if budget is not None and required > budget:
            engine_name = "density-matrix" if density else engine.replace("_", " ")
            per_unit = required / (1 << (2 * n if density else n))
            feasible = int(math.floor(math.log2(budget / per_unit) / (2 if density else 1))) \
                if budget > per_unit else 0
            raise ResourceLimitError(
                f"a dense {n}-qubit {precision} {engine_name} run needs ~{_fmt_bytes(required)} but only "
                f"{_fmt_bytes(budget)} is available to this worker; at most {feasible} simulated qubits "
                f"fit ({'use complex64 precision, ' if precision != 'complex64' else ''}raise the container memory "
                "limit, or touch fewer qubits)",
                details={"simulated_qubits": n, "required_bytes": required, "available_bytes": budget,
                         "max_feasible_qubits": feasible, "precision": precision, "device": self.spec.id,
                         "engine": engine})
        if options.return_statevector:
            if program.branching:
                raise UnsupportedFeatureError(
                    "return_statevector requires a circuit whose measurements are all terminal "
                    "(no mid-circuit measurement, reset or classical control)")
            if n > self.config.max_statevector_qubits:
                raise InvalidRequestError(
                    f"return_statevector is limited to {self.config.max_statevector_qubits} simulated qubits",
                    details={"simulated_qubits": n,
                             "max_statevector_qubits": self.config.max_statevector_qubits})
        if options.return_density_matrix:
            assert instrs is not None
            if needs_branching(program, instrs):
                raise UnsupportedFeatureError(
                    "return_density_matrix requires a circuit whose measurements are all terminal "
                    "(no mid-circuit measurement or classical control)")
            if n > self.config.max_density_matrix_qubits:
                raise InvalidRequestError(
                    f"return_density_matrix is limited to {self.config.max_density_matrix_qubits} "
                    "simulated qubits",
                    details={"simulated_qubits": n,
                             "max_density_matrix_qubits": self.config.max_density_matrix_qubits})
        trajectories = 0
        if engine == "trajectory":
            assert noise is not None
            trajectories = noise.trajectories or min(options.shots, self.config.max_trajectories)
        return _Plan(circuit, native, program, dtype, required, budget, self.max_sim_qubits,
                     routed, routing_stats, schedule, engine, model, instrs, trajectories, in_place)

    def execute(self, qasm: str, options: ExecutionOptions | None = None,
                timer: SpanTimer | None = None) -> ExecutionResult:
        """Run ``qasm`` end-to-end. Spans recorded in ``timer``: parse, compile, simulation, sampling."""
        options = options or ExecutionOptions()
        timer = timer if timer is not None else SpanTimer()
        plan = self.plan(qasm, options, timer)
        seed = options.seed if options.seed is not None else secrets.randbits(63)
        rng = np.random.default_rng(seed)
        sv = dm = noise_report = None
        model = plan.noise_model
        readout = model.readout_errors if model is not None else (lambda q: (0.0, 0.0))
        groups = model.readout_groups if model is not None else None
        with in_place_mode(plan.in_place):
            if plan.engine == "density_matrix":
                assert plan.instrs is not None
                out, diag = run_density(plan.program, plan.instrs, shots=options.shots, rng=rng, readout=readout,
                                        groups=groups, dtype=plan.dtype, timer=timer,
                                        want_density_matrix=options.return_density_matrix,
                                        want_memory=options.memory)
                if out.statevector is not None:
                    dm = np.stack([out.statevector.real, out.statevector.imag], axis=-1).astype(float).tolist()
            elif plan.engine == "trajectory":
                assert plan.instrs is not None and model is not None
                detuning = {plan.program.qmap[q]: model.quasi_static_khz(q) for q in plan.program.active}
                out, diag = run_trajectories(plan.program, plan.instrs, shots=options.shots,
                                             trajectories=plan.trajectories, rng=rng, readout=readout,
                                             groups=groups, seepage=model.seepage(), detuning_khz=detuning,
                                             dtype=plan.dtype, timer=timer, want_memory=options.memory,
                                             memory_budget=plan.budget_bytes)
            else:
                diag = {}
                out = run(plan.program, shots=options.shots, rng=rng, dtype=plan.dtype, timer=timer,
                          want_statevector=options.return_statevector, want_memory=options.memory)
                if out.statevector is not None:
                    sv = np.stack([out.statevector.real, out.statevector.imag], axis=1).astype(float).tolist()
        if plan.engine != "statevector":
            measured = sorted({op.qubits[0] for op in plan.native.ops if op.name == "measure"})
            noise_report = model.report(plan.native, measured) if model is not None else {"model": "ideal"}
            noise_report = {**noise_report, "engine": plan.engine, **diag}
        native = plan.native
        circuit = plan.circuit
        n = plan.program.n_sim
        compiled_from = plan.routing.circuit if plan.routing is not None else circuit
        itemsize = np.dtype(plan.dtype).itemsize
        compilation = {
            "mode": native.mode,
            "qasm_version": circuit.version,
            "declared_qubits": circuit.num_qubits,
            "physical_qubits_used": compiled_from.touched_qubits(),
            "simulated_qubits": n,
            "idle_measured_qubits": sorted(plan.program.idle_measured),
            **native.stats,
            "routing": plan.routing_stats,
            "hardware_time_estimate": self._time_estimate(plan, options.shots),
        }
        workload = {
            "simulated_qubits": n,
            "engine": plan.engine,
            "precision": np.dtype(plan.dtype).name,
            "statevector_bytes": (1 << n) * itemsize,
            "state_bytes": (1 << (2 * n if plan.density else n)) * itemsize,
            "estimated_peak_bytes": plan.required_bytes,
            "in_place": plan.in_place,
            "memory_budget_bytes": plan.budget_bytes,
            "max_sim_qubits": plan.max_sim_qubits,
            "shots": options.shots,
            "execution_path": out.execution_path,
            "branches": out.branches,
            "native_op_count": native.stats["native_op_count"],
            "native_two_qubit_count": native.stats["native_two_qubit_count"],
            "noisy": plan.noise_model is not None,
            "routed": plan.routing is not None,
        }
        return ExecutionResult(
            shots=options.shots, seed=seed, counts=out.counts, memory=out.memory, statevector=sv,
            statevector_qubits=plan.program.active if (sv is not None or dm is not None) else None,
            num_clbits=circuit.num_clbits,
            cregs={k: [o, s] for k, (o, s) in circuit.cregs.items()},
            compilation=compilation, workload=workload, density_matrix=dm, noise=noise_report)

    def _time_estimate(self, plan: _Plan, shots: int) -> dict[str, Any] | None:
        cal = self.spec.calibration
        if plan.schedule is None or cal is None:
            return None
        rep = duration_report(plan.schedule, cal, shots)
        paths = set()
        for op in plan.native.ops:
            if op.name == "measure":
                paths.add("readout.duration_ns")
            elif op.name == "reset":
                paths.add("reset.duration_ns")
            elif op.is_gate and not op.virtual:
                paths.add(f"gates.{op.name}.duration_ns")
        if cal.max_parallel_one_qubit is not None or cal.max_parallel_two_qubit is not None:
            paths.add("parallelism")
        if rep["includes_qccd_transport"]:
            paths.add("qccd")
        rep["assumed_parameters"] = sorted(p for p in paths if cal.provenance_class(p) != "published")
        return rep

    def describe(self, *, include_edges: bool = True) -> dict[str, Any]:
        d = self.spec.to_dict(include_edges=include_edges)
        d["emulator"] = {
            "simulator_class": type(self).__name__,
            "max_sim_qubits": self.max_sim_qubits,
            "precision": self.config.precision,
            "memory_guard": self.config.memory_guard,
            "available_memory_bytes": available_memory_bytes(),
            "max_shots": self.config.max_shots,
            "max_statevector_qubits": self.config.max_statevector_qubits,
            "max_density_matrix_qubits": self.config.max_density_matrix_qubits,
            "engines": ["statevector", "density_matrix", "trajectory"],
            "max_trajectories": self.config.max_trajectories,
            "density_auto_max_qubits": self.config.density_auto_max_qubits,
            "require_published_calibration": self.config.require_published_calibration,
            "noise_model": self.spec.calibration is not None,
            "routing": ["none", "sabre"],
        }
        return d


class TrappedIonSimulator(QPUSimulator):
    """All-to-all (``K_n``) connectivity: any pair of distinct qubits may interact."""

    technology = "trapped_ion"

    def check_connectivity(self, op: Operation) -> None:
        n = self.spec.n_physical
        for q in op.qubits:
            if not 0 <= q < n:   # pragma: no cover - check_resources catches this first
                raise ResourceLimitError(f"qubit {q} does not exist on {self.spec.id}")


class SuperconductingSimulator(QPUSimulator):
    """Nearest-neighbour connectivity enforced through the device adjacency matrix."""

    technology = "superconducting"

    def __init__(self, spec: DeviceSpec, config: SimulatorConfig | None = None) -> None:
        super().__init__(spec, config)
        self.adjacency = spec.coupling.adjacency          # symmetric boolean n×n numpy array

    def check_connectivity(self, op: Operation) -> None:
        a = self.adjacency
        for p, q in itertools.combinations(op.qubits, 2):
            if not a[p, q]:
                origin = op.origin
                gate = origin.gate if origin else op.name
                line = origin.line if origin else None
                text = origin.text if origin else f"{op.name} {p},{q}"
                cm = self.spec.coupling
                if cm.kind == "grid":            # e.g. "grid 6x9"
                    shape = f"{cm.kind} {cm.layout['rows']}x{cm.layout['cols']}"
                else:                            # e.g. "heavy-hex, 156 qubits, 175 couplers"
                    shape = f"{cm.layout.get('kind', cm.kind).replace('_', '-')}, {cm.n_qubits} qubits, {cm.n_edges} couplers"
                raise TopologicalCompilationError(
                    f"'{text}' (line {line}) requires a two-qubit interaction between physical qubits "
                    f"{p} and {q}, which are not coupled on {self.spec.id} "
                    f"({shape}, nearest-neighbour only; no routing)",
                    details={
                        "device": self.spec.id, "gate": gate, "primitive": op.name,
                        "qubits": [p, q], "line": line, "statement": text,
                        "neighbors": {str(p): self.spec.coupling.neighbors(p),
                                      str(q): self.spec.coupling.neighbors(q)},
                        "adjacency_entry": int(a[p, q]),
                    })


def simulator_for(spec: DeviceSpec, config: SimulatorConfig | None = None) -> QPUSimulator:
    cls = TrappedIonSimulator if spec.technology == "trapped_ion" else SuperconductingSimulator
    return cls(spec, config)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if n < 1024 or unit == "EiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} EiB"  # pragma: no cover
