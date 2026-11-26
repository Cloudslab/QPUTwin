"""Request and response models of the orchestrator API.

Usage: from qputwin_orchestrator.schemas import JobRequest

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RoutingSpec(BaseModel):
    """Opt-in SABRE routing (default: none — topology violations are rejected)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["none", "sabre"] = Field("sabre", description="'sabre' inserts SWAPs; 'none' rejects.")
    initial_layout: Literal["sabre", "trivial"] | list[int] = Field(
        "sabre", description="'sabre' (exact SWAP-free layout if one exists, else bidirectional layout search), "
                             "'trivial' (logical i → physical i), or an explicit list mapping logical qubit i "
                             "to a physical qubit.")
    exact_layout: bool = Field(True, description="Search for a SWAP-free embedding of the interaction graph first.")
    exact_layout_budget: int = Field(200_000, ge=0, le=10_000_000,
                                     description="Exact-layout search steps before falling back to SABRE.")
    noise_aware: bool = Field(True, description="After routing, move the circuit to the physical qubits whose "
                                                "calibrated gates and read-out cost the least (twins with "
                                                "per-qubit calibration; ignored with a pinned layout).")
    noise_aware_budget: int = Field(50_000, ge=0, le=10_000_000,
                                    description="Search steps of the calibration-aware re-layout.")
    trials: int = Field(0, ge=0, le=64, description="SABRE layout trials; 0 = automatic (16 / 8 / 4 / 2 by size). "
                                                    "Each trial is fully routed and the fewest 2Q gates wins.")
    layout_iterations: int = Field(2, ge=0, le=16, description="Forward+backward refinement rounds per trial.")
    seed: int = Field(0, ge=0, le=2 ** 63 - 1, description="Router RNG seed (independent of the shot seed).")
    lookahead: int = Field(20, ge=0, le=1000, description="Size of the extended (look-ahead) set.")
    lookahead_weight: float = Field(0.5, ge=0.0, le=10.0)
    decay: float = Field(0.001, ge=0.0, le=1.0)
    decay_reset: int = Field(5, ge=1, le=1000)
    bridge: bool = Field(False, description="Execute distance-2 CX as a bridge when the best SWAP does not help.")
    optimize: bool = Field(True, description="Commutation-aware cancellation of self-inverse 2Q pairs after routing.")


class NoiseSpec(BaseModel):
    """Calibration-driven noise (density-matrix engine). Omit or null for the ideal emulator."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["ideal", "device"] = Field("device", description="'device': use the backend's calibration.")
    scale: float = Field(1.0, ge=0.0, le=100.0, description="Multiplies every error rate, decay rate and angle.")
    depolarizing: bool = Field(True, description="Gate depolarising channels matched to published fidelities.")
    relaxation: bool = Field(True, description="Amplitude/phase damping (T1/T2) during gates.")
    idle: bool = Field(True, description="T1/T2 relaxation during idle windows of the ASAP schedule.")
    readout: bool = Field(True, description="Classical readout assignment errors.")
    coherent: bool = Field(True, description="Coherent over-rotation / residual Z(Z) phase errors.")
    crosstalk: bool = Field(True, description="Spectator ZZ crosstalk during two-qubit gates.")
    reset_errors: bool = Field(True, description="Imperfect reset (left in |1⟩ with probability 1 − F_reset).")
    correlated_readout: bool = Field(True, description="Joint readout matrices for correlated qubit groups.")
    leakage: bool = Field(True, description="Leakage out of the qubit subspace (trajectory engine).")
    quasi_static: bool = Field(True, description="Quasi-static, shot-correlated detuning (trajectory engine).")
    one_qubit_fidelity: float | None = Field(None, ge=0.5, le=1.0, description="Override for all physical 1Q gates.")
    two_qubit_fidelity: float | None = Field(None, ge=0.5, le=1.0, description="Override for all 2Q gates.")
    readout_fidelity: float | None = Field(None, ge=0.5, le=1.0, description="Override (symmetric).")
    reset_fidelity: float | None = Field(None, ge=0.5, le=1.0, description="Override the reset fidelity.")
    t1_us: float | None = Field(None, gt=0.0, description="Override T1 for every qubit (µs).")
    t2_us: float | None = Field(None, gt=0.0, description="Override T2 for every qubit (µs).")
    overrotation: float | None = Field(None, ge=-0.5, le=0.5, description="Fractional over-rotation ε of every "
                                                                          "physical gate (U → U^(1+ε)).")
    coherent_z_rad: float | None = Field(None, ge=-3.15, le=3.15, description="Residual Z / ZZ phase per gate.")
    crosstalk_zz_rad: float | None = Field(None, ge=-3.15, le=3.15, description="Spectator ZZ angle per 2Q gate.")
    leakage_rate: float | None = Field(None, ge=0.0, le=0.5, description="Leak probability per gate and qubit.")
    seepage_rate: float | None = Field(None, ge=0.0, le=1.0, description="Return probability per gate.")
    quasi_static_khz: float | None = Field(None, ge=0.0, le=1e6, description="Std. dev. of the quasi-static "
                                                                            "detuning (kHz).")
    trajectories: int | None = Field(None, ge=1, le=1_000_000, description="Trajectory count (trajectory engine); "
                                                                          "default min(shots, QPUTWIN_MAX_TRAJECTORIES).")


class JobOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compilation: Literal["native", "strict"] = Field(
        "native", description="'native': compile standard gates to the device's native set. "
                              "'strict': reject any gate that is not native to the device.")
    return_statevector: bool = Field(False, description="Return the pre-measurement statevector "
                                                        "(terminal-measurement circuits only).")
    memory: bool = Field(False, description="Return the per-shot outcome list.")
    precision: Literal["complex128", "complex64"] | None = Field(
        None, description="Override the worker's statevector precision for this job.")
    routing: RoutingSpec | None = Field(None, description="Opt-in SABRE routing; null rejects topology violations.")
    noise: NoiseSpec | None = Field(None, description="Noise model; null runs the ideal emulator.")
    engine: Literal["auto", "statevector", "density_matrix", "trajectory"] = Field(
        "auto", description="'auto': density matrix for narrow noisy circuits, trajectories for wide ones or "
                            "for time-correlated noise.")
    return_density_matrix: bool = Field(False, description="Return the final density matrix "
                                                           "(terminal-measurement circuits, ≤ 8 qubits).")


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "backend": "ionq_forte",
        "qasm": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n'
                "measure q -> c;\n",
        "shots": 1024, "seed": 42, "options": {"compilation": "native"},
    }]})

    backend: str = Field(..., description="Backend ID, e.g. 'quantinuum_helios', 'ionq_forte', "
                                          "'rigetti_ankaa_3', 'iqm_emerald', 'ibm_heron', 'ibm_nighthawk', "
                                          "'google_willow'.")
    qasm: str = Field(..., min_length=1, description="OpenQASM 2.0 or 3.x source.")
    shots: int = Field(1024, ge=1, le=1_000_000)
    seed: int | None = Field(None, ge=0, le=2 ** 63 - 1,
                             description="PRNG seed (PCG64). If omitted, one is drawn and echoed back.")
    options: JobOptions = Field(default_factory=JobOptions)
    priority: int = Field(5, ge=0, le=9, description="Asynchronous jobs only: 9 runs first. Capped by the "
                                                     "API key's maximum priority when authentication is on.")


class Breakdown(BaseModel):
    ingress_ms: float
    validation_ms: float
    routing_ms: float
    orchestrator_queue_ms: float
    dispatch_rtt_ms: float
    network_transit_ms: float
    worker_service_ms: float
    worker_queue_ms: float
    worker_parse_ms: float
    worker_route_ms: float
    worker_compile_ms: float
    simulation_ms: float
    sampling_ms: float
    worker_unattributed_ms: float
    response_assembly_ms: float
    unattributed_ms: float


class Telemetry(BaseModel):
    schema_version: str
    orchestration_overhead_ms: float = Field(..., description="total_roundtrip_ms − worker_service_ms")
    qpu_compute_time_ms: float = Field(..., description="Time inside the numpy engine (simulation + sampling)")
    total_roundtrip_ms: float = Field(..., description="Middleware entry → response ready (pre-serialisation)")
    engine_invoked: bool
    execution_mode: Literal["sync", "async"]
    queue_clock: Literal["perf_counter", "realtime"] | None = None
    attempt: int = 1
    breakdown: Breakdown
    workload: dict[str, Any]
    environment: dict[str, Any]
    clock: str
    timestamp_utc: str


class JobResult(BaseModel):
    shots: int
    seed: int
    counts: dict[str, int]
    num_clbits: int
    cregs: dict[str, list[int]]
    memory: list[str] | None = None
    statevector: list[list[float]] | None = None
    statevector_qubits: list[int] | None = None
    density_matrix: list[list[list[float]]] | None = None
    noise: dict[str, Any] | None = None


class ErrorBody(BaseModel):
    type: str
    message: str
    details: dict[str, Any] = {}


class JobResponse(BaseModel):
    job_id: str
    backend: str
    status: Literal["COMPLETED"]
    result: JobResult
    compilation: dict[str, Any]
    telemetry: Telemetry


class JobErrorResponse(BaseModel):
    job_id: str | None
    backend: str | None
    status: Literal["REJECTED", "FAILED"]
    error: ErrorBody
    telemetry: Telemetry | None


class JobLinks(BaseModel):
    self: str


class JobAccepted(BaseModel):
    """``202 Accepted`` body of an asynchronous submission."""

    job_id: str
    backend: str
    status: Literal["QUEUED"]
    execution_mode: Literal["async"]
    client: str
    priority: int
    submitted_at: str
    queue_position: int | None
    links: JobLinks


class JobStatus(BaseModel):
    """``GET /v1/jobs/{id}``: the summary while pending; the full sync-shaped body once finished."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    backend: str
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "REJECTED", "FAILED", "CANCELLED"]
    execution_mode: Literal["async"]
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    http_status: int | None = None
    queue_position: int | None = None
