# QPUTwin: Calibration-Faithful Digital Twins of Trapped-Ion and Superconducting Quantum Processors

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![OpenQASM](https://img.shields.io/badge/OpenQASM-2.0%20%7C%203.x-6f42c1)
![Tests](https://img.shields.io/badge/tests-616%20unit%20%2B%2047%20live-brightgreen)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

**QPUTwin** is an open-source **quantum digital twin** and **QPU emulator**. It reproduces seven real quantum processors behind one OpenQASM API, modelling each one's native gate set, qubit connectivity, calibrated noise and gate timing. Every processor runs as its own containerised twin; a FastAPI orchestrator routes OpenQASM 2.0/3.x circuits to them and returns results with exact, additive latency telemetry.

| Digital twin | Processor | Technology | Qubits | Connectivity |
|---|---|---|---|---|
| `quantinuum_helios` | Quantinuum Helios | Trapped ion (QCCD) | 98 | All-to-all |
| `ionq_forte` | IonQ Forte | Trapped ion | 36 | All-to-all |
| `rigetti_ankaa_3` | Rigetti Ankaa-3 | Superconducting | 84 | 7 × 12 square lattice |
| `iqm_emerald` | IQM Emerald | Superconducting | 54 | 6 × 9 square lattice |
| `ibm_heron` | IBM Heron r3 (`ibm_boston`) | Superconducting | 156 | Heavy-hex (IBM's map) |
| `ibm_nighthawk` | IBM Nighthawk (`ibm_miami`) | Superconducting | 120 | Square lattice (IBM's map) |
| `google_willow` | Google Willow (`willow_pink`) | Superconducting | 105 | Square grid (Google's map) |

The IBM and Google twins are built from the vendors' own published calibration snapshots, with every qubit and coupler.

## Highlights

* **Calibration-faithful noise.** Each gate's error channel is solved to reproduce its published fidelity exactly, and every calibration value is tagged `published` or `assumed`.
* **Hardware constraints enforced.** Phase-exact native compilation; topology violations are rejected, or routed with an opt-in exact-layout-first, calibration-aware SABRE router.
* **Three engines, built from scratch.** Fused statevector, density-matrix and batched quantum-trajectory engines in `numpy`, chosen per job by a cost model, with no vendor or third-party quantum SDK.
* **A real service.** Durable asynchronous jobs with leases, fair sharing and priorities, worker auto-discovery, API keys, Prometheus metrics and an HA gateway.

![QPUTwin system architecture](https://koustavmondal.com/qputwin-docs/assets/diagrams/architecture.svg)

## Quick start

Prerequisites: Docker with Compose v2, and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Cloudslab/QPUTwin.git qputwin && cd qputwin
docker compose up -d --build                                   # orchestrator + seven twins
curl -s localhost:8000/v1/backends | python3 -m json.tool      # list the emulated QPUs

curl -s localhost:8000/v1/jobs -H 'content-type: application/json' -d '{
  "backend": "ionq_forte",
  "qasm": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;",
  "shots": 1000, "seed": 7
}' | python3 -m json.tool
```

Interactive API documentation is served at <http://localhost:8000/docs>. Run the tests with `uv run pytest`.

## Results at a glance

Measured on a 16-vCPU, 62.8 GiB server against Qiskit Aer, the Amazon Braket local simulators, Cirq, qsim and Qulacs ([details](https://koustavmondal.com/qputwin-docs/research/simulator-comparison/)):

* **Vendors' own workflows.** For the same OpenQASM program, a QPUTwin job was faster than IBM's (Qiskit transpile + Aer `from_backend`) and Google's (Cirq + qsim with Google's noise model) emulation workflows in 107 of 108 cases, by a geometric mean of 7.1× (Heron r3), 10.2× (Nighthawk) and 4.8× (Willow).
* **Noisy density matrix.** From 10 qubits, 3.2–15× faster than Aer on one thread; Aer is level or ahead on smaller jobs.
* **Ideal statevector.** Fastest on shallow GHZ circuits at 24 qubits (3.9× qsim on one thread); on deep circuits qsim leads, by 1.9–2.2× at matched single precision on one thread.
* **Vendors' own noise models.** The IBM and Google twins reproduce them to a median |Δ| of 0.0011–0.0014.
* **Saturation.** With 21 workers on all 16 vCPUs for ten minutes, all 90 326 light and 5 756 heavy jobs completed, oversized jobs were refused, and no container ran out of memory.

## Documentation

Everything else (installation, the full API, the physics, operations and the benchmark results) is in the documentation site: **<https://koustavmondal.com/qputwin-docs/>**.

| Start here | |
|---|---|
| [Getting started](https://koustavmondal.com/qputwin-docs/getting-started/) | installation, quick start, tutorial |
| [Concepts](https://koustavmondal.com/qputwin-docs/concepts/) | digital twins, architecture, execution pipeline, devices |
| [User guide](https://koustavmondal.com/qputwin-docs/guide/) | jobs, OpenQASM, compilation, noise, engines, routing, timing, calibration, telemetry |
| [Reference](https://koustavmondal.com/qputwin-docs/reference/) | HTTP API, parameters, responses, errors, configuration |
| [Operations](https://koustavmondal.com/qputwin-docs/operations/) | deployment, scaling, security, monitoring |
| [Research](https://koustavmondal.com/qputwin-docs/research/) | benchmarks, simulator comparison, validation, reproducibility, limitations |

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Koustav Kumar Mondal · koustav@koustavmondal.com
