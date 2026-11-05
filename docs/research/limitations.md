---
title: Limitations
description: Documented limitations of QPUTwin — vendor data gaps, modelling boundaries of the noise model, and deployment limits.
---

# Limitations

## Data that has to come from the vendors

* **Superconducting coupling maps.** The packaged maps are synthetic lattices matching published qubit counts and connectivity class. A dated, calibrated map is one command away (`scripts/import_braket_device.py`), but it needs an AWS export.
* **Calibration values.** Many packaged values are `assumed`, mainly gate durations and superconducting readout fidelity. Every job lists the assumed values it used; strict provenance mode refuses to run on them.
* **Per-shot overhead and feed-forward latency** are not published for any of the four devices, so job-time estimates exclude them unless a calibration supplies them.

## Modelling boundaries

* **Gate-level model.** There is no pulse-level simulation or control-electronics model.
* **Leakage** is phenomenological: leak, inert, partner depolarisation, seepage. An exact model would need a qutrit engine.
* **Non-Markovian noise** is limited to quasi-static detuning; 1/f spectra with finite correlation times are not modelled.
* **Crosstalk** covers spectator ZZ during two-qubit gates; static always-on ZZ during idle windows is not modelled.
* **Correlated readout** applies to terminal measurements; mid-circuit reads use per-qubit confusion.
* **QCCD timing.** Transport and cooling are a fixed cost per two-qubit gate; trap zones and ion-reordering paths are not modelled.
* **Trajectory sampling** with fewer trajectories than shots samples the empirical mixture of the drawn trajectories, and is reported as such.
* **ESP** is a first-order predictor that ignores idle decoherence; use the simulated success instead.

## Deployment

* The SQLite job store coordinates instances on **one host**. Multi-host orchestrators need a networked store behind the `JobStore` interface.
* `/v1/telemetry` ring buffers are per instance; the JSONL dataset is shared and complete.
* Delivery is at-least-once, not exactly-once (retries are identical because seeds are fixed).
* There is no built-in rate limiting or TLS; use an edge proxy.
* Large statevectors are memory-bandwidth bound; replicas on one host give limited speed-up for 20+ qubit jobs.
* **Server-side telemetry starts at the orchestrator's middleware.** Time spent in the kernel accept queue or waiting for a saturated event loop is invisible to `total_roundtrip_ms`, or shows up as `network_transit_ms`. Under overload, measure latency at the client as well ([measured](load-testing.md#where-the-queue-lives)).
* **One Python process per orchestrator and per worker.** A worker sustains about 270 light jobs/s and an orchestrator about 370. Far past saturation, waiting connections cost event-loop and GIL time, and throughput sags (about −50 % at 256 clients on one worker). Bound concurrency with the asynchronous queue, or add replicas behind the gateway ([measured](load-testing.md#orchestrator-scale-out)).
* **Large circuits are parsed twice**: once for validation in the orchestrator and once in the worker (about 0.9 s each at 30 000 gates).

## Emulation cost

* **Routing can enlarge the simulated register.** On sparse lattices SABRE may move logical qubits through physical qubits the circuit never named, and all of them are simulated. A 20-qubit QFT adder needs 22 simulated qubits on Emerald and 24 on Ankaa-3; on Emerald the job takes 6.9× the worker time of all-to-all Helios ([measured](circuit-benchmarks.md#where-the-workers-time-goes)). Circuits whose interaction graph embeds in the lattice (chains, small stars, the HEA ladder) get an exact SWAP-free layout and cost the same as on Helios. Check `compilation.simulated_qubits`.
* **Not the fastest ideal simulator everywhere.** On deep ideal circuits qsim's hand-vectorised C++ is 1.8–3.0× faster than QPUTwin's default double-precision engine (1.2–1.75× at matched single precision), and on circuits under ~16 qubits Qulacs is faster, because QPUTwin plans its fused blocks in Python. The trajectory engine still applies gates one at a time ([comparison](simulator-comparison.md)).
