# QPUTwin examples

Jupyter notebooks that demonstrate QPUTwin live. Every job goes through the HTTP API to the Docker stack, so what you see
is what a client of the service gets. The notebooks are saved with their outputs, so you can also read them without
running anything.

## Run them

From the project root, start the stack:

```bash
docker compose up -d --wait
```

Then start Jupyter in this folder's own uv environment:

```bash
cd Example
uv sync
uv run jupyter notebook
```

Open a notebook and choose **Run > Run All Cells**.

## Notebooks

| Notebook | Shows |
|---|---|
| `00_quickstart` | Connecting, a first circuit, the response, seeds, and one circuit on every twin |
| `01_devices_and_calibration` | Native gates, coupling maps, calibration, and which numbers are published or assumed |
| `02_native_compilation` | One circuit drawn in four native gate sets, the exactness certificate, gate costs, hardware time, strict mode |
| `03_noise_and_fidelity` | Noise reports, exact against estimated success, error budgets, noise scaling, density matrices, trajectories |
| `04_topology_and_routing` | Rejection, opt-in routing, layouts on the lattice, exact embedding, connectivity cost, register growth |
| `05_algorithms` | Bernstein-Vazirani, Grover, QFT, mirror circuits, teleportation, QAOA, and a scoreboard across twins |
| `06_async_jobs_and_scheduling` | Asynchronous jobs, a visible queue, priority, cancellation, fair share |
| `07_telemetry_and_load` | Latency spans, additive identities, a live load test, and the resources it used |
| `08_fault_tolerance` | **Kills the orchestrator with `SIGKILL`**, restarts it, and shows no job was lost |
| `09_ibm_and_google_twins` | IBM Heron r3, IBM Nighthawk and Google Willow: published maps, per-qubit calibration, native gates, the poor qubit the calibration-aware re-layout avoids, and IBM's repetition delay |
| `10_openqasm3` | OpenQASM 3 declarations, parameterised gate definitions, `ctrl`/`inv`/`pow` modifiers, feed-forward, portable vendor-native gates, typed parse errors |
| `11_engines` | What `engine: auto` chooses, the two exact noisy engines agreeing, returned statevectors and density matrices, the memory guard |
| `12_precision` | complex64 against complex128: identical counts, amplitude error, half the memory, speed on deep circuits |
| `13_noise_channels` | The per-gate noise report, one channel at a time, an error budget on every twin, idle decoherence, estimated against executed success |
| `14_noise_what_if` | Overriding calibration: better gates, readout and coherence, scaling all errors, the fidelity a device would need |
| `15_coherent_errors` | Over-rotation growing quadratically, coherent against stochastic error, echoes, ZZ crosstalk on a spectator |
| `16_leakage_and_drift` | Leakage and seepage, Markovian against quasi-static Ramsey decay, spin echoes, imperfect reset |
| `17_trajectories` | Convergence with the number of trajectories, exact sampling, wide noisy circuits, batched and sequential trajectories |
| `18_readout` | Readout error per twin and with width, calibrating confusion matrices, mitigating a GHZ result |
| `19_timing` | Hardware time per shot and per job, IBM's repetition delay, parallel gates, busy time, shallow against deep GHZ circuits |
| `20_routing_options` | SABRE trials, exact layout, initial layouts, cancellation, bridge gates, calibration-aware re-layout |
| `21_entanglement_protocols` | Deutsch-Jozsa, superdense coding, a CHSH Bell test, entanglement swapping with feed-forward, Simon's algorithm |
| `22_phase_estimation` | Phase estimation of exact and non-binary phases, under device noise, precision against depth |
| `23_variational_h2` | A variational ground-state energy from counts in three bases, a parameter scan, each twin against the exact energy |
| `24_grover` | Grover search on 2 to 4 qubits, the oscillation with iterations, every twin under noise, when noise changes the best iteration count |
| `25_ghz_fidelity_witness` | GHZ fidelity from populations and parity oscillations, every twin, the width at which entanglement can no longer be certified |
| `26_mirror_benchmarking` | Mirror circuits, decay with depth, an effective error per layer, the cost of routing |
| `27_quantum_volume` | Heavy outputs, heavy-output probability on every twin, pass or fail by width, why few circuits are only a comparison |
| `28_async_job_arrays` | A job array over every twin, long polling, listing and filtering jobs, asynchronous against one by one, queue timestamps |
| `29_errors_and_limits` | A catalogue of typed errors, machine-readable details, validation errors, cancelling, each twin's limits |
| `30_telemetry_analysis` | The `Server-Timing` header, exact span identities, latency distributions per twin, the telemetry summary, time by circuit width |
| `31_reproducibility` | Seeds, replaying a server-drawn seed, per-shot memory, the calibration a result used, the software environment |

Notebook 07 reads Prometheus when the [monitoring stack](../monitoring/README.md) is running, and still works without it.
Open the Grafana dashboards beside it for the best live view.

## Reading the results

- The twins follow published medians, and they have not been validated against hardware.
- The Rigetti and IQM coupling maps are synthetic lattices, not vendor maps. The IBM and Google twins use the vendors' published maps and calibration snapshots.
- Notebooks 00 to 08 compare the first four twins (`BACKENDS`). Notebook 09 and most of notebooks 10 to 31 use all seven (`ALL_BACKENDS`).
- Durations are mostly assumed, so times are order-of-magnitude estimates. Notebook 01 lists every assumption.
- Notebook 07's load test uses Python threads on the same machine as the stack. It is a live illustration, not a benchmark.
- Run the notebooks one at a time on a laptop: each twin's worker may use up to 12 GiB by default. Set `QPUTWIN_WORKER_MEM` lower before `docker compose up` on a machine with little memory.

## How compiled circuits are drawn

The API returns compilation statistics rather than the compiled circuit. To draw one, `qputwin_demo.py` calls the same
compiler the workers use, locally, and stops with an error if the drawing's gate counts differ from the job's API response.
Everything else in the notebooks runs in Docker.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `QPUTWIN_URL` | `http://localhost:8000` | Orchestrator address |
| `QPUTWIN_API_KEY` | none | API key, when authentication is on |
| `QPUTWIN_PROMETHEUS_URL` | `http://localhost:9090` | Prometheus address |
| `QPUTWIN_GRAFANA_URL` | `http://localhost:3000` | Grafana address |
