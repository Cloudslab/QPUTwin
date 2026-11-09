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
| `09_ibm_and_google_twins` | IBM Heron r3, IBM Nighthawk and Google Willow: published maps, per-qubit calibration, native gates, a poor qubit the router finds, and IBM's repetition delay |

Notebook 07 reads Prometheus when the [monitoring stack](../monitoring/README.md) is running, and still works without it.
Open the Grafana dashboards beside it for the best live view.

## Reading the results

- The twins follow published medians, and they have not been validated against hardware.
- The Rigetti and IQM coupling maps are synthetic lattices, not vendor maps. The IBM and Google twins use the vendors' published maps and calibration snapshots.
- Notebooks 00 to 08 compare the first four twins (`BACKENDS`). `ALL_BACKENDS` adds the three IBM and Google twins.
- Durations are mostly assumed, so times are order-of-magnitude estimates. Notebook 01 lists every assumption.
- Notebook 07's load test uses Python threads on the same machine as the stack. It is a live illustration, not a benchmark.

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
