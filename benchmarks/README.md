# QPUTwin benchmarks

Koustav Kumar Mondal · koustav@koustavmondal.com

| Path | Contents |
|---|---|
| `run_benchmarks.py` | collects the data (21 experiments; live HTTP and in-process) |
| `circuits.py` | the nine benchmark circuit families (OpenQASM 2.0) and fidelity / heavy-output helpers |
| `plot_figures.py` | renders the publication figures from `results/` |
| `results/*.json` | raw measurements with environment metadata |
| `results/*.junit.xml`, `*.log` | unit (616) and live-stack (47) test reports, run logs |
| `figures/*.pdf`, `*.svg`, `*.png` | 27 figures: vector PDF/SVG and 300-dpi PNG |
| `figures/data/*.csv` | exactly the values plotted in each figure |

```bash
docker compose up -d --wait
uv run --group bench python benchmarks/run_benchmarks.py all     # or: latency circuits load_closed ...
uv run --group bench python benchmarks/plot_figures.py           # or: plot_figures.py qv soak ...
```

`QPUTWIN_BENCH_FAMILIES=qft,qv` re-runs only those families of the `circuits` experiment and keeps the rest of the stored results. `load_closed` and `load_ha` restart the Compose stack with different replica counts and restore the defaults afterwards.

## Core

| Figure | Experiment |
|---|---|
| fig01 / fig11 | latency decomposition and distribution (live, 100 jobs per backend) |
| fig02 | engine cost vs width (statevector c128/c64, density matrix, trajectories) |
| fig03 / fig03b | cost of routing: entanglers, predicted success, live latency |
| fig04 | GHZ success under calibrated noise vs width |
| fig05 | estimated hardware time per shot |
| fig06 | error budget, one channel at a time |
| fig07 | trajectory convergence to the exact density matrix |
| fig08 | async throughput vs worker replicas (cache-resident vs DRAM-bound) |
| fig09 | noise-scale sensitivity |
| fig10 | Ramsey: Markovian vs quasi-static dephasing |

## Circuit families (`circuits`, `qv`)

| Figure | Experiment |
|---|---|
| fig12 | fidelity and native 2Q cost, nine families × four QPUs, n = 8 |
| fig13 | fidelity vs width (3–10 qubits), one panel per family |
| fig14 | routing overhead (2Q gates ÷ all-to-all) by family and width, Ankaa-3 and Emerald |
| fig15 | worker time at 20 qubits, Helios vs Emerald, with the simulated width |
| fig16 | QV-style heavy-output probability vs width (12 circuits per width) |

## Load and stress (`load_closed`, `load_ha`, `load_open`, `soak`, `async_burst`, `fairshare`, `shots`, `gate_scaling`)

| Figure | Experiment |
|---|---|
| fig17 | closed loop: throughput and client latency vs 1–256 concurrent clients |
| fig18 | latency–throughput curve and Little's-law check |
| fig25 | where the queue lives: latency share by telemetry component |
| fig26 | orchestrator scale-out: direct vs nginx gateway with 1 or 3 orchestrators |
| fig19 | open-loop Poisson load vs the M/G/1 Pollaczek–Khinchine model |
| fig20 | five-minute soak: latency over time and container memory |
| fig21 | 2 000-job async burst: queue depth, drain rate, queue-wait distribution |
| fig22 | fair-share isolation from a flooding client |
| fig23 | latency and sampling time vs shots (1–10⁶) |
| fig24 | parse / route / compile time vs circuit size (100–30 000 gates) |

Interpretation and numbers: [benchmarks](https://koustavmondal.com/qputwin-docs/research/benchmarks/), [circuit benchmarks](https://koustavmondal.com/qputwin-docs/research/circuit-benchmarks/) and [load testing](https://koustavmondal.com/qputwin-docs/research/load-testing/) in the documentation site.
