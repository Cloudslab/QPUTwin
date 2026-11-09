# QPUTwin monitoring

Watches QPUTwin's jobs and the resources behind them: per-job latency and workload, each container, and the whole host.
Everything runs in Docker except Node Exporter, which runs on the host so that it sees the real machine.

## Start

Start QPUTwin first, then this stack:

```bash
docker compose up -d --wait     # from the project root
cd monitoring
./up.sh
```

`up.sh` is safe to run again. The first run generates passwords, installs Node Exporter, starts the containers and prints
the addresses with their logins. To see the logins again, read `monitoring/.env` and
`monitoring/secrets/portainer_admin_password`. Neither is tracked by git.

| Interface | Address | Login |
|---|---|---|
| Grafana | <http://localhost:3000> | `admin`, password in `.env` |
| Portainer CE | <https://localhost:9443> | `admin`, password in `secrets/`. Accept the self-signed certificate warning |
| Prometheus | <http://localhost:9090> | none |
| Alertmanager | <http://localhost:9093> | none |
| cAdvisor | <http://localhost:8081> | none |

All of them listen on `127.0.0.1` only. cAdvisor and Portainer can reach the Docker socket, so do not publish them on a
network without adding protection.

To stop:

```bash
./down.sh                              # keeps stored metrics
./down.sh --volumes                    # deletes them too
./node-exporter/install.sh --uninstall # removes Node Exporter from the host
```

## Components

| Component | Runs in | Job |
|---|---|---|
| Prometheus | Docker | Scrapes every source each 5 s, keeps 15 days, evaluates alerts |
| Grafana | Docker | Six provisioned dashboards |
| cAdvisor | Docker | CPU, memory, network and disk per container |
| Node Exporter | **Host** | CPU, memory, disks, network and temperatures of the machine |
| qputwin-exporter | Docker | Per-job metrics from the telemetry dataset |
| Blackbox Exporter | Docker | Probes `/ready` and each worker's `/health` from inside the QPUTwin network |
| Alertmanager | Docker | Groups and shows alerts |
| Portainer CE | Docker | Container management, logs and shell |

Node Exporter runs as a systemd user service called `qputwin-node-exporter`, so it needs no root access. It listens on
loopback and on the Docker bridge gateway, and not on your LAN. Check it with:

```bash
systemctl --user status qputwin-node-exporter
```

## Dashboards

Open them from the **QPUTwin** folder in Grafana. They refresh every 5 s.

| Dashboard | Answers |
|---|---|
| QPUTwin Overview | Is it healthy? Job rates, outcomes, p50/p95/p99 latency, queues, firing alerts |
| QPUTwin Latency Decomposition | Where does a job's time go, across the fourteen additive spans? |
| QPUTwin Workload | Which engines, how many qubits, shots and gates, what was refused, what was retried? |
| QPUTwin Jobs and Resources | For each twin, job load beside its container's CPU and memory at the same moment |
| Containers | CPU, throttling, memory against limits, network and disk for every container |
| Host | The whole machine: CPU by mode and core, memory, disks, network, temperatures, CPU frequency |

## Where the metrics come from

The orchestrator's `/metrics` endpoint gives job counts, three latency histograms, queues and replica load. The
per-job telemetry holds much more, so `qputwin-exporter/exporter.py` follows `data/telemetry.jsonl` and publishes it as
`qputwin_tel_*` metrics:

- the fourteen latency spans, per twin, and finer round-trip buckets;
- jobs by engine, execution path, noisy and routed;
- simulated qubits, peak memory, shots and native gates;
- rejections by error type, and jobs re-executed after a lost lease;
- `qputwin_tel_identity_violations_total`, which checks live that the spans sum to the round trip.

Reading the file means no QPUTwin code changed, nothing is added to the measured request path, and all orchestrator
replicas are covered, because they all append to the same file. The exporter counts jobs from the moment it starts.
Set `QPUTWIN_TEL_START=begin` in `.env` to replay the whole file instead.

Recording rules are in `prometheus/rules/recording.yml`.

## Alerts

`prometheus/rules/alerts.yml` covers the service, containers, host and the monitoring stack itself. Alerts appear in
Alertmanager and on the Overview dashboard, and nothing is sent anywhere until you add a receiver in
`alertmanager/alertmanager.yml`.

`QPUTwinSlowJobs` and `ContainerCPUThrottled` can fire during a deliberately heavy test. That is expected.

## Benchmarking with monitoring on

This stack shares the machine with QPUTwin, and with Docker's own overhead. Stop it with `./down.sh` before you measure
results for publication. Keep it running for demonstrations and for investigating behaviour.

## Changing things

| To change | Edit | Then |
|---|---|---|
| A dashboard | `grafana/build_dashboards.py` | Run `python3 grafana/build_dashboards.py`. Grafana reloads within 30 s |
| Scrape targets | `prometheus/prometheus.yml` | `curl -X POST localhost:9090/-/reload` |
| Rules | `prometheus/rules/` | `curl -X POST localhost:9090/-/reload` |
| Ports, retention | `.env` | `./up.sh` |

Replicas of the orchestrator and workers are found automatically through Compose DNS. With several orchestrators, the
dashboards sum across them.
