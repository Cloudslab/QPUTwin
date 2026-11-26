#!/usr/bin/env python3
"""Generates the Grafana dashboard JSON files in this folder.

Usage: python3 monitoring/grafana/build_dashboards.py

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}
TWINS = [("quantinuum_helios", "Helios", "worker-quantinuum-helios"), ("ionq_forte", "Forte", "worker-ionq-forte"),
         ("rigetti_ankaa_3", "Ankaa-3", "worker-rigetti-ankaa3"), ("iqm_emerald", "Emerald", "worker-iqm-emerald"),
         ("ibm_heron", "Heron r3", "worker-ibm-heron"), ("ibm_nighthawk", "Nighthawk", "worker-ibm-nighthawk"),
         ("google_willow", "Willow", "worker-google-willow")]
# Dark-surface steps of the notebooks' palette, one fixed colour per twin (checked for colour-blind separation).
TWIN_COLORS = {"quantinuum_helios": "#3987E5", "ionq_forte": "#D95926", "rigetti_ankaa_3": "#199E70",
               "iqm_emerald": "#C98500", "ibm_heron": "#D55181", "ibm_nighthawk": "#008300", "google_willow": "#9085E9"}
JOBS = "suffix: jobs/s"


def target(expr: str, legend: str = "", instant: bool = False) -> dict:
    return {"datasource": DS, "expr": expr, "legendFormat": legend or "__auto", "instant": instant, "range": not instant}


def twin_overrides() -> list[dict]:
    """The same colour for a twin on every panel, matching the notebooks."""
    return [{"matcher": {"id": "byName", "options": b},
             "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": TWIN_COLORS[b]}}]}
            for b, _, _ in TWINS]


class Board:
    def __init__(self, uid: str, title: str, description: str, variables: list[dict] | None = None) -> None:
        self.uid, self.title, self.description = uid, title, description
        self.variables = variables or []
        self.panels: list[dict] = []
        self.x = self.y = self.row_h = 0

    def _place(self, w: int, h: int) -> dict:
        if self.x + w > 24:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0
        pos = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return pos

    def _add(self, panel: dict, w: int, h: int) -> None:
        panel["id"] = len(self.panels) + 1
        panel["gridPos"] = self._place(w, h)
        panel["datasource"] = DS
        self.panels.append(panel)

    def row(self, title: str) -> None:
        self.x, self.y, self.row_h = 0, self.y + self.row_h, 1
        self.panels.append({"id": len(self.panels) + 1, "type": "row", "title": title, "collapsed": False,
                            "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1}, "panels": []})
        self.x, self.y, self.row_h = 0, self.y + 1, 0

    def stat(self, title: str, expr: str, unit: str = "none", w: int = 4, h: int = 4, description: str = "",
             decimals: int | None = None, thresholds: list[tuple[float | None, str]] | None = None,
             mappings: list[dict] | None = None) -> None:
        steps = [{"value": v, "color": c} for v, c in (thresholds or [(None, "text")])]
        defaults: dict = {"unit": unit, "thresholds": {"mode": "absolute", "steps": steps}, "mappings": mappings or []}
        if decimals is not None:
            defaults["decimals"] = decimals
        self._add({"type": "stat", "title": title, "description": description,
                   "targets": [target(expr, instant=True)],
                   "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                               "colorMode": "value", "graphMode": "none", "textMode": "value", "justifyMode": "center"},
                   "fieldConfig": {"defaults": defaults, "overrides": []}}, w, h)

    def series(self, title: str, targets: list[dict], unit: str = "none", w: int = 12, h: int = 8, stacked: bool = False,
               description: str = "", min_zero: bool = True, max_value: float | None = None,
               fill: int | None = None, decimals: int | None = None, legend_right: bool = False) -> None:
        custom = {"drawStyle": "line", "lineWidth": 2, "fillOpacity": fill if fill is not None else (70 if stacked else 8),
                  "showPoints": "never", "spanNulls": True, "lineInterpolation": "linear",
                  "stacking": {"mode": "normal" if stacked else "none", "group": "A"}}
        defaults: dict = {"unit": unit, "custom": custom, "color": {"mode": "palette-classic"}}
        if min_zero:
            defaults["min"] = 0
        if max_value is not None:
            defaults["max"] = max_value
        if decimals is not None:
            defaults["decimals"] = decimals
        # Narrow panels use a compact legend, so that no series is cut off.
        legend = ({"displayMode": "list", "placement": "bottom", "calcs": []} if w <= 8 else
                  {"displayMode": "table", "placement": "right" if legend_right else "bottom", "calcs": ["lastNotNull", "max"]})
        twins = any(t["legendFormat"].startswith("{{backend}}") for t in targets)
        self._add({"type": "timeseries", "title": title, "description": description, "targets": targets,
                   "options": {"legend": legend, "tooltip": {"mode": "multi", "sort": "desc"}},
                   "fieldConfig": {"defaults": defaults, "overrides": twin_overrides() if twins else []}}, w, h)

    def heatmap(self, title: str, expr: str, w: int = 12, h: int = 8, description: str = "") -> None:
        self._add({"type": "heatmap", "title": title, "description": description,
                   "targets": [{**target(expr, "{{le}}"), "format": "heatmap"}],
                   "options": {"calculate": False, "yAxis": {"axisPlacement": "left"},
                               "color": {"scheme": "Viridis", "mode": "scheme"}, "cellGap": 1,
                               "tooltip": {"mode": "single", "yHistogram": False}, "legend": {"show": True}},
                   "fieldConfig": {"defaults": {}, "overrides": []}}, w, h)

    def table(self, title: str, targets: list[dict], w: int = 12, h: int = 8, description: str = "",
              rename: dict[str, str] | None = None, hide: list[str] | None = None, overrides: list[dict] | None = None) -> None:
        transforms: list[dict] = [{"id": "merge", "options": {}}]
        excluded = dict.fromkeys(hide or ["Time"], True)
        transforms.append({"id": "organize", "options": {"excludeByName": excluded, "renameByName": rename or {}}})
        self._add({"type": "table", "title": title, "description": description,
                   "targets": [{**t, "format": "table"} for t in targets], "transformations": transforms,
                   "options": {"showHeader": True, "cellHeight": "sm"},
                   "fieldConfig": {"defaults": {}, "overrides": overrides or []}}, w, h)

    def alerts(self, title: str, w: int = 24, h: int = 6) -> None:
        self._add({"type": "alertlist", "title": title,
                   "options": {"alertInstanceLabelFilter": "", "alertName": "", "dashboardAlerts": False,
                               "groupBy": [], "groupMode": "default", "maxItems": 20, "sortOrder": 1,
                               "stateFilter": {"error": True, "firing": True, "noData": False, "normal": False, "pending": True},
                               "viewMode": "list", "datasource": "Prometheus"}}, w, h)

    def text(self, content: str, w: int = 24, h: int = 3) -> None:
        self._add({"type": "text", "title": "", "transparent": True,
                   "options": {"mode": "markdown", "content": content}}, w, h)

    def write(self, filename: str) -> None:
        dashboard = {"uid": self.uid, "title": self.title, "description": self.description, "tags": ["qputwin"],
                     "editable": False, "graphTooltip": 1, "refresh": "5s", "schemaVersion": 41, "version": 1,
                     "time": {"from": "now-15m", "to": "now"}, "timezone": "browser",
                     "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m"]},
                     "annotations": {"list": []}, "templating": {"list": self.variables}, "panels": self.panels,
                     "links": [{"type": "dashboards", "tags": ["qputwin"], "asDropdown": True, "title": "QPUTwin dashboards",
                                "includeVars": False, "keepTime": True, "icon": "external link"}]}
        (OUT / filename).write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")


def query_variable(name: str, label: str, query: str) -> dict:
    # ".+" and not ".*": cAdvisor's whole-machine series has an empty service label.
    return {"name": name, "label": label, "type": "query", "datasource": DS, "query": {"query": query, "refId": "v"},
            "refresh": 2, "sort": 1, "includeAll": True, "multi": True, "allValue": ".+",
            "current": {"text": "All", "value": "$__all"}}


UP_DOWN = [{"type": "value", "options": {"0": {"text": "DOWN", "color": "red"}, "1": {"text": "UP", "color": "green"}}}]
GOOD_BAD = [(None, "red"), (1, "green")]
COUNTED = "Since the telemetry exporter started, so an orchestrator restart does not reset it."


def overview() -> None:
    b = Board("qputwin-overview", "QPUTwin Overview", "Service health, job rates, latency and queues.")
    b.row("Service")
    b.stat("Orchestrators", 'sum(up{job="qputwin-orchestrator"})', w=3, thresholds=GOOD_BAD)
    b.stat("Twins up", 'sum(probe_success{job=~"qputwin-workers.*"})', w=3, thresholds=[(None, "red"), (4, "green")],
           description="Worker replicas answering /health. Four twins with one replica each gives 4.")
    b.stat("Jobs/s", "sum(qputwin:jobs:rate1m)", "none", w=3, decimals=1)
    b.stat("Completed", 'sum(qputwin_tel_jobs_total{status="COMPLETED"}) or vector(0)', "short", w=3, description=COUNTED)
    b.stat("Rejected", 'sum(qputwin_tel_jobs_total{status="REJECTED"}) or vector(0)', "short", w=3,
           description="Refused on purpose, for example a topology violation. A rejection is correct behaviour, not a fault. " + COUNTED)
    b.stat("Failed", 'sum(qputwin_tel_jobs_total{status="FAILED"}) or vector(0)', "short", w=3,
           thresholds=[(None, "green"), (1, "red")], description=COUNTED)
    b.stat("Jobs queued", 'sum(max by (backend) (qputwin_async_jobs{state="queued"})) or vector(0)', "short", w=3)
    b.stat("p50 latency", 'histogram_quantile(0.5, sum by (le) (rate(qputwin_tel_roundtrip_ms_bucket[1m])))', "ms", w=3, decimals=2,
           description="Over the last minute, including any queue wait.")
    b.row("Job rate")
    b.series("Jobs per second, by twin", [target('sum by (backend) (qputwin:jobs:rate1m{status="COMPLETED"})', "{{backend}}")], JOBS,
             w=12, stacked=True)
    b.series("Jobs per second, by outcome", [target("sum by (status) (qputwin:jobs:rate1m)", "{{status}}")], JOBS, w=12)
    b.row("Latency, measured by the server")
    for quantile in ("p50", "p95", "p99"):
        b.series(f"Round trip {quantile}, by twin", [target(f"qputwin:roundtrip_ms:{quantile}", "{{backend}}")], "ms", w=8,
                 decimals=2)
    b.row("Queues and replicas")
    b.series("Async jobs queued", [target('max by (backend) (qputwin_async_jobs{state="queued"})', "{{backend}}")], "short", w=8)
    b.series("Async jobs running", [target('max by (backend) (qputwin_async_jobs{state="running"})', "{{backend}}")], "short", w=8)
    b.series("In-flight requests per replica", [target("qputwin_replica_in_flight", "{{backend}} {{replica}}")], "short", w=8)
    b.row("Alerts")
    b.alerts("Firing and pending alerts")
    b.write("01-qputwin-overview.json")


BACKEND_VAR = query_variable("backend", "Twin", "label_values(qputwin_tel_span_ms_count, backend)")


def latency() -> None:
    b = Board("qputwin-latency", "QPUTwin Latency Decomposition",
              "The fourteen additive telemetry spans of every completed job.", [BACKEND_VAR])
    sel = 'backend=~"$backend"'
    mean = (f'sum by (span) (rate(qputwin_tel_span_ms_sum{{{sel}}}[1m])) '
            f'/ ignoring(span) group_left sum(rate(qputwin_tel_roundtrip_ms_count{{{sel}}}[1m]))')
    no_queue = f'sum by (span) (rate(qputwin_tel_span_ms_sum{{{sel},span!="orchestrator_queue"}}[1m])) ' \
               f'/ ignoring(span) group_left sum(rate(qputwin_tel_roundtrip_ms_count{{{sel}}}[1m]))'
    total = f'sum(rate(qputwin_tel_span_ms_sum{{{sel},span!="orchestrator_queue"}}[1m]))'
    b.text("Each span is measured in one process, so the spans add up exactly to the round trip. "
           "The **identity violations** panel counts any job where they do not, and should stay at 0.", h=2)
    b.stat("Identity violations", "sum(qputwin_tel_identity_violations_total) or vector(0)", "short", w=6,
           thresholds=[(None, "green"), (1, "red")])
    b.stat("Parse errors", "qputwin_tel_parse_errors_total", "short", w=6, thresholds=[(None, "green"), (1, "red")])
    b.stat("Engine share", f'sum(rate(qputwin_tel_span_ms_sum{{{sel},span=~"simulation|sampling"}}[1m])) / {total}',
           "percentunit", w=6, decimals=0,
           description="Simulation and sampling, as a share of the job excluding queue wait. Below about half, the emulation layer costs more than the simulation.")
    b.stat("Worker share", f'sum(rate(qputwin_tel_span_ms_sum{{{sel},span=~"worker.*|simulation|sampling"}}[1m])) / {total}',
           "percentunit", w=6, decimals=0, description="Everything inside the worker, as a share of the job excluding queue wait.")
    b.series("Where a job's time goes, excluding queue wait", [target(no_queue, "{{span}}")], "ms", w=24, h=11, stacked=True, legend_right=True,
             description="Mean milliseconds per job in each span.")
    b.series("Including queue wait", [target(mean, "{{span}}")], "ms", w=24, h=9, stacked=True, legend_right=True,
             description="The stack height is the mean round trip. An asynchronous backlog dominates this view.")
    b.row("Spans in detail")
    b.series("Orchestrator spans", [target(f'sum by (span) (rate(qputwin_tel_span_ms_sum{{{sel},span=~"ingress|validation|backend_selection|orchestrator_queue|network_transit|response_assembly|orchestrator_unattributed"}}[1m])) / sum by (span) (rate(qputwin_tel_span_ms_count{{{sel}}}[1m]))', "{{span}}")], "ms", w=12)
    b.series("Worker spans", [target(f'sum by (span) (rate(qputwin_tel_span_ms_sum{{{sel},span=~"worker_queue|worker_parse|worker_route|worker_compile|simulation|sampling|worker_unattributed"}}[1m])) / sum by (span) (rate(qputwin_tel_span_ms_count{{{sel}}}[1m]))', "{{span}}")], "ms", w=12)
    b.heatmap("Round trip distribution", f'sum by (le) (increase(qputwin_tel_roundtrip_ms_bucket{{{sel}}}[30s]))', w=12,
              description="Jobs per latency bucket over time. A band that climbs means a queue is forming.")
    b.series("Span p95", [target(f'histogram_quantile(0.95, sum by (span, le) (rate(qputwin_tel_span_ms_bucket{{{sel}}}[1m])))', "{{span}}")], "ms", w=12)
    b.write("02-qputwin-latency.json")


def workload() -> None:
    b = Board("qputwin-workload", "QPUTwin Workload", "What the twins are being asked to run.", [BACKEND_VAR])
    sel = 'backend=~"$backend"'
    b.row("Mix")
    for title, label in (("Engine", "engine"), ("Execution path", "execution_path"), ("Noisy jobs", "noisy"), ("Routed jobs", "routed")):
        b.series(title, [target(f'sum by ({label}) (rate(qputwin_tel_jobs_total{{{sel},status="COMPLETED"}}[1m]))', f"{{{{{label}}}}}")],
                 JOBS, w=6, stacked=True)
    b.row("Size")
    b.heatmap("Simulated qubits", f'sum by (le) (increase(qputwin_tel_simulated_qubits_bucket{{{sel}}}[30s]))', w=12,
              description="Qubits actually simulated, after routing and compaction.")
    b.series("Estimated peak engine memory, p95",
             [target(f'histogram_quantile(0.95, sum by (backend, le) (rate(qputwin_tel_estimated_peak_bytes_bucket{{{sel}}}[1m])))', "{{backend}}")],
             "bytes", w=12)
    b.series("Shots per second", [target(f"sum by (backend) (rate(qputwin_tel_shots_total{{{sel}}}[1m]))", "{{backend}}")], "short", w=8)
    b.series("Native gates per second", [target(f"sum by (backend) (rate(qputwin_tel_native_ops_total{{{sel}}}[1m]))", "{{backend}}")], "short", w=8)
    b.series("Native two-qubit gates per job",
             [target(f"sum by (backend) (rate(qputwin_tel_native_two_qubit_ops_total{{{sel}}}[1m])) / sum by (backend) (rate(qputwin_tel_roundtrip_ms_count{{{sel}}}[1m]))", "{{backend}}")],
             "short", w=8)
    b.row("Refusals and retries")
    b.series("Rejected or failed jobs, by error type", [target(f"sum by (error_type) (rate(qputwin_tel_errors_total{{{sel}}}[1m]))", "{{error_type}}")], JOBS, w=12)
    b.series("Jobs re-executed after a lost lease", [target(f"sum by (backend) (qputwin_tel_retried_jobs_total{{{sel}}})", "{{backend}}")], "short", w=12)
    b.write("03-qputwin-workload.json")


def resources() -> None:
    b = Board("qputwin-resources", "QPUTwin Jobs and Resources",
              "Each twin's job load next to what its container used at the same moment.")
    b.row("Whole system")
    b.series("Jobs per second", [target('sum(qputwin:jobs:rate1m{status="COMPLETED"})', "jobs/s")], JOBS, w=8, h=7)
    b.series("Host CPU busy", [target("host:cpu_busy:ratio1m", "host")], "percentunit", w=8, h=7, max_value=1)
    b.series("Orchestrator", [target('service:cpu_cores:rate1m{service="orchestrator"}', "CPU cores")], "short", w=8, h=7, decimals=2)
    for backend, name, service in TWINS:
        b.row(name)
        b.series("Jobs per second", [target(f'sum(qputwin:jobs:rate1m{{backend="{backend}",status="COMPLETED"}})', "jobs/s")], JOBS, w=6, h=6)
        b.series("Engine time per job", [target(f'sum(rate(qputwin_job_qpu_compute_ms_sum{{backend="{backend}"}}[1m])) / sum(rate(qputwin_job_qpu_compute_ms_count{{backend="{backend}"}}[1m]))', "ms")], "ms", w=6, h=6, decimals=2)
        b.series("Container CPU", [target(f'service:cpu_cores:rate1m{{service="{service}"}}', "cores used"),
                                   target(f'sum(container_spec_cpu_quota{{service="{service}"}}) / sum(container_spec_cpu_period{{service="{service}"}})', "limit")],
                 "short", w=6, h=6, decimals=2)
        b.series("Container memory", [target(f'service:memory_working_set:bytes{{service="{service}"}}', "working set")], "bytes", w=6, h=6)
    b.write("04-qputwin-resources.json")


def containers() -> None:
    service_var = query_variable("service", "Service", 'label_values(container_memory_working_set_bytes{service!=""}, service)')
    project_var = query_variable("project", "Stack", 'label_values(container_memory_working_set_bytes{service!=""}, project)')
    b = Board("qputwin-containers", "Containers", "cAdvisor metrics per container.", [project_var, service_var])
    sel = 'service=~"$service",project=~"$project"'
    b.row("Now")
    rename = {"service": "Service", "Value #A": "CPU cores", "Value #B": "Memory", "Value #C": "Memory limit", "Value #D": "Throttled"}
    b.table("Containers right now", [
        {**target(f"sum by (service) (rate(container_cpu_usage_seconds_total{{{sel}}}[1m]))", instant=True), "refId": "A"},
        {**target(f"sum by (service) (container_memory_working_set_bytes{{{sel}}})", instant=True), "refId": "B"},
        {**target(f"sum by (service) (container_spec_memory_limit_bytes{{{sel}}})", instant=True), "refId": "C"},
        {**target(f"sum by (service) (service:cpu_throttled:ratio1m{{{sel.split(',')[0]}}})", instant=True), "refId": "D"},
    ], w=24, h=19, rename=rename, overrides=[
        {"matcher": {"id": "byName", "options": "CPU cores"}, "properties": [{"id": "decimals", "value": 3}]},
        {"matcher": {"id": "byName", "options": "Memory"}, "properties": [{"id": "unit", "value": "bytes"}]},
        {"matcher": {"id": "byName", "options": "Memory limit"}, "properties": [
            {"id": "unit", "value": "bytes"},
            {"id": "mappings", "value": [{"type": "value", "options": {"0": {"text": "none"}}}]}]},
        {"matcher": {"id": "byName", "options": "Throttled"}, "properties": [{"id": "unit", "value": "percentunit"}]},
    ])
    b.row("CPU")
    b.series("CPU, by service", [target(f"sum by (service) (rate(container_cpu_usage_seconds_total{{{sel}}}[1m]))", "{{service}}")], "short", w=12, decimals=2,
             description="One unit is one full core.")
    b.series("CPU throttling", [target(f"sum by (service) (rate(container_cpu_cfs_throttled_periods_total{{{sel}}}[1m])) / sum by (service) (rate(container_cpu_cfs_periods_total{{{sel}}}[1m]))", "{{service}}")],
             "percentunit", w=12, max_value=1, description="Share of scheduler periods in which a container hit its CPU quota.")
    b.row("Memory")
    b.series("Working set", [target(f"sum by (service) (container_memory_working_set_bytes{{{sel}}})", "{{service}}")], "bytes", w=12)
    b.series("Share of memory limit", [target(f"sum by (service) (container_memory_working_set_bytes{{{sel}}}) / sum by (service) (container_spec_memory_limit_bytes{{{sel}}} > 0)", "{{service}}")],
             "percentunit", w=12, max_value=1)
    b.row("Network and disk")
    b.series("Network received", [target(f"sum by (service) (rate(container_network_receive_bytes_total{{{sel}}}[1m]))", "{{service}}")], "Bps", w=12)
    b.series("Network sent", [target(f"sum by (service) (rate(container_network_transmit_bytes_total{{{sel}}}[1m]))", "{{service}}")], "Bps", w=12)
    b.series("Disk writes", [target(f"sum by (service) (rate(container_fs_writes_bytes_total{{{sel}}}[1m]))", "{{service}}")], "Bps", w=12)
    b.series("Container restarts", [target(f"changes(container_start_time_seconds{{{sel}}}[15m])", "{{name}}")], "short", w=12)
    b.write("05-containers.json")


def host() -> None:
    b = Board("qputwin-host", "Host", "Whole-machine resources, from Node Exporter running on the host.")
    root = 'mountpoint="/"'
    b.row("Now")
    b.stat("CPU busy", "host:cpu_busy:ratio1m", "percentunit", w=4, decimals=0, thresholds=[(None, "green"), (0.7, "orange"), (0.9, "red")])
    b.stat("Memory used", "host:memory_used:ratio", "percentunit", w=4, decimals=0, thresholds=[(None, "green"), (0.75, "orange"), (0.9, "red")])
    b.stat("Disk used, /", f"1 - node_filesystem_avail_bytes{{{root}}} / node_filesystem_size_bytes{{{root}}}", "percentunit", w=4, decimals=0,
           thresholds=[(None, "green"), (0.8, "orange"), (0.9, "red")])
    b.stat("Load, 1 min", "node_load1", "short", w=4, decimals=2)
    b.stat("Hottest sensor", "max(node_hwmon_temp_celsius)", "celsius", w=4, decimals=0, thresholds=[(None, "green"), (80, "orange"), (90, "red")])
    b.stat("Uptime", "time() - node_boot_time_seconds", "s", w=4, decimals=0)
    b.row("CPU")
    b.series("CPU by mode", [target('sum by (mode) (rate(node_cpu_seconds_total{mode!="idle"}[1m])) / scalar(count(node_cpu_seconds_total{mode="idle"}))', "{{mode}}")],
             "percentunit", w=12, stacked=True, max_value=1)
    b.series("CPU per core", [target('1 - rate(node_cpu_seconds_total{mode="idle"}[1m])', "cpu {{cpu}}")], "percentunit", w=12, max_value=1, fill=0)
    b.series("Load average", [target("node_load1", "1 min"), target("node_load5", "5 min"), target("node_load15", "15 min")], "short", w=12, decimals=2)
    b.series("CPU frequency", [target("avg(node_cpu_scaling_frequency_hertz)", "mean"), target("max(node_cpu_scaling_frequency_hertz)", "max")], "hertz", w=12, min_zero=False,
             description="A fall under load is a sign of thermal throttling, which affects timing results.")
    b.row("Memory")
    b.series("Memory", [target("node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes", "used"), target("node_memory_MemTotal_bytes", "total")], "bytes", w=12)
    b.series("Swap used", [target("node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes", "swap")], "bytes", w=12)
    b.row("Disk")
    b.series("Disk throughput", [target('rate(node_disk_read_bytes_total{device=~"nvme.+|sd.+"}[1m])', "read {{device}}"),
                                 target('rate(node_disk_written_bytes_total{device=~"nvme.+|sd.+"}[1m])', "write {{device}}")], "Bps", w=12)
    b.series("Filesystem used", [target('1 - node_filesystem_avail_bytes{fstype!~"tmpfs|overlay|squashfs"} / node_filesystem_size_bytes{fstype!~"tmpfs|overlay|squashfs"}', "{{mountpoint}}")],
             "percentunit", w=12, max_value=1)
    b.row("Network and temperature")
    b.series("Network", [target('rate(node_network_receive_bytes_total{device!~"lo|veth.*|br-.*|docker.*"}[1m])', "in {{device}}"),
                         target('rate(node_network_transmit_bytes_total{device!~"lo|veth.*|br-.*|docker.*"}[1m])', "out {{device}}")], "Bps", w=12)
    b.series("Temperatures", [target("max by (chip) (node_hwmon_temp_celsius)", "{{chip}}")], "celsius", w=12, min_zero=False)
    b.write("06-host.json")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for build in (overview, latency, workload, resources, containers, host):
        build()
    print("Wrote", len(list(OUT.glob("*.json"))), "dashboards to", OUT)
