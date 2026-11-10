"""Helpers for the QPUTwin notebooks: API client, example circuits and plots.

Every job in the notebooks runs in the Docker stack through the HTTP API. The one local
computation is :func:`native_plan`, which calls the same compiler the workers use so
that a compiled circuit can be drawn, because the API returns its statistics rather
than the circuit itself. :func:`draw_native` checks the two agree.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Iterable, Sequence
from typing import Any

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ClassicalRegister, Gate

API_URL = os.environ.get("QPUTWIN_URL", "http://localhost:8000")
PROMETHEUS_URL = os.environ.get("QPUTWIN_PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL = os.environ.get("QPUTWIN_GRAFANA_URL", "http://localhost:3000")
API_KEY = os.environ.get("QPUTWIN_API_KEY")

# Notebooks 00-08 compare the first four twins; notebook 09 covers the IBM and Google twins.
BACKENDS = ["quantinuum_helios", "ionq_forte", "rigetti_ankaa_3", "iqm_emerald"]
VENDOR_SNAPSHOT_TWINS = ["ibm_heron", "ibm_nighthawk", "google_willow"]
ALL_BACKENDS = BACKENDS + VENDOR_SNAPSHOT_TWINS
NAMES = {"quantinuum_helios": "Helios", "ionq_forte": "Forte", "rigetti_ankaa_3": "Ankaa-3", "iqm_emerald": "Emerald",
         "ibm_heron": "Heron r3", "ibm_nighthawk": "Nighthawk", "google_willow": "Willow"}
# One fixed colour per twin, in this order (checked for colour-blind separation; same hues as the Grafana boards).
COLORS = {"quantinuum_helios": "#2A78D6", "ionq_forte": "#EB6834", "rigetti_ankaa_3": "#1BAF7A", "iqm_emerald": "#EDA100",
          "ibm_heron": "#E87BA4", "ibm_nighthawk": "#008300", "google_willow": "#6250D6"}

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

_http = httpx.Client(base_url=API_URL, timeout=300.0,
                     headers={"authorization": f"Bearer {API_KEY}"} if API_KEY else None)


# ------------------------------------------------------------------------ API
def connect() -> dict[str, Any]:
    """Check that the stack answers, and say so clearly when it does not."""
    try:
        ready = _http.get("/ready").json()
    except httpx.ConnectError as exc:
        raise RuntimeError(f"QPUTwin is not reachable at {API_URL}. Start it from the project root with: "
                           "docker compose up -d --wait") from exc
    print(f"QPUTwin at {API_URL}: {ready['status']}, {len(ready['backends_reachable'])} twins reachable")
    return ready


def backends() -> pd.DataFrame:
    rows = _http.get("/v1/backends").json()["backends"]
    return pd.DataFrame(rows)[["id", "status", "vendor", "technology", "n_physical", "connectivity", "calibration"]]


def device(backend: str) -> dict[str, Any]:
    return _http.get(f"/v1/backends/{backend}").json()


def body(backend: str, qasm: str, *, shots: int = 1024, seed: int | None = 7, noise: Any = None, routing: Any = None,
         priority: int = 5, **options: Any) -> dict[str, Any]:
    """Build a ``POST /v1/jobs`` body. ``noise=True`` means the device noise model; ``routing=True`` means SABRE."""
    opts = dict(options)
    if noise is not None and noise is not False:
        opts["noise"] = {"model": "device"} if noise is True else noise
    if routing is not None and routing is not False:
        opts["routing"] = {"method": "sabre"} if routing is True else routing
    return {"backend": backend, "qasm": qasm, "shots": shots, "seed": seed, "priority": priority, "options": opts}


def run(backend: str, qasm: str, **kwargs: Any) -> dict[str, Any]:
    """Submit one synchronous job and return the JSON response, including any error body."""
    response = _http.post("/v1/jobs", json=body(backend, qasm, **kwargs))
    payload = response.json()
    payload["http_status"] = response.status_code
    return payload


def submit(backend: str, qasm: str, client: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Submit one asynchronous job. The server answers 202 at once. ``client`` names the fair-share client."""
    headers = {"x-qputwin-client": client} if client else None
    response = _http.post("/v1/jobs", params={"mode": "async"}, json=body(backend, qasm, **kwargs), headers=headers)
    response.raise_for_status()
    return response.json()


def job(job_id: str, wait: float = 0.0) -> dict[str, Any]:
    return _http.get(f"/v1/jobs/{job_id}", params={"wait": wait}).json()


def queues() -> dict[str, dict[str, int]]:
    """Queued and running asynchronous jobs per twin."""
    return _http.get("/v1/jobs", params={"limit": 1}).json()["queues"]


def cancel(job_id: str) -> dict[str, Any]:
    return _http.delete(f"/v1/jobs/{job_id}").json()


def recent_telemetry(limit: int = 100, backend: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if backend:
        params["backend"] = backend
    return _http.get("/v1/telemetry", params=params).json()["records"]


def metrics_text(prefix: str = "qputwin_jobs_total") -> str:
    lines = _http.get("/metrics").text.splitlines()
    return "\n".join(line for line in lines if line.startswith(prefix))


def prometheus(query: str) -> pd.DataFrame | None:
    """Instant Prometheus query, or ``None`` when the monitoring stack is not running."""
    try:
        data = httpx.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=5).json()["data"]["result"]
    except httpx.HTTPError:
        return None
    return pd.DataFrame([{**s["metric"], "value": float(s["value"][1])} for s in data])


def prometheus_range(query: str, seconds: float, step: float = 2.0) -> pd.DataFrame | None:
    """The last ``seconds`` of a Prometheus query, one row per sample."""
    end = time.time()
    try:
        data = httpx.get(f"{PROMETHEUS_URL}/api/v1/query_range", timeout=10, params={
            "query": query, "start": end - seconds, "end": end, "step": step}).json()["data"]["result"]
    except httpx.HTTPError:
        return None
    return pd.DataFrame([{**s["metric"], "t": float(t) - end, "value": float(v)}
                         for s in data for t, v in s["values"]])


# ------------------------------------------------------------------- circuits
HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _qasm(n: int, lines: Iterable[str], measure: bool = True) -> str:
    text = HEADER + f"qreg q[{n}];\ncreg c[{n}];\n" + "".join(f"{line};\n" for line in lines)
    return text + "measure q -> c;\n" if measure else text


def program(n: int, lines: Iterable[str], measure: bool = True) -> str:
    """An OpenQASM 2 program on ``n`` qubits from gate lines (without semicolons), measuring all qubits by default."""
    return _qasm(n, lines, measure)


def bell() -> str:
    return _qasm(2, ["h q[0]", "cx q[0],q[1]"])


def ghz(n: int) -> str:
    """GHZ on a chain: embeds in every twin without a SWAP."""
    return _qasm(n, ["h q[0]", *(f"cx q[{i}],q[{i + 1}]" for i in range(n - 1))])


def star_ghz(n: int) -> str:
    """GHZ on a star: every gate touches qubit 0, so a lattice cannot embed it above degree 4."""
    return _qasm(n, ["h q[0]", *(f"cx q[0],q[{i}]" for i in range(1, n))])


def bernstein_vazirani(secret: str) -> str:
    """Recovers ``secret`` in one query. Only the data qubits are measured."""
    n = len(secret)
    lines = [f"x q[{n}]", *(f"h q[{i}]" for i in range(n + 1))]
    lines += [f"cx q[{i}],q[{n}]" for i, bit in enumerate(reversed(secret)) if bit == "1"]
    lines += [f"h q[{i}]" for i in range(n)]
    return (HEADER + f"qreg q[{n + 1}];\ncreg c[{n}];\n" + "".join(f"{line};\n" for line in lines)
            + "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(n)))


def grover_3(marked: str = "101") -> str:
    """Two Grover iterations on three qubits, marking one bitstring (written q2 q1 q0)."""
    flips = [f"x q[{i}]" for i, bit in enumerate(reversed(marked)) if bit == "0"]
    oracle = [*flips, "h q[2]", "ccx q[0],q[1],q[2]", "h q[2]", *flips]
    hs = [f"h q[{i}]" for i in range(3)]
    xs = [f"x q[{i}]" for i in range(3)]
    diffuser = [*hs, *xs, "h q[2]", "ccx q[0],q[1],q[2]", "h q[2]", *xs, *hs]
    return _qasm(3, [*hs, *oracle, *diffuser, *oracle, *diffuser])


def qft_roundtrip(n: int, value: int) -> str:
    """Prepares ``value``, then applies QFT and inverse QFT: ideally ``value`` comes back exactly."""
    prepare = [f"x q[{i}]" for i in range(n) if value >> i & 1]
    forward: list[str] = []
    for i in reversed(range(n)):
        forward.append(f"h q[{i}]")
        forward += [f"cu1(pi/{2 ** (i - j)}) q[{j}],q[{i}]" for j in reversed(range(i))]
    backward: list[str] = []
    for i in range(n):
        backward += [f"cu1(-pi/{2 ** (i - j)}) q[{j}],q[{i}]" for j in range(i)]
        backward.append(f"h q[{i}]")
    return _qasm(n, [*prepare, *forward, "barrier q", *backward])


def qaoa_ring(n: int, gamma: float, beta: float) -> str:
    """One QAOA layer for MaxCut on a ring of ``n`` nodes."""
    lines = [f"h q[{i}]" for i in range(n)]
    lines += [f"rzz({2 * gamma:.8f}) q[{i}],q[{(i + 1) % n}]" for i in range(n)]
    lines += [f"rx({2 * beta:.8f}) q[{i}]" for i in range(n)]
    return _qasm(n, lines)


def teleportation(theta: float = 1.1) -> str:
    """Teleports Ry(theta)|0⟩ from q0 to q2 with feed-forward on two mid-circuit measurements."""
    return (HEADER + "qreg q[3];\ncreg a[1];\ncreg b[1];\ncreg out[1];\n"
            f"ry({theta}) q[0];\nh q[1];\ncx q[1],q[2];\ncx q[0],q[1];\nh q[0];\n"
            "measure q[0] -> a[0];\nmeasure q[1] -> b[0];\n"
            "if(b==1) x q[2];\nif(a==1) z q[2];\nmeasure q[2] -> out[0];\n")


def mirror(n: int, layers: int, seed: int = 1) -> str:
    """Random layers followed by their inverse: an ideal device returns all zeros."""
    rng = random.Random(seed)
    forward: list[tuple[str, ...]] = []
    for _ in range(layers):
        forward += [("ry", f"{rng.uniform(0, math.pi):.6f}", str(q)) for q in range(n)]
        order = list(range(n))
        rng.shuffle(order)
        forward += [("cx", str(a), str(b)) for a, b in zip(order[::2], order[1::2], strict=False)]

    def text(op: tuple[str, ...], invert: bool) -> str:
        if op[0] == "cx":
            return f"cx q[{op[1]}],q[{op[2]}]"
        return f"ry({'-' if invert else ''}{op[1]}) q[{op[2]}]"

    return _qasm(n, [*(text(op, False) for op in forward), "barrier q", *(text(op, True) for op in reversed(forward))])


def scrambler(n: int, layers: int, seed: int = 1) -> str:
    """Random layers with no inverse: the outcome is spread widely, and costly to simulate."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(layers):
        lines += [f"ry({rng.uniform(0, math.pi):.6f}) q[{q}]" for q in range(n)]
        order = list(range(n))
        rng.shuffle(order)
        lines += [f"cx q[{a}],q[{b}]" for a, b in zip(order[::2], order[1::2], strict=False)]
    return _qasm(n, lines)


# ------------------------------------------------------------------- analysis
def hellinger(p: dict[str, float], q: dict[str, float]) -> float:
    """Hellinger fidelity of two distributions or count dictionaries."""
    sp, sq = sum(p.values()), sum(q.values())
    return sum(math.sqrt(p[k] / sp * q.get(k, 0) / sq) for k in p) ** 2


def ideal_counts(qasm: str, backend: str = "quantinuum_helios", shots: int = 200_000) -> dict[str, int]:
    """A reference distribution from the ideal engine of an all-to-all twin."""
    return run(backend, qasm, shots=shots)["result"]["counts"]


def success(result: dict[str, Any], outcomes: Sequence[str]) -> float:
    """Fraction of shots that landed on ``outcomes``. Raises with the server's message if the job was rejected."""
    if "result" not in result:
        error = result.get("error", {})
        raise RuntimeError(f"job rejected ({result.get('http_status')}): {error.get('type')}: {error.get('message')}")
    counts = result["result"]["counts"]
    return sum(counts.get(k, 0) for k in outcomes) / sum(counts.values())


# ---------------------------------------------------------------------- plots
def draw(qasm: str, **kwargs: Any) -> Any:
    """Draw an OpenQASM 2 program with Qiskit's circuit drawer."""
    return QuantumCircuit.from_qasm_str(qasm).draw("mpl", fold=kwargs.pop("fold", 40), **kwargs)


def native_plan(backend: str, qasm: str, routing: bool = False) -> Any:
    """Compile locally with the same code the worker runs. Used only to draw the result."""
    from qputwin_core.devices import get_device
    from qputwin_core.simulators import ExecutionOptions, SuperconductingSimulator, TrappedIonSimulator
    from qputwin_core.timing import SpanTimer

    spec = get_device(backend)
    simulator = (TrappedIonSimulator if spec.technology == "trapped_ion" else SuperconductingSimulator)(spec)
    options = ExecutionOptions(shots=1, routing={"method": "sabre"} if routing else None)
    return simulator.plan(qasm, options, SpanTimer())


def native_circuit(backend: str, qasm: str, routing: bool = False) -> QuantumCircuit:
    """The compiled circuit as a Qiskit object, labelled with physical qubit numbers."""
    native = native_plan(backend, qasm, routing).native
    used = sorted({q for op in native.ops for q in op.qubits})
    registers = [QuantumRegister(1, f"p{q}") for q in used]
    wire = {q: registers[i][0] for i, q in enumerate(used)}
    clbits = ClassicalRegister(native.num_clbits, "c")
    circuit = QuantumCircuit(*registers, clbits)
    for op in native.ops:
        qubits = [wire[q] for q in op.qubits]
        if op.name == "measure":
            circuit.measure(qubits[0], clbits[op.clbits[0]])
        elif op.name == "cz":
            circuit.cz(*qubits)
        elif op.name == "barrier":
            circuit.barrier(*qubits)
        else:
            params = [round(float(p), 3) for p in op.params]
            circuit.append(Gate(op.name, len(qubits), params), qubits)
    return circuit


def draw_native(result: dict[str, Any], qasm: str, routing: bool = False, **kwargs: Any) -> Any:
    """Draw the circuit a twin executed, after checking it matches the job's API response."""
    backend = result["backend"]
    native = native_plan(backend, qasm, routing).native
    local: dict[str, int] = {}
    for op in native.ops:
        if op.is_gate:
            local[op.name] = local.get(op.name, 0) + 1
    remote = result["compilation"]["native_gate_histogram"]
    if local != remote:
        raise AssertionError(f"local compile {local} differs from the job's {remote}")
    return native_circuit(backend, qasm, routing).draw("mpl", fold=kwargs.pop("fold", 30), **kwargs)


def plot_counts(counts: dict[str, int], title: str = "", top: int = 16, ax: Any = None, color: str = "#2E5AA8") -> Any:
    ax = ax or plt.subplots(figsize=(7, 3))[1]
    total = sum(counts.values())
    items = sorted(sorted(counts.items()), key=lambda kv: -kv[1])[:top]
    items.sort()
    ax.bar([k for k, _ in items], [v / total for _, v in items], color=color)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90 if len(items) > 8 else 0)
    ax.grid(axis="x", visible=False)
    return ax


def plot_each_twin(results: dict[str, dict[str, Any]], top: int = 12) -> None:
    """One histogram per twin, four to a row."""
    cols = min(4, len(results))
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows), sharey=True, squeeze=False)
    for ax, (backend, result) in zip(axes.flat, results.items(), strict=False):
        plot_counts(result["result"]["counts"], NAMES[backend], top=top, ax=ax, color=COLORS[backend])
    for ax in list(axes.flat)[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def plot_topology(backend: str, ax: Any = None, highlight: Iterable[int] = (), arrows: Sequence[tuple[int, int]] = ()) -> Any:
    """Draw a coupling map: the chip's own positions where published, grid positions for a lattice, a circle for all-to-all."""
    info = device(backend)
    cmap = info["coupling_map"]
    n = info["n_physical"]
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    layout = cmap.get("layout") or {}
    if "coordinates" in layout:
        pos = {q: (c, -r) for q, (r, c) in enumerate(layout["coordinates"])}
    elif "rows" in layout:
        cols = cmap["layout"]["cols"]
        pos = {q: (q % cols, -(q // cols)) for q in range(n)}
    else:
        pos = {q: (math.cos(2 * math.pi * q / n), math.sin(2 * math.pi * q / n)) for q in range(n)}
    if "edges" in cmap:
        edges = [tuple(e) for e in cmap["edges"]]
    else:
        edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    dense = len(edges) > 500
    ax.add_collection(LineCollection([(pos[a], pos[b]) for a, b in edges], colors="#9AA3B2",
                                     linewidths=0.2 if dense else 1.2, alpha=0.5 if dense else 1, zorder=1))
    picked = set(highlight)
    ax.scatter(*zip(*pos.values(), strict=True), s=14 if n > 60 else 40, zorder=2,
               color=[COLORS[backend] if q in picked else "#2D3142" for q in range(n)])
    for q in picked:
        ax.scatter(*pos[q], s=130, color=COLORS[backend], zorder=3)
        ax.annotate(str(q), pos[q], ha="center", va="center", color="white", fontsize=7, zorder=4)
    for a, b in arrows:
        ax.plot(*zip(pos[a], pos[b], strict=True), color=COLORS[backend], linewidth=2.5, zorder=2)
    ax.set_title(f"{NAMES[backend]}: {n} qubits, {len(edges):,} couplers")
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# Leaf spans of the telemetry breakdown, in request order. They sum to the round trip.
SPANS = {
    "ingress_ms": "ingress", "validation_ms": "validation", "routing_ms": "backend selection",
    "orchestrator_queue_ms": "orchestrator queue", "network_transit_ms": "network",
    "worker_queue_ms": "worker queue", "worker_parse_ms": "parse", "worker_route_ms": "SABRE routing",
    "worker_compile_ms": "native compile", "simulation_ms": "simulation", "sampling_ms": "sampling",
    "worker_unattributed_ms": "worker other", "response_assembly_ms": "response", "unattributed_ms": "orchestrator other",
}


def span_table(telemetry: dict[str, Any]) -> pd.DataFrame:
    """The latency breakdown of one job, with its additive check."""
    b = telemetry["breakdown"]
    frame = pd.DataFrame({"span": list(SPANS.values()), "ms": [round(b[k], 3) for k in SPANS]})
    frame["share %"] = (100 * frame["ms"] / telemetry["total_roundtrip_ms"]).round(1)
    return frame.set_index("span")


def plot_spans(rows: dict[str, dict[str, float]], title: str = "Where the time goes") -> None:
    """Stacked horizontal bars of mean span durations, one bar per label."""
    fig, ax = plt.subplots(figsize=(10, 0.65 * len(rows) + 2.2))
    labels = list(rows)
    left = np.zeros(len(labels))
    palette = plt.colormaps["tab20"](np.linspace(0, 1, len(SPANS)))
    small = np.zeros(len(labels))            # spans under 1 % of the largest bar, drawn together at the end
    largest = max(1e-9, max(sum(rows[label].get(k, 0.0) for k in SPANS) for label in labels))
    for color, (key, name) in zip(palette, SPANS.items(), strict=True):
        values = np.array([rows[label].get(key, 0.0) for label in labels])
        if values.max() < 0.01 * largest:
            small += values
            continue
        ax.barh(labels, values, left=left, label=name, color=color)
        left += values
    if small.any():
        ax.barh(labels, small, left=left, label="other small spans", color="#BBBBBB")
    ax.invert_yaxis()
    ax.set_xlabel("Milliseconds")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.legend(ncols=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    plt.show()
