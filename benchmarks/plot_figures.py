"""Publication figures from ``benchmarks/results/*.json``.

Every figure is written as vector PDF + SVG and a 300-dpi PNG to ``benchmarks/figures/``,
together with a CSV of exactly the values plotted (``benchmarks/figures/data/``) so each
chart has a table view. Figures are sized for a two-column paper (3.5 in single, 7.2 in
double column) and copied into the documentation site.

Colour: the seven devices take the first seven slots of the validated categorical palette
in fixed order (adjacent-pair CVD ΔE ≥ 9.1, normal-vision ΔE ≥ 19.6 on white); every series
also has its own marker and line style, so identity never relies on colour alone.

    uv run --group bench python benchmarks/plot_figures.py

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "benchmarks" / "figures"
DATA = OUT / "data"
DOCS = ROOT / "docs" / "assets" / "figures"

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#1f2328", "#52514e", "#8a8984", "#e3e2dd"
DEVICES = ["quantinuum_helios", "ionq_forte", "rigetti_ankaa_3", "iqm_emerald", "ibm_heron", "ibm_nighthawk",
           "google_willow"]
LABEL = {"quantinuum_helios": "Quantinuum Helios", "ionq_forte": "IonQ Forte",
         "rigetti_ankaa_3": "Rigetti Ankaa-3", "iqm_emerald": "IQM Emerald", "ibm_heron": "IBM Heron r3",
         "ibm_nighthawk": "IBM Nighthawk", "google_willow": "Google Willow"}
SHORT = {"quantinuum_helios": "Helios", "ionq_forte": "Forte", "rigetti_ankaa_3": "Ankaa-3", "iqm_emerald": "Emerald",
         "ibm_heron": "Heron r3", "ibm_nighthawk": "Nighthawk", "google_willow": "Willow"}
COLOR = dict(zip(DEVICES, SLOTS, strict=False))
MARK = {"quantinuum_helios": "o", "ionq_forte": "s", "rigetti_ankaa_3": "^", "iqm_emerald": "D", "ibm_heron": "v",
        "ibm_nighthawk": "P", "google_willow": "X"}
STYLE = {"quantinuum_helios": "-", "ionq_forte": "--", "rigetti_ankaa_3": "-", "iqm_emerald": "--", "ibm_heron": "-.",
         "ibm_nighthawk": ":", "google_willow": "-."}
LATTICES = [d for d in DEVICES if d not in ("quantinuum_helios", "ionq_forte")]
SINGLE, DOUBLE = 3.5, 7.2

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 5,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150,
})


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / f"{name}.json").read_text())


def save(fig: plt.Figure, name: str, rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    shutil.copy(OUT / f"{name}.svg", DOCS / f"{name}.svg")
    shutil.copy(OUT / f"{name}.png", DOCS / f"{name}.png")
    if rows:
        with (DATA / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))   # tables can mix row kinds
            w.writeheader()
            w.writerows(rows)
    plt.close(fig)
    print(f"  {name}")


def ring(ax: plt.Axes, x, y, color: str, marker: str, **kw) -> None:
    """Line with markers that carry a 2-px surface ring so overlaps stay legible."""
    ax.plot(x, y, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.9, **kw)


def end_label(ax: plt.Axes, x: float, y: float, text: str, dy: float = 0.0) -> None:
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", fontsize=6.8, color=INK2, va="center")


# ================================================================ figures
def fig_latency() -> None:
    rows = load("latency")["data"]
    comps = [
        ("Orchestrator (ingress, validation, routing, assembly)",
         lambda r: r["ingress_ms"] + r["validation_ms"] + r["routing_ms"] + r["response_assembly_ms"]
         + r["unattributed_ms"] + r["orchestrator_queue_ms"]),
        ("Network transit", lambda r: r["network_transit_ms"]),
        ("Worker parse + compile + overhead",
         lambda r: r["worker_queue_ms"] + r["worker_parse_ms"] + r["worker_route_ms"] + r["worker_compile_ms"]
         + r["worker_unattributed_ms"]),
        ("Engine: simulation", lambda r: r["simulation_ms"]),
        ("Engine: sampling", lambda r: r["sampling_ms"]),
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE, 0.42 * len(DEVICES) + 1.0))
    table = []
    ys = np.arange(len(DEVICES))[::-1]
    for y, dev in zip(ys, DEVICES, strict=True):
        rs = [r for r in rows if r["backend"] == dev]
        left = 0.0
        med = {name: statistics.median(f(r) for r in rs) for name, f in comps}
        for k, (name, _) in enumerate(comps):
            ax.barh(y, med[name], left=left, height=0.56, color=SLOTS[k], edgecolor="white", linewidth=1.2,
                    label=name if y == ys[0] else None)
            left += med[name]
        total = statistics.median(r["total_roundtrip_ms"] for r in rs)
        p95 = float(np.percentile([r["total_roundtrip_ms"] for r in rs], 95))
        ax.annotate(f"{total:.2f} ms  (p95 {p95:.2f})", (left, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
        table.append({"backend": dev, **{n: round(v, 4) for n, v in med.items()}, "total_median_ms": round(total, 4),
                      "total_p95_ms": round(p95, 4), "n": len(rs)})
    ax.set_yticks(ys, [LABEL[d] for d in DEVICES])
    ax.set_xlabel("Median latency per job (ms) — stacked components sum to the server round trip")
    ax.set_xlim(0, max(t["total_median_ms"] for t in table) * 1.35)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.2, columnspacing=1.2)
    ax.set_title("Latency decomposition of a 12-qubit GHZ job (1024 shots, n = 100 per backend)", loc="left")
    save(fig, "fig01_latency_decomposition", table)


def fig_scaling() -> None:
    rows = load("scaling")["data"]
    series = [("statevector_c128", "Statevector (complex128)", "o", "-"),
              ("statevector_c64", "Statevector (complex64)", "s", "--"),
              ("density_matrix", "Density matrix (exact noise)", "^", "-"),
              ("trajectory_T32", "Trajectories (T = 32, noisy)", "D", "--")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    table = []
    for k, (key, name, m, ls) in enumerate(series):
        rs = sorted((r for r in rows if r["engine"] == key), key=lambda r: r["qubits"])
        x, y = [r["qubits"] for r in rs], [r["qpu_ms_median"] for r in rs]
        ring(ax, x, y, SLOTS[k], m, linestyle=ls, label=name)
        table += [{"engine": key, "qubits": r["qubits"], "qpu_ms_median": round(r["qpu_ms_median"], 5)} for r in rs]
    # reference slopes anchored at the largest statevector / density-matrix points
    sv = sorted((r for r in rows if r["engine"] == "statevector_c128"), key=lambda r: r["qubits"])[-1]
    dm = sorted((r for r in rows if r["engine"] == "density_matrix"), key=lambda r: r["qubits"])[-1]
    xs = np.arange(14, sv["qubits"] + 1)
    ax.plot(xs, sv["qpu_ms_median"] * 2.0 ** (xs - sv["qubits"]), color=MUTED, lw=0.8, ls=":", zorder=0)
    ax.annotate(r"$\propto 2^n$", (xs[0], sv["qpu_ms_median"] * 2.0 ** (xs[0] - sv["qubits"])), xytext=(-4, 3),
                textcoords="offset points", fontsize=7, color=INK2, ha="right", va="bottom")
    xd = np.arange(6, dm["qubits"] + 1)
    ax.plot(xd, dm["qpu_ms_median"] * 4.0 ** (xd - dm["qubits"]), color=MUTED, lw=0.8, ls=":", zorder=0)
    xl = xd[len(xd) // 2]
    ax.annotate(r"$\propto 4^n$", (xl, dm["qpu_ms_median"] * 4.0 ** (xl - dm["qubits"])), xytext=(-6, 0),
                textcoords="offset points", fontsize=7, color=INK2, ha="right", va="center")
    ax.set_yscale("log")
    ax.set_xlabel("Simulated qubits n")
    ax.set_ylabel("Engine time per job (ms)")
    ax.set_title("Engine cost vs width (GHZ chain, 1024 shots)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, columnspacing=1.0)
    save(fig, "fig02_engine_scaling", table)


def fig_routing() -> None:
    rows = load("routing")["data"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    table = []
    for dev in DEVICES:
        rs = sorted((r for r in rows if r["backend"] == dev), key=lambda r: r["qubits"])
        x = [r["qubits"] for r in rs]
        ring(axes[0], x, [r["native_two_qubit"] for r in rs], COLOR[dev], MARK[dev], linestyle=STYLE[dev],
             label=LABEL[dev] + (" (routed)" if not dev.startswith(("quantinuum", "ionq")) else " (native)"))
        ring(axes[1], x, [r["esp"] for r in rs], COLOR[dev], MARK[dev], linestyle=STYLE[dev], label=LABEL[dev])
        table += [{"backend": dev, "qubits": r["qubits"], "native_two_qubit": r["native_two_qubit"],
                   "swaps": r["swaps"], "cancelled": r["cancelled"], "two_qubit_depth": r["two_qubit_depth"],
                   "esp": round(r["esp"], 5), "ghz_success": round(r.get("ghz_success", float("nan")), 5)}
                  for r in rs]
    axes[0].annotate("Helios and Forte coincide: k − 1 entanglers", (16, 3), xytext=(0, 0),
                     textcoords="offset points", fontsize=6.8, color=INK2, ha="right")
    axes[0].set_xlabel("Star-GHZ width k (hub = qubit 0)")
    axes[0].set_ylabel("Native two-qubit gates")
    axes[0].set_title("(a) Entangler count: all-to-all vs SABRE-routed lattice", loc="left")
    axes[1].set_xlabel("Star-GHZ width k")
    axes[1].set_ylabel("Estimated success probability")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("(b) Predicted success under calibrated noise", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=2.5)
    save(fig, "fig03_routing_cost", table)


def fig_routing_latency() -> None:
    rows = load("routing_latency")["data"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    table = []
    for k, (mode, name, m) in enumerate((("reject", "Rejected (TopologicalCompilationError)", "o"),
                                         ("route", "Routed with SABRE", "s"))):
        rs = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["qubits"])
        x = [r["qubits"] for r in rs]
        med = [r["total_ms_median"] for r in rs]
        lo = [float(np.percentile(r["total_ms"], 25)) for r in rs]
        hi = [float(np.percentile(r["total_ms"], 75)) for r in rs]
        ax.fill_between(x, lo, hi, color=SLOTS[k], alpha=0.12, linewidth=0)
        ring(ax, x, med, SLOTS[k], m, label=name)
        table += [{"mode": mode, "qubits": r["qubits"], "total_ms_median": round(r["total_ms_median"], 4),
                   "route_ms_median": round(r["route_ms_median"], 4)} for r in rs]
    ax.set_xlabel("Star-GHZ width k")
    ax.set_ylabel("Server round trip (ms, median, IQR band)")
    ax.set_ylim(bottom=0)
    ax.set_title("Cost of routing vs rejection (IQM Emerald, live)", loc="left")
    ax.legend(loc="upper left")
    save(fig, "fig03b_routing_latency", table)


def fig_fidelity() -> None:
    rows = load("fidelity")["data"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    table = []
    for dev in DEVICES:
        rs = sorted((r for r in rows if r["backend"] == dev), key=lambda r: r["qubits"])
        x = [r["qubits"] for r in rs]
        ring(ax, x, [r["ghz_success"] for r in rs], COLOR[dev], MARK[dev], linestyle=STYLE[dev], label=LABEL[dev])
        ax.plot(x, [r["esp"] for r in rs], color=COLOR[dev], lw=0.8, ls=":", alpha=0.9)
        end_label(ax, x[-1], rs[-1]["ghz_success"], f"{rs[-1]['ghz_success']:.2f}")
        table += [{"backend": dev, "qubits": r["qubits"], "ghz_success": round(r["ghz_success"], 5),
                   "esp": round(r["esp"], 5), "purity": round(r["purity"], 5)} for r in rs]
    ax.plot([], [], color=MUTED, lw=0.8, ls=":", label="dotted: ESP prediction")
    ax.set_xlabel("GHZ width n (native nearest-neighbour path)")
    ax.set_ylabel("P(0…0) + P(1…1)")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(1.6, 11.2)
    ax.set_title("GHZ success under calibrated noise (exact)", loc="left")
    ax.legend(loc="lower left")
    save(fig, "fig04_ghz_fidelity", table)


def fig_hwtime() -> None:
    rows = load("hwtime")["data"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    table = []
    for dev in DEVICES:
        rs = sorted((r for r in rows if r["backend"] == dev), key=lambda r: r["qubits"])
        x = [r["qubits"] for r in rs]
        y = [r["circuit_duration_us"] for r in rs]
        ring(ax, x, y, COLOR[dev], MARK[dev], linestyle=STYLE[dev], label=LABEL[dev], markevery=2)
        table += [{"backend": dev, "qubits": r["qubits"], "circuit_duration_us": round(r["circuit_duration_us"], 4),
                   "job_duration_s_1000_shots": round(r["job_duration_s_1000_shots"], 6)} for r in rs]
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: (f"{v / 1000:g} ms" if v >= 1000 else f"{v:g} µs")))
    ax.set_xlabel("GHZ width n")
    ax.set_ylabel("Estimated hardware time per shot")
    ax.set_title("Gate-duration model (ASAP schedule)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    save(fig, "fig05_hardware_time", table)


def fig_budget() -> None:
    rows = load("budget")["data"]
    conds = ["gate depolarising", "T1/T2 relaxation", "readout", "all channels"]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.7))
    width = 0.19
    xs = np.arange(len(DEVICES))
    table = []
    for k, c in enumerate(conds):
        vals = [next(r["infidelity"] for r in rows if r["backend"] == d and r["condition"] == c) for d in DEVICES]
        ax.bar(xs + (k - 1.5) * width, vals, width=width, color=SLOTS[k], edgecolor="white", linewidth=0.5,
               label=c[0].upper() + c[1:])
        table += [{"backend": d, "condition": c, "infidelity": round(v, 5)} for d, v in zip(DEVICES, vals, strict=True)]
        if c == "all channels":
            for x, v in zip(xs, vals, strict=True):
                ax.annotate(f"{v:.3f}", (x + (k - 1.5) * width, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(xs, [SHORT[d] for d in DEVICES])
    ax.set_ylabel("GHZ₆ infidelity  1 − P(success)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Error budget: one noise channel at a time (GHZ₆, exact)", loc="left")
    ax.legend(ncol=2, loc="upper left")
    save(fig, "fig06_error_budget", table)


def fig_convergence() -> None:
    doc = load("convergence")
    rows = doc["data"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    t = np.array([r["trajectories"] for r in rows])
    m = np.array([r["tv_mean"] for r in rows])
    s = np.array([r["tv_std"] for r in rows])
    ax.axhspan(0, doc["shot_noise_floor_p95"], color=GRID, alpha=0.6, linewidth=0)
    ax.axhline(doc["shot_noise_floor_mean"], color=MUTED, lw=0.8)
    ax.annotate(f"shot-noise floor ({doc['shots']:,} shots)", (t[0], doc["shot_noise_floor_mean"]),
                xytext=(0, -9), textcoords="offset points", fontsize=6.8, color=INK2)
    ax.errorbar(t, m, yerr=s, color=SLOTS[0], marker="o", markeredgecolor="white", markeredgewidth=0.9,
                capsize=2, elinewidth=0.8)
    ref = m[0] * np.sqrt(t[0] / t)
    ax.plot(t, ref, color=MUTED, lw=0.8, ls=":")
    ax.annotate(r"$\propto T^{-1/2}$", (t[5], ref[5]), xytext=(-4, -4), textcoords="offset points",
                fontsize=7, color=INK2, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trajectories T (shots fixed)")
    ax.set_ylabel("Total-variation distance to exact")
    ax.set_title("Trajectory engine convergence (GHZ₄, IQM Emerald, noise ×3)", loc="left")
    save(fig, "fig07_trajectory_convergence",
         [{"trajectories": int(a), "tv_mean": round(float(b), 6), "tv_std": round(float(c), 6)} for a, b, c in
          zip(t, m, s, strict=True)])


def fig_async() -> None:
    rows = load("async_scaling")["data"]
    widths = sorted({r["qubits"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    table = []
    for k, n in enumerate(widths):
        rs = sorted((r for r in rows if r["qubits"] == n), key=lambda r: r["replicas"])
        x = [r["replicas"] for r in rs]
        base = rs[0]["throughput_jobs_s"]
        w0 = statistics.median(rs[0]["worker_service_ms"])
        name = f"{n} qubits ({'256 KiB' if n == 14 else '16 MiB'} state)"
        ring(axes[0], x, [r["throughput_jobs_s"] / base for r in rs], SLOTS[k], "os"[k], label=name)
        if all("throughput_runs" in r for r in rs):                # range over the repetitions
            lo = [r["throughput_jobs_s"] / base - min(r["throughput_runs"]) / base for r in rs]
            hi = [max(r["throughput_runs"]) / base - r["throughput_jobs_s"] / base for r in rs]
            axes[0].errorbar(x, [r["throughput_jobs_s"] / base for r in rs], yerr=[lo, hi], fmt="none",
                             ecolor=SLOTS[k], elinewidth=1.0, capsize=2.5, zorder=2)
        ring(axes[1], x, [statistics.median(r["worker_service_ms"]) / w0 for r in rs], SLOTS[k], "os"[k], label=name)
        for r in rs:
            table.append({"qubits": n, "replicas": r["replicas"], "throughput_jobs_s": round(r["throughput_jobs_s"], 4),
                          "speedup": round(r["throughput_jobs_s"] / base, 4),
                          "throughput_runs": ";".join(f"{v:.3f}" for v in r.get("throughput_runs", [])),
                          "worker_service_ms_median": round(statistics.median(r["worker_service_ms"]), 3),
                          "workers_used": r["workers_used"]})
        end_label(axes[0], x[-1], rs[-1]["throughput_jobs_s"] / base, f"{rs[-1]['throughput_jobs_s']:.1f} jobs/s")
    xs = [1, 2, 3]
    axes[0].plot(xs, xs, color=MUTED, lw=0.8, ls=":")
    axes[0].text(2.25, 2.05, "ideal (linear)", fontsize=6.4, color=INK2, rotation=0, ha="left", va="top")
    axes[0].set_xticks(xs)
    axes[0].set_xlabel("Worker replicas (one host)")
    axes[0].set_ylabel("Throughput speed-up")
    axes[0].set_xlim(0.8, 3.6)
    axes[0].set_title("(a) Async throughput vs replicas", loc="left")
    axes[0].legend(loc="upper left")
    axes[1].set_xticks(xs)
    axes[1].set_xlabel("Worker replicas (one host)")
    axes[1].set_ylabel("Per-job worker time (× 1 replica)")
    axes[1].set_title("(b) Per-job worker time", loc="left")
    axes[1].axhline(1.0, color=MUTED, lw=0.8, ls=":")
    lo1, hi1 = axes[1].get_ylim()
    axes[1].set_ylim(min(lo1, 0.9), max(hi1, 1.1))
    axes[1].legend(loc="upper center")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig08_async_scaling", table)


def fig_noise_sweep() -> None:
    rows = load("noise_sweep")["data"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    table = []
    for dev in DEVICES:
        rs = sorted((r for r in rows if r["backend"] == dev), key=lambda r: r["scale"])
        ring(ax, [r["scale"] for r in rs], [r["ghz_success"] for r in rs], COLOR[dev], MARK[dev],
             linestyle=STYLE[dev], label=LABEL[dev])
        table += [{"backend": dev, "scale": r["scale"], "ghz_success": round(r["ghz_success"], 5)} for r in rs]
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.annotate("published calibration", (1.0, 0.37), xytext=(4, 0), textcoords="offset points", fontsize=6.8,
                color=INK2, va="bottom")
    ax.set_xlabel("Noise scale (× every error and decay rate)")
    ax.set_ylabel("GHZ₅ success probability")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("Sensitivity to noise strength", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    save(fig, "fig09_noise_sweep", table)


def fig_ramsey() -> None:
    doc = load("ramsey")
    rows = doc["data"]
    t2 = doc["t2_us"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    tt = np.linspace(0, max(r["delay_us"] for r in rows), 200)
    ax.plot(tt, np.exp(-tt / t2), color=SLOTS[0], lw=0.9, alpha=0.8)
    ax.plot(tt, np.exp(-((tt / t2) ** 2)), color=SLOTS[1], lw=0.9, alpha=0.8, ls="--")
    table = []
    for k, (model, name, m) in enumerate((("markovian", "Markovian T₂ (exp. decay)", "o"),
                                          ("quasi_static", "Quasi-static detuning (Gaussian)", "s"))):
        rs = sorted((r for r in rows if r["model"] == model), key=lambda r: r["delay_us"])
        ax.plot([r["delay_us"] for r in rs], [r["contrast"] for r in rs], linestyle="none", marker=m,
                color=SLOTS[k], markeredgecolor="white", markeredgewidth=0.9, label=name)
        table += [{"model": model, "delay_us": r["delay_us"], "contrast": round(r["contrast"], 5)} for r in rs]
    ax.plot([], [], color=MUTED, lw=0.9, label="lines: analytic e^(−t/T₂), e^(−(t/T₂)²)")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Ramsey delay t (µs)")
    ax.set_ylabel("Ramsey contrast  1 − 2·P(1)")
    ax.set_title(f"Markovian vs non-Markovian dephasing (T₂ = {t2:g} µs)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    save(fig, "fig10_ramsey", table)


def fig_latency_ecdf() -> None:
    rows = load("latency")["data"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    table = []
    for dev in DEVICES:
        xs = np.sort([r["total_roundtrip_ms"] for r in rows if r["backend"] == dev])
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=COLOR[dev], linestyle=STYLE[dev], label=LABEL[dev])
        table.append({"backend": dev, "p50_ms": round(float(np.percentile(xs, 50)), 4),
                      "p90_ms": round(float(np.percentile(xs, 90)), 4), "p99_ms": round(float(np.percentile(xs, 99)), 4),
                      "max_ms": round(float(xs[-1]), 4)})
    ax.set_xlabel("Server round trip (ms)")
    ax.set_ylabel("Cumulative fraction of jobs")
    ax.set_title("Latency distribution (12-qubit GHZ, n = 100)", loc="left")
    ax.legend(loc="lower right")
    save(fig, "fig11_latency_ecdf", table)


# ===================================================== circuit families
FAMILIES = ["ghz", "bv", "hea", "ising", "qaoa", "adder", "mirror", "qft", "qv"]
FAM_LABEL = {"ghz": "GHZ", "qft": "QFT adder", "qv": "QV-style", "qaoa": "QAOA (p=2)", "hea": "HEA (VQE)",
             "ising": "Ising (Trotter)", "bv": "Bernstein–Vazirani", "adder": "Cuccaro adder", "mirror": "Mirror"}
SEQ = LinearSegmentedColormap.from_list("qputwin_blue", ["#f3f8fe", "#a9c9f0", "#2a78d6", "#0b3a73"])
OTHER = [SLOTS[7], INK, MUTED, "#b9b8b3"]                   # non-device categories: never reuse device hues


def heatmap(ax: plt.Axes, mat: np.ndarray, rows: list[str], cols: list[str], fmt: str, norm=None) -> None:
    im = ax.imshow(mat, cmap=SEQ, aspect="auto", norm=norm)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6.6,
                    color="white" if lum < 0.5 else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_circuit_heatmap() -> None:
    rows = load("circuits")["data"]
    n = 8
    pick = {(r["family"], r["backend"]): r for r in rows if r["qubits"] == n}
    fid = np.array([[pick[(f, d)]["hellinger"] for d in DEVICES] for f in FAMILIES])
    twoq = np.array([[pick[(f, d)]["native_two_qubit"] for d in DEVICES] for f in FAMILIES], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.3))
    short = [SHORT[d] for d in DEVICES]
    im0 = heatmap(axes[0], fid, [FAM_LABEL[f] for f in FAMILIES], short, ".2f",
                  norm=matplotlib.colors.Normalize(0, 1))
    im1 = heatmap(axes[1], twoq, [FAM_LABEL[f] for f in FAMILIES], short, ".0f",
                  norm=matplotlib.colors.LogNorm(max(1, twoq.min()), twoq.max()))
    axes[1].set_yticklabels([])
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    fig.colorbar(im0, ax=axes[0], fraction=0.05, pad=0.03).set_label("Hellinger fidelity", fontsize=7)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.05, pad=0.03)
    cb1.set_label("native 2Q gates (log)", fontsize=7)
    cb1.set_ticks([5, 10, 30, 100, 200], labels=["5", "10", "30", "100", "200"])
    cb1.minorticks_off()
    axes[0].set_title(f"(a) Output fidelity under calibrated noise, n = {n}", loc="left")
    axes[1].set_title(f"(b) Compiled entangling cost, n = {n}", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig12_circuit_fidelity_heatmap",
         [{"family": f, "backend": d, "qubits": n, "hellinger": round(pick[(f, d)]["hellinger"], 5),
           "native_two_qubit": pick[(f, d)]["native_two_qubit"], "swaps": pick[(f, d)]["swaps"],
           "esp": round(pick[(f, d)]["esp"], 5)} for f in FAMILIES for d in DEVICES])


def fig_circuit_width() -> None:
    rows = load("circuits")["data"]
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE, 5.6), sharex=True, sharey=True)
    table = []
    for ax, fam in zip(axes.flat, FAMILIES, strict=True):
        for dev in DEVICES:
            rs = sorted((r for r in rows if r["family"] == fam and r["backend"] == dev
                         and r["hellinger"] is not None), key=lambda r: r["qubits"])   # None: beyond exact DM
            x = [r["qubits"] for r in rs]
            ring(ax, x, [r["hellinger"] for r in rs], COLOR[dev], MARK[dev], linestyle=STYLE[dev],
                 label=LABEL[dev], markersize=3.8, linewidth=1.2)
            table += [{"family": fam, "backend": dev, "qubits": r["qubits"], "hellinger": round(r["hellinger"], 5),
                       "tv_distance": round(r["tv_distance"], 5), "native_two_qubit": r["native_two_qubit"]}
                      for r in rs]
        ax.set_title(FAM_LABEL[fam], loc="left")
        ax.set_ylim(0, 1.03)
        ax.set_xticks([3, 4, 5, 6, 7, 8, 10])
    for ax in axes[-1]:
        ax.set_xlabel("Qubits n")
    for ax in axes[:, 0]:
        ax.set_ylabel("Hellinger fidelity")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=0.9, w_pad=0.8)
    save(fig, "fig13_circuit_fidelity_vs_width", table)


def fig_routing_overhead() -> None:
    cost = load("circuits")["cost"]
    widths = sorted({r["qubits"] for r in cost})
    base = {(r["family"], r["qubits"]): r["native_two_qubit"] for r in cost if r["backend"] == "quantinuum_helios"}
    fig, grid = plt.subplots(2, 3, figsize=(DOUBLE, 5.4))
    axes = list(grid.flat[:len(LATTICES)])
    grid.flat[-1].axis("off")
    table = []
    mats = {}
    for dev in LATTICES:
        pick = {(r["family"], r["qubits"]): r for r in cost if r["backend"] == dev}
        mats[dev] = np.array([[pick[(f, n)]["native_two_qubit"] / max(1, base[(f, n)])
                               if (f, n) in pick else np.nan for n in widths] for f in FAMILIES])
        table += [{"backend": dev, "family": f, "qubits": n, "native_two_qubit": pick[(f, n)]["native_two_qubit"],
                   "helios_two_qubit": base[(f, n)], "swaps": pick[(f, n)]["swaps"],
                   "overhead": round(pick[(f, n)]["native_two_qubit"] / max(1, base[(f, n)]), 4)}
                  for f in FAMILIES for n in widths if (f, n) in pick]
    vmax = np.nanmax([m for m in mats.values()])
    norm = matplotlib.colors.LogNorm(1, vmax)
    for k, (ax, (dev, mat)) in enumerate(zip(axes, mats.items(), strict=True)):
        im = heatmap(ax, mat, [FAM_LABEL[f] for f in FAMILIES], [str(n) for n in widths], ".1f", norm=norm)
        if k >= 2:                                   # bottom panel of its column
            ax.set_xlabel("Qubits n")
        ax.set_title(f"({'abcde'[k]}) {SHORT[dev]} (routed)", loc="left")
        if k % 3:
            ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=list(grid.flat), fraction=0.025, pad=0.02)
    cb.set_label("native 2Q gates ÷ all-to-all (Helios)", fontsize=7)
    ticks = [t for t in (1, 1.5, 2, 3, 4) if t <= vmax]
    cb.set_ticks(ticks, labels=[f"{t:g}×" for t in ticks])
    cb.minorticks_off()
    save(fig, "fig14_routing_overhead_by_family", table)


def fig_pipeline_by_family() -> None:
    cost = load("circuits")["cost"]
    n = max(r["qubits"] for r in cost)
    stages = ("parse_ms", "route_ms", "compile_ms", "simulation_ms", "sampling_ms")
    devs = [("quantinuum_helios", "o"), ("iqm_emerald", "D")]
    fams = [f for f in FAMILIES if any(r["family"] == f and r["qubits"] == n for r in cost)]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.9))
    ys = np.arange(len(fams))[::-1]
    table = []
    for k, (dev, _mk) in enumerate(devs):
        pick = {r["family"]: r for r in cost if r["backend"] == dev and r["qubits"] == n}
        tot = np.array([sum(pick[f][s] for s in stages) for f in fams])
        off = 0.19 if k == 0 else -0.19
        ax.barh(ys + off, tot, height=0.36, color=COLOR[dev], edgecolor="white", linewidth=0.6, label=LABEL[dev])
        for y, f, t in zip(ys, fams, tot, strict=True):
            txt = (f"{t / 1000:.1f} s" if t >= 1000 else f"{t:.0f} ms") + f"  · {pick[f]['simulated_qubits']} q"
            ax.annotate(txt, (t, y + off), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.3,
                        color=INK2)
            table.append({"backend": dev, "family": f, "qubits": n, "simulated_qubits": pick[f]["simulated_qubits"],
                          "native_two_qubit": pick[f]["native_two_qubit"],
                          **{s: round(pick[f][s], 3) for s in stages}, "total_ms": round(float(t), 3)})
    ax.set_xscale("log")
    ax.set_xlim(1, max(t["total_ms"] for t in table) * 40)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"))
    ax.set_yticks(ys, [FAM_LABEL[f] for f in fams])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Worker time per job (log; {n}-qubit circuit, ideal statevector, 1024 shots) — label: simulated qubits")
    ax.set_title("Same logical circuit, different cost: routing widens the simulated register on the lattice", loc="left")
    ax.legend(loc="upper right")
    save(fig, "fig15_pipeline_time_by_family", table)


def fig_qv() -> None:
    rows = load("qv")["data"]
    fig, grid = plt.subplots(2, 4, figsize=(DOUBLE, 4.6), sharey=True)
    axes = list(grid.flat)
    axes[-1].axis("off")
    table = []
    for ax, dev in zip(axes, DEVICES, strict=False):
        widths = sorted({r["width"] for r in rows})
        best = None
        means, lows, ideals = [], [], []
        for m in widths:
            hs = [r["hop"] for r in rows if r["backend"] == dev and r["width"] == m]
            ih = [r["ideal_hop"] for r in rows if r["backend"] == dev and r["width"] == m]
            mean = statistics.fmean(hs)
            low = mean - 2 * statistics.stdev(hs) / np.sqrt(len(hs))
            means.append(mean)
            lows.append(low)
            ideals.append(statistics.fmean(ih))
            if low > 2 / 3 and (best is None and m == widths[0] or best == m - 1):
                best = m                                   # largest width with every smaller width passing
            jit = (np.arange(len(hs)) - len(hs) / 2) * 0.025
            ax.scatter(m + jit, hs, s=5, color=COLOR[dev], alpha=0.35, linewidths=0)
            table.append({"backend": dev, "width": m, "hop_mean": round(mean, 5), "hop_lower_2sigma": round(low, 5),
                          "ideal_hop_mean": round(statistics.fmean(ih), 5), "circuits": len(hs),
                          "pass": low > 2 / 3})
        ax.errorbar(widths, means, yerr=np.array(means) - np.array(lows), color=COLOR[dev], marker=MARK[dev],
                    markeredgecolor="white", markeredgewidth=0.8, capsize=2, elinewidth=0.8, lw=1.3)
        ax.plot(widths, ideals, color=MUTED, lw=0.9, ls="--")
        ax.axhline(2 / 3, color=INK2, lw=0.7, ls=":")
        sup = str(best).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")) if best else ""
        geq = "≥ " if best == widths[-1] else ""             # every tested width passes
        ax.set_title(f"{SHORT[dev]}\nQV-style {geq}2{sup} = {2 ** best}" if best
                     else f"{SHORT[dev]}\nno width passes", loc="left")
        ax.set_xticks(widths)
        ax.set_xlabel("Width = depth m")
        ax.set_ylim(0.45, 0.92)
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("Heavy-output probability")
    for ax in axes[:3]:
        ax.set_xlabel("")
    axes[2].annotate("ideal (dashed)", (8, table[-1]["ideal_hop_mean"]), xytext=(0, 4), textcoords="offset points",
                     fontsize=6.6, color=INK2, ha="right")
    axes[0].annotate("2/3 threshold", (8, 2 / 3), xytext=(0, -9), textcoords="offset points", fontsize=6.6,
                     color=INK2, ha="right")
    fig.tight_layout(w_pad=0.6, h_pad=1.0)
    save(fig, "fig16_quantum_volume", table)


# ============================================================ load tests
LOAD_COND = {"light_emerald": "Light GHZ₈ · Emerald · 1 worker", "heavy_forte": "Heavy QV₁₂ · Forte · 1 worker",
             "heavy_forte_x4": "Heavy QV₁₂ · Forte · 4 workers"}
LOAD_MARK = dict(zip(LOAD_COND, "os^", strict=True))
LOAD_COLOR = dict(zip(LOAD_COND, OTHER[:3], strict=True))
HA_SETUP = {"direct_1_orch": "Direct · 1 orchestrator", "gateway_1_orch": "nginx gateway · 1 orchestrator",
            "gateway_3_orch": "nginx gateway · 3 orchestrators"}


def fig_load_closed() -> None:
    rows = load("load_closed")["data"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))
    table = []
    for cond, name in LOAD_COND.items():
        rs = sorted((r for r in rows if r["condition"] == cond), key=lambda r: r["concurrency"])
        x = [r["concurrency"] for r in rs]
        ring(axes[0], x, [r["throughput_jobs_s"] for r in rs], LOAD_COLOR[cond], LOAD_MARK[cond], label=name)
        ring(axes[1], x, [r["client_p50_ms"] for r in rs], LOAD_COLOR[cond], LOAD_MARK[cond], label=name)
        axes[1].plot(x, [r["client_p99_ms"] for r in rs], color=LOAD_COLOR[cond], lw=0.8, ls=":")
        peak = max(rs, key=lambda r: r["throughput_jobs_s"])
        axes[0].annotate(f"peak {peak['throughput_jobs_s']:.0f}/s", (peak["concurrency"], peak["throughput_jobs_s"]),
                         xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.8, color=INK2)
        table += [{"condition": cond, "concurrency": r["concurrency"],
                   "throughput_jobs_s": round(r["throughput_jobs_s"], 3),
                   "client_p50_ms": round(r["client_p50_ms"], 3), "client_p99_ms": round(r["client_p99_ms"], 3),
                   "server_p50_ms": round(r["p50_ms"], 3), "server_p99_ms": round(r["p99_ms"], 3),
                   "errors": r["errors"]} for r in rs]
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrent clients N (closed loop)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Throughput (jobs/s)")
    axes[0].set_title("(a) Throughput saturates at the serial QPU slot", loc="left")
    axes[0].set_xlim(0.7, 900)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Client-measured latency (ms)")
    axes[1].set_title("(b) End-to-end latency: p50 (markers), p99 (dotted)", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
    save(fig, "fig17_load_closed_loop", table)


def fig_littles_law() -> None:
    rows = load("load_closed")["data"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    table = []
    for cond, name in LOAD_COND.items():
        rs = sorted((r for r in rows if r["condition"] == cond), key=lambda r: r["concurrency"])
        ring(axes[0], [r["throughput_jobs_s"] for r in rs], [r["client_p50_ms"] for r in rs], LOAD_COLOR[cond],
             LOAD_MARK[cond], label=name)
        lx = [r["throughput_jobs_s"] * r["client_mean_ms"] / 1e3 for r in rs]
        axes[1].plot([r["concurrency"] for r in rs], lx, linestyle="none", marker=LOAD_MARK[cond],
                     color=LOAD_COLOR[cond], markeredgecolor="white", markeredgewidth=0.8, label=name)
        table += [{"condition": cond, "concurrency": r["concurrency"], "throughput_jobs_s": round(r["throughput_jobs_s"], 3),
                   "client_mean_ms": round(r["client_mean_ms"], 3), "little_N": round(v, 3)}
                  for r, v in zip(rs, lx, strict=True)]
    lim = [0.8, 400]
    axes[1].plot(lim, lim, color=MUTED, lw=0.8, ls=":")
    axes[1].annotate("N = X · R", (60, 60), xytext=(-4, 4), textcoords="offset points", fontsize=6.8, color=INK2,
                     ha="right")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].set_xlabel("Throughput X (jobs/s)")
    axes[0].set_ylabel("p50 client-measured latency (ms)")
    axes[0].set_title("(a) Latency–throughput curve", loc="left")
    axes[1].set_xlabel("Concurrent clients N")
    axes[1].set_ylabel("X × mean client latency R")
    axes[1].set_title("(b) Little's law holds at every level", loc="left")
    axes[0].legend(loc="center left", bbox_to_anchor=(0.2, 0.3), fontsize=6.4)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig18_latency_throughput_littles_law", table)


def fig_queue_location() -> None:
    rows = load("load_closed")["data"]
    parts = [("Before the orchestrator's clock (client stack, socket backlog)", ()),
             ("Orchestrator (ingress, validation, routing, assembly)",
              ("ingress_ms", "validation_ms", "routing_ms", "response_assembly_ms", "unattributed_ms")),
             ("Dispatch transit (network + socket / event-loop queueing)", ("network_transit_ms",)),
             ("Worker admission queue", ("worker_queue_ms",)),
             ("Worker service (parse, route, compile, engine)",
              ("worker_parse_ms", "worker_route_ms", "worker_compile_ms", "qpu_compute_time_ms",
               "worker_unattributed_ms"))]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.3), sharey=True)
    table = []
    ha = load("load_ha")["data"]
    for ax, cond in zip(axes, ("light_emerald", "direct_1_orch"), strict=True):
        src = rows if cond == "light_emerald" else ha
        rs = sorted((r for r in src if r["condition"] == cond), key=lambda r: r["concurrency"])
        x = np.arange(len(rs))
        comp = np.array([[sum(r["spans_mean_ms"].get(kk, 0.0) for kk in keys) for r in rs] for _, keys in parts])
        comp[0] = [max(0.0, r["client_mean_ms"] - comp[1:, i].sum()) for i, r in enumerate(rs)]
        total = comp.sum(axis=0)
        bottom = np.zeros(len(rs))
        for k, (name, _) in enumerate(parts):
            ax.bar(x, comp[k] / total, bottom=bottom, width=0.72, color=[MUTED, *SLOTS][k], edgecolor="white",
                   linewidth=0.6, label=name)
            bottom += comp[k] / total
            table += [{"condition": cond, "concurrency": r["concurrency"], "component": name, "mean_ms": round(v, 4),
                       "share": round(v / t, 4)} for r, v, t in zip(rs, comp[k], total, strict=True)]
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xticks(x, [f"{r['concurrency']}\n{t:.0f}" if t >= 10 else f"{r['concurrency']}\n{t:.1f}"
                          for r, t in zip(rs, total, strict=True)], fontsize=6.6)
        ax.set_xlabel("Concurrent clients N  /  mean latency (ms)")
        ax.grid(axis="x", visible=False)
        ax.set_title("(a) Worker-bound: light jobs on one QPU" if cond == "light_emerald"
                     else "(b) Orchestrator-bound: light jobs over seven QPUs", loc="left")
    axes[0].set_ylabel("Share of mean client latency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, handlelength=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=2.0)
    save(fig, "fig25_queue_location", table)


def fig_orchestrator_scaling() -> None:
    ha = load("load_ha")["data"]
    series = [(name, [r for r in ha if r["condition"] == key], col, mk)
              for (key, name), col, mk in zip(HA_SETUP.items(), (OTHER[0], OTHER[1], OTHER[2]), "os^", strict=True)]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    table = []
    for name, rows, col, mk in series:
        rs = sorted(rows, key=lambda r: r["concurrency"])
        x = [r["concurrency"] for r in rs]
        ring(axes[0], x, [r["throughput_jobs_s"] for r in rs], col, mk, label=name)
        ring(axes[1], x, [r["client_p99_ms"] for r in rs], col, mk, label=name)
        peak = max(rs, key=lambda r: r["throughput_jobs_s"])
        axes[0].annotate(f"{peak['throughput_jobs_s']:.0f}/s", (peak["concurrency"], peak["throughput_jobs_s"]),
                         xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.8, color=INK2)
        table += [{"setup": name, "concurrency": r["concurrency"], "throughput_jobs_s": round(r["throughput_jobs_s"], 3),
                   "client_p50_ms": round(r["client_p50_ms"], 3), "client_p99_ms": round(r["client_p99_ms"], 3),
                   "errors": r["errors"]} for r in rs]
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Concurrent clients N (light jobs over seven QPUs)")
    axes[0].set_ylim(0, max(r["throughput_jobs_s"] for r in rows) * 1.15)
    axes[0].set_ylabel("Throughput (jobs/s)")
    axes[0].set_title("(a) Orchestrator replicas lift the ceiling", loc="left")
    axes[0].legend(loc="upper left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("p99 client latency (ms)")
    axes[1].set_title("(b) Tail latency", loc="left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig26_orchestrator_scaling", table)


def fig_open_loop() -> None:
    doc = load("load_open")
    rows = [r for r in doc["data"] if r["rho_target"] < 1]
    over = [r for r in doc["data"] if r["rho_target"] >= 1]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    rho = np.array([r["rho_target"] for r in rows])
    ax = axes[0]
    rr = np.linspace(0.05, 0.97, 200)
    es, es2 = doc["service_mean_ms"] / 1e3, doc["service_second_moment_s2"]
    pk = doc["response_c1_ms"] + rr / es * es2 / (2 * (1 - rr)) * 1e3
    ax.plot(rr, pk, color=MUTED, lw=1.0, label="M/G/1 Pollaczek–Khinchine")
    ring(ax, rho, [r["mean_ms"] for r in rows], SLOTS[0], "o", linestyle="none", label="measured mean")
    ring(ax, rho, [r["p99_ms"] for r in rows], SLOTS[1], "s", linestyle="none", label="measured p99")
    ax.set_yscale("log")
    ax.set_yticks([100, 200, 500, 1000, 2000], ["100", "200", "500", "1000", "2000"])
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Utilisation ρ = λ · E[S]")
    ax.set_ylabel("Server round trip (ms)")
    ax.set_title(f"(a) Queueing delay vs load (E[S] = {doc['service_mean_ms']:.0f} ms, "
                 f"cv = {doc['service_cv']:.2f})", loc="left")
    ax.legend(loc="upper left")
    ax2 = axes[1]
    for k, r in enumerate(over + rows[-2:][::-1]):
        pts = np.array(doc["latencies_ms"][str(r["rho_target"])])
        ax2.plot(pts[:, 0], pts[:, 1] / 1e3, ".", ms=2.2, color=[SLOTS[7], SLOTS[0], SLOTS[2]][k], alpha=0.7,
                 label=f"ρ = {r['rho_target']:g}")
    ax2.set_xlabel("Arrival time (s)")
    ax2.set_ylabel("Round trip (s)")
    ax2.set_title("(b) Stable below ρ = 1, unbounded growth above", loc="left")
    ax2.legend(loc="upper left", markerscale=3)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig19_open_loop_queueing",
         [{"rho": r["rho_target"], "arrival_rate": round(r["arrival_rate"], 3), "mean_ms": round(r["mean_ms"], 3),
           "p99_ms": round(r["p99_ms"], 3), "pk_mean_ms": round(r["pk_mean_ms"], 3) if r["pk_mean_ms"] else None,
           "throughput_jobs_s": round(r["throughput_jobs_s"], 3)} for r in doc["data"]])


def fig_soak() -> None:
    doc = load("soak")
    pts = [p for p in doc["data"] if p[1] is not None]
    t = np.array([p[0] for p in pts])
    lat = np.array([p[1] for p in pts])
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 3.8), sharex=True)
    bins = np.arange(0, doc["duration_s"] + 10, 10)
    mid, p50, p99, thr = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:], strict=True):
        sel = lat[(t >= a) & (t < b)]
        if len(sel):
            mid.append((a + b) / 2)
            p50.append(np.percentile(sel, 50))
            p99.append(np.percentile(sel, 99))
            thr.append(len(sel) / (b - a))
    ax = axes[0]
    ax.plot(t, lat, ".", ms=1.0, color=GRID, zorder=0, rasterized=True)   # 36k points: raster keeps the SVG small
    ring(ax, mid, p50, SLOTS[0], "o", markersize=3.5, label="p50 per 10 s")
    ring(ax, mid, p99, SLOTS[1], "s", markersize=3.5, label="p99 per 10 s")
    ax.set_yscale("log")
    ax.set_ylabel("Round trip (ms)")
    errs = sum(1 for p in doc["data"] if p[2] != 200)
    s = doc["summary"]
    ax.set_title(f"(a) {len(pts):,} jobs at {doc['rate_jobs_s']:g} jobs/s for {doc['duration_s']:.0f} s — "
                 f"p50 {s['p50_ms']:.1f} ms, p99 {s['p99_ms']:.1f} ms, {errs} errors", loc="left")
    ax.legend(loc="upper right", ncol=2)
    ax2 = axes[1]
    names = sorted({c["name"] for smp in doc["stats"] for c in smp["containers"]})
    table = []
    for k, nm in enumerate(names):
        xs = [smp["t"] for smp in doc["stats"] for c in smp["containers"] if c["name"] == nm]
        ys = [c["mem_mib"] for smp in doc["stats"] for c in smp["containers"] if c["name"] == nm]
        short = nm.replace("qputwin-", "").replace("worker-", "").rsplit("-", 1)[0]
        col = {"orchestrator": INK, "quantinuum-helios": COLOR["quantinuum_helios"],
               "ionq-forte": COLOR["ionq_forte"], "rigetti-ankaa3": COLOR["rigetti_ankaa_3"],
               "iqm-emerald": COLOR["iqm_emerald"], "ibm-heron": COLOR["ibm_heron"],
               "ibm-nighthawk": COLOR["ibm_nighthawk"], "google-willow": COLOR["google_willow"]}.get(short, MUTED)
        ax2.plot(xs, ys, color=col, lw=1.3, label=short)
        table += [{"container": short, "t_s": round(x, 2), "mem_mib": y} for x, y in zip(xs, ys, strict=True)]
    ax2.set_ylabel("Container memory (MiB)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(bottom=0)
    ax2.set_title("(b) Memory plateaus once the telemetry ring buffer is full — no leak", loc="left")
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig20_soak", table)


def fig_async_burst() -> None:
    doc = load("async_burst")
    rows = [r for r in doc["data"] if r["status"] == "COMPLETED"]

    def ts(s: str) -> float:
        from datetime import datetime
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

    t0 = min(ts(r["submitted_at"]) for r in rows)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    table = []
    grid = np.linspace(0, max(ts(r["finished_at"]) for r in rows) - t0, 300)
    sub = np.sort([ts(r["submitted_at"]) - t0 for r in rows])
    st = np.sort([ts(r["started_at"]) - t0 for r in rows])
    fin = np.sort([ts(r["finished_at"]) - t0 for r in rows])
    queued = np.searchsorted(sub, grid, "right") - np.searchsorted(st, grid, "right")
    done = np.searchsorted(fin, grid, "right")
    ax = axes[0]
    ax.fill_between(grid, queued, color=SLOTS[3], alpha=0.25, linewidth=0)
    ax.plot(grid, queued, color=SLOTS[3], lw=1.3, label="queued")
    ax.plot(grid, done, color=SLOTS[2], lw=1.5, label="completed (cumulative)")
    ax.set_xlabel("Time since first submission (s)")
    ax.set_ylabel("Jobs")
    drain = len(rows) / (fin[-1] - st[0])
    ax.set_title(f"(a) {len(rows):,} async jobs: drained at {drain:.0f} jobs/s", loc="left")
    ax.legend(loc="upper left")
    ax2 = axes[1]
    for dev in DEVICES:
        q = np.sort([r["queue_ms"] / 1e3 for r in rows if r["backend"] == dev])
        ax2.step(q, np.arange(1, len(q) + 1) / len(q), where="post", color=COLOR[dev], linestyle=STYLE[dev],
                 label=LABEL[dev])
        table.append({"backend": dev, "jobs": len(q), "queue_p50_s": round(float(np.median(q)), 4),
                      "queue_p99_s": round(float(np.percentile(q, 99)), 4), "queue_max_s": round(float(q[-1]), 4)})
    ax2.set_xlabel("Orchestrator queue wait (s)")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_title("(b) Queue wait per QPU (one dispatch slot each)", loc="left")
    ax2.legend(loc="center left", bbox_to_anchor=(0.02, 0.62))
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig21_async_burst", table)


def fig_fairshare() -> None:
    rows = load("fairshare")["data"]
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    table = []
    for k, c in enumerate(conds):
        light = np.array([r["queue_ms"] for r in rows if r["condition"] == c and r["kind"] == "light"])
        heavy = np.array([r["queue_ms"] for r in rows if r["condition"] == c and r["kind"] == "heavy"])
        y = len(conds) - 1 - k
        jit = (np.random.default_rng(k).random(len(light)) - 0.5) * 0.3
        ax.scatter(light, y + jit, s=10, color=OTHER[k], edgecolors="white", linewidths=0.5, zorder=3)
        ax.plot([np.median(light)] * 2, [y - 0.28, y + 0.28], color=INK, lw=1.4, zorder=4)
        ax.plot([np.median(heavy)], [y], marker="|", ms=12, color=MUTED, mew=1.2, zorder=2)
        table.append({"condition": c, "light_p50_ms": round(float(np.median(light)), 3),
                      "light_max_ms": round(float(light.max()), 3), "heavy_p50_ms": round(float(np.median(heavy)), 3)})
    ax.set_xscale("log")
    ax.set_yticks(range(len(conds)), conds[::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Light-client queue wait (ms, log)")
    ax.set_title("Isolation from a flooding client", loc="left")
    ax.plot([], [], color=INK, lw=1.4, label="light median")
    ax.plot([], [], marker="|", ms=8, color=MUTED, ls="none", label="heavy median")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2)
    save(fig, "fig22_fair_share", table)


def fig_shots() -> None:
    rows = load("shots")["data"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    table = []
    for dev in DEVICES:
        shots = sorted({r["shots"] for r in rows})
        tot = [statistics.median(r["total_ms"] for r in rows if r["backend"] == dev and r["shots"] == s) for s in shots]
        smp = [statistics.median(r["sampling_ms"] for r in rows if r["backend"] == dev and r["shots"] == s)
               for s in shots]
        ring(axes[0], shots, tot, COLOR[dev], MARK[dev], linestyle=STYLE[dev], label=LABEL[dev])
        ring(axes[1], shots, smp, COLOR[dev], MARK[dev], linestyle=STYLE[dev], label=LABEL[dev])
        table += [{"backend": dev, "shots": s, "total_ms_median": round(a, 4), "sampling_ms_median": round(b, 4)}
                  for s, a, b in zip(shots, tot, smp, strict=True)]
    ref = np.array([1e3, 1e6])
    m = statistics.median(r["sampling_ms"] for r in rows if r["shots"] == 1_000_000)
    axes[1].plot(ref, m * ref / 1e6, color=MUTED, lw=0.8, ls=":")
    axes[1].annotate(r"$\propto$ shots", (1e5, m * 1e-1), xytext=(6, -6), textcoords="offset points", fontsize=7,
                     color=INK2, ha="left", va="top")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Shots")
    axes[0].set_yticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    axes[0].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    axes[0].set_ylabel("Server round trip (ms, median)")
    axes[0].set_title("(a) End-to-end latency vs shots (GHZ₁₂)", loc="left")
    axes[1].set_ylabel("Sampling time (ms, median)")
    axes[1].set_title("(b) Sampling cost is linear in shots", loc="left")
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig23_shots_scaling", table)


def fig_gate_scaling() -> None:
    rows = load("gate_scaling")["data"]
    gates = sorted({r["gates"] for r in rows})
    stages = [("worker_parse_ms", "Parse + flatten", "o"), ("worker_route_ms", "SABRE route", "s"),
              ("worker_compile_ms", "Native compile", "^"), ("simulation_ms", "Simulation", "D")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))
    table = []

    def med(dev: str, key: str, g: int) -> float:
        return statistics.median(r[key] for r in rows if r["backend"] == dev and r["gates"] == g)

    for k, (key, name, mk) in enumerate(stages):
        ring(axes[0], gates, [max(med("iqm_emerald", key, g), 1e-3) for g in gates], SLOTS[k], mk, label=name)
    ref = np.array(gates, dtype=float)
    y0 = med("iqm_emerald", "worker_parse_ms", gates[-1])
    axes[0].plot(ref, y0 * ref / gates[-1], color=MUTED, lw=0.8, ls=":")
    axes[0].annotate("dotted: linear", (gates[2], y0 * gates[2] / gates[-1]), xytext=(4, -10),
                     textcoords="offset points", fontsize=6.8, color=INK2)
    ends: dict[str, list[tuple[str, float]]] = {}
    for dev in DEVICES:
        fe = [med(dev, "worker_parse_ms", g) + med(dev, "worker_route_ms", g) + med(dev, "worker_compile_ms", g)
              for g in gates]
        ring(axes[1], gates, fe, COLOR[dev], MARK[dev], linestyle=STYLE[dev], label=LABEL[dev])
        ends.setdefault("lattices" if dev in LATTICES else "ions", []).append((dev, fe[-1]))
        table += [{"backend": dev, "gates": g, **{k2: round(med(dev, k2, g), 4) for k2, _, _ in stages},
                   "front_end_ms": round(f, 4)} for g, f in zip(gates, fe, strict=True)]
    for group, devs in ends.items():                       # one label per connectivity class
        rates = sorted(gates[-1] / v for _, v in devs)
        fmt = (lambda r: f"{r:.0f}") if rates[0] >= 10 else (lambda r: f"{r:.1f}")
        rate = fmt(rates[0]) if fmt(rates[0]) == fmt(rates[-1]) else f"{fmt(rates[0])}–{fmt(rates[-1])}"
        end_label(axes[1], gates[-1], max(v for _, v in devs) if group == "lattices" else min(v for _, v in devs),
                  f"{rate} gates/ms ({'five lattices' if group == 'lattices' else 'both ions'})")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Gates in the source circuit (10 qubits)")
    axes[0].set_ylabel("Worker time (ms, median)")
    axes[0].set_title("(a) Stage cost on IQM Emerald", loc="left")
    axes[0].legend(loc="upper left")
    axes[1].set_ylabel("Parse + route + compile (ms)")
    axes[1].set_xlim(right=gates[-1] * 9)
    axes[1].set_title("(b) Front-end time per QPU", loc="left")
    axes[1].legend(loc="lower right", fontsize=6.2)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig24_gate_scaling", table)


def fig_saturation() -> None:
    """E22 — the whole host under mixed load: CPU and memory over time, and latency per job class."""
    doc = load("saturation")
    d = doc["data"]
    host = d["host"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 3.0), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    t = [h["t"] for h in host]
    ax = axes[0]
    ax.plot(t, [h["cpu_pct"] for h in host], color=SLOTS[0], lw=1.2)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Host CPU busy (%)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"(a) CPU, {doc['worker_cpus']} vCPUs (median {doc['summary']['host_cpu_pct_median']:.0f} %)",
                 loc="left")
    ax = axes[1]
    total = host[0]["mem_total_gib"]
    ax.plot(t, [h["mem_used_gib"] for h in host], color=SLOTS[1], lw=1.2)
    ax.axhline(total, color=MUTED, lw=0.8, ls=":")
    ax.text(0.02, total / (total * 1.12) + 0.01, f"host RAM {total:.1f} GiB", transform=ax.transAxes, fontsize=6.4,
            color=INK2, va="bottom")
    ax.set_ylim(0, total * 1.12)
    ax.set_ylabel("Host memory in use (GiB)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"(b) Memory (max {doc['summary']['host_mem_gib_max']:.1f} GiB)", loc="left")
    ax = axes[2]
    classes = [("light sync, GHZ$_8$", [p[1] / 1e3 for p in d["light"] if p[2] == 200 and p[1] is not None],
                SLOTS[0], "-"),
               ("heavy: dense state", [h["latency_s"] for h in d["heavy"] if h["stream"] == "memory"
                                       and h["status"] == "COMPLETED"], SLOTS[1], "-"),
               ("heavy: DM / QV", [h["latency_s"] for h in d["heavy"] if h["stream"] == "compute"
                                    and h["status"] == "COMPLETED"], SLOTS[2], "--"),
               ("oversized: refused", [r["latency_s"] for r in d["oversized"]], SLOTS[3], ":")]
    table = []
    for name, v, col, ls in classes:
        if not v:
            continue
        v = np.sort(np.asarray(v))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=1.3, label=f"{name} (n={len(v):,})")
        table += [{"class": name, "n": len(v), "p50_s": round(float(np.median(v)), 4),
                   "p95_s": round(float(np.percentile(v, 95)), 4), "max_s": round(float(v.max()), 4)}]
    ax.set_xscale("log")
    ax.set_xlabel("Response time (s)")
    ax.set_ylabel("Cumulative fraction")
    sm = doc["summary"]
    ax.set_title(f"(c) {sm['oom_kills']} OOM kills, {sm['restarts']} restarts", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.09, 1, 1))
    save(fig, "fig27_saturation", table)


FIGURES = [fig_latency, fig_scaling, fig_routing, fig_routing_latency, fig_fidelity, fig_hwtime, fig_budget,
           fig_convergence, fig_async, fig_noise_sweep, fig_ramsey, fig_latency_ecdf, fig_circuit_heatmap,
           fig_circuit_width, fig_routing_overhead, fig_pipeline_by_family, fig_qv, fig_load_closed,
           fig_littles_law, fig_queue_location, fig_orchestrator_scaling, fig_open_loop, fig_soak, fig_async_burst,
           fig_fairshare, fig_shots, fig_gate_scaling, fig_saturation]


def main(argv: list[str] | None = None) -> None:
    import sys
    only = set(argv if argv is not None else sys.argv[1:])
    for f in FIGURES:
        if only and not any(o in f.__name__ for o in only):
            continue
        try:
            f()
        except FileNotFoundError as e:                    # experiment not run yet
            print(f"  skip {f.__name__}: {e.filename}")
    print(f"wrote {len(FIGURES)} figures to {OUT.relative_to(ROOT)} (+ docs/assets/figures)")


if __name__ == "__main__":
    main()
