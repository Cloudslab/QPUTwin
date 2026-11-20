"""Draws the simulator-comparison figures from comparison/results/*.json.

Usage: cd comparison && uv run python plot_comparison.py

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
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, OUT = HERE / "results", HERE / "figures"
DATA = OUT / "data"
DOCS = HERE.parent / "docs" / "assets" / "figures" / "comparison"      # copied for the documentation site

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#1f2328", "#52514e", "#8a8984", "#e3e2dd"
SIMS = ["QPUTwin", "Qiskit Aer", "Braket local", "Cirq", "qsim", "Qulacs"]
COLOR = {**dict(zip(SIMS, SLOTS, strict=False)), "QPUTwin (in-process)": SLOTS[0], "QPUTwin (complex64)": SLOTS[6]}
MARK = {**dict(zip(SIMS, "osD^vP", strict=True)), "QPUTwin (in-process)": "o", "QPUTwin (complex64)": "h"}
DEVICES = ["quantinuum_helios", "ionq_forte", "rigetti_ankaa_3", "iqm_emerald", "ibm_heron", "ibm_nighthawk",
           "google_willow"]
DEV_LABEL = {"quantinuum_helios": "Quantinuum Helios", "ionq_forte": "IonQ Forte",
             "rigetti_ankaa_3": "Rigetti Ankaa-3", "iqm_emerald": "IQM Emerald", "ibm_heron": "IBM Heron r3",
             "ibm_nighthawk": "IBM Nighthawk", "google_willow": "Google Willow"}
DEV_COLOR = dict(zip(DEVICES, SLOTS, strict=False))
DEV_MARK = dict(zip(DEVICES, "os^DvPX", strict=True))
FAM_LABEL = {"ghz": "GHZ", "qft": "QFT adder", "qv": "QV-style", "qaoa": "QAOA (p=2)", "hea": "HEA (VQE)",
             "ising": "Ising (Trotter)", "bv": "Bernstein–Vazirani", "adder": "Cuccaro adder", "mirror": "Mirror"}
SINGLE, DOUBLE = 3.5, 7.2

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 5, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150,
})


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / f"{name}.json").read_text())


def save(fig: plt.Figure, name: str, rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    for ext in ("svg", "png"):
        shutil.copy(OUT / f"{name}.{ext}", DOCS / f"{name}.{ext}")
    if rows:
        with (DATA / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))   # tables can mix row kinds
            w.writeheader()
            w.writerows(rows)
    plt.close(fig)
    print(f"  {name}")


def ring(ax: plt.Axes, x, y, color: str, marker: str, **kw) -> None:
    ax.plot(x, y, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.8, **kw)


def best(r: dict[str, Any], engine: bool = False) -> float:
    """Best of the repetitions (3 below 2 s, else 1): drops first-call warm-up."""
    if engine:
        return r["engine_s_best"]
    return r["run_s_best"] if "run_s_best" in r else min(r["run_s"])


def secs(v: float, _=None) -> str:
    return f"{v * 1e3:g} ms" if v < 1 else f"{v:g} s"


# ================================================================= figures
def fig_correctness() -> None:
    rows = load("correctness")["data"]
    fams = list(dict.fromkeys(r["family"] for r in rows))
    sims = [s for s in SIMS if s != "Qiskit Aer"]
    mat = np.array([[max(r["max_abs_dp_vs_aer"] for r in rows if r["family"] == f and r["simulator"] == s)
                     for s in sims] for f in fams])
    fig, ax = plt.subplots(figsize=(SINGLE * 1.45, 3.0))
    cmap = LinearSegmentedColormap.from_list("q", ["#f3f8fe", "#a9c9f0", "#2a78d6", "#0b3a73"])
    shown = np.where(mat > 0, mat, 1e-18)
    im = ax.imshow(shown, cmap=cmap, norm=LogNorm(1e-18, 1e-6), aspect="auto")
    for i in range(len(fams)):
        for j in range(len(sims)):
            v = mat[i, j]
            txt = "0" if v == 0 else f"{v:.0e}".replace("e-0", "e-").replace("e-", "e−")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.4,
                    color="white" if shown[i, j] > 1e-10 else INK)
    ax.set_xticks(range(len(sims)), sims, rotation=20, ha="right")
    ax.set_yticks(range(len(fams)), [FAM_LABEL[f] for f in fams])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("max |Δp| vs Qiskit Aer (4–16 qubits)", fontsize=7)
    ax.set_title("Ideal output distributions agree to machine precision", loc="left")
    save(fig, "figC1_correctness", [{"family": f, "simulator": s, "max_abs_dp_vs_aer": float(mat[i, j])}
                                    for i, f in enumerate(fams) for j, s in enumerate(sims)])


def _series(ax: plt.Axes, rows: list[dict[str, Any]], fam: str, table: list[dict[str, Any]]) -> None:
    for sim in ["QPUTwin", "QPUTwin (in-process)", "QPUTwin (complex64)", *SIMS[1:]]:
        rs = sorted((r for r in rows if r["family"] == fam and r["simulator"] == sim), key=lambda r: r["qubits"])
        if not rs:
            continue
        x = [r["qubits"] for r in rs]
        if sim.startswith("QPUTwin"):
            label = {"QPUTwin": "QPUTwin engine", "QPUTwin (in-process)": "QPUTwin engine",
                     "QPUTwin (complex64)": "QPUTwin engine, complex64"}[sim]
            ring(ax, x, [best(r, engine=True) for r in rs], COLOR[sim], MARK[sim], label=label, zorder=5)
            if sim == "QPUTwin":
                ax.plot(x, [best(r) for r in rs], color=COLOR[sim], lw=1.0, ls=":", label="QPUTwin end-to-end (HTTP)")
        else:
            ring(ax, x, [best(r) for r in rs], COLOR[sim], MARK[sim], label=sim, markersize=4)
        table += [{"family": fam, "simulator": sim, "qubits": r["qubits"], "run_s_best": round(best(r), 6),
                   "engine_s_best": round(r["engine_s_best"], 6) if r.get("engine_s_best") is not None else ""}
                  for r in rs]


def _speed_panels(rows: list[dict[str, Any]], fams: list[str], name: str, title: str,
                  xticks: tuple[int, ...] = (6, 10, 14, 18, 22)) -> None:
    fig, axes = plt.subplots(1, len(fams), figsize=(DOUBLE, 2.6), sharey=True)
    table: list[dict[str, Any]] = []
    for ax, fam in zip(axes, fams, strict=True):
        _series(ax, rows, fam, table)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(secs))
        ax.set_xlabel("Qubits n")
        ax.set_xticks(list(xticks))
        ax.set_title(FAM_LABEL[fam], loc="left")
    axes[0].set_ylabel("Time per job (best of 3)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97), w_pad=0.8)
    save(fig, name, table)


def fig_sv_speed() -> None:
    _speed_panels(load("sv_speed_t1")["data"], ["ghz", "qv", "qft"], "figC2_statevector_speed_1thread",
                  "Ideal statevector simulation + 1024 shots, one thread each")


def fig_threads() -> None:
    _speed_panels(load("sv_speed_t4")["data"], ["ghz", "qv", "qft"], "figC3_statevector_speed_4threads",
                  "Ideal statevector simulation + 1024 shots, four threads each (QPUTwin: the running stack)")


def fig_threads16() -> None:
    _speed_panels(load("sv_speed_t16")["data"], ["ghz", "qv", "qft"], "figC10_statevector_speed_16threads",
                  "Ideal statevector simulation + 1024 shots, 16 threads each (all in-process)",
                  xticks=(6, 10, 14, 18, 22, 26))


def fig_dm_speed() -> None:
    budgets = [("t1", "one thread"), ("t4", "four threads")]
    if (RES / "dm_speed_t16.json").exists():
        budgets.append(("t16", "16 threads"))
    fig, axes = plt.subplots(len(budgets), 2, figsize=(DOUBLE, 2.3 * len(budgets)), sharey=True)
    table: list[dict[str, Any]] = []
    for row, (tag, title) in enumerate(budgets):
        rows = load(f"dm_speed_{tag}")["data"]
        for col, fam in enumerate(("ghz", "qv")):
            ax = axes[row, col]
            _series(ax, rows, fam, table)
            for t in table:
                t.setdefault("threads", tag)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(secs))
            ax.set_title(f"({'abcdef'[2 * row + col]}) {FAM_LABEL[fam]}, {title}", loc="left")
            if row == len(budgets) - 1:
                ax.set_xlabel("Qubits n")
        axes[row, 0].set_ylabel("Time per job (best of 3)")
    seen: dict[str, Any] = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Noisy density-matrix simulation + 1024 shots (QPUTwin: its full calibrated noise model)",
                 x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08 if len(budgets) == 2 else 0.06, 1, 0.97), h_pad=1.0)
    save(fig, "figC4_density_matrix_speed", table)


def fig_ranking() -> None:
    """Every simulator's time relative to QPUTwin, per workload and thread budget (>1 = slower than QPUTwin)."""
    work = [("sv", "ghz", 24), ("sv", "qv", 20), ("sv", "qv", 24), ("sv", "qft", 20), ("sv", "qft", 24),
            ("dm", "ghz", 12), ("dm", "qv", 12)]
    c64 = "QPUTwin (complex64)"                       # qsim runs in single precision only: also compare like for like
    sims_by = {"t1": ["Qiskit Aer", "qsim", ("qsim", c64), "Qulacs", "Braket local", "Cirq"],
               "t4": ["Qiskit Aer", "qsim", ("qsim", c64), "Qulacs"],
               "t16": ["Qiskit Aer", "qsim", ("qsim", c64), "Qulacs"]}
    tags = ["t1", "t4"] + (["t16"] if (RES / "sv_speed_t16.json").exists() else [])
    fig, axes = plt.subplots(1, len(tags), figsize=(DOUBLE, 3.1),
                             gridspec_kw={"width_ratios": [len(sims_by[t]) for t in tags]})
    table = []
    cmap = LinearSegmentedColormap.from_list("rk", ["#e34948", "#f6f6f4", "#2a78d6"])
    titles = {"t1": "one thread", "t4": "four threads", "t16": "16 threads"}
    for k, (ax, tag) in enumerate(zip(axes, tags, strict=True)):
        data = {"sv": load(f"sv_speed_{tag}")["data"], "dm": load(f"dm_speed_{tag}")["data"]}
        twin_name = "QPUTwin" if tag == "t4" else "QPUTwin (in-process)"
        sims = [(s, twin_name) if isinstance(s, str) else s for s in sims_by[tag]]
        mat = np.full((len(work), len(sims)), np.nan)
        for i, (kind, fam, n) in enumerate(work):
            for j, (sim, twin) in enumerate(sims):
                tw = next((r for r in data[kind] if r["family"] == fam and r["qubits"] == n and r["simulator"] == twin),
                          None)
                r = next((r for r in data[kind] if r["family"] == fam and r["qubits"] == n and r["simulator"] == sim),
                         None)
                if r is not None and tw is not None:
                    mat[i, j] = best(r) / best(tw, engine=True)
                    table.append({"threads": tag, "workload": f"{kind} {fam} {n}", "simulator": sim,
                                  "qputwin_precision": "complex64" if twin == c64 else "complex128",
                                  "time_relative_to_qputwin": round(float(mat[i, j]), 3)})
        im = ax.imshow(np.log10(mat), cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto")
        for i in range(len(work)):
            for j in range(len(sims)):
                v = mat[i, j]
                txt = "—" if np.isnan(v) else (f"{v:.1f}×" if v < 10 else f"{v:.0f}×")
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.4, color=INK)
        short = {"Qiskit Aer": "Aer", "Braket local": "Braket"}
        ax.set_xticks(range(len(sims)), [short.get(s, s) + ("\nvs c64" if t == c64 else "") for s, t in sims])
        ax.set_yticks(range(len(work)), [f"{'SV' if k == 'sv' else 'DM'} {FAM_LABEL[f]} {n}q" for k, f, n in work])
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({'abc'[k]}) {titles[tag]}", loc="left")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02)
    cb.set_ticks([-1, 0, 1], labels=["QPUTwin 10× slower", "equal", "QPUTwin 10× faster"])
    cb.set_label("other simulator's time ÷ QPUTwin engine time", fontsize=7)
    save(fig, "figC8_relative_speed", table)


def fig_noise_xval() -> None:
    rows = load("noise_xval")["data"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.9))
    table = []
    for dev in DEVICES:
        rs = [r for r in rows if r["backend"] == dev]
        kw = dict(s=14, marker=DEV_MARK[dev], linewidths=0.6, facecolors=DEV_COLOR[dev], edgecolors="white",
                  label=DEV_LABEL[dev])
        axes[0].scatter([r["aer_matched"] for r in rs], [r["qputwin_no_idle"] for r in rs], **kw)
        axes[1].scatter([r["aer_matched"] for r in rs], [r["qputwin_full"] for r in rs], **kw)
        axes[2].scatter([r["qiskit_two_qubit"] for r in rs], [r["qputwin_two_qubit"] for r in rs], **kw)
        table += [{k: r[k] for k in ("backend", "family", "qubits", "aer_matched", "qputwin_no_idle", "qputwin_full",
                                     "tvd_no_idle_vs_aer", "qiskit_two_qubit", "qputwin_two_qubit")} for r in rs]
    d0 = np.array([r["qputwin_no_idle"] - r["aer_matched"] for r in rows])
    d1 = np.array([r["qputwin_full"] - r["aer_matched"] for r in rows])
    for ax, d, title in ((axes[0], d0, "(a) QPUTwin without idle noise"),
                         (axes[1], d1, "(b) QPUTwin full model (+ idle)")):
        ax.plot([0, 1.02], [0, 1.02], color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Aer, independent per-qubit model")
        ax.set_title(title, loc="left")
        ax.annotate(f"identical circuits (n = {len(d)}):\nmedian |Δ| = {np.median(np.abs(d)):.4f}\n"
                    f"mean Δ = {d.mean():+.4f}", (0.04, 0.96), xycoords="axes fraction", va="top",
                    fontsize=6.6, color=INK2)
    axes[0].set_ylabel("QPUTwin (Hellinger fidelity)")
    lim = max(max(r["qiskit_two_qubit"] for r in rows), max(r["qputwin_two_qubit"] for r in rows)) * 1.2
    axes[2].plot([0.8, lim], [0.8, lim], color=MUTED, lw=0.8, ls=":")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Qiskit transpiler (O1)")
    axes[2].set_ylabel("QPUTwin compiler")
    axes[2].set_title("(c) Native 2Q gate count", loc="left")
    axes[2].set_aspect("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), columnspacing=1.0,
               handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.2)
    save(fig, "figC5_noise_cross_validation", table)


def fig_noise_by_family() -> None:
    rows = load("noise_xval")["data"]
    fams = list(dict.fromkeys(r["family"] for r in rows))
    fig, grid = plt.subplots(2, 4, figsize=(DOUBLE, 4.6), sharey=True)
    axes = list(grid.flat)
    table = []
    for ax, dev in zip(axes, DEVICES, strict=False):
        x = np.arange(len(fams))
        for k, (key, name) in enumerate((("aer_matched", "Aer, independent model"),
                                         ("qputwin_no_idle", "QPUTwin, no idle"), ("qputwin_full", "QPUTwin, full"))):
            vals = [statistics.fmean(r[key] for r in rows if r["backend"] == dev and r["family"] == f) for f in fams]
            ax.bar(x + (k - 1) * 0.27, vals, width=0.27, color=[SLOTS[1], SLOTS[6], SLOTS[0]][k],
                   edgecolor="white", linewidth=0.5, label=name)
            table += [{"backend": dev, "family": f, "model": key, "mean_hellinger": round(v, 5)}
                      for f, v in zip(fams, vals, strict=True)]
        short = {"ghz": "GHZ", "bv": "BV", "mirror": "Mirror", "adder": "Adder", "qft": "QFT", "qv": "QV"}
        ax.set_xticks(x, [short[f] for f in fams], rotation=40, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        ax.set_title(DEV_LABEL[dev], loc="left")
    axes[-1].axis("off")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("Mean Hellinger fidelity\n(3–8 qubits)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False)
    fig.tight_layout(w_pad=0.6, h_pad=1.2)
    save(fig, "figC6_noise_by_family", table)


VENDOR_LABEL = {"ibm_heron": "IBM Heron r3 vs Qiskit Aer\nfrom_backend(ibm_boston)",
                "ibm_nighthawk": "IBM Nighthawk vs Qiskit Aer\nfrom_backend(ibm_miami)",
                "google_willow": "Google Willow vs Cirq\nwillow_pink noise model"}


def fig_vendor_xval() -> None:
    """The IBM and Google twins against the vendors' own noise models of the same snapshot."""
    rows = load("vendor_xval")["data"]
    devs = [d for d in VENDOR_LABEL if any(r["backend"] == d for r in rows)]
    fig = plt.figure(figsize=(DOUBLE, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.62, wspace=0.32)
    table = []
    for k, dev in enumerate(devs):
        ax = fig.add_subplot(gs[0, k])
        rs = [r for r in rows if r["backend"] == dev]
        if dev == "google_willow":
            # like for like: both apply idle decoherence; Cirq's extra decay during the 600 ns read-out removed
            x = [r["vendor_no_measurement_decay"] for r in rs]
            ax.scatter(x, [r["qputwin_full"] for r in rs], s=16, marker=DEV_MARK[dev], color=DEV_COLOR[dev],
                       edgecolors="white", linewidths=0.6, label="QPUTwin (full model)", zorder=3)
            ax.scatter(x, [r["vendor"] for r in rs], s=12, marker="+", color=INK2, linewidths=0.8,
                       label="Cirq as shipped", zorder=2)
            xl = "Cirq, no decay during read-out"
        else:
            # like for like: neither model idles an unscheduled circuit
            x = [r["vendor"] for r in rs]
            ax.scatter(x, [r["qputwin_no_idle"] for r in rs], s=16, marker=DEV_MARK[dev], color=DEV_COLOR[dev],
                       edgecolors="white", linewidths=0.6, label="QPUTwin (no idle)", zorder=3)
            ax.scatter(x, [r["median_model"] for r in rs], s=13, marker="o", facecolors="none",
                       edgecolors=INK2, linewidths=0.8, label="device medians only", zorder=2)
            xl = "IBM's Aer noise model"
        ax.plot([0, 1.02], [0, 1.02], color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel(xl, fontsize=7)
        ax.set_title(f"({'abc'[k]}) " + VENDOR_LABEL[dev].replace("\n", "\n     "), loc="left", fontsize=7.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.2, handletextpad=0.2,
                  borderaxespad=0.0, ncol=1)
        if k == 0:
            ax.set_ylabel("Hellinger fidelity to the ideal")
        table += [{key: r.get(key) for key in ("backend", "family", "qubits", "vendor", "vendor_no_measurement_decay",
                                               "qputwin_full", "qputwin_no_idle", "median_model",
                                               "tvd_no_idle_vs_vendor", "tvd_full_vs_vendor_no_meas",
                                               "tvd_median_vs_vendor", "tvd_sampling_floor")} for r in rs]
    # distance to the vendor model, per device: QPUTwin, median-only, and the sampling floor
    ax = fig.add_subplot(gs[1, :])
    keys = [("QPUTwin (like-for-like variant)", SLOTS[0], "o"), ("device medians only", SLOTS[1], "s"),
            ("sampling floor (vendor vs itself)", MUTED, "D")]
    for k, (name, col, mk) in enumerate(keys):
        for j, dev in enumerate(devs):
            rs = [r for r in rows if r["backend"] == dev]
            if k == 0:
                v = [r["tvd_full_vs_vendor_no_meas"] if dev == "google_willow" else r["tvd_no_idle_vs_vendor"]
                     for r in rs]
            elif k == 1:
                if dev == "google_willow":
                    continue          # the median Aer model has no idle term, unlike Cirq: not comparable
                v = [r["tvd_median_vs_vendor"] for r in rs]
            else:
                v = [r["tvd_sampling_floor"] for r in rs]
            y = j + (k - 1) * 0.25
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([q1, q3], [y, y], color=col, lw=2.4, solid_capstyle="butt")
            ax.plot([med], [y], marker=mk, color=col, ms=5.5, mec="white", mew=0.7,
                    label=name if j == 0 else None, ls="none")
    ax.set_yticks(range(len(devs)), [DEV_LABEL[d] for d in devs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    per_twin = sorted({sum(r["backend"] == d for r in rows) for d in devs})
    count = f"{per_twin[0]}" if len(per_twin) == 1 else f"{per_twin[0]}-{per_twin[-1]}"
    ax.set_xlabel("Total-variation distance to the vendor model's distribution (log; median and interquartile range "
                  f"over {count} circuits per twin)", fontsize=7)
    ax.set_title("(d) Distance to the vendor's model", loc="left", fontsize=7.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.4, handletextpad=0.3)
    save(fig, "figC9_vendor_cross_validation", table)


CAP_LABEL = {"qputwin_c128": "QPUTwin (complex128)", "qputwin_c64": "QPUTwin (complex64)",
             "aer_c128": "Qiskit Aer (complex128)", "aer_c64": "Qiskit Aer (complex64)", "qsim": "qsim (complex64)",
             "qulacs": "Qulacs (complex128)", "qputwin_dm": "QPUTwin", "aer_dm": "Qiskit Aer"}
CAP_STYLE = {"qputwin_c128": (SLOTS[0], "o", "-"), "qputwin_c64": (SLOTS[6], "h", "-"),
             "aer_c128": (SLOTS[1], "s", "-"), "aer_c64": (SLOTS[1], "D", "--"), "qsim": (SLOTS[4], "v", "-"),
             "qulacs": (SLOTS[5], "P", "-"), "qputwin_dm": (SLOTS[0], "o", "-"), "aer_dm": (SLOTS[1], "s", "-")}


def fig_capacity() -> None:
    """Widest runs on the whole host: time and peak memory per width, with the first failure marked."""
    d = load("capacity")
    rows, threads = d["data"], d.get("threads")
    gib = 2 ** 30
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 3.0), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    table = []
    panels = [(axes[0], "sv", "qv", "time", "(a) QV-style statevector: time"),
              (axes[1], "sv", "ghz", "mem", "(b) GHZ statevector: peak memory"),
              (axes[2], "dm", "qv", "time", "(c) Noisy density matrix (QV-style)")]
    # how the first failing width ended: a clean refusal (the simulator's own memory check), killed by the
    # kernel for memory, or over the time cap
    fail_mk = {"refused": ("o", "none"), "failed": ("o", "none"), "killed": ("X", None),
               "out_of_memory": ("X", None), "timeout": ("s", "none")}
    for ax, kind, fam, what, title in panels:
        sims = [s for s in CAP_LABEL if any(r["simulator"] == s and r["kind"] == kind for r in rows)]
        for si, sim in enumerate(sims):
            rs = sorted((r for r in rows if r["simulator"] == sim and r["kind"] == kind and r["family"] == fam),
                        key=lambda r: r["qubits"])
            ok = [r for r in rs if r["status"] == "ok"]
            col, mk, ls = CAP_STYLE[sim]
            y = [r["run_s"] if what == "time" else r["peak_rss_bytes"] / gib for r in ok]
            if ok:
                ring(ax, [r["qubits"] for r in ok], y, col, mk, ls=ls, label=CAP_LABEL[sim], markersize=4)
            fail = next((r for r in rs if r["status"] != "ok"), None)
            if fail is not None:                    # first width that failed, on the top edge, offset per simulator
                fm, fc = fail_mk.get(fail["status"], ("X", None))
                off = (si - (len(sims) - 1) / 2) * 0.13
                ax.plot([fail["qubits"] + off], [0.96], marker=fm, color=col, mfc=col if fc is None else fc,
                        mec=col, ms=5, mew=1.2, ls="none", transform=ax.get_xaxis_transform(), clip_on=False)
            table += [{"kind": kind, "family": fam, "simulator": sim, "qubits": r["qubits"], "status": r["status"],
                       "run_s": round(r.get("run_s", r.get("wall_s", 0.0)), 3),
                       "peak_rss_gib": round(r["peak_rss_bytes"] / gib, 3) if r.get("peak_rss_bytes") else "",
                       "in_place": r.get("in_place", "")} for r in rs]
        ax.set_title(title, loc="left")
        ax.set_xlabel("Qubits n")
        if what == "time":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(secs))
        else:
            host = 62.8
            ax.axhline(host, color=MUTED, lw=0.8, ls=":")
            ax.text(0.02, host / 70 + 0.01, "host RAM 62.8 GiB", fontsize=6.4, color=INK2, va="bottom",
                    transform=ax.transAxes)
            ax.set_ylabel("Peak resident memory (GiB)")
            ax.set_ylim(0, 70)
    axes[0].set_ylabel("Time per job")
    seen: dict[str, Any] = {}
    for ax in axes[:2]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(lab, h)
    from matplotlib.lines import Line2D
    for lab, (fm, fc) in (("first failure: refused cleanly", ("o", "none")), ("killed for memory", ("X", None)),
                          ("over the 20 min cap", ("s", "none"))):
        seen[lab] = Line2D([], [], marker=fm, color=INK2, mfc=INK2 if fc is None else fc, mec=INK2, ls="none", ms=5)
    fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Widest runs on the whole host ({threads} threads, 62.8 GiB RAM; markers on the top edge: "
                 "first width that failed)", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.96), w_pad=1.0)
    save(fig, "figC11_capacity", table)


WF_LABEL = {"qputwin": "QPUTwin", "qputwin_plain": "QPUTwin, connectivity-only layout",
            "vendor": "vendor workflow", "google_dm": "vendor workflow, Cirq density matrix"}
WF_VENDOR = {"ibm_heron": "IBM Heron r3: Qiskit transpile\n+ Aer from_backend",
             "ibm_nighthawk": "IBM Nighthawk: Qiskit transpile\n+ Aer from_backend",
             "google_willow": "Google Willow: Cirq RouteCQC\n+ qsim, Google noise model"}
WF_STYLE = {"qputwin": (SLOTS[0], "o"), "qputwin_plain": (SLOTS[6], "h"), "vendor": (SLOTS[1], "s"),
            "google_dm": (SLOTS[3], "D")}


def fig_vendor_workflow() -> None:
    """Device emulation end to end: time per job, and the predicted fidelity against the vendor's workflow."""
    d = load("vendor_workflow_t4")
    rows = [r for r in d["data"] if r.get("status", "ok") == "ok"]
    devs = [b for b in WF_VENDOR if any(r["backend"] == b for r in rows)]
    fig, axes = plt.subplots(2, len(devs), figsize=(DOUBLE, 4.9), squeeze=False)
    table = []
    for j, b in enumerate(devs):
        ax = axes[0, j]
        for p in ("qputwin", "vendor", "google_dm"):
            rs = [r for r in rows if r["backend"] == b and r["pipeline"] == p]
            if not rs:
                continue
            widths = sorted({r["qubits"] for r in rs})
            gm, lo, hi = [], [], []
            for n in widths:
                v = [r["run_s_best"] for r in rs if r["qubits"] == n]
                gm.append(float(np.exp(np.mean(np.log(v)))))
                lo.append(min(v))
                hi.append(max(v))
            col, mk = WF_STYLE[p]
            ax.fill_between(widths, lo, hi, color=col, alpha=0.13, lw=0)
            ring(ax, widths, gm, col, mk, label=WF_LABEL[p], markersize=4)
            table += [{"backend": b, "pipeline": p, "qubits": n, "time_s_geomean": round(g, 4),
                       "time_s_min": round(a, 4), "time_s_max": round(c, 4)}
                      for n, g, a, c in zip(widths, gm, lo, hi, strict=True)]
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(secs))
        ax.set_title(f"({'abcdef'[j]}) {WF_VENDOR[b]}", loc="left", fontsize=7.4)
        ax.set_xlabel("Logical qubits n")
        ax.set_xticks(sorted({r["qubits"] for r in rows}))
        # predicted fidelity: QPUTwin (with and without calibration-aware layout) vs the vendor workflow
        ax2 = axes[1, j]
        ven = {(r["family"], r["qubits"]): r["hellinger"] for r in rows if r["backend"] == b and r["pipeline"] == "vendor"}
        devs_med: dict[str, float] = {}
        for p in ("qputwin_plain", "qputwin"):
            pts = [(ven[(r["family"], r["qubits"])], r["hellinger"]) for r in rows
                   if r["backend"] == b and r["pipeline"] == p and (r["family"], r["qubits"]) in ven]
            if not pts:
                continue
            col, mk = WF_STYLE[p]
            x, y = zip(*pts, strict=True)
            dev = float(np.median(np.abs(np.subtract(y, x))))
            devs_med[p] = dev
            if p == "qputwin":
                ax2.plot(x, y, marker=mk, ls="none", color=col, mec="white", mew=0.6, ms=4.5,
                         label="QPUTwin, calibration-aware layout")
            else:
                ax2.plot(x, y, marker=mk, ls="none", mfc="none", mec=col, mew=0.9, ms=4.5,
                         label="QPUTwin, connectivity-only layout")
            table += [{"backend": b, "pipeline": p, "vendor_hellinger": round(a, 4), "qputwin_hellinger": round(c, 4)}
                      for a, c in pts]
        ax2.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        ax2.set_xlim(0, 1.02)
        ax2.set_ylim(0, 1.02)
        ax2.set_aspect("equal")
        ax2.set_xlabel("Vendor workflow: predicted $F_H$")
        ax2.set_title(f"({'abcdef'[len(devs) + j]}) median |Δ$F_H$| to the vendor:\n"
                      f"{devs_med.get('qputwin', float('nan')):.3f} calibrated, "
                      f"{devs_med.get('qputwin_plain', float('nan')):.3f} connectivity",
                      loc="left", fontsize=7.0)
    axes[0, 0].set_ylabel("Time per job (geometric mean,\nrange over six families)")
    axes[1, 0].set_ylabel("QPUTwin: predicted $F_H$")
    seen: dict[str, Any] = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.2, w_pad=0.8)
    save(fig, "figC12_vendor_workflow", table)


def fig_workflow_throughput() -> None:
    """96 device-emulation jobs on the whole host: QPUTwin worker replicas vs the vendor workflow in a process pool."""
    rows = load("workflow_throughput")["data"]
    devs = [b for b in WF_VENDOR if any(r["backend"] == b for r in rows)]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    x = np.arange(len(devs))
    table = []
    vals: dict[str, list[float]] = {}
    for k, p in enumerate(("qputwin", "vendor")):
        v = [next((r["jobs_per_s"] for r in rows if r["backend"] == b and r["pipeline"] == p), np.nan) for b in devs]
        vals[p] = v
        col, _ = WF_STYLE[p]
        bars = ax.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=col, label=WF_LABEL[p], edgecolor="white", lw=0.6)
        for rect, val in zip(bars, v, strict=True):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.4, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=6.4, color=INK2)
        table += [{"backend": b, "pipeline": p, "jobs_per_s": round(val, 4)} for b, val in zip(devs, v, strict=True)]
    top = np.nanmax(vals["qputwin"])
    for i, (q, ven) in enumerate(zip(vals["qputwin"], vals["vendor"], strict=True)):
        ax.text(i, top * 1.18, f"{q / ven:.1f}×", ha="center", fontsize=7, color=INK, fontweight="bold")
    ax.set_xticks(x, [DEV_LABEL[b] for b in devs])
    ax.set_ylabel("Jobs per second (16 vCPUs)")
    ax.set_ylim(0, top * 1.32)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=6.4, bbox_to_anchor=(0.0, 0.88))
    save(fig, "figC13_workflow_throughput", table)


FIGURES = [fig_correctness, fig_sv_speed, fig_threads, fig_threads16, fig_dm_speed, fig_noise_xval,
           fig_noise_by_family, fig_vendor_xval, fig_ranking, fig_capacity, fig_vendor_workflow,
           fig_workflow_throughput]


def main() -> None:
    import sys
    only = set(sys.argv[1:])
    for f in FIGURES:
        if only and not any(o in f.__name__ for o in only):
            continue
        try:
            f()
        except FileNotFoundError as e:
            print(f"  skip {f.__name__}: missing {Path(e.filename).name}")


if __name__ == "__main__":
    main()
