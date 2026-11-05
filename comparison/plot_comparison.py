"""Publication figures for the simulator comparison (``results/*.json`` → ``figures/``).

Same conventions as ``benchmarks/plot_figures.py``: vector PDF + SVG and a 300-dpi PNG
per figure, a CSV of exactly the plotted values in ``figures/data/``, DejaVu Sans at
7–8.5 pt, and a validated categorical palette in fixed order with a distinct marker
per series.

    uv run python plot_comparison.py

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
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, OUT = HERE / "results", HERE / "figures"
DATA = OUT / "data"
DOCS = HERE.parent / "docs" / "assets" / "figures" / "comparison"      # copied for the documentation site

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#1f2328", "#52514e", "#8a8984", "#e3e2dd"
SIMS = ["QPUTwin", "Qiskit Aer", "Braket local", "Cirq", "qsim", "Qulacs"]
COLOR = {**dict(zip(SIMS, SLOTS, strict=False)), "QPUTwin (in-process)": SLOTS[0], "QPUTwin (complex64)": SLOTS[6]}
MARK = {**dict(zip(SIMS, "osD^vP", strict=True)), "QPUTwin (in-process)": "o", "QPUTwin (complex64)": "h"}
DEVICES = ["quantinuum_helios", "ionq_forte", "rigetti_ankaa_3", "iqm_emerald"]
DEV_LABEL = {"quantinuum_helios": "Quantinuum Helios", "ionq_forte": "IonQ Forte",
             "rigetti_ankaa_3": "Rigetti Ankaa-3", "iqm_emerald": "IQM Emerald"}
DEV_COLOR = dict(zip(DEVICES, SLOTS, strict=False))
DEV_MARK = dict(zip(DEVICES, "os^D", strict=True))
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
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
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


def _speed_panels(rows: list[dict[str, Any]], fams: list[str], name: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(fams), figsize=(DOUBLE, 2.6), sharey=True)
    table: list[dict[str, Any]] = []
    for ax, fam in zip(axes, fams, strict=True):
        _series(ax, rows, fam, table)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(secs))
        ax.set_xlabel("Qubits n")
        ax.set_xticks([6, 10, 14, 18, 22])
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


def fig_dm_speed() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.6), sharey=True)
    table: list[dict[str, Any]] = []
    for row, (tag, title) in enumerate((("t1", "one thread"), ("t4", "four threads"))):
        rows = load(f"dm_speed_{tag}")["data"]
        for col, fam in enumerate(("ghz", "qv")):
            ax = axes[row, col]
            _series(ax, rows, fam, table)
            for t in table:
                t.setdefault("threads", tag)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(secs))
            ax.set_title(f"({'abcd'[2 * row + col]}) {FAM_LABEL[fam]}, {title}", loc="left")
            if row == 1:
                ax.set_xlabel("Qubits n")
        axes[row, 0].set_ylabel("Time per job (best of 3)")
    seen: dict[str, Any] = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Noisy density-matrix simulation + 1024 shots (QPUTwin: its full calibrated noise model)",
                 x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97), h_pad=1.0)
    save(fig, "figC4_density_matrix_speed", table)


def fig_improvement() -> None:
    """QPUTwin's engine before (v1, per-gate kernels, 1 thread) and after (fused; 1 and 4 threads)."""
    v1sv = json.loads((RES / "v1" / "sv_speed_t1.json").read_text())["data"]
    v1dm = json.loads((RES / "v1" / "dm_speed.json").read_text())["data"]
    cases = [("sv", "ghz", 24), ("sv", "qv", 20), ("sv", "qft", 20), ("sv", "qv", 22), ("sv", "qft", 22),
             ("dm", "ghz", 12), ("dm", "qv", 10), ("dm", "qv", 12)]
    new = {"t1": {"sv": load("sv_speed_t1")["data"], "dm": load("dm_speed_t1")["data"]},
           "t4": {"sv": load("sv_speed_t4")["data"], "dm": load("dm_speed_t4")["data"]}}

    def engine(rows, fam, n, sim):
        r = next((r for r in rows if r["family"] == fam and r["qubits"] == n and r["simulator"] == sim), None)
        if r is None:
            return None
        return r.get("engine_s_best") or (min(r["engine_s"]) if isinstance(r["engine_s"], list) else r["engine_s"])

    table = []
    labels, before, after1, after4 = [], [], [], []
    for kind, fam, n in cases:
        b = engine(v1sv if kind == "sv" else v1dm, fam, n, "QPUTwin")
        a1 = engine(new["t1"][kind], fam, n, "QPUTwin (in-process)")
        a4 = engine(new["t4"][kind], fam, n, "QPUTwin")
        if None in (b, a1, a4):
            continue
        labels.append(f"{'SV' if kind == 'sv' else 'DM'} {FAM_LABEL[fam]} {n}q")
        before.append(b)
        after1.append(a1)
        after4.append(a4)
        table.append({"case": labels[-1], "before_s": round(b, 4), "after_1thread_s": round(a1, 4),
                      "after_4threads_s": round(a4, 4), "speedup_1thread": round(b / a1, 2),
                      "speedup_4threads": round(b / a4, 2)})
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.7))
    x = np.arange(len(labels))
    for k, (vals, name, col) in enumerate(((before, "before (per-gate kernels, 1 thread)", MUTED),
                                           (after1, "now, 1 thread", SLOTS[0]),
                                           (after4, "now, 4 threads", SLOTS[6]))):
        ax.bar(x + (k - 1) * 0.27, vals, width=0.27, color=col, edgecolor="white", linewidth=0.5, label=name)
    for xi, b, a4 in zip(x, before, after4, strict=True):
        ax.annotate(f"{b / a4:.0f}×", (xi + 0.27, a4), xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=6.6, color=INK2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(secs))
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("QPUTwin engine time")
    ax.set_title("Engine speed-up from fusion, lazy activation and threading (label: before ÷ now, 4 threads)",
                 loc="left")
    ax.legend(loc="upper left", ncol=3)
    save(fig, "figC7_engine_speedup", table)


def fig_ranking() -> None:
    """Every simulator's time relative to QPUTwin, per workload and thread budget (>1 = slower than QPUTwin)."""
    work = [("sv", "ghz", 24), ("sv", "qv", 20), ("sv", "qv", 24), ("sv", "qft", 20), ("sv", "qft", 24),
            ("dm", "ghz", 12), ("dm", "qv", 12)]
    c64 = "QPUTwin (complex64)"                       # qsim runs in single precision only: also compare like for like
    sims_by = {"t1": ["Qiskit Aer", "qsim", ("qsim", c64), "Qulacs", "Braket local", "Cirq"],
               "t4": ["Qiskit Aer", "qsim", ("qsim", c64), "Qulacs"]}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.1), gridspec_kw={"width_ratios": [6, 4]})
    table = []
    cmap = LinearSegmentedColormap.from_list("rk", ["#e34948", "#f6f6f4", "#2a78d6"])
    for ax, tag in zip(axes, ("t1", "t4"), strict=True):
        data = {"sv": load(f"sv_speed_{tag}")["data"], "dm": load(f"dm_speed_{tag}")["data"]}
        twin_name = "QPUTwin (in-process)" if tag == "t1" else "QPUTwin"
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
        ax.set_title(f"({'ab'[tag == 't4']}) {'one thread' if tag == 't1' else 'four threads'}", loc="left")
    axes[1].set_yticklabels([])
    cb = fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02)
    cb.set_ticks([-1, 0, 1], labels=["QPUTwin 10× slower", "equal", "QPUTwin 10× faster"])
    cb.set_label("other simulator's time ÷ QPUTwin engine time", fontsize=7)
    save(fig, "figC8_relative_speed", table)


def fig_noise_xval() -> None:
    rows = load("noise_xval")["data"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.7))
    table = []
    for dev in DEVICES:
        rs = [r for r in rows if r["backend"] == dev]
        for same in (True, False):
            sub = [r for r in rs if (r["qputwin_two_qubit"] == r["qiskit_two_qubit"]) == same]
            kw = dict(s=15, marker=DEV_MARK[dev], linewidths=0.9,
                      facecolors=DEV_COLOR[dev] if same else "none",
                      edgecolors="white" if same else DEV_COLOR[dev],
                      label=DEV_LABEL[dev] if same else None)
            axes[0].scatter([r["aer_matched"] for r in sub], [r["qputwin_no_idle"] for r in sub], **kw)
            axes[1].scatter([r["aer_matched"] for r in sub], [r["qputwin_full"] for r in sub], **kw)
            axes[2].scatter([r["qiskit_two_qubit"] for r in sub], [r["qputwin_two_qubit"] for r in sub], **kw)
        table += [{k: r[k] for k in ("backend", "family", "qubits", "aer_matched", "qputwin_no_idle", "qputwin_full",
                                     "qiskit_two_qubit", "qputwin_two_qubit")} for r in rs]
    same = [r for r in rows if r["qputwin_two_qubit"] == r["qiskit_two_qubit"]]
    d0 = np.array([r["qputwin_no_idle"] - r["aer_matched"] for r in same])
    d1 = np.array([r["qputwin_full"] - r["aer_matched"] for r in same])
    for ax, d, title in ((axes[0], d0, "(a) QPUTwin without idle noise"),
                         (axes[1], d1, "(b) QPUTwin full model (+ idle)")):
        ax.plot([0, 1.02], [0, 1.02], color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Aer, calibration-matched model")
        ax.set_title(title, loc="left")
        ax.annotate(f"same circuit (n = {len(d)}):\nmedian |Δ| = {np.median(np.abs(d)):.4f}\n"
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
    handles.append(plt.Line2D([], [], marker="o", ls="none", mfc="none", mec=MUTED, ms=4.5))
    labels.append("hollow: the compilers produced different circuits")
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), columnspacing=1.0,
               handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    save(fig, "figC5_noise_cross_validation", table)


def fig_noise_by_family() -> None:
    rows = load("noise_xval")["data"]
    fams = list(dict.fromkeys(r["family"] for r in rows))
    fig, axes = plt.subplots(1, len(DEVICES), figsize=(DOUBLE, 2.5), sharey=True)
    table = []
    for ax, dev in zip(axes, DEVICES, strict=True):
        x = np.arange(len(fams))
        for k, (key, name) in enumerate((("aer_matched", "Aer matched"), ("qputwin_no_idle", "QPUTwin, no idle"),
                                         ("qputwin_full", "QPUTwin, full"))):
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
    axes[0].set_ylabel("Mean Hellinger fidelity\n(3–8 qubits)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    save(fig, "figC6_noise_by_family", table)


FIGURES = [fig_correctness, fig_sv_speed, fig_threads, fig_dm_speed, fig_noise_xval, fig_noise_by_family,
           fig_improvement, fig_ranking]


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
