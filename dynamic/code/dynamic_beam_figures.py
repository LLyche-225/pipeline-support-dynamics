"""Vector PDF figures for the dynamic transfer beam (6.5 in wide, 10 pt).

Run: PYTHONPATH=code:vendor OPENBLAS_NUM_THREADS=1 \
    python code/dynamic_beam_figures.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "beam"
FIG = ROOT / "manuscript" / "figures_v5"
METHODS = ("calibrated", "fixed", "free", "oracle")
COLORS = {"calibrated": "#0072B2", "fixed": "#D55E00", "free": "#009E73",
          "oracle": "#CC79A7"}
TAU_STYLE = {0.01: "-", 0.1: "--"}
CHANNEL_LABEL = {"force": "force channel",
                 "ground": "foundation-motion channel"}

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                     "legend.fontsize": 8.5, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "lines.linewidth": 1.4,
                     "figure.dpi": 200})


def fig_medians():
    paired = {}
    for r in csv.DictReader(open(OUT / "beam_paired.csv")):
        if r["shape"] != "asymmetric":
            continue
        key = (r["channel"], float(r["f_over_f0"]), float(r["tau"]))
        paired.setdefault(key, {})[r["seed"]] = {m: float(r[m]) for m in METHODS}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for ax, ch in zip(axes, ("force", "ground")):
        for m in METHODS:
            for tau in (0.01, 0.1):
                xs, ys = [], []
                for ratio in (0.25, 0.75, 1.0, 1.15):
                    per_seed = paired[(ch, ratio, tau)]
                    xs.append(ratio)
                    ys.append(np.median([per_seed[s][m] for s in per_seed]))
                ax.plot(xs, 100 * np.array(ys), TAU_STYLE[tau], color=COLORS[m],
                        marker="o", ms=3.5,
                        label=f"{m}, $\\tau$={int(tau * 100)}%")
        ax.set_yscale("log")
        ax.set_xlabel("$f/f_0$ (–)")
        ax.set_xticks([0.25, 0.75, 1.0, 1.15])
        ax.set_title(CHANNEL_LABEL[ch], fontsize=9)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("median relative gauge-strain error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(right=0.70)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.72, 0.5),
               frameon=False, title="method", title_fontsize=9)
    fig.savefig(FIG / "beam_medians.pdf")
    plt.close(fig)


def fig_transient():
    data = np.load(OUT / "beam_transient.npz")
    t_ms = 1e3 * data["time"]
    ref = data["reference"]
    truth = data["strain_truth_production"]
    i_gauge = int(np.argmax(np.abs(truth).max(axis=0)))
    i_peak = int(np.argmax(np.abs(truth[:, i_gauge])))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    ax = axes[0]
    for m in METHODS:
        ax.plot(t_ms, 1e6 * data[f"strain_{m}"][:, i_gauge], color=COLORS[m],
                label=m)
    ax.plot(t_ms, 1e6 * truth[:, i_gauge], "k-", lw=1.6, label="truth (production)")
    ax.plot(t_ms, 1e6 * ref[:, i_gauge], "k:", lw=1.2, label="reference (skfem fine)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(f"gauge strain at x={data['gauges'][i_gauge]:.1f} m (µɛ)")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax = axes[1]
    for m in METHODS:
        ax.plot(data["gauges"], 1e6 * data[f"strain_{m}"][i_peak], color=COLORS[m],
                label=m)
    ax.plot(data["gauges"], 1e6 * truth[i_peak], "k-", lw=1.6, label="truth (production)")
    ax.plot(data["gauges"], 1e6 * ref[i_peak], "k:", lw=1.2, label="reference (skfem fine)")
    ax.set_xlabel("gauge position x (m)")
    ax.set_ylabel(f"strain at t={t_ms[i_peak]:.1f} ms (µɛ)")
    ax.grid(True, lw=0.3, alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(right=0.70)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.72, 0.5),
               frameon=False)
    fig.savefig(FIG / "beam_transient.pdf")
    plt.close(fig)


def fig_verification():
    import json
    ver = json.loads((ROOT / "audit" / "beam_verification.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    ax = axes[0]
    ss = ver["ss_sine_analytic"]["cases"]
    for ratio in (0.25, 0.75, 1.0, 1.15):
        pts = [(c["mesh_h"], c["rel_error"]) for c in ss if c["f_over_f0"] == ratio]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3.5,
                label=f"$f/f_0$={ratio}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("mesh size h (m)")
    ax.set_ylabel("relative gauge-strain error (–)")
    ax.set_title("simply supported sine mode vs exact", fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax = axes[1]
    matched = ver["matched_reference_production_vs_skfem"]["cases"]
    finer = ver["reference_fine_vs_finer"]["cases"]
    shapes = ("intact", "asymmetric", "broad_shallow", "double")
    x = np.arange(len(shapes))
    for k, (cases, name) in enumerate(((matched, "production vs skfem fine"),
                                       (finer, "fine vs finer"))):
        worst = [max(c["rel_error"] for c in cases if c["shape"] == s) for s in shapes]
        ax.bar(x + 0.2 * k - 0.1, worst, width=0.18, label=name,
               color=["#0072B2", "#D55E00"][k])
    ax.axhline(0.01, color="k", ls="--", lw=0.8, label="1% target")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, rotation=20)
    ax.set_ylabel("worst relative error (–)")
    ax.set_title("harmonic transfer verification", fontsize=9)
    ax.grid(True, axis="y", which="both", lw=0.3, alpha=0.5)
    handles, labels = [], []
    for a in axes:
        h_, l_ = a.get_legend_handles_labels()
        handles += h_
        labels += l_
    fig.tight_layout()
    fig.subplots_adjust(right=0.68)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.70, 0.5),
               frameon=False)
    fig.savefig(FIG / "beam_verification.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_medians()
    fig_transient()
    fig_verification()
    print("wrote", [p.name for p in sorted(FIG.glob("beam_*.pdf"))])
