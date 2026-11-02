"""Final publication figures for the dynamic transfer beam (vector PDF, 6.5 in, 10 pt CM).

Reads results/beam/beam_metrics.csv and results/beam/beam_transient.npz and writes
manuscript/figures_v5/transfer_publication.pdf and pulse_publication.pdf.

Run: PYTHONPATH=code:vendor OPENBLAS_NUM_THREADS=1 \
    python code/plot_vibration_transfer.py
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-plot-vibration-transfer")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "beam"
FIGDIR = ROOT / "manuscript" / "figures_v5"

FONT_PT = 10
METHODS = ("fixed", "free", "calibrated", "oracle")
STYLE = {
    "fixed": dict(color="black", linestyle="-", marker="o"),
    "free": dict(color="#005AB5", linestyle="--", marker="s"),
    "calibrated": dict(color="#1B9E57", linestyle="-.", marker="^"),
    "oracle": dict(color="#E69F00", linestyle=":", marker="D"),
}
FREQS = (0.25, 0.75, 1.0, 1.15)
TAUS = (0.01, 0.10)
CHANNELS = ("force", "ground")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_medians():
    """6-seed medians of rel_error keyed from beam_metrics.csv (no interpolation)."""
    grouped = defaultdict(list)
    with open(RESULTS / "beam_metrics.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "selected" or row["shape"] != "asymmetric":
                continue
            key = (float(row["tau"]), row["channel"], float(row["f_over_f0"]), row["method"])
            grouped[key].append(float(row["rel_error"]))
    medians = {}
    for tau in TAUS:
        for channel in CHANNELS:
            for freq in FREQS:
                for method in METHODS:
                    values = grouped[(tau, channel, freq, method)]
                    if len(values) != 6:
                        raise RuntimeError(f"expected 6 seeds for {(tau, channel, freq, method)}, got {len(values)}")
                    medians_key = (tau, channel, freq, method)
                    medians[medians_key] = float(np.median(values))
    return medians


def fig_transfer(medians):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), sharex=True, sharey=True)
    letters = "abcd"
    for row, tau in enumerate(TAUS):
        for col, channel in enumerate(CHANNELS):
            ax = axes[row, col]
            for method in METHODS:
                ys = [100.0 * medians[(tau, channel, freq, method)] for freq in FREQS]
                st = STYLE[method]
                ax.plot(
                    FREQS,
                    ys,
                    linestyle=st["linestyle"],
                    color=st["color"],
                    marker=st["marker"],
                    markersize=4.5,
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                    linewidth=1.1,
                    label=method,
                )
            ax.set_yscale("log")
            ax.set_title(
                f"({letters[2 * row + col]}) {channel}, $\\tau={int(tau * 100)}\\%$",
                loc="left",
            )
            ax.grid(True, which="both", lw=0.3, alpha=0.5)
            ax.set_xticks(FREQS)
            if row == len(TAUS) - 1:
                ax.set_xlabel("$f/f_0$")
            if col == 0:
                ax.set_ylabel("Strain error (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.10, right=0.985, top=0.955, bottom=0.205, hspace=0.34, wspace=0.08)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )
    fig.savefig(FIGDIR / "transfer_publication.pdf")
    plt.close(fig)


def fig_pulse():
    data = np.load(RESULTS / "beam_transient.npz")
    time_ms = 1e3 * data["time"]
    gauges = data["gauges"]
    ref = data["reference"]
    i_x = int(np.argmin(np.abs(gauges - 6.0)))
    if not np.isclose(gauges[i_x], 6.0):
        raise RuntimeError(f"no gauge at x=6.0 m (nearest {gauges[i_x]})")
    i_t = int(np.unravel_index(np.argmax(np.abs(ref)), ref.shape)[0])
    t_peak_ms = time_ms[i_t]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.6))
    ax = axes[0]
    ax.plot(time_ms, 1e6 * ref[:, i_x], color="0.35", linestyle="-", linewidth=1.5, label="reference")
    for method in METHODS:
        st = STYLE[method]
        ax.plot(
            time_ms,
            1e6 * data[f"strain_{method}"][:, i_x],
            linestyle=st["linestyle"],
            color=st["color"],
            marker=st["marker"],
            markevery=95,
            markersize=4.5,
            markerfacecolor="none",
            markeredgewidth=1.0,
            linewidth=1.0,
            label=method,
        )
    ax.set_title("(a)", loc="left")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Strain at $x=6.0$ m ($\\mu\\varepsilon$)")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.margins(x=0.01)

    ax = axes[1]
    ax.axhline(0.0, color="0.35", linestyle="-", linewidth=1.0)
    for method in METHODS:
        st = STYLE[method]
        err = 1e6 * (data[f"strain_{method}"][i_t] - ref[i_t])
        ax.plot(
            gauges,
            err,
            linestyle=st["linestyle"],
            color=st["color"],
            marker=st["marker"],
            markevery=5,
            markersize=4.5,
            markerfacecolor="none",
            markeredgewidth=1.0,
            linewidth=1.0,
            label=method,
        )
    ax.set_title("(b)", loc="left")
    ax.set_xlabel("Position $x$ (m)")
    ax.set_ylabel(f"Strain error at $t={t_peak_ms:.1f}$ ms ($\\mu\\varepsilon$)")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.margins(x=0.01)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.115, right=0.985, top=0.855, bottom=0.105, hspace=0.62)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.savefig(FIGDIR / "pulse_publication.pdf")
    plt.close(fig)
    return t_peak_ms, gauges[i_x]


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    medians = load_medians()
    fig_transfer(medians)
    t_peak_ms, x_m = fig_pulse()
    print(f"transfer_publication.pdf: asymmetric shape, taus 1%/10%, channels force/ground, "
          f"4 methods x 4 frequencies, 6-seed medians, log strain error (%)")
    print(f"pulse_publication.pdf: x={x_m:.1f} m history; signed error at t={t_peak_ms:.1f} ms "
          f"(time of max |reference strain|)")


if __name__ == "__main__":
    main()
