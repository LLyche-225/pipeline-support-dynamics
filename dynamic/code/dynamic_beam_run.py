"""Step-3 driver: dynamic transfer of the 192 static support fields.

For every selected field: production harmonic response at f/f0 in
{0.25, 0.75, 1.00, 1.15} for both excitation channels, compared against
independently assembled skfem fine-grid references with the prescribed
continuous loss profile (no retuning). Plus the seed-1101 asymmetric tau=10%
smooth-pulse transient with independent reference.

Outputs (results/beam/): beam_responses.npz, beam_metrics.csv,
beam_medians.csv, beam_paired.csv, beam_transient.npz,
beam_transient_peaks.csv; audit/beam_report.json.

Run: PYTHONPATH=code:vendor OPENBLAS_NUM_THREADS=1 \
    python code/dynamic_beam_run.py
"""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from shape_factory import get_shape_function
import dynamic_beam as db
import dynamic_beam_skfem as ds
from dynamic_beam_export import load_fields
from dynamic_beam_verify import pulse_factor, rel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "beam"
PROFILES = ("intact", "asymmetric", "broad_shallow", "double")
EDGE_HINTS = {"asymmetric": (1.2,)}
CHANNELS = ("force", "ground")
METHODS = ("calibrated", "fixed", "free", "oracle")


def build_references():
    """skfem fine-grid (h=0.125) references per continuous loss profile."""
    refs = {}
    for shape in PROFILES:
        func = None if shape == "intact" else get_shape_function(shape)
        model = ds.SkfemDynamics(db.aligned_mesh(0.125, EDGE_HINTS.get(shape, ())), func)
        refs[shape] = {ratio: model.solve_frequency(ratio * db.OMEGA0)
                       for ratio in db.RATIOS}
    return refs


def run_frequency_matrix(fields, refs):
    """192 fields x 4 frequencies x 2 channels = 1536 metric rows."""
    mesh = db.aligned_mesh(0.25, (1.2,))
    rows, pred, refm = [], [], []
    t0 = time.perf_counter()
    for i, field in enumerate(fields):
        beam = db.BeamDynamics(mesh, support=np.asarray(field["coefficients"], float))
        shape = field["shape"]
        for ratio in db.RATIOS:
            sol = beam.solve_frequency(ratio * db.OMEGA0)
            for ch in CHANNELS:
                p, r = sol[ch]["strain"], refs[shape][ratio][ch]["strain"]
                rows.append({"field_index": i, "key": field["path_key"],
                             "shape": shape, "seed": field["seed"], "tau": field["tau"],
                             "method": field["method"], "status": field["status"],
                             "selected_origin": field["selected_origin"],
                             "f_over_f0": ratio, "freq_hz": ratio * db.F0, "channel": ch,
                             "rel_error": rel(p, r),
                             "pred_norm": float(np.linalg.norm(p)),
                             "ref_norm": float(np.linalg.norm(r))})
                pred.append(p)
                refm.append(r)
        if (i + 1) % 48 == 0:
            print(f"  {i + 1}/{len(fields)} fields, {time.perf_counter() - t0:.1f}s")
    return rows, np.array(pred), np.array(refm)


def run_transient(fields):
    """Seed 1101, asymmetric, tau=10%: smooth pulse, four methods + truth."""
    T = 1.0 / db.F0
    dt = T / 480
    n = int(np.ceil(3 * T / dt))
    factor = pulse_factor(T)
    selected = [f for f in fields if f["shape"] == "asymmetric" and f["seed"] == 1101
                and abs(f["tau"] - 0.1) < 1e-12]
    selected.sort(key=lambda f: METHODS.index(f["method"]))
    mesh = db.aligned_mesh(0.25, (1.2,))
    curves, peaks, labels = {}, {}, []
    for field in selected:
        beam = db.BeamDynamics(mesh, support=np.asarray(field["coefficients"], float))
        t, rec = beam.newmark(beam.b_force, factor, dt, n)
        label = field["method"]
        curves[label] = rec
        labels.append(label)
        peaks[label] = float(np.abs(rec).max())
    truth = db.BeamDynamics(mesh, support=get_shape_function("asymmetric"))
    t, rec_truth = truth.newmark(truth.b_force, factor, dt, n)
    curves["truth_production"] = rec_truth
    peaks["truth_production"] = float(np.abs(rec_truth).max())
    ref = ds.SkfemDynamics(db.aligned_mesh(0.125, (1.2,)), get_shape_function("asymmetric"))
    dt_ref = T / 1920
    t_ref, rec_ref = ref.newmark(ref.b_force, factor, dt_ref, int(np.ceil(3 * T / dt_ref)))
    ref_sub = rec_ref[::4]
    peaks["reference_skfem_fine"] = float(np.abs(rec_ref).max())
    errors = {label: rel(curves[label], ref_sub) for label in curves}
    return {
        "time": t, "gauges": db.GAUGES, "curves": curves,
        "reference_time_subsampled": t_ref[::4], "reference": ref_sub,
        "dt": dt, "dt_reference": dt_ref, "horizon_s": float(t[-1]), "T": T,
        "pulse": "sin^2(pi t/T), 0<t<T, T=1/f0",
        "peaks": peaks, "rel_errors_vs_reference": errors,
        "fields": [{"method": f["method"], "key": f["path_key"],
                    "selected_origin": f["selected_origin"]} for f in selected],
    }


def write_tables(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "beam_metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # medians per (shape, method, tau, channel, ratio) over the six seeds
    groups = {}
    for r in rows:
        groups.setdefault((r["shape"], r["method"], r["tau"], r["channel"],
                           r["f_over_f0"]), []).append(r["rel_error"])
    with open(OUT / "beam_medians.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["shape", "method", "tau", "channel", "f_over_f0",
                         "median_rel_error", "n_seeds"])
        for key in sorted(groups):
            vals = groups[key]
            writer.writerow([*key, float(np.median(vals)), len(vals)])
    # wide per-seed table for within-seed paired differences
    wide = {}
    for r in rows:
        wide.setdefault((r["shape"], r["tau"], r["channel"], r["f_over_f0"],
                         r["seed"]), {})[r["method"]] = r["rel_error"]
    with open(OUT / "beam_paired.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["shape", "tau", "channel", "f_over_f0", "seed",
                         *METHODS])
        for key in sorted(wide):
            writer.writerow([*key, *[wide[key].get(m, "") for m in METHODS]])


def main():
    t0 = time.perf_counter()
    fields, provenance = load_fields()
    assert len(fields) == 192, f"expected 192 fields, got {len(fields)}"
    print("references ...")
    refs = build_references()
    print("frequency matrix ...")
    rows, pred, refm = run_frequency_matrix(fields, refs)
    write_tables(rows)
    np.savez(OUT / "beam_responses.npz",
             gauges=db.GAUGES, pred=pred, ref=refm,
             field_index=np.array([r["field_index"] for r in rows]),
             f_over_f0=np.array([r["f_over_f0"] for r in rows]),
             channel=np.array([r["channel"] for r in rows]),
             keys=np.array([r["key"] for r in rows]),
             rel_error=np.array([r["rel_error"] for r in rows]))
    print("transient ...")
    tr = run_transient(fields)
    np.savez(OUT / "beam_transient.npz",
             time=tr["time"], gauges=tr["gauges"], reference=tr["reference"],
             **{f"strain_{k}": v for k, v in tr["curves"].items()})
    with open(OUT / "beam_transient_peaks.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["field", "peak_abs_strain", "rel_error_vs_reference"])
        for label in [*tr["peaks"]]:
            err = tr["rel_errors_vs_reference"].get(label, "")
            writer.writerow([label, tr["peaks"][label], err])
    errors = np.array([r["rel_error"] for r in rows])
    verification = json.loads((ROOT / "audit" / "beam_verification.json").read_text())
    report = {
        "schema_version": "beam-report-v1.0",
        "runtime_s": time.perf_counter() - t0,
        "fields_provenance": provenance,
        "verification": "audit/beam_verification.json",
        "verification_gate": verification["gate"],
        "verification_all_pass": verification["all_pass"],
        "frequencies": {"ratios": list(db.RATIOS), "f0_hz": db.F0,
                        "omega0_rad_s": db.OMEGA0},
        "metric_rows": len(rows),
        "worst_rel_error": {
            "overall": float(errors.max()),
            "by_channel": {ch: float(errors[[i for i, r in enumerate(rows)
                                             if r["channel"] == ch]].max())
                           for ch in CHANNELS},
        },
        "median_rel_error": float(np.median(errors)),
        "transient": {k: tr[k] for k in
                      ("dt", "dt_reference", "horizon_s", "T", "pulse", "peaks",
                       "rel_errors_vs_reference", "fields")},
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "platform": platform.platform()},
        "output_meanings": {
            "beam_metrics.csv": "one row per field x frequency x channel; "
                "rel_error = ||pred-ref||/||ref|| over 61 complex gauges",
            "beam_responses.npz": "complex 61-gauge strain vectors per metric row",
            "beam_medians.csv": "median rel_error over six seeds per group",
            "beam_paired.csv": "per-seed rel_error wide table for paired diffs",
            "beam_transient.npz": "pulse gauge-strain curves (4 methods + truth) "
                "and independent skfem fine reference",
            "beam_transient_peaks.csv": "peak |strain| and transient rel errors",
        },
        "reproduce": {
            "verify": "PYTHONPATH=code:vendor OPENBLAS_NUM_THREADS=1 "
                "python code/dynamic_beam_verify.py",
            "run": "PYTHONPATH=code:vendor OPENBLAS_NUM_THREADS=1 "
                "python code/dynamic_beam_run.py",
            "figures": "PYTHONPATH=code:vendor "
                "python code/dynamic_beam_figures.py",
            "tests": "PYTHONPATH=code:vendor "
                "python -m pytest "
                "tests/test_dynamic_beam.py -q",
        },
        "deviations": [],
        "notes": ("Finite 100 m clamped fixture; six seeds are repetitions, not "
                  "population inference; no dynamic inverse or field validation "
                  "is claimed."),
    }
    import scipy, skfem
    report["environment"]["scipy"] = scipy.__version__
    report["environment"]["skfem"] = skfem.__version__
    (ROOT / "audit" / "beam_report.json").write_text(json.dumps(report, indent=1))
    print("rows:", len(rows), "worst:", errors.max(), "median:", np.median(errors))
    print("wrote", ROOT / "audit" / "beam_report.json")


if __name__ == "__main__":
    main()
