"""Verification battery for the dynamic transfer beam (protocol step 2).

Writes audit/beam_verification.json. All errors are complex response-vector
relative L2 norms over the 61 gauges; denominators are reference norms, never
point values.

Run: PYTHONPATH=code:vendor OPENBLAS_NUM_THREADS=1 \
    python code/dynamic_beam_verify.py
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from scipy.sparse.linalg import splu

from beam_inverse import HermiteBeamFE
from shape_factory import get_shape_function
import dynamic_beam as db
import dynamic_beam_skfem as ds

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("intact", "asymmetric", "broad_shallow", "double")
EDGE_HINTS = {"asymmetric": (1.2,)}


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))


def check_ss_analytic():
    """Simply supported sine-mode: production dynamics incl. gauge + phase."""
    p0 = db.FORCE_TOTAL / db.FOOTPRINT
    rows = []
    for h in (0.25, 0.125, 0.0625):
        pts = np.arange(db.DOMAIN[0], db.DOMAIN[1] + h / 2.0, h)
        beam = db.BeamDynamics(pts, support=None, bc="ss")
        b = beam.sine_load(p0)
        for ratio in db.RATIOS:
            w = ratio * db.OMEGA0
            A = (beam.K + 1j * w * beam.C - w**2 * beam.M)[beam.free][:, beam.free].tocsc()
            u = np.zeros(2 * pts.size, dtype=complex)
            u[beam.free] = splu(A).solve(b[beam.free])
            exact = db.ss_sine_exact(db.GAUGES, w, p0)
            rows.append({"mesh_h": h, "f_over_f0": ratio,
                         "rel_error": rel(beam.G @ u, exact)})
    return {"cases": rows, "worst": max(r["rel_error"] for r in rows), "target": 1e-2}


def check_zero_frequency_static():
    """omega=0 production solve vs verified static HermiteBeamFE (units/signs)."""
    coeffs = np.full(29, 0.25)
    fe = HermiteBeamFE([db.FORCE_CENTER], n_elements=400, force=db.FORCE_TOTAL)
    strain_static, _ = fe.strain_and_jacobian(coeffs, [db.GAUGES])
    beam = db.BeamDynamics(db.aligned_mesh(0.25), support=coeffs)
    out = beam.solve_frequency(0.0)
    force_err = rel(out["force"]["strain"], strain_static)
    dominant = np.argmax(np.abs(strain_static))
    return {
        "force_channel_rel_error": force_err,
        "force_strain_sign_matches_static":
            bool(np.sign(out["force"]["strain"][dominant].real)
                 == np.sign(strain_static[dominant])),
        "ground_channel_at_zero_is_real":
            bool(np.allclose(out["ground"]["strain"].imag, 0.0, atol=1e-18)),
        "target": 1e-3,
    }


def check_damped_phase_sign():
    """With exp(+iwt) and positive viscous damping the response lags: Im<0."""
    beam = db.BeamDynamics(db.aligned_mesh(0.25), support=None)
    out = beam.solve_frequency(0.5 * db.OMEGA0)
    s = out["force"]["strain"]
    dominant = np.argmax(np.abs(s))
    return {"dominant_gauge_strain": [float(s[dominant].real), float(s[dominant].imag)],
            "imaginary_part_negative": bool(s[dominant].imag < 0.0)}


def check_matched_reference():
    """Production (truth callable, h=0.25) vs independent skfem fine (h=0.125)."""
    rows = []
    refs = {}
    for shape in PROFILES:
        func = get_shape_function(shape)
        ref = ds.SkfemDynamics(db.aligned_mesh(0.125, EDGE_HINTS.get(shape, ())),
                               None if shape == "intact" else func)
        prod = db.BeamDynamics(db.aligned_mesh(0.25, EDGE_HINTS.get(shape, ())),
                               support=None if shape == "intact" else func)
        refs[shape] = ref
        for ratio in db.RATIOS:
            w = ratio * db.OMEGA0
            po, ro = prod.solve_frequency(w), ref.solve_frequency(w)
            for ch in ("force", "ground"):
                rows.append({"shape": shape, "f_over_f0": ratio, "channel": ch,
                             "rel_error": rel(po[ch]["strain"], ro[ch]["strain"])})
    return {"cases": rows, "worst": max(r["rel_error"] for r in rows),
            "target": 1e-2, "_refs": refs}


def check_fine_vs_finer(refs):
    """skfem h=0.125 vs skfem h=0.0625 (numerical convergence of reference)."""
    rows = []
    for shape in PROFILES:
        func = get_shape_function(shape)
        finer = ds.SkfemDynamics(db.aligned_mesh(0.0625, EDGE_HINTS.get(shape, ())),
                                 None if shape == "intact" else func)
        for ratio in db.RATIOS:
            w = ratio * db.OMEGA0
            rf, rr = refs[shape].solve_frequency(w), finer.solve_frequency(w)
            for ch in ("force", "ground"):
                rows.append({"shape": shape, "f_over_f0": ratio, "channel": ch,
                             "rel_error": rel(rf[ch]["strain"], rr[ch]["strain"])})
    return {"cases": rows, "worst": max(r["rel_error"] for r in rows), "target": 1e-2}


def pulse_factor(T):
    def f(t):
        t = np.asarray(t, dtype=float)
        return np.where((t > 0.0) & (t < T), np.sin(np.pi * t / T) ** 2, 0.0)
    return f


def check_newmark_modal():
    """Newmark vs exact modal oscillator (uniform SS beam, sine load)."""
    p0 = db.FORCE_TOTAL / db.FOOTPRINT
    T = 1.0 / db.F0
    pts = np.arange(db.DOMAIN[0], db.DOMAIN[1] + 0.125, 0.25)
    beam = db.BeamDynamics(pts, support=None, bc="ss")
    b = beam.sine_load(p0)
    out = {}
    dt = T / 240
    n = int(np.ceil(3 * T / dt))
    times, rec = beam.newmark(b, pulse_factor(T), dt, n)
    q = db.ss_modal_oscillator(None, p0, times, pulse_T=T)
    exact = db.ss_gauge_from_q(db.GAUGES, q)
    scale = np.abs(exact).max()
    out["pulse_max_abs_rel_error"] = float(np.abs(rec - exact).max() / scale)
    periods = 6
    w = 0.75 * db.OMEGA0
    dt_h = (2 * np.pi / w) / 240
    n_h = int(np.ceil(periods * 2 * np.pi / w / dt_h))
    times_h, rec_h = beam.newmark(b, lambda t: np.cos(w * t), dt_h, n_h)
    q_h = db.ss_modal_oscillator(w, p0, times_h)
    exact_h = db.ss_gauge_from_q(db.GAUGES, q_h)
    out["harmonic_0.75_max_abs_rel_error"] = float(
        np.abs(rec_h - exact_h).max() / np.abs(exact_h).max())
    out["target"] = 1e-2
    return out


def check_newmark_pulse_convergence():
    """Clamped asymmetric truth pulse: step halving + independent reference."""
    T = 1.0 / db.F0
    func = get_shape_function("asymmetric")
    prod = db.BeamDynamics(db.aligned_mesh(0.25, (1.2,)), support=func)
    dt1, dt2 = T / 240, T / 480
    t1, r1 = prod.newmark(prod.b_force, pulse_factor(T), dt1, int(np.ceil(3 * T / dt1)))
    t2, r2 = prod.newmark(prod.b_force, pulse_factor(T), dt2, int(np.ceil(3 * T / dt2)))
    denom = np.linalg.norm(r2)
    halving = float(np.linalg.norm(r1 - r2[::2]) / denom)
    ref = ds.SkfemDynamics(db.aligned_mesh(0.125, (1.2,)), func)
    dt_ref = T / 1920
    t_ref, r_ref = ref.newmark(ref.b_force, pulse_factor(T), dt_ref,
                               int(np.ceil(3 * T / dt_ref)))
    indep = float(np.linalg.norm(r2 - r_ref[::4]) / np.linalg.norm(r_ref[::4]))
    return {"dt_coarse": dt1, "dt_fine": dt2, "dt_reference": dt_ref,
            "horizon_s": float(t1[-1]),
            "step_halving_rel_change": halving,
            "production_vs_independent_reference_rel_error": indep,
            "targets": {"step_halving": 2e-2, "independent_reference": 1e-2},
            "peak_gauge_strain_production": float(np.abs(r2).max()),
            "peak_gauge_strain_reference": float(np.abs(r_ref).max())}


def main():
    report = {
        "schema_version": "beam-verification-v1.0",
        "boundary_treatment": ("100 m clamped-clamped finite fixture at x=+-50 m; "
                               "ends are finite clamps, never claimed as infinite"),
        "error_definition": ("complex L2 norm over 61 rotation-difference gauges "
                             "(-12..12 m), relative to reference norm"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    import scipy, skfem
    report["environment"]["scipy"] = scipy.__version__
    report["environment"]["skfem"] = skfem.__version__
    report["ss_sine_analytic"] = check_ss_analytic()
    print("ss_sine_analytic worst:", report["ss_sine_analytic"]["worst"])
    report["zero_frequency_static"] = check_zero_frequency_static()
    print("zero_frequency_static:", report["zero_frequency_static"]["force_channel_rel_error"])
    report["damped_phase_sign"] = check_damped_phase_sign()
    matched = check_matched_reference()
    refs = matched.pop("_refs")
    report["matched_reference_production_vs_skfem"] = matched
    print("matched worst:", matched["worst"])
    report["reference_fine_vs_finer"] = check_fine_vs_finer(refs)
    print("fine vs finer worst:", report["reference_fine_vs_finer"]["worst"])
    report["newmark_modal"] = check_newmark_modal()
    print("newmark modal:", report["newmark_modal"])
    report["newmark_pulse"] = check_newmark_pulse_convergence()
    print("newmark pulse:", {k: v for k, v in report["newmark_pulse"].items()})

    targets = [
        (report["ss_sine_analytic"]["worst"], 1e-2, "ss_sine_analytic"),
        (report["zero_frequency_static"]["force_channel_rel_error"], 1e-3,
         "zero_frequency_static"),
        (matched["worst"], 1e-2, "matched_reference"),
        (report["reference_fine_vs_finer"]["worst"], 1e-2, "fine_vs_finer"),
        (report["newmark_modal"]["pulse_max_abs_rel_error"], 1e-2, "newmark_modal_pulse"),
        (report["newmark_modal"]["harmonic_0.75_max_abs_rel_error"], 1e-2,
         "newmark_modal_harmonic"),
        (report["newmark_pulse"]["step_halving_rel_change"], 2e-2, "step_halving"),
        (report["newmark_pulse"]["production_vs_independent_reference_rel_error"], 1e-2,
         "newmark_independent"),
    ]
    report["gate"] = {name: {"value": v, "target": t, "pass": bool(v < t)}
                      for v, t, name in targets}
    report["all_pass"] = bool(all(v < t for v, t, _ in targets))
    out = ROOT / "audit" / "beam_verification.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out, "all_pass:", report["all_pass"])


if __name__ == "__main__":
    main()
