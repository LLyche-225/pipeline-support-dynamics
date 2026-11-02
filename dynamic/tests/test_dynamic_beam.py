"""Checks for the dynamic transfer beam. Runnable by pytest or directly.

Run: PYTHONPATH=code:vendor OPENBLAS_NUM_THREADS=1 \
    python -m pytest tests/test_dynamic_beam.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

import numpy as np
from scipy.sparse.linalg import splu

import dynamic_beam as db
import dynamic_beam_skfem as ds
from beam_inverse import HermiteBeamFE
from shape_factory import get_shape_function
from dynamic_beam_verify import pulse_factor


def _solve_ss(beam, load, omega):
    A = (beam.K + 1j * omega * beam.C - omega**2 * beam.M
         )[beam.free][:, beam.free].tocsc()
    u = np.zeros(2 * beam.points.size, dtype=complex)
    u[beam.free] = splu(A).solve(load[beam.free])
    return u


def test_ss_sine_analytic_frequency():
    """Production dynamics (gauge matrix on complex state) vs exact SS mode."""
    p0 = db.FORCE_TOTAL / db.FOOTPRINT
    pts = np.arange(db.DOMAIN[0], db.DOMAIN[1] + 0.125, 0.25)
    beam = db.BeamDynamics(pts, support=None, bc="ss")
    b = beam.sine_load(p0)
    for ratio in (0.25, 1.0):
        u = _solve_ss(beam, b, ratio * db.OMEGA0)
        exact = db.ss_sine_exact(db.GAUGES, ratio * db.OMEGA0, p0)
        err = np.linalg.norm(beam.G @ u - exact) / np.linalg.norm(exact)
        assert err < 1e-4, f"ratio {ratio}: {err}"


def test_zero_frequency_matches_static_inverse_kernel():
    coeffs = np.full(29, 0.25)
    fe = HermiteBeamFE([db.FORCE_CENTER], n_elements=400, force=db.FORCE_TOTAL)
    strain_static, _ = fe.strain_and_jacobian(coeffs, [db.GAUGES])
    beam = db.BeamDynamics(db.aligned_mesh(0.25), support=coeffs)
    out = beam.solve_frequency(0.0)
    err = np.linalg.norm(out["force"]["strain"] - strain_static) / np.linalg.norm(strain_static)
    assert err < 1e-3
    dominant = np.argmax(np.abs(strain_static))
    assert np.sign(out["force"]["strain"][dominant].real) == np.sign(strain_static[dominant])


def test_gauge_matrix_real_and_damping_phase():
    beam = db.BeamDynamics(db.aligned_mesh(0.25), support=None)
    assert np.isrealobj(beam.G.data)
    out = beam.solve_frequency(0.5 * db.OMEGA0)
    s = out["force"]["strain"]
    assert np.iscomplexobj(s)
    assert s[np.argmax(np.abs(s))].imag < 0.0  # response lags for exp(+iwt)
    zero = beam.solve_frequency(0.0)
    assert np.allclose(zero["ground"]["strain"].imag, 0.0, atol=1e-18)


def test_newmark_vs_modal_exact():
    """Average-acceleration Newmark vs exact modal oscillator (pulse)."""
    p0 = db.FORCE_TOTAL / db.FOOTPRINT
    T = 1.0 / db.F0
    pts = np.arange(db.DOMAIN[0], db.DOMAIN[1] + 0.125, 0.25)
    beam = db.BeamDynamics(pts, support=None, bc="ss")
    b = beam.sine_load(p0)
    dt = T / 240
    times, rec = beam.newmark(b, pulse_factor(T), dt, int(np.ceil(3 * T / dt)))
    q = db.ss_modal_oscillator(None, p0, times, pulse_T=T)
    exact = db.ss_gauge_from_q(db.GAUGES, q)
    assert np.abs(rec - exact).max() / np.abs(exact).max() < 1e-2


def test_newmark_step_halving():
    T = 1.0 / db.F0
    beam = db.BeamDynamics(db.aligned_mesh(0.25, (1.2,)),
                           support=get_shape_function("asymmetric"))
    f = pulse_factor(T)
    _, r1 = beam.newmark(beam.b_force, f, T / 240, int(np.ceil(1.5 * T / (T / 240))))
    _, r2 = beam.newmark(beam.b_force, f, T / 480, int(np.ceil(1.5 * T / (T / 480))))
    change = np.linalg.norm(r1 - r2[::2]) / np.linalg.norm(r2)
    assert change < 2e-2


def test_production_matches_skfem_reference():
    func = get_shape_function("asymmetric")
    prod = db.BeamDynamics(db.aligned_mesh(0.25, (1.2,)), support=func)
    ref = ds.SkfemDynamics(db.aligned_mesh(0.125, (1.2,)), func)
    for ratio in (0.25, 1.0):
        po, ro = prod.solve_frequency(ratio * db.OMEGA0), ref.solve_frequency(ratio * db.OMEGA0)
        for ch in ("force", "ground"):
            err = np.linalg.norm(po[ch]["strain"] - ro[ch]["strain"]) / np.linalg.norm(ro[ch]["strain"])
            assert err < 1e-2, f"{ch} ratio {ratio}: {err}"


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_")}.items()):
        fn()
        print("PASS", name)
