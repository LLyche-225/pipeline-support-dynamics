"""Tests for the independent published benchmark (Banushi 2026, Table 4-5)."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

import published_modes as pm  # noqa: E402


def test_rigid_modes_are_sqrt_k_over_m():
    omega = pm.analytic_frequencies(pm.INPUTS["E_Pa"] * pm.exact_second_moment(
        pm.INPUTS["D_m"], pm.INPUTS["t_m"]))
    rigid = np.sqrt(pm.INPUTS["k_N_per_m2"] / pm.INPUTS["m_kg_per_m"])
    assert omega[0] == pytest.approx(rigid, rel=0, abs=1e-12)
    assert omega[1] == pytest.approx(rigid, rel=0, abs=1e-12)


def test_free_free_roots_and_mode_indexing():
    roots = pm.free_free_roots(4)
    known = [4.730040744862704, 7.853204624095838,
             10.995607838001671, 14.137165491257207]
    assert roots == pytest.approx(known, rel=0, abs=1e-12)
    # Mode 3 is the first flexural root (mode n uses root b_{n-2}).
    EI = pm.INPUTS["E_Pa"] * pm.exact_second_moment(
        pm.INPUTS["D_m"], pm.INPUTS["t_m"])
    omega = pm.analytic_frequencies(EI)
    k, m, L = (pm.INPUTS["k_N_per_m2"], pm.INPUTS["m_kg_per_m"],
               pm.INPUTS["L_m"])
    expect3 = np.sqrt((k + EI * (roots[0] / L) ** 4) / m)
    assert omega[2] == pytest.approx(expect3, rel=0, abs=1e-12)
    assert np.all(np.diff(omega[1:]) > 0)  # modes 2..20 strictly increasing


def test_exact_geometry_vs_table_J():
    I_exact = pm.exact_second_moment(1.067, 0.0079)
    assert I_exact == pytest.approx(3.685717496e-3, rel=1e-12)
    # Rounding J = 0.0037 changes frequencies by well under 1%.
    E = pm.INPUTS["E_Pa"]
    diff = (pm.analytic_frequencies(E * pm.INPUTS["J_table_m4"])
            - pm.analytic_frequencies(E * I_exact))
    assert np.all(np.abs(diff[2:]) > 0)  # flexural modes shift
    assert np.all(np.abs(diff) < 0.01 * pm.analytic_frequencies(E * I_exact))


def test_published_values_are_angular_rad_s():
    # Recorded Table 5 values match the task's five compared modes exactly,
    # and sit on the angular scale: the cyclic value sqrt(k/m)/(2*pi) ~= 17.5
    # would be inconsistent with the tabulated mode-1 frequency.
    assert pm.PUBLISHED_FEM == {1: 109.805, 2: 109.811, 6: 116.107,
                                12: 228.878, 20: 585.863}
    rigid = np.sqrt(pm.INPUTS["k_N_per_m2"] / pm.INPUTS["m_kg_per_m"])
    assert abs(pm.PUBLISHED_FEM[1] - rigid) < 0.05  # rad/s, not Hz


def test_skfem_matches_reduced_model_and_converges():
    EI = pm.INPUTS["E_Pa"] * pm.exact_second_moment(
        pm.INPUTS["D_m"], pm.INPUTS["t_m"])
    analytic = pm.analytic_frequencies(EI)
    coarse = pm.skfem_frequencies(100, EI)
    fine = pm.skfem_frequencies(400, EI)
    # Rigid modes: discretization-exact (constant mode is in the space).
    assert coarse[0] == pytest.approx(analytic[0], rel=1e-7)
    assert fine[1] == pytest.approx(analytic[1], rel=1e-7)
    # Discretization error shrinks with refinement for flexural modes.
    for i in (5, 11, 19):  # modes 6, 12, 20
        assert abs(fine[i] - analytic[i]) <= abs(coarse[i] - analytic[i])
        assert abs(fine[i] - analytic[i]) < 2e-3 * analytic[i]


def test_run_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "OUT_DIR", tmp_path)
    res = pm.run()
    assert (tmp_path / "published_modes.json").exists()
    assert (tmp_path / "published_modes.csv").exists()
    rows = {r["mode"]: r for r in res["modes"]}
    assert set(rows) == {1, 2, 6, 12, 20}
    # Model gap stays at the few-percent Timoshenko/rounding scale, not zero.
    assert abs(rows[1]["analytic_minus_published_fem_rad_s"]) < 0.05
    assert abs(rows[20]["analytic_minus_published_fem_rad_s"]) > 1.0
