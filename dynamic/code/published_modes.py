"""Independent published benchmark: 100 m free-free unfilled buried pipeline.

Source: G. Banushi, "Dynamic analysis of free-free Timoshenko beams on elastic
foundation under transverse transient ground deformation", Engineering Structures 352 (2026) 122080,
DOI 10.1016/j.engstruct.2025.122080; author manuscript arXiv:2507.22850v3,
Tables 4-5. The published PDF/tables are not redistributed; only the few
numeric inputs (Table 4) and compared outputs (Table 5, L = 100 m, rad/s)
needed to reproduce this benchmark are recorded below.

Reduced model (free-free Euler-Bernoulli beam on a Winkler foundation with
distributed consistent translational mass):
  - both rigid modes have omega = sqrt(k / m);
  - flexural mode n >= 3 uses root b_{n-2} of cos(b) cosh(b) = 1,
    omega^2 = (k + EI (b / L)^4) / m.

The pipe second moment of area I comes from exact circular geometry
(pi/64 (D^4 - (D - 2t)^4)); the discrepancy from using the Table 4 rounded
J = 0.0037 m^4 is reported separately. The tabulated k = 2.503e6 N/m^2 is
used as published (not recomputed from the rounded p-y description), and no
inputs are adjusted to match. An independent scikit-fem generalized
eigensolve (Hermite beam elements, free-free, consistent mass) at two mesh
refinements validates the reduced-model frequencies; its residual against
the analytic values is the discretization error, while the offset against
the published FEM values is the Timoshenko-vs-Euler-Bernoulli plus rounding
discrepancy of the source model. All frequencies are angular (rad/s).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

from skfem import BilinearForm, Basis, ElementLineHermite, MeshLine

# Exact published inputs (Banushi 2026, Table 4; unfilled 100 m pipe).
INPUTS = {
    "L_m": 100.0,
    "k_N_per_m2": 2.503e6,  # tabulated lateral soil-spring stiffness (controls)
    "m_kg_per_m": 207.56,
    "E_Pa": 210e9,
    "D_m": 1.067,
    "t_m": 0.0079,
    "J_table_m4": 0.0037,  # Table 4 rounded value, reported for comparison only
}

# Compared published values (Table 5, L = 100 m column, angular rad/s).
COMPARED_MODES = [1, 2, 6, 12, 20]
PUBLISHED_SA = {1: 109.804, 2: 109.813, 6: 116.147, 12: 229.197, 20: 585.293}
PUBLISHED_FEM = {1: 109.805, 2: 109.811, 6: 116.107, 12: 228.878, 20: 585.863}

N_MODES = max(COMPARED_MODES)
REFINEMENTS = [100, 400]  # element counts; 100 matches the paper's 1 m mesh

CITATION = (
    "G. Banushi, Engineering Structures 352 (2026) 122080, "
    "DOI 10.1016/j.engstruct.2025.122080, arXiv:2507.22850v3, Tables 4-5"
)

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "published"


def exact_second_moment(D: float, t: float) -> float:
    """Second moment of area of the exact circular pipe cross section (m^4)."""
    return np.pi / 64.0 * (D**4 - (D - 2.0 * t) ** 4)


def free_free_roots(n_roots: int) -> np.ndarray:
    """First n_roots positive roots of cos(b) cosh(b) = 1 (free-free beam)."""
    roots = []
    for j in range(1, n_roots + 1):
        b0 = (2 * j + 1) * np.pi / 2.0
        f = lambda b: np.cos(b) * np.cosh(b) - 1.0  # noqa: E731
        roots.append(brentq(f, b0 - 0.4, b0 + 0.4, xtol=1e-14, rtol=1e-15))
    return np.array(roots)


def analytic_frequencies(EI: float) -> np.ndarray:
    """Reduced-model angular frequencies (rad/s) for modes 1..N_MODES."""
    k, m, L = INPUTS["k_N_per_m2"], INPUTS["m_kg_per_m"], INPUTS["L_m"]
    omega = np.full(N_MODES, np.sqrt(k / m))  # modes 1-2: rigid
    roots = free_free_roots(N_MODES - 2)
    for j, b in enumerate(roots, start=1):  # mode n = j + 2
        omega[j + 1] = np.sqrt((k + EI * (b / L) ** 4) / m)
    return omega


@BilinearForm
def _stiffness(u, v, w):
    return w["ei"] * u.hess[0, 0] * v.hess[0, 0] + w["foundation"] * u * v


@BilinearForm
def _mass(u, v, w):
    return w["m"] * u * v


def skfem_frequencies(n_elem: int, EI: float) -> np.ndarray:
    """Free-free scikit-fem generalized eigensolve, modes 1..N_MODES (rad/s)."""
    mesh = MeshLine(np.linspace(0.0, INPUTS["L_m"], n_elem + 1))
    basis = Basis(mesh, ElementLineHermite())
    K = _stiffness.assemble(basis, ei=EI, foundation=INPUTS["k_N_per_m2"])
    M = _mass.assemble(basis, m=INPUTS["m_kg_per_m"])
    # Dense generalized solve: matrices are small and K is SPD (Winkler
    # foundation), so this is exact and avoids ARPACK non-convergence on the
    # near-degenerate rigid modes.
    vals = eigh(K.toarray(), M.toarray(), eigvals_only=True)
    return np.sqrt(np.sort(vals)[:N_MODES])


def run() -> dict:
    """Run the benchmark and write results/published outputs."""
    E = INPUTS["E_Pa"]
    I_exact = exact_second_moment(INPUTS["D_m"], INPUTS["t_m"])
    I_table = INPUTS["J_table_m4"]

    omega_exact = analytic_frequencies(E * I_exact)
    omega_table = analytic_frequencies(E * I_table)
    omega_fem = {n: skfem_frequencies(n, E * I_exact) for n in REFINEMENTS}

    roots = free_free_roots(N_MODES - 2)
    rows = []
    for mode in COMPARED_MODES:
        i = mode - 1
        fem_by_h = {str(n): float(omega_fem[n][i]) for n in REFINEMENTS}
        rows.append(
            {
                "mode": mode,
                "kind": "rigid" if mode <= 2 else "flexural",
                "root_b": None if mode <= 2 else float(roots[mode - 3]),
                "omega_analytic_exact_I_rad_s": float(omega_exact[i]),
                "omega_analytic_J0.0037_rad_s": float(omega_table[i]),
                "omega_skfem_nelem100_rad_s": fem_by_h[str(REFINEMENTS[0])],
                "omega_skfem_nelem400_rad_s": fem_by_h[str(REFINEMENTS[1])],
                "omega_published_sa_rad_s": PUBLISHED_SA[mode],
                "omega_published_fem_rad_s": PUBLISHED_FEM[mode],
                # Discretization: skfem fine minus analytic reduced model.
                "discretization_fine_minus_analytic_rad_s": float(
                    omega_fem[REFINEMENTS[-1]][i] - omega_exact[i]
                ),
                # Model gap: analytic EB reduced model minus published FEM
                # (Timoshenko/Euler-Bernoulli plus source rounding).
                "analytic_minus_published_fem_rad_s": float(
                    omega_exact[i] - PUBLISHED_FEM[mode]
                ),
                "rounding_J_minus_exact_rad_s": float(
                    omega_table[i] - omega_exact[i]
                ),
            }
        )

    result = {
        "citation": CITATION,
        "units": "rad/s (angular frequency, not cyclic Hz)",
        "inputs": INPUTS,
        "I_exact_m4": float(I_exact),
        "EI_exact": float(E * I_exact),
        "EI_table_J0.0037": float(E * I_table),
        "free_free_roots_first18": free_free_roots(N_MODES - 2).tolist(),
        "skfem_refinements_nelem": REFINEMENTS,
        "modes": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "published_modes.json", "w") as fh:
        json.dump(result, fh, indent=2)
    with open(OUT_DIR / "published_modes.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    res = run()
    for row in res["modes"]:
        print(
            f"mode {row['mode']:>2}: analytic {row['omega_analytic_exact_I_rad_s']:9.3f}  "
            f"skfem400 {row['omega_skfem_nelem400_rad_s']:9.3f}  "
            f"published FEM {row['omega_published_fem_rad_s']:9.3f} rad/s"
        )
    print(f"wrote {OUT_DIR / 'published_modes.json'} and .csv")
