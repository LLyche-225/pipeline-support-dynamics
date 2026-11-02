"""Dynamic elastic-layer versus locally calibrated springs (protocol step 4).

Reference system: the static coupled Hermite-beam / plane-strain-layer
assembly of ``continuum_reference.py`` plus a consistent plane-strain soil
mass (density ``rho``, out-of-plane width ``B``) and a consistent beam mass
(line mass ``mline``).  Comparator: an independently assembled Euler-Bernoulli
beam on a uniform Winkler foundation with the v3 intact-calibrated stiffness
``k0``; for the asymmetric interface the tested transfer assumption is the
same dimensionless reduction applied to that equivalent stiffness,
``k(x) = k0 * (1 - d(x))`` -- not equality of physical interface fields.

Harmonic convention: complex amplitudes with exp(+i omega t),
``D(omega) = K + i omega C - omega^2 M``.  Rayleigh damping
``C = alpha M + beta K`` with alpha, beta fixed so the modal damping ratio is
exactly ``zeta`` at ``f1`` and ``f2``; the identical rule and coefficients are
applied to both model families, each with its own consistent M and K.

Boundary conditions (caveat, recorded in every report): beam clamped at both
ends; soil bottom fixed in both components; soil sides are vertical rollers
(horizontal displacement fixed, vertical free).  This is a finite fixture, not
a half-space impedance model.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from skfem import Basis, BilinearForm, ElementLineHermite, MeshLine
from skfem.helpers import dot
from skfem.models.elasticity import lame_parameters, linear_elasticity

import continuum_reference as cr


# Frozen numerical inputs for this benchmark (DYNAMIC_PROTOCOL step 4 and
# audit/task_layer.md).  These are numerical test inputs, not field data.
M_LINE_KG_M = 400.0
RHO_SOIL_KG_M3 = 1800.0
ZETA_TARGET = 0.02
F1_HZ = 2.0
F2_HZ = 20.0
FORCE_N = 1000.0
FOOTPRINT_M = 4.3
LOAD_CENTER_M = 6.0
GAUGE_LENGTH_M = 0.2
GAUGE_POINTS = np.linspace(-12.0, 12.0, 61)
EI_N_M2 = 162321230.0
RADIUS_M = 0.3
INTERFACE_STIFFNESS_N_M2 = 36.0e6
SOIL_NU = 0.3
WIDTH_M = 0.6
LAYER_DEPTH_M = 10.0
HALF_DOMAIN_M = 50.0
SOIL_MODULI_PA = (50.0e6, 200.0e6)
FREQUENCIES_HZ = (2.0, 5.0, 10.0, 20.0)

_GAUSS_X = cr._GAUSS_X
_GAUSS_W = cr._GAUSS_W


def rayleigh_coefficients(
    zeta: float = ZETA_TARGET, f1_hz: float = F1_HZ, f2_hz: float = F2_HZ
) -> tuple[float, float]:
    """alpha, beta such that zeta(omega) = alpha/(2 omega) + beta omega / 2 is
    exactly ``zeta`` at f1 and f2."""
    w1 = 2.0 * np.pi * f1_hz
    w2 = 2.0 * np.pi * f2_hz
    beta = 2.0 * zeta / (w1 + w2)
    alpha = 2.0 * zeta * w1 * w2 / (w1 + w2)
    return float(alpha), float(beta)


@BilinearForm
def _beam_mass(u, v, w):
    return w["mline"] * u * v


@BilinearForm
def _soil_mass(u, v, w):
    return w["rho_area"] * dot(u, v)


def _beam_spring_matrix(
    beam_basis: Basis, points: np.ndarray, stiffness: np.ndarray
) -> sparse.csr_matrix:
    """Beam-only spring matrix from per-quadrature-point stiffness values.

    Mirrors the beam-interface quadrature of
    ``continuum_reference._assemble_interface`` (4-point Gauss-Legendre per
    interval) so the comparator foundation uses the same integration rule as
    the layer interface.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    jacobians = 0.5 * np.diff(points)
    for interval in range(points.size - 1):
        beam_trace = cr._beam_quadrature_values(beam_basis, interval)
        weight = stiffness[:, interval] * jacobians[interval] * _GAUSS_W
        for first in range(4):
            global_first = int(beam_basis.element_dofs[first, interval])
            for second in range(first, 4):
                global_second = int(beam_basis.element_dofs[second, interval])
                integral = float(
                    np.sum(weight * beam_trace[first] * beam_trace[second])
                )
                rows.append(global_first)
                cols.append(global_second)
                data.append(integral)
                if global_first != global_second:
                    rows.append(global_second)
                    cols.append(global_first)
                    data.append(integral)
    return sparse.coo_matrix(
        (data, (rows, cols)), shape=(beam_basis.N, beam_basis.N)
    ).tocsr()


def _quadrature_coordinates(points: np.ndarray) -> np.ndarray:
    interval_sizes = np.diff(points)
    interval_midpoints = 0.5 * (points[:-1] + points[1:])
    return interval_midpoints[None, :] + 0.5 * interval_sizes[None, :] * _GAUSS_X[:, None]


def _fixed_and_free(
    points: np.ndarray, depth_nodes: np.ndarray, soil_mesh, n_beam: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicates the boundary conditions of continuum_forward."""
    beam_fixed = np.array([0, 1, 2 * points.size - 2, 2 * points.size - 1], dtype=int)
    x_nodes, z_nodes = soil_mesh.p
    bottom_nodes = np.flatnonzero(
        np.isclose(z_nodes, depth_nodes[-1], rtol=0.0, atol=1.0e-12)
    ).astype(int)
    side_nodes = np.flatnonzero(
        np.isclose(x_nodes, points[0], rtol=0.0, atol=1.0e-12)
        | np.isclose(x_nodes, points[-1], rtol=0.0, atol=1.0e-12)
    ).astype(int)
    soil_fixed = np.unique(
        np.concatenate([2 * bottom_nodes, 2 * bottom_nodes + 1, 2 * side_nodes])
    ).astype(int)
    soil_fixed += n_beam
    fixed = np.unique(np.concatenate([beam_fixed, soil_fixed]))
    free = np.setdiff1d(np.arange(n_beam + soil_mesh.p.shape[1] * 2), fixed)
    return fixed, free, bottom_nodes


def assemble_layer_system(
    points: np.ndarray,
    depth_nodes: np.ndarray,
    support_callable,
    *,
    ei: float = EI_N_M2,
    interface_stiffness: float = INTERFACE_STIFFNESS_N_M2,
    soil_E: float,
    soil_nu: float = SOIL_NU,
    width: float = WIDTH_M,
    rho: float = RHO_SOIL_KG_M3,
    mline: float = M_LINE_KG_M,
    support_discontinuity_points: np.ndarray | None = None,
) -> dict:
    """Assemble the dynamic coupled beam / plane-strain-layer system.

    Returns K, M (full, unconstrained), fixed/free dof sets, the bases and
    the rigid-translation mass check values.
    """
    (
        beam_interface,
        coupling,
        soil_interface,
        support,
        beam_basis,
        soil_mesh,
        soil_basis,
    ) = cr._assemble_interface(
        points, depth_nodes, support_callable, interface_stiffness,
        support_discontinuity_points,
    )
    beam_k = cr._beam_bending.assemble(beam_basis, ei=ei).tocsr() + beam_interface
    lame_lambda, lame_mu = lame_parameters(soil_E, soil_nu)
    soil_k = (
        width * linear_elasticity(Lambda=lame_lambda, Mu=lame_mu).assemble(soil_basis).tocsr()
        + soil_interface
    )
    beam_m = _beam_mass.assemble(beam_basis, mline=mline).tocsr()
    soil_m = _soil_mass.assemble(soil_basis, rho_area=rho * width).tocsr()
    stiffness = sparse.bmat([[beam_k, coupling], [coupling.T, soil_k]], format="csr")
    mass = sparse.bmat([[beam_m, None], [None, soil_m]], format="csr")
    fixed, free, bottom_nodes = _fixed_and_free(
        points, depth_nodes, soil_mesh, beam_basis.N
    )
    return {
        "K": stiffness,
        "M": mass,
        "K_beam": beam_k,
        "M_beam": beam_m,
        "M_soil": soil_m,
        "fixed": fixed,
        "free": free,
        "bottom_nodes": bottom_nodes,
        "beam_basis": beam_basis,
        "soil_basis": soil_basis,
        "soil_mesh": soil_mesh,
        "support": support,
    }


def assemble_local_beam(
    points: np.ndarray,
    support_callable,
    *,
    k0: float,
    ei: float = EI_N_M2,
    mline: float = M_LINE_KG_M,
) -> dict:
    """Independent equivalent local beam comparator.

    k(x) = k0 * (1 - d(x)): the intact-calibrated uniform equivalent stiffness
    with the same dimensionless reduction as the layer interface loss.
    """
    points = np.asarray(points, dtype=float)
    beam_mesh = MeshLine(points)
    beam_basis = Basis(beam_mesh, ElementLineHermite())
    quadrature = _quadrature_coordinates(points)
    support = cr._evaluate_support(support_callable, quadrature)
    stiffness_q = k0 * (1.0 - support)
    spring = _beam_spring_matrix(beam_basis, points, stiffness_q)
    bending = cr._beam_bending.assemble(beam_basis, ei=ei).tocsr()
    mass = _beam_mass.assemble(beam_basis, mline=mline).tocsr()
    fixed = np.array([0, 1, 2 * points.size - 2, 2 * points.size - 1], dtype=int)
    free = np.setdiff1d(np.arange(beam_basis.N), fixed)
    return {
        "K": bending + spring,
        "M": mass,
        "K_bending": bending,
        "K_spring": spring,
        "fixed": fixed,
        "free": free,
        "beam_basis": beam_basis,
        "support": support,
    }


def beam_load_vector(beam_basis: Basis, centers, load_width: float, forces) -> np.ndarray:
    """Production footprint force assembly reused from continuum_reference."""
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    forces = np.atleast_1d(np.asarray(forces, dtype=float))
    rhs, _ = cr._beam_load_vectors(beam_basis, centers, load_width, forces)
    return rhs


def gauge_strain_matrix(beam_basis: Basis, evaluation: np.ndarray, gauge_length: float):
    """Real sparse matrix mapping beam dofs to finite-gauge rotation-difference
    strain, reusing the production probe assembly."""
    evaluation = np.asarray(evaluation, dtype=float)
    _, right_rotation = cr._line_probe_matrices(beam_basis, evaluation + gauge_length / 2.0)
    _, left_rotation = cr._line_probe_matrices(beam_basis, evaluation - gauge_length / 2.0)
    return (-RADIUS_M / gauge_length) * (right_rotation - left_rotation)


def solve_harmonic(
    K: sparse.csr_matrix,
    M: sparse.csr_matrix,
    rhs: np.ndarray,
    free: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, dict]:
    """Solve (K + i omega C - omega^2 M) u = rhs on free dofs, C = alpha M + beta K.

    Complex sparse LU with diagonal scaling and one iterative refinement step,
    mirroring the static solver of continuum_reference.  Returns the full state
    (zeros on fixed dofs) and residual / work-dissipation diagnostics.
    """
    a_k = 1.0 + 1j * omega * beta
    a_m = 1j * omega * alpha - omega**2
    dynamic = (a_k * K + a_m * M).tocsc()
    A = dynamic[free][:, free].tocsc()
    b = np.asarray(rhs, dtype=complex)[free]
    diagonal = A.diagonal()
    magnitude = np.abs(diagonal)
    if np.any(~np.isfinite(magnitude)) or np.any(magnitude <= 0.0):
        raise RuntimeError("dynamic matrix has a zero or non-finite diagonal")
    scale = 1.0 / np.sqrt(magnitude)
    scaling = sparse.diags(scale)
    scaled = (scaling @ A @ scaling).tocsc()
    factorization = splu(scaled)
    solution = scale * factorization.solve(scale * b)
    solution = solution + scale * factorization.solve(scale * (b - A @ solution))

    u = np.zeros(K.shape[0], dtype=complex)
    u[free] = solution
    residual = A @ solution - b
    c_free_u = alpha * (M[free][:, free] @ solution) + beta * (K[free][:, free] @ solution)
    dissipated = 0.5 * omega**2 * float(np.vdot(solution, c_free_u).real)
    external = 0.5 * float(np.vdot(b, 1j * omega * solution).real)
    rhs_norm = float(np.max(np.abs(b))) if b.size else 0.0
    diagnostics = {
        "residual_inf_norm": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "residual_relative": float(np.max(np.abs(residual)) / max(rhs_norm, np.finfo(float).tiny)),
        "external_power_W": external,
        "dissipated_power_W": dissipated,
        "power_balance_relative_error": abs(external - dissipated)
        / max(abs(dissipated), np.finfo(float).tiny),
    }
    return u, diagnostics


def complex_l2_relative(first: np.ndarray, second: np.ndarray) -> float:
    """||first - second||_2 / ||second||_2 with complex parts retained."""
    first = np.asarray(first, dtype=complex)
    second = np.asarray(second, dtype=complex)
    return float(np.linalg.norm(first - second) / np.linalg.norm(second))


def amplitude_ratio(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Per-gauge |first| / |second|."""
    return np.abs(np.asarray(first, dtype=complex)) / np.abs(np.asarray(second, dtype=complex))
