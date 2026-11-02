"""Verification for the dynamic layer benchmark (protocol step 4).

All tests use small meshes so the suite stays fast; the driver repeats the
zero-frequency and power-balance checks at production resolution.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import continuum_reference as cr
import dynamic_layer as dl
from beam_inverse import HermiteBeamFE, _hermite_shapes
from shape_factory import asymmetric_shape, intact_shape
from skfem_reference import mesh_with_load_edges

H_TEST = 1.0
DZ_TEST = 1.0
CENTERS = np.array([dl.LOAD_CENTER_M])
K0_TEST = 4_899_684.222523651  # v3 intact calibration, E = 50 MPa


def _mesh(h: float, half_domain: float, asymmetric: bool) -> np.ndarray:
    centres = np.unique(np.concatenate([CENTERS, -CENTERS]))
    return mesh_with_load_edges(
        -half_domain,
        half_domain,
        h,
        centres,
        load_width=dl.FOOTPRINT_M,
        extra_edges=np.array([1.2]) if asymmetric else None,
    )


def _depth(dz: float) -> np.ndarray:
    return np.linspace(0.0, dl.LAYER_DEPTH_M, round(dl.LAYER_DEPTH_M / dz) + 1)


def _layer(h=H_TEST, dz=DZ_TEST, soil_E=50.0e6, shape="intact", half_domain=50.0):
    points = _mesh(h, half_domain, shape == "asymmetric")
    support = asymmetric_shape if shape == "asymmetric" else intact_shape
    return points, dl.assemble_layer_system(
        points,
        _depth(dz),
        support,
        soil_E=soil_E,
        support_discontinuity_points=np.array([1.2]) if shape == "asymmetric" else None,
    )


def test_rayleigh_rule_exact():
    alpha, beta = dl.rayleigh_coefficients()
    for f in (dl.F1_HZ, dl.F2_HZ):
        w = 2.0 * np.pi * f
        zeta = alpha / (2.0 * w) + beta * w / 2.0
        assert abs(zeta - dl.ZETA_TARGET) < 1.0e-14


def test_mass_rigid_translation():
    points, system = _layer()
    n_beam_nodes = points.size
    # Beam: rigid vertical translation, one metre at every displacement dof.
    u_beam = np.zeros(system["M_beam"].shape[0])
    u_beam[0::2] = 1.0
    total_beam = float(np.ones(u_beam.size) @ (system["M_beam"] @ u_beam))
    expected_beam = dl.M_LINE_KG_M * (points[-1] - points[0])
    assert abs(total_beam - expected_beam) / expected_beam < 1.0e-12
    # Pure bending stiffness must be zero under rigid translation.
    bending_only = system["K_beam"] - _beam_interface_part(system, points)
    assert np.max(np.abs(bending_only @ u_beam)) < 1.0e-6 * dl.EI_N_M2
    # Soil: rigid vertical and horizontal translations give rho * B * volume.
    soil_mesh = system["soil_mesh"]
    n_soil_dofs = system["M_soil"].shape[0]
    area = (points[-1] - points[0]) * dl.LAYER_DEPTH_M
    expected_soil = dl.RHO_SOIL_KG_M3 * dl.WIDTH_M * area
    for component in (0, 1):
        u_soil = np.zeros(n_soil_dofs)
        u_soil[component::2] = 1.0
        total_soil = float(np.ones(n_soil_dofs) @ (system["M_soil"] @ u_soil))
        assert abs(total_soil - expected_soil) / expected_soil < 1.0e-12


def _beam_interface_part(system, points):
    """Beam-interface block of the layer K, recovered via the public assembler."""
    beam_interface, _, _, _, _, _, _ = cr._assemble_interface(
        points, _depth(DZ_TEST), intact_shape, dl.INTERFACE_STIFFNESS_N_M2
    )
    return beam_interface


@pytest.mark.parametrize("shape", ["intact", "asymmetric"])
def test_zero_frequency_matches_continuum_forward(shape):
    points, system = _layer(shape=shape)
    support = asymmetric_shape if shape == "asymmetric" else intact_shape
    gauges = dl.GAUGE_POINTS
    static = cr.continuum_forward(
        points,
        _depth(DZ_TEST),
        CENTERS,
        support,
        gauges,
        dl.GAUGE_LENGTH_M,
        dl.EI_N_M2,
        dl.INTERFACE_STIFFNESS_N_M2,
        50.0e6,
        dl.SOIL_NU,
        dl.WIDTH_M,
        dl.FORCE_N,
        dl.FOOTPRINT_M,
        dl.RADIUS_M,
        support_discontinuity_points=np.array([1.2]) if shape == "asymmetric" else None,
    )
    rhs = np.zeros(system["K"].shape[0])
    rhs[: system["beam_basis"].N] = dl.beam_load_vector(
        system["beam_basis"], CENTERS, dl.FOOTPRINT_M, [dl.FORCE_N]
    )[:, 0]
    u, diag = dl.solve_harmonic(
        system["K"], system["M"], rhs, system["free"], 0.0, 0.0, 0.0
    )
    assert diag["residual_relative"] < 1.0e-10
    strain_operator = dl.gauge_strain_matrix(
        system["beam_basis"], gauges, dl.GAUGE_LENGTH_M
    )
    strain_dynamic = strain_operator @ u[: system["beam_basis"].N]
    strain_static = np.asarray(static["strain"]).reshape(-1)
    scale = np.max(np.abs(strain_static))
    assert np.max(np.abs(strain_dynamic - strain_static)) / scale < 1.0e-8
    displacement_static = np.asarray(static["displacement"]).reshape(-1)
    probe, _ = cr._line_probe_matrices(system["beam_basis"], gauges)
    displacement_dynamic = probe @ u[: system["beam_basis"].N]
    dscale = np.max(np.abs(displacement_static))
    assert np.max(np.abs(displacement_dynamic - displacement_static)) / dscale < 1.0e-8


def test_comparator_static_matches_hermite_beam_fe_uniform():
    # Existing production inverse code, intact uniform foundation k0.
    # n_elements=800 matches the comparator h=0.125 base spacing; finer
    # HermiteBeamFE meshes show solver roundoff above 1e-6 (no refinement
    # step in that legacy solver), so the matched-spacing pair is the
    # tightest honest cross-code check.
    reference = HermiteBeamFE(
        CENTERS,
        x_min=-50.0,
        x_max=50.0,
        n_elements=800,
        gauge_length=dl.GAUGE_LENGTH_M,
        foundation=K0_TEST,
        force=dl.FORCE_N,
        load_width=dl.FOOTPRINT_M,
        ei=dl.EI_N_M2,
        radius=dl.RADIUS_M,
    )
    variables = np.zeros(reference.basis.n_variables)
    strain_reference, _ = reference.strain_and_jacobian(variables, [dl.GAUGE_POINTS])

    points = _mesh(0.125, 50.0, asymmetric=False)
    comparator = dl.assemble_local_beam(points, intact_shape, k0=K0_TEST)
    rhs = dl.beam_load_vector(
        comparator["beam_basis"], CENTERS, dl.FOOTPRINT_M, [dl.FORCE_N]
    )[:, 0]
    u, diag = dl.solve_harmonic(
        comparator["K"], comparator["M"], rhs, comparator["free"], 0.0, 0.0, 0.0
    )
    assert diag["residual_relative"] < 1.0e-6
    strain_operator = dl.gauge_strain_matrix(
        comparator["beam_basis"], dl.GAUGE_POINTS, dl.GAUGE_LENGTH_M
    )
    strain_comparator = strain_operator @ u
    assert dl.complex_l2_relative(strain_comparator, np.asarray(strain_reference).reshape(-1)) < 1.0e-6


@pytest.mark.parametrize("shape", ["intact", "asymmetric"])
def test_comparator_independent_quadrature(shape):
    # Hand-assembled Hermite beam (analytic bending + pointwise 4-pt Gauss
    # spring), independent of the scikit-fem assembly path.
    support = asymmetric_shape if shape == "asymmetric" else intact_shape
    points = _mesh(0.5, 50.0, asymmetric=(shape == "asymmetric"))
    comparator = dl.assemble_local_beam(points, support, k0=K0_TEST)

    n = points.size
    gauss, gw = np.polynomial.legendre.leggauss(4)
    t = (gauss + 1.0) / 2.0
    wq = gw / 2.0
    rows, cols, data = [], [], []
    bending4 = None
    for element in range(n - 1):
        h = points[element + 1] - points[element]
        bending = dl.EI_N_M2 / h**3 * np.array(
            [[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
             [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]]
        )
        xq = points[element] + h * t
        kq = K0_TEST * (1.0 - np.asarray(support(xq), dtype=float))
        shapes = _hermite_shapes(t, h)
        spring = h * np.einsum("q,q,qi,qj->ij", wq, kq, shapes, shapes)
        local = bending + spring
        dofs = [2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3]
        for i in range(4):
            for j in range(4):
                rows.append(dofs[i])
                cols.append(dofs[j])
                data.append(local[i, j])
    K_hand = sparse.coo_matrix((data, (rows, cols)), shape=(2 * n, 2 * n)).tocsr()
    difference = (K_hand - comparator["K"]).tocoo()
    scale = comparator["K"].diagonal().max()
    assert np.max(np.abs(difference.data)) / scale < 1.0e-8


def test_harmonic_power_balance_and_residual():
    points, system = _layer()
    rhs = np.zeros(system["K"].shape[0])
    rhs[: system["beam_basis"].N] = dl.beam_load_vector(
        system["beam_basis"], CENTERS, dl.FOOTPRINT_M, [dl.FORCE_N]
    )[:, 0]
    alpha, beta = dl.rayleigh_coefficients()
    for f_hz in dl.FREQUENCIES_HZ:
        omega = 2.0 * np.pi * f_hz
        u, diag = dl.solve_harmonic(
            system["K"], system["M"], rhs, system["free"], omega, alpha, beta
        )
        assert diag["residual_relative"] < 1.0e-9
        assert diag["power_balance_relative_error"] < 1.0e-8
        assert diag["dissipated_power_W"] > 0.0
