from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu


CODE = Path(__file__).resolve().parents[1] / "code"
VENDOR = Path(__file__).resolve().parents[1] / "vendor"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(VENDOR))

from continuum_reference import (  # noqa: E402
    continuum_forward,
    _assemble_interface,
    _line_probe_matrices,
    _uniform_plane_strain_layer,
)
from skfem import (  # noqa: E402
    BilinearForm,
    Basis,
    ElementLineHermite,
    LinearForm,
    MeshLine,
)
from skfem.models.elasticity import lame_parameters  # noqa: E402
from shape_factory import get_shape_function  # noqa: E402
from skfem_reference import mesh_with_load_edges  # noqa: E402


def mesh_points(n_points: int, domain: float = 6.0, load_width: float = 1.2):
    return np.unique(
        np.concatenate(
            [
                np.linspace(-domain, domain, n_points),
                [-0.5 * load_width, 0.5 * load_width],
            ]
        )
    )


def solve_reference(**overrides):
    defaults = dict(
        points=mesh_points(25),
        depth_nodes=np.array([0.0, 0.5, 1.5, 3.0]),
        load_centers=np.array([0.0]),
        support_callable=lambda x: np.zeros_like(x),
        evaluation_points=np.linspace(-4.0, 4.0, 9),
        gauge_length=0.2,
        ei=1.0e6,
        interface_stiffness=1.0e6,
        soil_E=100.0e6,
        soil_nu=0.3,
        width=1.0,
        force=1.0e4,
        load_width=1.2,
        radius=0.3,
    )
    defaults.update(overrides)
    return continuum_forward(**defaults)


def test_plane_strain_layer_has_oedometric_strain_and_bottom_reaction():
    E, nu, width, pressure = 100.0e6, 0.3, 0.6, 100.0e3
    layer = _uniform_plane_strain_layer(
        np.linspace(-2.0, 2.0, 9),
        np.array([0.0, 1.0, 2.0]),
        E,
        nu,
        width,
        pressure,
    )
    lam, mu = lame_parameters(E, nu)
    constrained_modulus = lam + 2.0 * mu
    nodes = layer["nodes"]
    surface = np.flatnonzero(np.isclose(nodes[:, 1], 0.0))
    bottom = np.flatnonzero(np.isclose(nodes[:, 1], 2.0))
    surface_uz = layer["uz"][surface]
    bottom_uz = layer["uz"][bottom]
    measured_strain = (surface_uz - bottom_uz) / 2.0
    assert np.allclose(measured_strain, pressure / constrained_modulus, rtol=2e-14)
    assert np.max(np.abs(layer["ux"])) < 1e-14
    reaction_error = (
        layer["bottom_vertical_reaction"] + layer["applied_resultant"]
    ) / layer["applied_resultant"]
    assert abs(reaction_error) < 2e-14


def test_interface_matrix_is_psd_and_zero_for_equal_constant_displacement():
    points = np.array([0.0, 1.0, 2.5])
    beam_interface, coupling, soil_interface, *_ = _assemble_interface(
        points,
        np.array([0.0, 1.0]),
        lambda x: 0.2 + 0.0 * x,
        interface_stiffness=2.0,
    )
    interface = np.asarray(sparse.bmat(
        [
            [beam_interface, coupling],
            [coupling.T, soil_interface],
        ],
        format="dense",
    ))
    equal_constant = np.zeros(interface.shape[0])
    equal_constant[: 2 * points.size : 2] = 1.0
    equal_constant[2 * points.size + 1 :: 2] = 1.0
    assert abs(equal_constant @ (interface @ equal_constant)) < 1e-12
    eigenvalues = np.linalg.eigvalsh(interface)
    assert eigenvalues[0] >= -1e-12 * max(1.0, eigenvalues[-1])


def test_reflection_symmetry_and_exact_force_linearity():
    first = solve_reference(force=1.0e4)
    second = solve_reference(force=1.7e4)
    assert first["diagnostics"]["symmetry_applicable"] is True
    assert first["diagnostics"]["max_reflection_symmetry_error"] < 1e-10
    ratio = 1.7
    for key in (
        "displacement",
        "strain",
        "beam_displacements",
        "beam_rotations",
        "soil_ux",
        "soil_uz",
    ):
        expected = ratio * first[key]
        relative = np.linalg.norm(second[key] - expected) / np.linalg.norm(expected)
        assert relative < 1e-12, f"{key}: {relative}"


def test_reaction_energy_resultant_gauge_formula_and_downward_sign():
    result = solve_reference(force=12_345.0)
    diagnostics = result["diagnostics"]
    assert diagnostics["load_resultant"][0] == pytest.approx(12_345.0, rel=1e-14)
    assert np.max(result["beam_displacements"]) > 0.0
    assert np.max(result["soil_uz"]) > 0.0
    assert np.isfinite(diagnostics["beam_support_vertical_reaction"]).all()
    assert diagnostics["soil_support_vertical_reaction"].sum() < 0.0
    assert (
        abs(diagnostics["vertical_reaction_error"][0]) / 12_345.0 < 2e-12
    )
    assert diagnostics["relative_energy_balance_error"][0] < 1e-10
    assert (
        diagnostics["free_residual_inf_norm"][0] / 12_345.0 < 1e-9
    )

    beam = Basis(MeshLine(result["beam_nodes"]), ElementLineHermite())
    beam_solution = np.empty(beam.N)
    beam_solution[0::2] = result["beam_displacements"][:, 0]
    beam_solution[1::2] = result["beam_rotations"][:, 0]
    left = result["evaluation_points"] - 0.5 * result["gauge_length"]
    right = result["evaluation_points"] + 0.5 * result["gauge_length"]
    _, left_rotations = _line_probe_matrices(beam, left)
    _, right_rotations = _line_probe_matrices(beam, right)
    expected_strain = -0.3 * (
        (right_rotations @ beam_solution) - (left_rotations @ beam_solution)
    ) / result["gauge_length"]
    assert np.allclose(result["strain"][:, 0], expected_strain, rtol=0, atol=1e-15)


def test_vector_probe_surface_extraction_multiple_points_and_loads():
    # Two asymmetric loads with different forces make both the horizontal and
    # vertical surface displacement fields nontrivial, so a component-major
    # versus point-major probe mistake changes the extracted values.
    points = mesh_points(25, load_width=1.0)
    centers = np.array([-1.5, 0.5])
    forces = np.array([1.0e4, 2.5e4])
    evaluation = np.linspace(-4.0, 4.0, 17)
    result = solve_reference(
        points=points,
        load_centers=centers,
        force=forces,
        load_width=1.0,
        evaluation_points=evaluation,
    )
    nodes = result["soil_nodes"]
    surface = np.flatnonzero(np.isclose(nodes[:, 1], 0.0))
    surface = surface[np.argsort(nodes[surface, 0])]
    extracted = result["soil_surface_vertical_displacement"]
    assert extracted.shape == (evaluation.size, centers.size)
    for load in range(centers.size):
        # Along a Quad1 surface edge the vertical displacement is exactly the
        # piecewise-linear interpolant of the nodal values.
        expected = np.interp(
            evaluation, nodes[surface, 0], result["soil_uz"][surface, load]
        )
        assert np.allclose(extracted[:, load], expected, rtol=0.0, atol=1e-13)
    # Guard against a degenerate test: both components must be nontrivial and
    # point-dependent, and the two load columns must differ.
    assert np.ptp(result["soil_uz"]) > 0.0
    assert np.ptp(result["soil_ux"][:, 0]) > 0.0
    assert np.max(np.abs(result["soil_ux"])) > 1e-6 * np.max(np.abs(result["soil_uz"]))
    assert np.linalg.norm(extracted[:, 0] - extracted[:, 1]) > 0.0


def test_force_may_be_scalar_or_one_value_per_load():
    centers = np.array([-1.0, 1.0])
    points = mesh_points(25, load_width=1.0)
    forces = np.array([100.0, 300.0])
    result = solve_reference(
        points=points,
        load_centers=centers,
        force=forces,
        load_width=1.0,
    )
    assert result["displacement"].shape == (9, 2)
    assert np.allclose(result["diagnostics"]["applied_force"], forces)
    assert np.allclose(result["diagnostics"]["load_resultant"], forces)
    single = solve_reference(
        points=points,
        load_centers=np.array([1.0]),
        force=100.0,
        load_width=1.0,
    )
    relative = np.linalg.norm(
        result["displacement"][:, 1] - 3.0 * single["displacement"][:, 0]
    ) / np.linalg.norm(result["displacement"][:, 1])
    assert relative < 1e-13


@BilinearForm
def _winkler_form(u, v, w):
    return w["ei"] * u.hess[0, 0] * v.hess[0, 0] + w["k"] * u * v


@LinearForm
def _winkler_load(v, w):
    return w["pressure"] * (np.abs(w.x[0] - w["center"]) <= w["half"]) * v


def independent_winkler_beam(points, ei, foundation, center, force, load_width):
    basis = Basis(MeshLine(points), ElementLineHermite())
    stiffness = _winkler_form.assemble(
        basis, ei=ei, k=foundation
    ).tocsr()
    rhs = _winkler_load.assemble(
        basis,
        center=center,
        half=0.5 * load_width,
        pressure=force / load_width,
    )
    free = np.arange(2, basis.N - 2)
    solution = np.zeros(basis.N)
    solution[free] = splu(stiffness[free][:, free].tocsc()).solve(rhs[free])
    return solution[0::2]


def test_stiff_soil_sequence_converges_to_independent_winkler_beam():
    points = mesh_points(25, load_width=1.0)
    reference = independent_winkler_beam(
        points, ei=1.0e6, foundation=1.0e6, center=0.0, force=1.0e4, load_width=1.0
    )
    errors = []
    for soil_E in (1.0e10, 1.0e11, 1.0e12):
        result = solve_reference(
            points=points,
            soil_E=soil_E,
            load_width=1.0,
        )
        errors.append(
            np.linalg.norm(result["beam_displacements"][:, 0] - reference)
            / np.linalg.norm(reference)
        )
    assert errors[0] < 5.0e-4
    assert errors[1] < errors[0] / 5.0
    assert errors[2] < errors[1] / 5.0
    assert errors[2] < 5.0e-5


def test_invalid_geometry_units_and_loss_are_rejected():
    valid = solve_reference()
    assert np.isfinite(valid["strain"]).all()

    duplicate_points = np.r_[mesh_points(25)[:-1], mesh_points(25)[-2]]
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_reference(points=duplicate_points)
    with pytest.raises(ValueError, match="footprint edges"):
        solve_reference(points=np.linspace(-6.0, 6.0, 25))
    with pytest.raises(ValueError, match="interface loss"):
        solve_reference(support_callable=lambda x: 0.951 + 0.0 * x)
    with pytest.raises(ValueError, match="physical parameters"):
        solve_reference(width=-0.6)
    with pytest.raises(ValueError, match="gauge_length"):
        solve_reference(gauge_length=0.0)
    with pytest.raises(ValueError, match="force"):
        solve_reference(force=np.array([1.0, 2.0]))


def test_smooth_interface_loss_mesh_refinement_is_bounded_and_convergent():
    smooth_loss = lambda x: 0.35 * np.exp(-((x / 1.2) ** 2))  # noqa: E731
    evaluation = np.linspace(-4.0, 4.0, 9)
    results = []
    for n_x, n_z in ((17, 4), (31, 5), (49, 7)):
        result = solve_reference(
            points=mesh_points(n_x),
            depth_nodes=np.linspace(0.0, 3.0, n_z),
            support_callable=smooth_loss,
            evaluation_points=evaluation,
        )
        assert result["mesh"]["soil_dofs"] < 10_000
        assert result["diagnostics"]["relative_energy_balance_error"][0] < 1e-10
        results.append(result)
    coarse_difference = np.linalg.norm(
        results[0]["strain"][:, 0] - results[1]["strain"][:, 0]
    ) / np.linalg.norm(results[1]["strain"][:, 0])
    fine_difference = np.linalg.norm(
        results[1]["strain"][:, 0] - results[2]["strain"][:, 0]
    ) / np.linalg.norm(results[2]["strain"][:, 0])
    assert coarse_difference < 0.05
    assert fine_difference < coarse_difference


def test_production_fine_mesh_force_and_energy_balance_within_gate():
    """Regression for scaled-LU roundoff at production mesh size.

    The coordinator mesh pilot found the finest production case (E = 50 MPa,
    h = dz = 0.125, asymmetric interface, ~1.3e5 free dofs) slightly over the
    1e-8 force/energy balance gate before one-step iterative refinement was
    added to ``_scaled_lu_solve`` (force 1.67e-8, energy 1.72e-8).  This test
    reruns exactly that configuration and enforces the gate.
    """
    centres = np.array([-4.0, 0.0, 4.0, 6.0])
    points = mesh_with_load_edges(
        -50.0, 50.0, 0.125, np.r_[centres, -centres], extra_edges=np.array([1.2])
    )
    depth_nodes = np.linspace(0.0, 10.0, 81)
    evaluation = np.unique(
        np.r_[np.linspace(-12.0, 12.0, 241), np.linspace(-12.0, 12.0, 20)]
    )
    result = continuum_forward(
        points,
        depth_nodes,
        centres,
        get_shape_function("asymmetric"),
        evaluation,
        0.2,
        162321230.0,
        36e6,
        50e6,
        0.3,
        0.6,
        750000.0,
        4.3,
        0.3,
    )
    diagnostics = result["diagnostics"]
    force_error = np.max(np.abs(diagnostics["vertical_reaction_error"])) / 750000.0
    energy_error = np.max(diagnostics["relative_energy_balance_error"])
    assert force_error <= 1e-8
    assert energy_error <= 1e-8
