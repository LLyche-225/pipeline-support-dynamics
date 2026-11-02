"""Independent Euler--Bernoulli beam / continuum-soil forward model.

The model is an idealized plane-strain reference.  It is not measured data and
is not a validated buried-pipe continuum.  Coordinates use x horizontal and z
positive downward; all vertical displacements and forces are positive downward.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
from skfem import (  # noqa: E402
    BilinearForm,
    Basis,
    ElementLineHermite,
    ElementQuad1,
    ElementVector,
    MeshLine,
    MeshQuad,
)
from skfem.models.elasticity import lame_parameters, linear_elasticity  # noqa: E402


_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)
_LOSS_LIMIT = 0.95


@BilinearForm
def _beam_bending(u, v, w):
    return w["ei"] * u.hess[0, 0] * v.hess[0, 0]


def _strictly_increasing(values: np.ndarray, label: str, scale: float) -> None:
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{label} must be a 1D array with at least two values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite")
    spacing = np.diff(values)
    if np.any(spacing <= 0.0):
        raise ValueError(f"{label} must be strictly increasing")
    if np.min(spacing) <= 1.0e-10 * scale:
        raise ValueError(f"{label} contains a sliver interval")


def _validate_geometry(
    points: np.ndarray,
    depth_nodes: np.ndarray,
    load_centers: np.ndarray,
    load_width: float,
    evaluation_points: np.ndarray,
    gauge_length: float,
    support_discontinuity_points: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    depth_nodes = np.asarray(depth_nodes, dtype=float)
    _strictly_increasing(points, "points", max(1.0, float(np.ptp(points))))
    _strictly_increasing(depth_nodes, "depth_nodes", max(1.0, float(np.ptp(depth_nodes))))
    if not np.isclose(depth_nodes[0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("depth_nodes[0] must be exactly zero (soil surface)")
    if not np.isclose(points[0] + points[-1], 0.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("horizontal nodes must span a symmetric domain [-L, L]")
    if points[-1] <= 0.0 or depth_nodes[-1] <= 0.0:
        raise ValueError("L and soil depth H must be positive")

    centers = np.atleast_1d(np.asarray(load_centers, dtype=float))
    if centers.ndim != 1 or centers.size == 0 or not np.all(np.isfinite(centers)):
        raise ValueError("load_centers must be a nonempty finite 1D array")
    half_width = load_width / 2.0
    if np.any(centers - half_width < points[0]) or np.any(centers + half_width > points[-1]):
        raise ValueError("every load footprint must lie inside the beam domain")

    coordinate_tolerance = 1.0e-11 * max(1.0, float(np.ptp(points)))
    required_edges = np.unique(
        np.concatenate([centers - half_width, centers + half_width])
    )
    for edge in required_edges:
        if not np.any(np.isclose(points, edge, rtol=0.0, atol=coordinate_tolerance)):
            raise ValueError(
                "load footprint edges must be present exactly in horizontal mesh points"
            )
    if support_discontinuity_points is not None:
        discontinuities = np.unique(np.asarray(support_discontinuity_points, dtype=float))
        if discontinuities.ndim != 1 or not np.all(np.isfinite(discontinuities)):
            raise ValueError("support_discontinuity_points must be finite and 1D")
        if np.any(discontinuities < points[0]) or np.any(discontinuities > points[-1]):
            raise ValueError("support discontinuities must lie inside the beam domain")
        for edge in discontinuities:
            if not np.any(np.isclose(points, edge, rtol=0.0, atol=coordinate_tolerance)):
                raise ValueError(
                    "support discontinuity edges must be present exactly in horizontal mesh points"
                )

    evaluation_points = np.atleast_1d(np.asarray(evaluation_points, dtype=float))
    if evaluation_points.ndim != 1 or not np.all(np.isfinite(evaluation_points)):
        raise ValueError("evaluation_points must be a finite 1D array")
    if not np.isfinite(gauge_length) or gauge_length <= 0.0:
        raise ValueError("gauge_length must be positive for finite-gauge strain")
    left = evaluation_points - gauge_length / 2.0
    right = evaluation_points + gauge_length / 2.0
    if np.any(left < points[0]) or np.any(right > points[-1]):
        raise ValueError("gauge endpoints must lie inside the beam domain")
    return points, depth_nodes, centers, evaluation_points


def _validate_physical_parameters(
    ei: float,
    interface_stiffness: float,
    soil_E: float,
    soil_nu: float,
    width: float,
    load_width: float,
    radius: float,
) -> None:
    values = {
        "ei": ei,
        "interface_stiffness": interface_stiffness,
        "soil_E": soil_E,
        "width": width,
        "load_width": load_width,
        "radius": radius,
    }
    if not all(np.isfinite(value) and value > 0.0 for value in values.values()):
        raise ValueError("physical parameters must be finite and positive")
    if not np.isfinite(soil_nu) or not -1.0 < soil_nu < 0.5:
        raise ValueError("soil_nu must be finite and lie in (-1, 0.5)")


def _validate_force(force: float | np.ndarray, n_loads: int) -> np.ndarray:
    forces = np.asarray(force, dtype=float)
    if forces.ndim == 0:
        forces = np.full(n_loads, float(forces))
    elif forces.ndim == 1 and forces.size == n_loads:
        forces = forces.copy()
    else:
        raise ValueError("force must be a scalar or have one value per load")
    if not np.all(np.isfinite(forces)) or np.any(forces <= 0.0):
        raise ValueError("forces must be finite and positive (downward)")
    return forces


def _evaluate_support(
    callable_: Callable[[np.ndarray], np.ndarray | float],
    quadrature_coordinates: np.ndarray,
) -> np.ndarray:
    evaluated = np.asarray(callable_(quadrature_coordinates))
    if evaluated.ndim == 0:
        evaluated = np.broadcast_to(float(evaluated), quadrature_coordinates.shape)
    if evaluated.shape != quadrature_coordinates.shape:
        raise ValueError(
            "support_callable must return a scalar or pointwise values matching its input"
        )
    if not np.all(np.isfinite(evaluated)):
        raise ValueError("support_callable returned non-finite interface loss")
    if np.any(evaluated < 0.0) or np.any(evaluated > _LOSS_LIMIT):
        raise ValueError("interface loss must lie in [0, 0.95]")
    return np.asarray(evaluated, dtype=float)


def _top_surface_facets(
    mesh: MeshQuad,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    surface_mask = np.all(np.isclose(mesh.p[1, mesh.facets], 0.0, rtol=0.0, atol=1.0e-12), axis=0)
    facet_ids = np.flatnonzero(surface_mask)
    if facet_ids.size != mesh.p.shape[1] // (np.unique(mesh.p[1]).size) - 1:
        raise RuntimeError("failed to identify all top-surface facets")
    facet_nodes = mesh.facets[:, facet_ids]
    surface_nodes = np.flatnonzero(np.isclose(mesh.p[1], 0.0, rtol=0.0, atol=1.0e-12))
    surface_x = mesh.p[0, surface_nodes]
    interval_for_facet: dict[tuple[int, int], int] = {}
    for facet, facet_id in enumerate(facet_ids):
        left, right = facet_nodes[:, facet]
        if mesh.p[0, left] > mesh.p[0, right]:
            left, right = right, left
        interval_for_facet[(int(left), int(right))] = int(facet_id)
    return facet_ids, surface_nodes, interval_for_facet


def _assemble_interface(
    points: np.ndarray,
    depth_nodes: np.ndarray,
    support_callable: Callable[[np.ndarray], np.ndarray | float],
    interface_stiffness: float,
    support_discontinuity_points: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, np.ndarray, Basis, MeshQuad, Basis]:
    if support_discontinuity_points is not None:
        # Geometry is validated by the public function before assembly. This
        # internal path is used only by focused verification tests.
        edges = np.asarray(support_discontinuity_points, dtype=float)
        tolerance = 1.0e-11 * max(1.0, float(np.ptp(points)))
        for edge in edges:
            if not np.any(np.isclose(points, edge, rtol=0.0, atol=tolerance)):
                raise ValueError("support discontinuity edges must coincide with mesh points")

    beam_mesh = MeshLine(points)
    beam_basis = Basis(beam_mesh, ElementLineHermite())
    soil_mesh = MeshQuad.init_tensor(points, depth_nodes)
    soil_basis = Basis(soil_mesh, ElementVector(ElementQuad1()))
    _, surface_nodes, facet_for_interval = _top_surface_facets(soil_mesh)
    surface_x = soil_mesh.p[0, surface_nodes]

    jacobians = 0.5 * np.diff(points)[None, None, :]
    interval_sizes = np.diff(points)
    interval_midpoints = 0.5 * (points[:-1] + points[1:])
    # Map reference coordinates s in [-1, 1] onto each interval as
    # midpoint + (h / 2) * s, i.e. x_left + h * (1 + s) / 2.
    quadrature_coordinates = (
        interval_midpoints[None, :] + 0.5 * interval_sizes[None, :] * _GAUSS_X[:, None]
    )
    support = _evaluate_support(support_callable, quadrature_coordinates)
    effective_k = interface_stiffness * (1.0 - support) * jacobians * _GAUSS_W[None, :, None]

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    soil_rows: list[int] = []
    soil_cols: list[int] = []
    soil_data: list[float] = []
    coupling_rows: list[int] = []
    coupling_cols: list[int] = []
    coupling_data: list[float] = []

    for interval in range(points.size - 1):
        beam_left, beam_right = int(beam_mesh.t[0, interval]), int(beam_mesh.t[1, interval])
        soil_left_surface = int(surface_nodes[np.argmin(np.abs(surface_x - points[beam_left]))])
        soil_right_surface = int(surface_nodes[np.argmin(np.abs(surface_x - points[beam_right]))])
        if (
            not np.isclose(soil_mesh.p[0, soil_left_surface], points[beam_left], rtol=0.0, atol=1.0e-12)
            or not np.isclose(soil_mesh.p[0, soil_right_surface], points[beam_right], rtol=0.0, atol=1.0e-12)
        ):
            raise RuntimeError("beam and soil top meshes do not share horizontal nodes")
        left_node, right_node = soil_left_surface, soil_right_surface
        try:
            facet_id = facet_for_interval[(left_node, right_node)]
        except KeyError as error:
            raise RuntimeError("beam and soil top meshes do not share horizontal facets") from error
        soil_left, soil_right = soil_mesh.facets[:, facet_id]
        if soil_mesh.p[0, soil_left] > soil_mesh.p[0, soil_right]:
            soil_left, soil_right = soil_right, soil_left
        soil_vertical = (2 * int(soil_left) + 1, 2 * int(soil_right) + 1)
        left_basis_value = 0.5 * (1.0 - _GAUSS_X)
        right_basis_value = 0.5 * (1.0 + _GAUSS_X)
        soil_trace = np.vstack(
            [left_basis_value[None, :], right_basis_value[None, :]]
        )[:, :, None]
        beam_trace = np.zeros((4, _GAUSS_X.size, 1))
        # `_beam_values_at_quadrature` uses one column per element.
        beam_trace[:, :, 0] = _beam_quadrature_values(beam_basis, interval)

        for first in range(4):
            global_first = int(beam_basis.element_dofs[first, interval])
            for second in range(first, 4):
                global_second = int(beam_basis.element_dofs[second, interval])
                integral = np.sum(effective_k[:, :, interval] * beam_trace[first, :, 0] * beam_trace[second, :, 0])
                rows.append(global_first)
                cols.append(global_second)
                data.append(integral)
                if global_first != global_second:
                    rows.append(global_second)
                    cols.append(global_first)
                    data.append(integral)
            for soil_local, soil_global in enumerate(soil_vertical):
                integral = -np.sum(
                    effective_k[:, :, interval]
                    * beam_trace[first, :, 0]
                    * soil_trace[soil_local, :, 0]
                )
                coupling_rows.append(global_first)
                coupling_cols.append(soil_global)
                coupling_data.append(integral)
        for first, first_global in enumerate(soil_vertical):
            for second in range(first, 2):
                second_global = soil_vertical[second]
                integral = np.sum(
                    effective_k[:, :, interval]
                    * soil_trace[first, :, 0]
                    * soil_trace[second, :, 0]
                )
                soil_rows.append(first_global)
                soil_cols.append(second_global)
                soil_data.append(integral)
                if first_global != second_global:
                    soil_rows.append(second_global)
                    soil_cols.append(first_global)
                    soil_data.append(integral)

    beam_interface = sparse.coo_matrix(
        (data, (rows, cols)), shape=(beam_basis.N, beam_basis.N)
    ).tocsr()
    soil_interface = sparse.coo_matrix(
        (soil_data, (soil_rows, soil_cols)), shape=(soil_basis.N, soil_basis.N)
    ).tocsr()
    coupling = sparse.coo_matrix(
        (coupling_data, (coupling_rows, coupling_cols)),
        shape=(beam_basis.N, soil_basis.N),
    ).tocsr()
    return (
        beam_interface,
        coupling,
        soil_interface,
        support,
        beam_basis,
        soil_mesh,
        soil_basis,
    )


def _beam_quadrature_values(beam_basis: Basis, interval: int) -> np.ndarray:
    local = 0.5 * (_GAUSS_X + 1.0)[None, None, :]
    values = np.empty((4, _GAUSS_X.size))
    for local_basis in range(4):
        field = beam_basis.elem.gbasis(
            beam_basis.mapping, local, local_basis, tind=np.array([interval])
        )[0]
        values[local_basis] = np.asarray(field.value).reshape(-1)
    return values


def _beam_load_vectors(
    beam_basis: Basis,
    centers: np.ndarray,
    load_width: float,
    forces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = beam_basis.mesh
    values = np.empty((4, _GAUSS_X.size, mesh.t.shape[1]))
    for interval in range(mesh.t.shape[1]):
        values[:, :, interval] = _beam_quadrature_values(beam_basis, interval)
    jacobians = 0.5 * np.diff(mesh.p[0])[None, None, :]
    rhs = np.zeros((beam_basis.N, centers.size))
    resultants = np.empty(centers.size)
    for load, center in enumerate(centers):
        pressure = forces[load] / load_width
        interval_centers = 0.5 * (mesh.p[0, :-1] + mesh.p[0, 1:])
        inside_intervals = np.abs(interval_centers - center) <= 0.5 * load_width
        inside = inside_intervals[None, None, :]
        weighted = pressure * inside * jacobians * _GAUSS_W[None, :, None]
        for local_basis in range(4):
            global_dofs = beam_basis.element_dofs[local_basis]
            np.add.at(
                rhs[:, load],
                global_dofs,
                np.sum(weighted * values[local_basis], axis=(0, 1)),
            )
        resultants[load] = np.sum(weighted)
    return rhs, resultants


def _line_probe_matrices(
    beam_basis: Basis,
    evaluation_points: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    points = beam_basis.mesh.p[0]
    value_rows: list[int] = []
    value_cols: list[int] = []
    value_data: list[float] = []
    rotation_rows: list[int] = []
    rotation_cols: list[int] = []
    rotation_data: list[float] = []
    for row, coordinate in enumerate(evaluation_points):
        interval = int(np.searchsorted(points, coordinate, side="right") - 1)
        interval = min(max(interval, 0), points.size - 2)
        length = points[interval + 1] - points[interval]
        local = (coordinate - points[interval]) / length
        for local_basis in range(4):
            field = beam_basis.elem.gbasis(
                beam_basis.mapping,
                np.array([[local]]),
                local_basis,
                tind=np.array([interval]),
            )[0]
            global_dof = int(beam_basis.element_dofs[local_basis, interval])
            basis_value = float(np.asarray(field.value).reshape(-1)[0])
            basis_rotation = float(np.asarray(field.grad[0]).reshape(-1)[0])
            if abs(basis_value) > 1.0e-15:
                value_rows.append(row)
                value_cols.append(global_dof)
                value_data.append(basis_value)
            if abs(basis_rotation) > 1.0e-15:
                rotation_rows.append(row)
                rotation_cols.append(global_dof)
                rotation_data.append(basis_rotation)
    size = beam_basis.N
    values = sparse.coo_matrix(
        (value_data, (value_rows, value_cols)), shape=(evaluation_points.size, size)
    ).tocsr()
    rotations = sparse.coo_matrix(
        (rotation_data, (rotation_rows, rotation_cols)),
        shape=(evaluation_points.size, size),
    ).tocsr()
    return values, rotations


def _scaled_lu_solve(matrix: sparse.csr_matrix, rhs: np.ndarray) -> tuple[np.ndarray, object]:
    diagonal = matrix.diagonal()
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise RuntimeError("free stiffness matrix has a non-positive diagonal")
    scale = 1.0 / np.sqrt(diagonal)
    scaling = sparse.diags(scale)
    scaled = scaling @ matrix @ scaling
    factorization = splu(scaled.tocsc())
    solution = scaling @ factorization.solve(scaling @ rhs)
    # One step of iterative refinement reusing the same factorization.  At
    # production mesh sizes (~1.3e5 free dofs) LU roundoff leaves relative
    # reaction/energy residuals slightly above 1e-8; a single correction solve
    # on the unscaled residual removes that roundoff without changing the
    # assembled model.  On smaller, well-resolved systems the correction is at
    # the backward-stable floor and leaves the solution unchanged in practice.
    solution = solution + scaling @ factorization.solve(scaling @ (rhs - matrix @ solution))
    return solution, factorization


def _reaction_diagnostics(
    stiffness: sparse.csr_matrix,
    rhs: np.ndarray,
    solution: np.ndarray,
    free: np.ndarray,
    forces: np.ndarray,
    n_beam_nodes: int,
    bottom_nodes: np.ndarray,
) -> dict[str, np.ndarray | float]:
    residual = stiffness @ solution - rhs
    energy = 0.5 * np.einsum("ij,ij->j", solution, stiffness @ solution)
    work = np.einsum("ij,ij->j", solution, rhs)
    beam_support_dofs = np.array([0, 2 * (n_beam_nodes - 1)], dtype=int)
    soil_bottom_vertical = 2 * n_beam_nodes + 2 * bottom_nodes + 1
    beam_reaction = residual[beam_support_dofs].sum(axis=0)
    soil_reaction = residual[soil_bottom_vertical].sum(axis=0)
    total_reaction = beam_reaction + soil_reaction
    return {
        "applied_force": forces,
        "beam_support_vertical_reaction": beam_reaction,
        "soil_support_vertical_reaction": soil_reaction,
        "total_vertical_reaction": total_reaction,
        "vertical_reaction_error": forces + total_reaction,
        "strain_energy": energy,
        "external_work": work,
        "energy_balance_error": 2.0 * energy - work,
        "relative_energy_balance_error": np.abs(2.0 * energy - work)
        / np.maximum(np.abs(work), np.finfo(float).tiny),
        "free_residual_inf_norm": np.max(np.abs(residual[free]), axis=0),
    }


def _symmetry_diagnostics(
    points: np.ndarray,
    soil_nodes: np.ndarray,
    centers: np.ndarray,
    support: np.ndarray,
    beam_displacements: np.ndarray,
    soil_ux: np.ndarray,
    soil_uz: np.ndarray,
) -> dict[str, bool | float | np.ndarray]:
    applicable = bool(
        np.allclose(points + points[::-1], 0.0, rtol=0.0, atol=1.0e-11 * max(1.0, float(np.ptp(points))))
        and centers.size == 1
        and np.isclose(centers[0], 0.0, rtol=0.0, atol=1.0e-11)
        and float(np.max(support)) <= 1.0e-14
    )
    if not applicable:
        return {
            "symmetry_applicable": False,
            "beam_reflection_symmetry_error": -1.0,
            "soil_uz_reflection_symmetry_error": -1.0,
            "soil_ux_reflection_symmetry_error": -1.0,
            "max_reflection_symmetry_error": -1.0,
        }

    def relative_error(first: np.ndarray, second: np.ndarray) -> float:
        scale = max(float(np.max(np.abs(second))), np.finfo(float).tiny)
        return float(np.max(np.abs(first - second)) / scale)

    beam_error = relative_error(beam_displacements, beam_displacements[::-1])

    x = soil_nodes[:, 0]
    z = soil_nodes[:, 1]
    keys = {(round(float(xx), 12), round(float(zz), 12)): i for i, (xx, zz) in enumerate(zip(x, z))}
    reflected_ux = np.empty_like(soil_ux)
    reflected_uz = np.empty_like(soil_uz)
    for node in range(x.size):
        reflected = keys[(round(float(-x[node]), 12), round(float(z[node]), 12))]
        reflected_ux[node] = -soil_ux[reflected]
        reflected_uz[node] = soil_uz[reflected]
    uz_error = relative_error(soil_uz, reflected_uz)
    ux_error = relative_error(soil_ux, reflected_ux)
    return {
        "symmetry_applicable": True,
        "beam_reflection_symmetry_error": beam_error,
        "soil_uz_reflection_symmetry_error": uz_error,
        "soil_ux_reflection_symmetry_error": ux_error,
        "max_reflection_symmetry_error": max(beam_error, uz_error, ux_error),
    }


def continuum_forward(
    points: np.ndarray,
    depth_nodes: np.ndarray,
    load_centers: np.ndarray,
    support_callable: Callable[[np.ndarray], np.ndarray | float],
    evaluation_points: np.ndarray,
    gauge_length: float,
    ei: float,
    interface_stiffness: float,
    soil_E: float,
    soil_nu: float,
    width: float,
    force: float | np.ndarray,
    load_width: float,
    radius: float,
    support_discontinuity_points: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int | str | bool | dict[str, int | str]]:
    """Solve the coupled beam / plane-strain layer model.

    ``points`` and ``depth_nodes`` define a tensor-product soil mesh and an
    independently assembled Hermite beam on the top surface.  ``support_callable``
    returns interface loss d(x), not soil damage.  ``force`` is positive downward
    and may be scalar or one value per load.  Unit footprint loads are assembled
    once and scaled by these forces; all loads reuse one sparse LU factorization.
    """
    assembly_start = time.perf_counter()
    points, depth_nodes, centers, evaluation = _validate_geometry(
        points,
        depth_nodes,
        load_centers,
        load_width,
        evaluation_points,
        gauge_length,
        support_discontinuity_points,
    )
    _validate_physical_parameters(
        ei,
        interface_stiffness,
        soil_E,
        soil_nu,
        width,
        load_width,
        radius,
    )
    forces = _validate_force(force, centers.size)

    beam_interface, coupling, soil_interface, support, beam_basis, soil_mesh, soil_basis = _assemble_interface(
        points,
        depth_nodes,
        support_callable,
        interface_stiffness,
        support_discontinuity_points,
    )
    # Runtime ordering check required before relying on vector probe output.
    # scikit-fem ``Basis.probes`` returns a component-major matrix: for a
    # vector element with ``comp`` components and ``n`` probe points the
    # product rows are ordered (component 0 at all points, ..., component
    # comp-1 at all points), matching ``Basis.interpolator`` which reshapes
    # the product to ``(comp, n)``.  A single probe point cannot distinguish
    # this from point-major ordering, so verify with several points and
    # distinct nontrivial horizontal and vertical marker values.
    n_marker_points = 3
    marker_x = np.linspace(points[0], points[-1], n_marker_points + 2)[1:-1]
    marker_coordinates = np.vstack((marker_x, np.zeros_like(marker_x)))
    probe_test = np.zeros(soil_basis.N)
    probe_test[0::2] = 2.0
    probe_test[1::2] = -3.0
    probe_result = soil_basis.probes(marker_coordinates) @ probe_test
    expected_marker = np.concatenate(
        (np.full(n_marker_points, 2.0), np.full(n_marker_points, -3.0))
    )
    if probe_result.shape != (2 * n_marker_points,) or not np.allclose(
        probe_result, expected_marker, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError("soil vector probe ordering could not be verified")

    beam_stiffness = _beam_bending.assemble(beam_basis, ei=ei).tocsr() + beam_interface
    lame_lambda, lame_mu = lame_parameters(soil_E, soil_nu)
    soil_stiffness = (
        width * linear_elasticity(Lambda=lame_lambda, Mu=lame_mu).assemble(soil_basis).tocsr()
        + soil_interface
    )
    stiffness = sparse.bmat(
        [[beam_stiffness, coupling], [coupling.T, soil_stiffness]], format="csr"
    )

    unit_rhs = np.zeros((beam_basis.N + soil_basis.N, centers.size))
    beam_load, resultants = _beam_load_vectors(
        beam_basis, centers, load_width, np.ones(centers.size)
    )
    unit_rhs[: beam_basis.N] = beam_load
    rhs = unit_rhs * forces[None, :]

    beam_fixed = np.array([0, 1, 2 * points.size - 2, 2 * points.size - 1], dtype=int)
    x_nodes, z_nodes = soil_mesh.p
    bottom_nodes = np.flatnonzero(np.isclose(z_nodes, depth_nodes[-1], rtol=0.0, atol=1.0e-12)).astype(int)
    side_nodes = np.flatnonzero(
        np.isclose(x_nodes, points[0], rtol=0.0, atol=1.0e-12)
        | np.isclose(x_nodes, points[-1], rtol=0.0, atol=1.0e-12)
    ).astype(int)
    soil_fixed = np.unique(
        np.concatenate(
            [2 * bottom_nodes, 2 * bottom_nodes + 1, 2 * side_nodes]
        )
    ).astype(int)
    soil_fixed += beam_basis.N
    fixed = np.unique(np.concatenate([beam_fixed, soil_fixed]))
    free = np.setdiff1d(np.arange(stiffness.shape[0]), fixed)

    solve_start = time.perf_counter()
    solution = np.zeros_like(rhs)
    solution[free], _ = _scaled_lu_solve(
        stiffness[free][:, free].tocsr(), rhs[free]
    )
    solve_seconds = time.perf_counter() - solve_start
    assembly_seconds = time.perf_counter() - assembly_start - solve_seconds

    beam_solution = solution[: beam_basis.N]
    soil_solution = solution[beam_basis.N :]
    beam_displacements = beam_solution[0::2]
    beam_rotations = beam_solution[1::2]
    soil_ux = soil_solution[0::2]
    soil_uz = soil_solution[1::2]

    value_probe, _ = _line_probe_matrices(beam_basis, evaluation)
    displacement_at_evaluation = value_probe @ beam_solution
    _, right_rotation_probe = _line_probe_matrices(
        beam_basis, evaluation + gauge_length / 2.0
    )
    _, left_rotation_probe = _line_probe_matrices(
        beam_basis, evaluation - gauge_length / 2.0
    )
    left_rotations = left_rotation_probe @ beam_solution
    right_rotations = right_rotation_probe @ beam_solution
    strain = -radius * (right_rotations - left_rotations) / gauge_length

    surface_probe = soil_basis.probes(
        np.vstack((evaluation, np.zeros_like(evaluation)))
    )
    # Component-major probe rows (verified above): the vertical component at
    # every evaluation point occupies the second ``evaluation.size`` rows.
    surface_values = surface_probe @ soil_solution
    soil_surface_vertical = surface_values[evaluation.size :]

    diagnostics = _reaction_diagnostics(
        stiffness,
        rhs,
        solution,
        free,
        forces,
        points.size,
        bottom_nodes,
    )
    diagnostics["load_resultant"] = resultants * forces
    diagnostics.update(
        _symmetry_diagnostics(
            points,
            np.column_stack((x_nodes, z_nodes)),
            centers,
            support,
            beam_displacements,
            soil_ux,
            soil_uz,
        )
    )

    return {
        "evaluation_points": evaluation.copy(),
        "gauge_length": float(gauge_length),
        "displacement": displacement_at_evaluation,
        "strain": strain,
        "beam_nodes": points.copy(),
        "beam_displacements": beam_displacements,
        "beam_rotations": beam_rotations,
        "soil_nodes": np.column_stack((x_nodes, z_nodes)),
        "soil_quads": soil_mesh.t.T.astype(int),
        "soil_ux": soil_ux,
        "soil_uz": soil_uz,
        "soil_surface_vertical_displacement": soil_surface_vertical,
        "diagnostics": diagnostics,
        "mesh": {
            "beam_dofs": int(beam_basis.N),
            "soil_dofs": int(soil_basis.N),
            "total_dofs": int(stiffness.shape[0]),
            "free_dofs": int(free.size),
            "soil_elements": int(soil_mesh.t.shape[1]),
            "interface_quadrature": "4-point Gauss-Legendre",
            "vector_probe_order_verified": True,
        },
        "solver": {
            "method": "symmetric-diagonal-scaled sparse LU",
            "iterative_refinement_steps": 1,
            "one_factorization_for_all_loads": True,
            "unit_force_scaling": True,
            "assembly_seconds": float(assembly_seconds),
            "solve_seconds": float(solve_seconds),
        },
        "sign_convention": "x horizontal; z and all vertical displacements/forces positive downward",
        "interface_loss_definition": "d_interface scales only the bilateral spring; soil E is unchanged",
    }


def _uniform_plane_strain_layer(
    points: np.ndarray,
    depth_nodes: np.ndarray,
    soil_E: float,
    soil_nu: float,
    width: float,
    pressure: float,
) -> dict[str, np.ndarray | float]:
    """Internal oedometric verification solve used by the test suite."""
    points = np.asarray(points, dtype=float)
    depth_nodes = np.asarray(depth_nodes, dtype=float)
    _strictly_increasing(points, "points", max(1.0, float(np.ptp(points))))
    _strictly_increasing(depth_nodes, "depth_nodes", max(1.0, float(np.ptp(depth_nodes))))
    if not np.isclose(depth_nodes[0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("depth_nodes[0] must be zero")
    _validate_physical_parameters(1.0, 1.0, soil_E, soil_nu, width, 1.0, 1.0)
    if not np.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure must be finite and positive downward")

    mesh = MeshQuad.init_tensor(points, depth_nodes)
    basis = Basis(mesh, ElementVector(ElementQuad1()))
    lame_lambda, lame_mu = lame_parameters(soil_E, soil_nu)
    stiffness = width * linear_elasticity(
        Lambda=lame_lambda, Mu=lame_mu
    ).assemble(basis).tocsr()
    rhs = np.zeros(basis.N)
    facet_ids, _, _ = _top_surface_facets(mesh)
    for facet_id in facet_ids:
        endpoint_ids = mesh.facets[:, facet_id]
        if mesh.p[0, endpoint_ids[0]] > mesh.p[0, endpoint_ids[1]]:
            endpoint_ids = endpoint_ids[::-1]
        length = mesh.p[0, endpoint_ids[1]] - mesh.p[0, endpoint_ids[0]]
        trace = np.vstack([0.5 * (1.0 - _GAUSS_X), 0.5 * (1.0 + _GAUSS_X)])
        for local, endpoint in enumerate(endpoint_ids):
            rhs[2 * int(endpoint) + 1] += pressure * width * np.sum(
                trace[local] * _GAUSS_W
            ) * 0.5 * length

    x_nodes, z_nodes = mesh.p
    bottom_nodes = np.flatnonzero(np.isclose(z_nodes, depth_nodes[-1], rtol=0.0, atol=1.0e-12))
    side_nodes = np.flatnonzero(
        np.isclose(x_nodes, points[0], rtol=0.0, atol=1.0e-12)
        | np.isclose(x_nodes, points[-1], rtol=0.0, atol=1.0e-12)
    )
    fixed = np.unique(
        np.concatenate([2 * bottom_nodes, 2 * bottom_nodes + 1, 2 * side_nodes])
    ).astype(int)
    free = np.setdiff1d(np.arange(basis.N), fixed)
    solution, _ = _scaled_lu_solve(stiffness[free][:, free].tocsr(), rhs[free])
    full_solution = np.zeros(basis.N)
    full_solution[free] = solution
    residual = stiffness @ full_solution - rhs
    return {
        "nodes": np.column_stack((x_nodes, z_nodes)),
        "ux": full_solution[0::2],
        "uz": full_solution[1::2],
        "bottom_vertical_reaction": float(residual[2 * bottom_nodes + 1].sum()),
        "applied_resultant": float(pressure * width * np.ptp(points)),
    }


__all__ = ["continuum_forward"]
