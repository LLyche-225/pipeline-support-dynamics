"""Independent scikit-fem 12.0.2 forward reference using ElementLineHermite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
from skfem import BilinearForm, Basis, ElementLineHermite, LinearForm, MeshLine  # noqa: E402


def mesh_with_load_edges(
    x_min: float,
    x_max: float,
    h: float,
    centers: np.ndarray,
    load_width: float = 4.3,
    extra_edges: np.ndarray | None = None,
):
    """Create mesh with load footprint edges and optional extra edges.

    Args:
        x_min: Minimum domain coordinate
        x_max: Maximum domain coordinate
        h: Target element size
        centers: Load center positions
        load_width: Load footprint width (default 4.3)
        extra_edges: Additional edges to include (e.g., rectangular boundaries)

    Returns:
        Unique sorted mesh points with duplicates removed
    """
    points = [x_min, x_max]
    half_width = load_width / 2.0

    # Add load footprint edges
    for center in np.atleast_1d(centers):
        points.extend([center - half_width, center + half_width])

    # Add extra edges if provided
    if extra_edges is not None:
        points.extend(np.atleast_1d(extra_edges).tolist())

    # Generate base mesh
    base = np.arange(x_min, x_max + h / 2.0, h)
    all_points = np.r_[base, points]

    # Clip to domain and remove duplicates with robust tolerance
    clipped = np.clip(all_points, x_min, x_max)
    sorted_points = np.sort(clipped)

    # Remove duplicates: keep points that differ by more than h/100
    if sorted_points.size == 0:
        return sorted_points

    unique_mask = np.ones(sorted_points.size, dtype=bool)
    unique_mask[1:] = np.diff(sorted_points) > h / 100.0

    return sorted_points[unique_mask]


@BilinearForm
def _beam_form(u, v, w):
    return (
        w["ei"] * u.hess[0, 0] * v.hess[0, 0]
        + w["foundation"] * u * v
        + w["shear"] * u.grad[0] * v.grad[0]
    )


@LinearForm
def _uniform_load(v, w):
    inside = np.abs(w.x[0] - w["center"]) <= w["half_width"]
    return (w["pressure"] * inside) * v


def skfem_forward(
    points: np.ndarray,
    load_centers: np.ndarray,
    support_coefficients: np.ndarray | None = None,
    support_callable: callable | None = None,
    *,
    evaluation_points: np.ndarray | None = None,
    gauge_length: float = 0.2,
    ei: float = 1.6232123e8,
    foundation: float = 18.0e6,
    shear: float = 0.0,
    force: float = 750_000.0,
    load_width: float = 4.3,
    radius: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward solve with scikit-fem, supporting arbitrary support or B-spline coefficients.

    Args:
        points: Mesh node coordinates (must be ordered)
        load_centers: Load center positions
        support_coefficients: Full B-spline coefficients (including endpoint zeros), or None
        support_callable: Function d(x) returning support loss in [0,1], or None
        evaluation_points: Points at which to evaluate displacement and strain (default: 81 points in [-20, 20])
        gauge_length: Gauge length for strain measurement
        ei: Bending stiffness (must be positive)
        foundation: Foundation modulus k0 (must be positive)
        shear: Pasternak shear modulus kG (must be non-negative)
        force: Total force per load (must be positive)
        load_width: Load footprint width (must be positive)
        radius: Outer fiber radius for strain (must be positive)

    Returns:
        (displacement, strain) at evaluation_points, each shape (n_eval, n_loads)
    """
    # Validate inputs
    if support_coefficients is not None and support_callable is not None:
        raise ValueError("Cannot specify both support_coefficients and support_callable")

    points_arr = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(points_arr)):
        raise ValueError("Mesh points must be finite")
    if not np.all(np.diff(points_arr) > 0):
        raise ValueError("Mesh points must be strictly increasing")

    centers = np.atleast_1d(np.asarray(load_centers, dtype=float))
    if not np.all(np.isfinite(centers)):
        raise ValueError("Load centers must be finite")

    if ei <= 0 or foundation <= 0 or shear < 0 or force <= 0 or load_width <= 0 or radius <= 0:
        raise ValueError("Physical constants must be positive (shear non-negative)")

    if gauge_length < 0:
        raise ValueError("Gauge length must be non-negative")

    if evaluation_points is None:
        evaluation = np.linspace(-20.0, 20.0, 81)
    else:
        evaluation = np.asarray(evaluation_points, dtype=float)
        if not np.all(np.isfinite(evaluation)):
            raise ValueError("Evaluation points must be finite")
        # Check if gauge endpoints are inside domain
        if gauge_length > 0:
            domain_min, domain_max = points_arr[0], points_arr[-1]
            if np.any(evaluation - gauge_length / 2.0 < domain_min) or np.any(evaluation + gauge_length / 2.0 > domain_max):
                raise ValueError("Evaluation points with gauge would exceed mesh domain")
    mesh = MeshLine(points_arr)
    basis = Basis(mesh, ElementLineHermite())

    # Evaluate support at quadrature points
    if support_coefficients is None and support_callable is None:
        support = np.zeros((basis.nelems, basis.X.shape[-1]))
    elif support_callable is not None:
        # Use arbitrary callable d(x)
        coordinates = basis.global_coordinates()[0].T  # Shape: (n_quad_per_elem, n_elements)
        support_values = support_callable(coordinates.ravel())
        # Validate support is in [0, 1]
        if not np.all((support_values >= 0) & (support_values <= 1)):
            raise ValueError("support_callable must return values in [0, 1]")
        support = support_values.reshape(coordinates.shape).T  # Shape: (n_elements, n_quad)
    else:
        # Use B-spline coefficients
        from beam_inverse import SupportBasis
        full_coeffs = np.asarray(support_coefficients, dtype=float)
        if not np.all(np.isfinite(full_coeffs)):
            raise ValueError("Support coefficients must be finite")
        if not np.all((full_coeffs >= 0) & (full_coeffs <= 1)):
            raise ValueError("Support coefficients must be in [0, 1]")
        spline = SupportBasis(n_coefficients=len(full_coeffs))
        coordinates = basis.global_coordinates()[0].T
        basis_matrix = spline.values(coordinates.ravel())
        support_values = basis_matrix @ full_coeffs
        support = support_values.reshape(coordinates.shape).T

    modulus = foundation * (1.0 - support)

    stiffness = _beam_form.assemble(
        basis, ei=ei, foundation=modulus, shear=shear
    ).tocsr()
    rhs = np.column_stack(
        [
            _uniform_load.assemble(
                basis, center=center, half_width=load_width / 2.0, pressure=force / load_width
            )
            for center in centers
        ]
    )
    free = np.arange(2, basis.N - 2)
    displacement = np.zeros((basis.N, centers.size))
    displacement[free] = splu(stiffness[free][:, free]).solve(rhs[free])

    # Evaluate displacement and strain at evaluation_points using actual basis functions
    finder = mesh.element_finder()
    elements = finder(evaluation)
    local = (evaluation - mesh.p[0, elements]) / (
        mesh.p[0, mesh.t[1, elements]] - mesh.p[0, mesh.t[0, elements]]
    )
    values = np.empty((evaluation.size, centers.size))
    slopes = np.empty_like(values)
    for load in range(centers.size):
        w_load = np.zeros(evaluation.size)
        rot_load = np.zeros(evaluation.size)
        for i in range(4):
            field = basis.elem.gbasis(
                basis.mapping, np.array([local]), i, elements
            )[0]
            global_dofs = basis.element_dofs[i, elements]
            w_load += displacement[global_dofs, load] * np.diag(field)
            rot_load += displacement[global_dofs, load] * np.diag(field.grad[0])
        values[:, load] = w_load
        slopes[:, load] = rot_load

    if gauge_length == 0:
        raise ValueError("point-strain scikit-fem reference is not implemented")

    # Compute gauge strain using actual basis gradients at gauge endpoints
    left_points = evaluation - gauge_length / 2.0
    right_points = evaluation + gauge_length / 2.0
    left_elements = finder(left_points)
    right_elements = finder(right_points)
    left_local = (left_points - mesh.p[0, left_elements]) / (
        mesh.p[0, mesh.t[1, left_elements]] - mesh.p[0, mesh.t[0, left_elements]]
    )
    right_local = (right_points - mesh.p[0, right_elements]) / (
        mesh.p[0, mesh.t[1, right_elements]] - mesh.p[0, mesh.t[0, right_elements]]
    )
    left_slopes = np.empty_like(values)
    right_slopes = np.empty_like(values)
    for load in range(centers.size):
        left_rot = np.zeros(evaluation.size)
        right_rot = np.zeros(evaluation.size)
        for i in range(4):
            left_field = basis.elem.gbasis(
                basis.mapping, np.array([left_local]), i, left_elements
            )[0]
            right_field = basis.elem.gbasis(
                basis.mapping, np.array([right_local]), i, right_elements
            )[0]
            global_left = basis.element_dofs[i, left_elements]
            global_right = basis.element_dofs[i, right_elements]
            left_rot += displacement[global_left, load] * np.diag(left_field.grad[0])
            right_rot += displacement[global_right, load] * np.diag(right_field.grad[0])
        left_slopes[:, load] = left_rot
        right_slopes[:, load] = right_rot
    strain = -radius * (right_slopes - left_slopes) / gauge_length
    return values, strain


__all__ = ["mesh_with_load_edges", "skfem_forward"]
