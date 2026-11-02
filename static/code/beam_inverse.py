"""Stable Hermite FE inverse kernel for effective pipeline support."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu


EI = 1.6232123e8
K0 = 18.0e6
RADIUS = 0.3
LOAD_FORCE = 750_000.0
LOAD_WIDTH = 4.3
GAMMA_LOWER = 0.7
GAMMA_UPPER = 1.3


def _hermite_shapes(t: np.ndarray, h: float) -> np.ndarray:
    """Hermite shape functions on [0, 1] interval."""
    return np.column_stack(
        [
            1.0 - 3.0 * t**2 + 2.0 * t**3,
            h * (t - 2.0 * t**2 + t**3),
            3.0 * t**2 - 2.0 * t**3,
            h * (-t**2 + t**3),
        ]
    )


def _hermite_derivatives(t: np.ndarray, h: float) -> np.ndarray:
    """First derivatives of Hermite shape functions w.r.t. local coordinate."""
    return np.column_stack(
        [
            (-6.0 * t + 6.0 * t**2) / h,
            (1.0 - 4.0 * t + 3.0 * t**2),
            (6.0 * t - 6.0 * t**2) / h,
            (-2.0 * t + 3.0 * t**2),
        ]
    )


def _hermite_second_derivatives(t: np.ndarray, h: float) -> np.ndarray:
    """Second derivatives of Hermite shape functions w.r.t. local coordinate."""
    return np.column_stack(
        [
            (-6.0 + 12.0 * t) / h**2,
            (-4.0 + 6.0 * t) / h,
            (6.0 - 12.0 * t) / h**2,
            (-2.0 + 6.0 * t) / h,
        ]
    )


@dataclass(frozen=True)
class SupportBasis:
    """Clamped cubic B-spline basis with fixed zero endpoint coefficients."""

    n_coefficients: int = 31
    support: tuple[float, float] = (-12.0, 12.0)

    def __post_init__(self) -> None:
        if self.n_coefficients < 5:
            raise ValueError("at least five B-spline coefficients are required")
        left, right = self.support
        # For degree 3, n coefficients require n+4 knots
        # Interior count = n-4, obtained from linspace(left, right, n-2)[1:-1]
        interior = np.linspace(left, right, self.n_coefficients - 2)[1:-1]
        knots = np.r_[np.full(4, left), interior, np.full(4, right)]
        object.__setattr__(self, "knots", knots)
        spline = BSpline(knots, np.eye(self.n_coefficients), 3, extrapolate=False)
        object.__setattr__(self, "_spline", spline)
        object.__setattr__(self, "_derivative", spline.derivative())

    @property
    def n_variables(self) -> int:
        return self.n_coefficients - 2

    def full_coefficients(self, variables: np.ndarray) -> np.ndarray:
        values = np.zeros(self.n_coefficients)
        values[1:-1] = np.asarray(variables, dtype=float)
        return values

    def values(self, x: np.ndarray) -> np.ndarray:
        """Evaluate basis at points x, returns (len(x), n_coefficients) array."""
        x_arr = np.asarray(x, dtype=float)
        result = self._spline(x_arr)
        # Handle extrapolation: set to zero outside support
        outside = (x_arr < self.support[0]) | (x_arr > self.support[1])
        result[outside] = 0.0
        return result

    def derivatives(self, x: np.ndarray) -> np.ndarray:
        """Evaluate basis derivatives, returns (len(x), n_coefficients) array."""
        x_arr = np.asarray(x, dtype=float)
        result = self._derivative(x_arr)
        # Handle extrapolation: set to zero outside support
        outside = (x_arr < self.support[0]) | (x_arr > self.support[1])
        result[outside] = 0.0
        return result

    def penalty_quadrature(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gauss points/weights and matching value/derivative matrices."""
        local, local_w = np.polynomial.legendre.leggauss(4)
        edges = np.unique(self.knots)
        points = (edges[:-1, None] + np.diff(edges)[:, None] * (local[None, :] + 1.0) / 2.0).ravel()
        lengths = np.repeat(np.diff(edges), local.size)
        weights = lengths * np.tile(local_w / 2.0, edges.size - 1)
        return points, weights, self.values(points), self.derivatives(points)


class HermiteBeamFE:
    """Vectorized Euler-Bernoulli beam with affine B-spline foundation loss."""

    def __init__(
        self,
        load_centers: list[float] | np.ndarray,
        *,
        x_min: float = -50.0,
        x_max: float = 50.0,
        n_elements: int = 400,
        n_coefficients: int = 31,
        gauge_length: float = 0.2,
        foundation: float = K0,
        pasternak_shear: float = 0.0,
        force: float = LOAD_FORCE,
        load_width: float = LOAD_WIDTH,
        ei: float = EI,
        radius: float = RADIUS,
    ) -> None:
        if x_min >= x_max or n_elements < 2:
            raise ValueError("invalid beam domain or element count")
        self.x = np.linspace(x_min, x_max, n_elements + 1)
        self.h = (x_max - x_min) / n_elements
        self.centers = np.atleast_1d(np.asarray(load_centers, dtype=float))
        self.basis = SupportBasis(n_coefficients=n_coefficients)
        self.gauge_length = float(gauge_length)
        self.foundation = float(foundation)
        self.shear = float(pasternak_shear)
        self.force = float(force)
        self.load_width = float(load_width)
        self.ei = float(ei)
        self.radius = float(radius)
        self._assemble()

    def _assemble(self) -> None:
        n = self.x.size
        h = self.h
        gauss, gw = np.polynomial.legendre.leggauss(4)
        t = (gauss + 1.0) / 2.0
        wq = gw / 2.0
        shapes = _hermite_shapes(t, h)
        dshapes = _hermite_derivatives(t, h)

        # Bending stiffness matrix (analytic)
        bending = self.ei / h**3 * np.array(
            [[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
             [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]]
        )
        foundation = self.foundation * h * np.einsum("q,qi,qj->ij", wq, shapes, shapes, optimize=True)
        shear = self.shear * h * np.einsum("q,qi,qj->ij", wq, dshapes, dshapes, optimize=True)
        intact_local = bending + foundation + shear

        dofs = np.column_stack([2 * np.arange(n - 1), 2 * np.arange(n - 1) + 1,
                                2 * np.arange(1, n), 2 * np.arange(1, n) + 1])
        rows = np.repeat(dofs, 4, axis=1).ravel()
        cols = np.tile(dofs, (1, 4)).ravel()
        self.k_intact = coo_matrix(
            (np.tile(intact_local.ravel(), n - 1), (rows, cols)), shape=(2 * n, 2 * n)
        ).tocsr()

        # Assemble support contribution matrices
        xq = self.x[:-1, None] + h * t[None, :]
        support = self.basis.values(xq.ravel())  # Shape: (n_elements * n_quad, n_coefficients)
        support = support.reshape(n - 1, len(t), self.basis.n_coefficients)

        self.k_coefficients: list[coo_matrix] = []
        for coefficient in range(self.basis.n_coefficients):
            local = np.empty((n - 1, 4, 4))
            values = support[:, :, coefficient]  # Shape: (n_elements, n_quad)
            for i in range(4):
                for j in range(4):
                    local[:, i, j] = self.foundation * h * np.sum(
                        wq * values * shapes[:, i] * shapes[:, j], axis=1
                    )
            matrix = coo_matrix((local.ravel(), (rows, cols)), shape=(2 * n, 2 * n)).tocsr()
            self.k_coefficients.append(matrix)

        # Assemble load vectors
        self.rhs = np.zeros((2 * n, self.centers.size))
        for element, left in enumerate(self.x[:-1]):
            ids = dofs[element]
            for load, center in enumerate(self.centers):
                lo = max(left, center - self.load_width / 2.0)
                hi = min(left + h, center + self.load_width / 2.0)
                if hi <= lo:
                    continue
                points = lo + (hi - lo) * t
                local_shapes = _hermite_shapes((points - left) / h, h)
                self.rhs[ids, load] += (
                    self.force / self.load_width * (hi - lo) * (wq @ local_shapes)
                )

    def _factor(self, variables: np.ndarray):
        """Factor the stiffness matrix for given support coefficients."""
        full = self.basis.full_coefficients(variables)
        matrix = self.k_intact.astype(float).tolil()
        for value, coefficient_matrix in zip(full, self.k_coefficients, strict=True):
            if value:
                matrix = matrix - value * coefficient_matrix
        matrix = matrix.tocsr()
        free = np.arange(2, 2 * self.x.size - 2)
        free_matrix = matrix[free][:, free]
        diagonal = free_matrix.diagonal()
        if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
            raise np.linalg.LinAlgError("non-positive diagonal in FE stiffness")
        # Diagonal scaling: S = diag(1/sqrt(Kii))
        scale = 1.0 / np.sqrt(diagonal)
        scale_matrix = diags(scale)
        scaled = scale_matrix @ free_matrix @ scale_matrix
        return splu(scaled.tocsc()), scale, free, scaled

    def _state(self, variables: np.ndarray, with_jacobian: bool = True):
        lu, scale, free, _ = self._factor(variables)
        # Solve: K_scaled @ z = S @ b, then u = S @ z
        scaled_rhs = scale[:, None] * self.rhs[free]
        z = lu.solve(scaled_rhs)
        displacement_free = scale[:, None] * z
        displacement = np.zeros((2 * self.x.size, self.centers.size))
        displacement[free] = displacement_free
        if not with_jacobian:
            return displacement, None, (lu, scale, free)

        # Sensitivity RHS: K_m @ u for each coefficient, stacked coefficient-major then load
        sensitivity_rhs = []
        for matrix in self.k_coefficients[1:-1]:
            for load in range(self.centers.size):
                rhs = matrix[free][:, free] @ displacement_free[:, load]
                sensitivity_rhs.append(rhs)
        sensitivity_rhs = np.column_stack(sensitivity_rhs)

        # Solve scaled system for all sensitivities
        scaled_sens_rhs = scale[:, None] * sensitivity_rhs
        z_sens = lu.solve(scaled_sens_rhs)
        sensitivity_free = scale[:, None] * z_sens

        # Reshape: columns are stacked coefficient-major then load
        # Shape after column_stack: (free_dofs, n_variables * n_loads)
        # Reshape to (free_dofs, n_variables, n_loads) then transpose to (free_dofs, n_loads, n_variables)
        n_variables = self.basis.n_variables
        sensitivity_free = sensitivity_free.reshape(free.size, n_variables, self.centers.size).transpose(0, 2, 1)

        sensitivity = np.zeros((2 * self.x.size, self.centers.size, n_variables))
        sensitivity[free] = sensitivity_free
        return displacement, sensitivity, (lu, scale, free)

    @staticmethod
    def _element_state(points: np.ndarray, x_min: float, h: float, n_elements: int):
        """Find elements and local coordinates for evaluation points."""
        element = np.clip(((points - x_min) / h).astype(int), 0, n_elements - 1)
        local = (points - x_min - element * h) / h
        shape = _hermite_shapes(local, h)
        deriv = _hermite_derivatives(local, h)
        second = _hermite_second_derivatives(local, h)
        dofs = np.column_stack([2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3])
        return shape, deriv, second, dofs

    def _evaluate(self, points: np.ndarray, states: np.ndarray, derivative: bool = False):
        """Evaluate displacement, rotation, and curvature at points."""
        points = np.asarray(points, dtype=float)
        shape, deriv, second, dofs = self._element_state(
            points, self.x[0], self.h, self.x.size - 1
        )
        selected = states[dofs, ...]
        result = np.empty((points.size,) + states.shape[1:])
        rotation = np.empty((points.size,) + states.shape[1:])
        curvature = np.empty((points.size,) + states.shape[1:])
        for point in range(points.size):
            result[point] = shape[point] @ selected[point]
            rotation[point] = deriv[point] @ selected[point]
            curvature[point] = second[point] @ selected[point]
        if derivative:
            return result, rotation, curvature
        else:
            return result, curvature

    def strain_and_jacobian(
        self, variables: np.ndarray, observation_points: list[np.ndarray] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute strain and its Jacobian w.r.t. support coefficients.

        Args:
            variables: Support coefficient variables (interior only)
            observation_points: List of sensor arrays, one per load (must match centers.size)

        Returns:
            (strain, jacobian) concatenated across all loads
        """
        # Validate observation_points structure
        if np.ndim(observation_points[0]) == 0:
            observation_points = [np.asarray(observation_points)]

        if len(observation_points) != self.centers.size:
            raise ValueError(
                f"observation_points must have {self.centers.size} arrays "
                f"(one per load), got {len(observation_points)}"
            )

        displacement, sensitivity, _ = self._state(variables)
        strains, jacobians = [], []
        half_gauge = self.gauge_length / 2.0
        for load, points in enumerate(observation_points):
            points = np.asarray(points, dtype=float)
            if self.gauge_length == 0:
                _, curvature = self._evaluate(points, displacement[:, load], derivative=False)
                _, curvature_jac = self._evaluate(points, sensitivity[:, load, :], derivative=False)
                strains.append(-self.radius * curvature)
                jacobians.append(-self.radius * curvature_jac)
            else:
                # Gauge strain: rotation difference over gauge length
                left_w, left_rot, _ = self._evaluate(
                    points - half_gauge, displacement[:, load], derivative=True
                )
                right_w, right_rot, _ = self._evaluate(
                    points + half_gauge, displacement[:, load], derivative=True
                )
                left_j_w, left_j_rot, _ = self._evaluate(
                    points - half_gauge, sensitivity[:, load, :], derivative=True
                )
                right_j_w, right_j_rot, _ = self._evaluate(
                    points + half_gauge, sensitivity[:, load, :], derivative=True
                )
                # Strain from rotation difference
                factor = -self.radius / self.gauge_length
                strains.append(factor * (right_rot - left_rot))
                jacobians.append(factor * (right_j_rot - left_j_rot))
        return np.concatenate(strains), np.vstack(jacobians)


def penalty_and_gradient(variables: np.ndarray, basis: SupportBasis, kind: str = "h1"):
    """Compute penalty value and gradient for regularization."""
    points, weights, values, derivatives = basis.penalty_quadrature()
    c = np.asarray(variables, dtype=float)
    full_values = values[:, 1:-1] @ c
    full_derivatives = derivatives[:, 1:-1] @ c
    if kind == "h1":
        value = float(np.sum(weights * full_derivatives**2))
        gradient = 2.0 * (derivatives[:, 1:-1].T * weights) @ full_derivatives
    elif kind == "tv":
        eta = 0.02
        area_ratio = 0.05
        epsilon = 1.0e-4
        derivative_root = np.sqrt(full_derivatives**2 + eta**2)
        area_root = np.sqrt(full_values**2 + epsilon**2)
        value = float(
            np.sum(weights * (derivative_root - eta + area_ratio * area_root))
        )
        gradient = (
            (derivatives[:, 1:-1].T * (weights * full_derivatives / derivative_root)) @ np.ones(points.size)
            + area_ratio * (values[:, 1:-1].T * (weights * full_values / area_root)) @ np.ones(points.size)
        )
    else:
        raise ValueError("penalty must be 'h1' or 'tv'")
    return value, gradient


def _penalty_residual(variables: np.ndarray, basis: SupportBasis, kind: str, weight: float):
    """Return distributed penalty residuals for least_squares."""
    if weight == 0:
        return np.empty(0), np.empty((0, basis.n_variables))

    points, weights_q, values, derivatives = basis.penalty_quadrature()
    c = np.asarray(variables, dtype=float)
    full_derivatives = derivatives[:, 1:-1] @ c

    if kind == "h1":
        # Distributed residual: sqrt(weight * w_q) * d'
        residual = np.sqrt(weight * weights_q) * full_derivatives
        jacobian = np.sqrt(weight * weights_q)[:, None] * derivatives[:, 1:-1]
    elif kind == "tv":
        eta = 0.02
        area_ratio = 0.05
        epsilon = 1.0e-4
        full_values = values[:, 1:-1] @ c
        # TV residual uses fourth-root form: sqrt(lambda*w) * (dprime^2 + eta^2)^0.25
        # and sqrt(lambda*rho*w) * (d^2 + eps^2)^0.25
        # Their squared norm equals lambda * sum(w * (sqrt(dprime^2+eta^2) - eta + rho*sqrt(d^2+eps^2))) + constant
        derivative_term = (full_derivatives**2 + eta**2)**0.25
        area_term = (full_values**2 + epsilon**2)**0.25
        residual = np.concatenate([
            np.sqrt(weight * weights_q) * derivative_term,
            np.sqrt(weight * area_ratio * weights_q) * area_term
        ])
        # Jacobian: d/dc [(d'^2 + eta^2)^0.25] = 0.5 * d' / (d'^2 + eta^2)^0.75
        #           d/dc [(d^2 + eps^2)^0.25] = 0.5 * d / (d^2 + eps^2)^0.75
        d_deriv = 0.5 * full_derivatives / (full_derivatives**2 + eta**2)**0.75
        d_value = 0.5 * full_values / (full_values**2 + epsilon**2)**0.75
        jac_deriv = np.sqrt(weight * weights_q)[:, None] * d_deriv[:, None] * derivatives[:, 1:-1]
        jac_area = np.sqrt(weight * area_ratio * weights_q)[:, None] * d_value[:, None] * values[:, 1:-1]
        jacobian = np.vstack([jac_deriv, jac_area])
    else:
        raise ValueError("penalty must be 'h1' or 'tv'")

    return residual, jacobian


def _projected_gradient(x, gradient, lower, upper, tol=1e-8):
    """Project gradient to enforce KKT conditions at bounds."""
    x = np.asarray(x, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if x.shape != gradient.shape or x.shape != lower.shape or x.shape != upper.shape:
        raise ValueError("x, gradient, lower, and upper must have the same shape")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(gradient)):
        raise ValueError("x and gradient must be finite")
    if np.any(lower >= upper):
        raise ValueError("lower bounds must be strictly less than upper bounds")

    result = gradient.copy()
    # Use tolerance for bound-active detection
    at_lower = x <= lower + tol
    at_upper = x >= upper - tol
    result[at_lower] = np.minimum(result[at_lower], 0.0)
    result[at_upper] = np.maximum(result[at_upper], 0.0)
    return result


def _load_information_vectors(load_mean, load_std, n_loads: int, gamma_mode: str):
    mean = np.asarray(load_mean, dtype=float)
    std = np.asarray(load_std, dtype=float)
    if mean.ndim == 0:
        mean = np.full(n_loads, float(mean))
    if std.ndim == 0:
        std = np.full(n_loads, float(std))
    if mean.ndim != 1 or mean.size != n_loads:
        raise ValueError(f"load_mean must be a scalar or a length-{n_loads} vector")
    if std.ndim != 1 or std.size != n_loads:
        raise ValueError(f"load_std must be a scalar or a length-{n_loads} vector")
    if not np.all(np.isfinite(mean)) or np.any(mean <= 0.0):
        raise ValueError("load means must be finite and strictly positive")
    if np.any(np.isnan(std)) or np.any(std < 0.0):
        raise ValueError("load standard deviations must be nonnegative (allowing +inf)")

    exact = std == 0.0
    outside_bounds = (mean < GAMMA_LOWER) | (mean > GAMMA_UPPER)
    if np.any(outside_bounds if gamma_mode == "fixed" else exact & outside_bounds):
        raise ValueError("exact load means must lie within the gamma bounds [0.7, 1.3]")
    return mean, std


def profile_load_amplitude(
    f: np.ndarray,
    df: np.ndarray,
    target: np.ndarray,
    load_mean: float = 1.0,
    load_std: float = np.inf,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Return the bounded amplitude and reduced derivatives for one whitened load.

    ``f`` and ``target`` are one-dimensional whitened response and observation
    vectors, while each row of ``df`` is the corresponding whitened derivative
    with respect to support coefficients.  The stable return contract is
    ``(gamma, dgamma, prior_residual, prior_jacobian)``.  ``dgamma`` and
    ``prior_jacobian`` have exactly ``df.shape[1]`` entries.  A finite positive
    ``load_std`` contributes the Gaussian amplitude residual
    ``(gamma-load_mean)/load_std``; zero and infinite standard deviations
    respectively denote an exact amplitude and no amplitude prior.  The interior
    formulas are scaled to avoid overflow when precision ``1/load_std**2`` is
    large.  Noisy means outside the physical gamma bounds are allowed, but an
    exact zero-uncertainty mean must be admissible.
    """
    f = np.asarray(f, dtype=float)
    df = np.asarray(df, dtype=float)
    target = np.asarray(target, dtype=float)
    if f.ndim != 1 or target.ndim != 1 or f.size != target.size:
        raise ValueError("f and target must be matching one-dimensional arrays")
    if df.ndim != 2 or df.shape[0] != f.size:
        raise ValueError("df must be a two-dimensional array with one row per observation")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(df)) or not np.all(np.isfinite(target)):
        raise ValueError("f, df, and target must be finite")
    mean = float(load_mean)
    std = float(load_std)
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("load_mean must be finite and strictly positive")
    if np.isnan(std) or std < 0.0:
        raise ValueError("load_std must be nonnegative (allowing +inf)")

    response_norm = float(f @ f)
    if not np.isfinite(response_norm):
        raise ValueError("whitened response norm must be finite")
    zero_dgamma = np.zeros(df.shape[1], dtype=float)

    if std == 0.0:
        if mean < GAMMA_LOWER or mean > GAMMA_UPPER:
            raise ValueError("an exact load mean must lie within [0.7, 1.3]")
        return mean, zero_dgamma.copy(), 0.0, zero_dgamma.copy()

    cross = float(f @ target)
    if not np.isfinite(cross):
        raise ValueError("whitened response/observation inner product must be finite")

    if np.isinf(std):
        if response_norm <= 0.0:
            raise ValueError("a positive finite whitened response norm is required without a prior")
        gamma = float(np.clip(cross / response_norm, GAMMA_LOWER, GAMMA_UPPER))
        if gamma == GAMMA_LOWER or gamma == GAMMA_UPPER:
            return gamma, zero_dgamma.copy(), 0.0, zero_dgamma.copy()
        derivative_numerator = df.T @ target - 2.0 * gamma * (f @ df)
        dgamma = derivative_numerator / response_norm
        if not np.all(np.isfinite(dgamma)):
            raise ValueError("profile amplitude derivatives must be finite")
        return gamma, dgamma, 0.0, zero_dgamma.copy()

    # For tau <= 1, factor tau**2 out of numerator and denominator.  For
    # tau > 1, compute precision by two divisions so squaring tau cannot
    # overflow.  These are the same closed form as (f.y + p*mu)/(f.f + p).
    if std <= 1.0:
        tau2 = std * std
        denominator = 1.0 + tau2 * response_norm
        gamma = float(np.clip((mean + tau2 * cross) / denominator, GAMMA_LOWER, GAMMA_UPPER))
    else:
        precision = (1.0 / std) / std
        denominator = response_norm + precision
        gamma = float(np.clip((cross + precision * mean) / denominator, GAMMA_LOWER, GAMMA_UPPER))
    if not np.isfinite(gamma):
        raise ValueError("profile amplitude prior produced a nonfinite value")

    active = gamma == GAMMA_LOWER or gamma == GAMMA_UPPER
    if active:
        prior_residual = (gamma - mean) / std
        if not np.isfinite(prior_residual):
            raise ValueError("profile amplitude derivatives must be finite")
        return gamma, zero_dgamma.copy(), prior_residual, zero_dgamma.copy()

    derivative_numerator = df.T @ target - 2.0 * gamma * (f @ df)
    if std <= 1.0:
        dgamma = (tau2 * derivative_numerator) / denominator
        prior_jacobian = (std * derivative_numerator) / denominator
        prior_residual = (std * (cross - mean * response_norm)) / denominator
    else:
        dgamma = derivative_numerator / denominator
        prior_jacobian = derivative_numerator / (std * denominator)
        prior_residual = (cross - mean * response_norm) / (std * denominator)
    if (
        not np.all(np.isfinite(dgamma))
        or not np.all(np.isfinite(prior_jacobian))
        or not np.isfinite(prior_residual)
    ):
        raise ValueError("profile amplitude prior produced nonfinite values")
    return gamma, dgamma, float(prior_residual), prior_jacobian


def fit_support(
    forward: HermiteBeamFE,
    observation_points: list[np.ndarray] | np.ndarray,
    observations: list[np.ndarray] | np.ndarray,
    sigma: list[np.ndarray] | np.ndarray,
    *,
    gamma_mode: str = "profile",
    penalty: str = "h1",
    penalty_lambda: float = 0.0,
    starts: tuple[float, float] = (0.05, 0.30),
    max_nfev: int = 200,
    load_mean: float | np.ndarray = 1.0,
    load_std: float | np.ndarray = np.inf,
) -> dict:
    """Fit support from observations using two deterministic starts.

    Args:
        forward: Assembled finite element model
        observation_points: List of sensor locations for each load
        observations: List of measured strains for each load
        sigma: List of measurement uncertainties for each load
        gamma_mode: 'fixed' (supplied exact gamma), 'profile' (per-load
            optimal), or 'joint' (optimize with coefficients)
        penalty: 'h1' or 'tv'
        penalty_lambda: Regularization weight
        starts: Tuple of two coefficient starting values (applied uniformly)
        max_nfev: Maximum function evaluations per start
        load_mean: Scalar or per-load exact/noisy amplitude mean
        load_std: Scalar or per-load amplitude standard deviation (0, finite,
            or +inf)

    Returns:
        Dictionary with coefficients, gamma, optimization metadata, and start records
    """
    if gamma_mode not in {"fixed", "profile", "joint"}:
        raise ValueError("gamma_mode must be fixed, profile, or joint")
    if np.ndim(observation_points[0]) == 0:
        observation_points = [np.asarray(observation_points)]

    # Validate observation_points list length matches number of loads
    if len(observation_points) != forward.centers.size:
        raise ValueError(
            f"observation_points must have {forward.centers.size} arrays "
            f"(one per load), got {len(observation_points)}"
        )

    # Handle observations and sigma as lists directly
    if not isinstance(observations, list):
        observations = [observations]
    if not isinstance(sigma, list):
        sigma = [sigma]
    y = [np.asarray(v, dtype=float) for v in observations]
    sd = [np.asarray(v, dtype=float) for v in sigma]
    if len(y) != forward.centers.size or len(sd) != forward.centers.size:
        raise ValueError("one observation vector and sigma vector are required per load")
    # Validate observation_points structure
    for idx, points in enumerate(observation_points):
        points_arr = np.asarray(points, dtype=float)
        if points_arr.size != y[idx].size:
            raise ValueError(f"observation_points[{idx}] size mismatch")
        if not np.all(np.isfinite(points_arr)):
            raise ValueError(f"observation_points[{idx}] must be finite")
        # Check gauge and domain boundaries
        if forward.gauge_length > 0:
            half_gauge = forward.gauge_length / 2.0
            if np.any(points_arr - half_gauge < forward.x[0]) or np.any(points_arr + half_gauge > forward.x[-1]):
                raise ValueError(f"observation_points[{idx}] with gauge would exceed domain")
    if any(v.size != points.size for v, points in zip(y, observation_points)):
        raise ValueError("each observation must match its observation points")
    if any(v.size != points.size or np.any(v <= 0) or not np.all(np.isfinite(v))
           for v, points in zip(sd, observation_points)):
        raise ValueError("each sigma must match its observations, be strictly positive, and be finite")
    if any(not np.all(np.isfinite(v)) for v in y):
        raise ValueError("observations must be finite")
    means, stds = _load_information_vectors(
        load_mean, load_std, forward.centers.size, gamma_mode
    )

    n = forward.basis.n_variables
    lower_c, upper_c = np.zeros(n), np.full(n, 0.95)

    # Cache penalty quadrature to avoid rebuilding
    penalty_quad = forward.basis.penalty_quadrature() if penalty_lambda > 0 else None

    # Simple cache for last evaluation using copied array
    _cache = {"x": None, "result": None}

    def data_terms(variables):
        prediction, prediction_jacobian = forward.strain_and_jacobian(
            variables, observation_points
        )
        pieces, jacobian_pieces = [], []
        offset = 0
        for load, points in enumerate(observation_points):
            count = points.size
            f = prediction[offset:offset + count] / sd[load]
            df = prediction_jacobian[offset:offset + count] / sd[load][:, None]
            target = y[load] / sd[load]
            if gamma_mode == "profile":
                gamma, dgamma, prior_residual, prior_jacobian = profile_load_amplitude(
                    f, df, target, means[load], stds[load]
                )
            elif gamma_mode == "fixed":
                gamma, dgamma = means[load], np.zeros(forward.basis.n_variables)
                prior_residual, prior_jacobian = None, None
            else:
                gamma, dgamma = None, None
                prior_residual, prior_jacobian = None, None
            pieces.append((f, df, target, gamma, dgamma, prior_residual, prior_jacobian))
            offset += count
        return pieces

    def penalty_terms(variables):
        if penalty_lambda == 0:
            return np.empty(0), np.empty((0, forward.basis.n_variables))
        points, weights_q, values, derivatives = penalty_quad
        c = np.asarray(variables, dtype=float)
        full_derivatives = derivatives[:, 1:-1] @ c

        if penalty == "h1":
            residual = np.sqrt(penalty_lambda * weights_q) * full_derivatives
            jacobian = np.sqrt(penalty_lambda * weights_q)[:, None] * derivatives[:, 1:-1]
        elif penalty == "tv":
            eta = 0.02
            area_ratio = 0.05
            epsilon = 1.0e-4
            full_values = values[:, 1:-1] @ c
            derivative_term = (full_derivatives**2 + eta**2)**0.25
            area_term = (full_values**2 + epsilon**2)**0.25
            residual = np.concatenate([
                np.sqrt(penalty_lambda * weights_q) * derivative_term,
                np.sqrt(penalty_lambda * area_ratio * weights_q) * area_term
            ])
            d_deriv = 0.5 * full_derivatives / (full_derivatives**2 + eta**2)**0.75
            d_value = 0.5 * full_values / (full_values**2 + epsilon**2)**0.75
            jac_deriv = np.sqrt(penalty_lambda * weights_q)[:, None] * d_deriv[:, None] * derivatives[:, 1:-1]
            jac_area = np.sqrt(penalty_lambda * area_ratio * weights_q)[:, None] * d_value[:, None] * values[:, 1:-1]
            jacobian = np.vstack([jac_deriv, jac_area])
        else:
            raise ValueError("penalty must be 'h1' or 'tv'")
        return residual, jacobian

    if gamma_mode in {"fixed", "profile"}:
        def residual(variables):
            # Simple cache: compare against copied last x using array_equal
            if _cache["x"] is not None and np.array_equal(variables, _cache["x"]):
                return _cache["result"]

            pieces = data_terms(variables)
            rows, jac = [], []
            for load, (f, df, target, gamma, dgamma, prior_residual, prior_jacobian) in enumerate(pieces):
                rows.append((gamma * f - target))
                jac.append(gamma * df + np.outer(f, dgamma))
                if gamma_mode == "profile" and np.isfinite(stds[load]):
                    rows.append(np.array([prior_residual]))
                    jac.append(np.asarray(prior_jacobian).reshape(1, -1))
            penalty, penalty_jac = penalty_terms(variables)
            result = (np.concatenate(rows + [penalty]), np.vstack(jac + [penalty_jac]))
            _cache["x"] = variables.copy()
            _cache["result"] = result
            return result

        start_list = [np.full(n, starts[0]), np.full(n, starts[1])]
        bounds = (lower_c, upper_c)
    else:
        # Joint mode optimizes only amplitudes with nonzero uncertainty.
        # Zero-uncertainty amplitudes remain fixed at their exact values.
        gamma_loads = np.flatnonzero(stds > 0.0)
        gamma_local = {load: local for local, load in enumerate(gamma_loads)}
        gammas = [np.ones(gamma_loads.size), np.ones(gamma_loads.size)]
        c_starts = [np.full(n, starts[0]), np.full(n, starts[1])]

        def residual(full_variables):
            # Simple cache: compare against copied last x using array_equal
            if _cache["x"] is not None and np.array_equal(full_variables, _cache["x"]):
                return _cache["result"]

            variables = full_variables[:n]
            gam = np.ones(forward.centers.size)
            gam[stds == 0.0] = means[stds == 0.0]
            gam[gamma_loads] = full_variables[n:]
            pieces = data_terms(variables)
            rows, jac_list = [], []
            for load, (f, df, target, _, _, _, _) in enumerate(pieces):
                if np.isinf(stds[load]):
                    response_norm = float(f @ f)
                    if not np.isfinite(response_norm) or response_norm <= 0.0:
                        raise ValueError(
                            "a positive finite whitened response norm is required without a prior"
                        )
                rows.append(gam[load] * f - target)
                # Jacobian: d/dc (gamma * f) and d/dgamma (gamma * f)
                jac_c = gam[load] * df
                jac_gamma = np.zeros((f.size, gamma_loads.size))
                if load in gamma_local:
                    jac_gamma[:, gamma_local[load]] = f
                jac_list.append(np.column_stack([jac_c, jac_gamma]))
                if np.isfinite(stds[load]) and stds[load] > 0.0:
                    prior_jac = np.zeros((1, n + gamma_loads.size))
                    prior_jac[0, n + gamma_local[load]] = 1.0 / stds[load]
                    rows.append(np.array([(gam[load] - means[load]) / stds[load]]))
                    jac_list.append(prior_jac)
            penalty, penalty_jac = penalty_terms(variables)
            # Penalty only depends on c, not gamma
            penalty_full_jac = np.column_stack(
                [penalty_jac, np.zeros((penalty.size, gamma_loads.size))]
            )
            jac_list.append(penalty_full_jac)
            result = (np.concatenate(rows + [penalty]), np.vstack(jac_list))
            _cache["x"] = full_variables.copy()
            _cache["result"] = result
            return result

        start_list = [np.r_[c_starts[0], gammas[0]], np.r_[c_starts[1], gammas[1]]]
        lower = np.r_[lower_c, np.full(gamma_loads.size, GAMMA_LOWER)]
        upper = np.r_[upper_c, np.full(gamma_loads.size, GAMMA_UPPER)]
        bounds = (lower, upper)

    def residual_metadata(coefficients, gammas_full):
        pieces = data_terms(coefficients)
        calibration_squared = 0.0
        strain_squared = 0.0
        prior_squared = 0.0
        for load, (f, df, target, _, _, prior_residual, _) in enumerate(pieces):
            difference = gammas_full[load] * f - target
            calibration_squared += float(difference @ difference)
            strain_squared += float(np.sum((difference * sd[load]) ** 2))
            if np.isfinite(stds[load]):
                if gamma_mode == "profile":
                    prior_squared += float(prior_residual**2)
                elif stds[load] > 0.0:
                    prior_squared += float(
                        ((gammas_full[load] - means[load]) / stds[load]) ** 2
                    )
        return {
            "calibration_residual_norm": float(np.sqrt(calibration_squared)),
            "strain_residual_norm": float(np.sqrt(strain_squared)),
            "amplitude_prior_residual_norm": float(np.sqrt(prior_squared)),
        }

    records = []
    for start in start_list:
        _cache["x"] = None
        _cache["result"] = None
        started = time.perf_counter()

        def jac_only(x):
            return residual(x)[1]

        solution = least_squares(
            lambda x: residual(x)[0], start, jac=jac_only, bounds=bounds,
            x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=max_nfev
        )
        vector, jacobian = residual(solution.x)
        if gamma_mode == "joint":
            gammas_full = np.ones(forward.centers.size)
            gammas_full[stds == 0.0] = means[stds == 0.0]
            gammas_full[gamma_loads] = solution.x[n:]
        else:
            gammas_full = np.asarray(
                [piece[3] for piece in data_terms(solution.x[:n])], dtype=float
            )
        if gamma_mode == "joint":
            # Include gamma variables in KKT projection
            projected = _projected_gradient(
                solution.x, 2.0 * jacobian.T @ vector, bounds[0], bounds[1]
            )
        else:
            # Only coefficients
            projected = _projected_gradient(
                solution.x[:n], 2.0 * jacobian[:, :n].T @ vector, lower_c, upper_c
            )
        records.append(
            {
                "x": solution.x.copy(), "cost": float(vector @ vector),
                "success": bool(solution.success), "status": int(solution.status),
                "optimality": float(solution.optimality), "nfev": int(solution.nfev),
                "projected_first_order": float(np.max(np.abs(projected))),
                "wall_time_s": time.perf_counter() - started,
                **residual_metadata(solution.x[:n], gammas_full),
            }
        )
    successful = [index for index, record in enumerate(records) if record["success"]]
    candidates = successful or list(range(len(records)))
    chosen = min(candidates, key=lambda index: records[index]["cost"])
    gammas = []
    if gamma_mode == "profile":
        for _, _, _, gamma, _, _, _ in data_terms(records[chosen]["x"][:n]):
            gammas.append(float(gamma))
    elif gamma_mode == "joint":
        gammas = np.ones(forward.centers.size)
        gammas[stds == 0.0] = means[stds == 0.0]
        gammas[gamma_loads] = records[chosen]["x"][n:]
    else:
        gammas = means.copy()
    return {
        "coefficients": records[chosen]["x"][:n].copy(), "gamma": np.asarray(gammas),
        "chosen_start": chosen, "successful_start": bool(successful),
        "starts": records, "gamma_mode": gamma_mode, "penalty": penalty,
        "penalty_lambda": penalty_lambda,
        "load_mean": means.copy(), "load_std": stds.copy(),
    }


__all__ = [
    "EI", "K0", "RADIUS", "LOAD_FORCE", "LOAD_WIDTH", "SupportBasis",
    "HermiteBeamFE", "penalty_and_gradient", "profile_load_amplitude",
    "fit_support", "GAMMA_LOWER", "GAMMA_UPPER",
]
