"""Independent Green's function solver for intact foundation verification.

Extracted from original/IJCE_Teacher_Handover_20261102/03_Code/project/
Step_01_Reference_Solver/src/reference_solver.py for verification only.

Provenance: Original intact Green solution using NumPy/SciPy analytical kernels.
Beta formula and theta kernel from original implementation.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeamParameters:
    """Physical beam parameters for Green's function solution."""
    ei: float
    k0: float
    outer_fiber_radius: float

    def __post_init__(self) -> None:
        if self.ei <= 0.0:
            raise ValueError("ei must be positive")
        if self.k0 <= 0.0:
            raise ValueError("k0 must be positive")
        if self.outer_fiber_radius <= 0.0:
            raise ValueError("outer_fiber_radius must be positive")

    @property
    def beta(self) -> float:
        """Characteristic parameter: beta = (k0 / (4*EI))^0.25."""
        return float((self.k0 / (4.0 * self.ei)) ** 0.25)


def midpoint_cells(
    x_left: float,
    x_right: float,
    requested_dx: float,
) -> tuple[np.ndarray, float]:
    """Generate uniform cell centers for integration."""
    if not x_right > x_left:
        raise ValueError("cell interval must have positive length")
    if requested_dx <= 0.0:
        raise ValueError("requested_dx must be positive")
    count = max(1, int(np.ceil((x_right - x_left) / requested_dx)))
    actual_dx = (x_right - x_left) / count
    centers = x_left + (np.arange(count, dtype=float) + 0.5) * actual_dx
    return centers, actual_dx


def green_point_kernels(
    separation: np.ndarray,
    *,
    beta: float,
    k0: float,
) -> dict[str, np.ndarray]:
    """Green's function kernels for point load on intact foundation.

    Original formula: theta(s) = -sign(s)*beta^2/k0*exp(-beta*|s|)*sin(beta*|s|)
    """
    if beta <= 0.0 or k0 <= 0.0:
        raise ValueError("beta and k0 must be positive")
    separation = np.asarray(separation, dtype=float)
    radius = np.abs(separation)
    z = beta * radius
    decay = np.exp(-z)
    cosine = np.cos(z)
    sine = np.sin(z)
    coordinate_scale = max(float(np.max(radius, initial=0.0)), 1.0)
    tolerance = 16.0 * np.finfo(float).eps * coordinate_scale
    sign = np.where(radius <= tolerance, 0.0, np.sign(separation))
    return {
        "w": beta / (2.0 * k0) * decay * (cosine + sine),
        "theta": -sign * beta**2 / k0 * decay * sine,
        "M": 1.0 / (4.0 * beta) * decay * (cosine - sine),
        "Q": -0.5 * sign * decay * cosine,
    }


def green_uniform_load(
    x: np.ndarray,
    *,
    parameters: BeamParameters,
    center: float,
    width: float,
    total_force: float,
    source_dx: float,
) -> dict[str, np.ndarray | float]:
    """Green's function solution for uniform load on intact foundation.

    Integrates point kernels over uniform footprint to obtain gauge strain
    by rotation difference.
    """
    x = np.asarray(x, dtype=float)
    sources, actual_dx = midpoint_cells(
        center - 0.5 * width,
        center + 0.5 * width,
        source_dx,
    )
    line_intensity = total_force / width
    cell_force = line_intensity * actual_dx

    w = np.zeros_like(x)
    theta = np.zeros_like(x)

    # Integrate in blocks to avoid memory explosion
    for start in range(0, sources.size, 128):
        block = sources[start : start + 128]
        kernels = green_point_kernels(
            x[:, None] - block[None, :],
            beta=parameters.beta,
            k0=parameters.k0,
        )
        w += np.sum(kernels["w"] * cell_force, axis=1)
        theta += np.sum(kernels["theta"] * cell_force, axis=1)

    return {
        "w": w,
        "theta": theta,
        "actual_source_dx": actual_dx,
        "source_cells": float(sources.size),
    }


def green_gauge_strain(
    evaluation_points: np.ndarray,
    *,
    parameters: BeamParameters,
    center: float,
    width: float,
    total_force: float,
    gauge_length: float,
    source_dx: float = 0.001,
) -> np.ndarray:
    """Compute gauge strain using Green's function rotation difference.

    Gauge strain = -radius * (theta_right - theta_left) / gauge_length
    """
    evaluation_points = np.asarray(evaluation_points, dtype=float)
    half_gauge = gauge_length / 2.0

    left_result = green_uniform_load(
        evaluation_points - half_gauge,
        parameters=parameters,
        center=center,
        width=width,
        total_force=total_force,
        source_dx=source_dx,
    )
    right_result = green_uniform_load(
        evaluation_points + half_gauge,
        parameters=parameters,
        center=center,
        width=width,
        total_force=total_force,
        source_dx=source_dx,
    )

    theta_left = left_result["theta"]
    theta_right = right_result["theta"]

    strain = -parameters.outer_fiber_radius * (theta_right - theta_left) / gauge_length
    return strain


__all__ = ["BeamParameters", "green_gauge_strain", "green_uniform_load", "green_point_kernels"]
