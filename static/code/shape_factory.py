"""Deterministic support shape generation for campaign."""

import numpy as np


def intact_shape(x: np.ndarray) -> np.ndarray:
    """Intact foundation: d=0 everywhere."""
    return np.zeros_like(x, dtype=float)


def rectangle_shape(x: np.ndarray) -> np.ndarray:
    """Rectangle: d=0.7 on |x-0.8|<=0.8, else 0."""
    x = np.asarray(x, dtype=float)
    mask = np.abs(x - 0.8) <= 0.8
    return np.where(mask, 0.7, 0.0)


def asymmetric_shape(x: np.ndarray) -> np.ndarray:
    """Asymmetric Gaussian: 0.7*exp(-(x-1.2)^2/(2*s^2)), s=0.55 for x<1.2 else 1.35."""
    x = np.asarray(x, dtype=float)
    s = np.where(x < 1.2, 0.55, 1.35)
    return 0.7 * np.exp(-((x - 1.2) ** 2) / (2 * s**2))


def double_shape(x: np.ndarray) -> np.ndarray:
    """Double peak: 0.7*exp(-(x+1.5)^2/(2*0.35^2))+0.7*exp(-(x-1.5)^2/(2*0.35^2))."""
    x = np.asarray(x, dtype=float)
    left = 0.7 * np.exp(-((x + 1.5) ** 2) / (2 * 0.35**2))
    right = 0.7 * np.exp(-((x - 1.5) ** 2) / (2 * 0.35**2))
    return left + right


def broad_shallow_shape(x: np.ndarray) -> np.ndarray:
    """Broad shallow: 0.25*exp(-(x+0.7)^2/(2*2.5^2))."""
    x = np.asarray(x, dtype=float)
    return 0.25 * np.exp(-((x + 0.7) ** 2) / (2 * 2.5**2))


def pilot_gaussian_shape(x: np.ndarray) -> np.ndarray:
    """Pilot Gaussian: d=0.45*exp(-(x+0.3)^2/(2*1.1^2))."""
    x = np.asarray(x, dtype=float)
    return 0.45 * np.exp(-((x + 0.3) ** 2) / (2 * 1.1**2))


SHAPES = {
    "intact": intact_shape,
    "rectangle": rectangle_shape,
    "asymmetric": asymmetric_shape,
    "double": double_shape,
    "broad_shallow": broad_shallow_shape,
    "pilot_gaussian": pilot_gaussian_shape,
}


def get_shape_function(name: str):
    """Return the shape function for the given name."""
    if name not in SHAPES:
        raise ValueError(f"Unknown shape: {name}")
    return SHAPES[name]


__all__ = ["get_shape_function", "SHAPES"]
