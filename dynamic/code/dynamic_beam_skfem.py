"""Independent scikit-fem dynamic assembly for reference solutions.

Same physics as dynamic_beam.py, but all matrices, load vectors and the
rotation-gauge matrix are assembled through scikit-fem 12.0.2 quadrature and
ElementLineHermite, on independent (finer, footprint-aligned) grids.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from skfem import BilinearForm, Basis, ElementLineHermite, LinearForm, MeshLine

from beam_inverse import EI, K0, RADIUS
from dynamic_beam import (C_LINE, FOOTPRINT, FORCE_CENTER, FORCE_TOTAL,
                          GAUGE_LENGTH, GAUGES, M_LINE, ground_profile,
                          newmark_average)


@BilinearForm
def _beam_form(u, v, w):
    return w["ei"] * u.hess[0, 0] * v.hess[0, 0] + w["foundation"] * u * v


@BilinearForm
def _density_form(u, v, w):
    return w["density"] * u * v


@LinearForm
def _force_load(v, w):
    inside = np.abs(w.x[0] - FORCE_CENTER) <= FOOTPRINT / 2.0
    return (FORCE_TOTAL / FOOTPRINT * inside) * v


@LinearForm
def _ground_k_load(v, w):
    return (w["modulus"] * ground_profile(w.x[0])) * v


@LinearForm
def _ground_c_load(v, w):
    return (C_LINE * ground_profile(w.x[0])) * v


def _rotation_matrix(mesh, basis, x_eval: np.ndarray):
    """Real rotation interpolation matrix at x_eval via element gbasis."""
    finder = mesh.element_finder()
    elements = finder(x_eval)
    h = mesh.p[0, mesh.t[1, elements]] - mesh.p[0, mesh.t[0, elements]]
    local = (x_eval - mesh.p[0, elements]) / h
    rows, cols, vals = [], [], []
    for i in range(4):
        field = basis.elem.gbasis(basis.mapping, np.array([local]), i, elements)[0]
        grads = np.diag(field.grad[0])
        rows.extend(range(x_eval.size))
        cols.extend(basis.element_dofs[i, elements].tolist())
        vals.extend(grads.tolist())
    return coo_matrix((vals, (rows, cols)),
                      shape=(x_eval.size, basis.N)).tocsr()


class SkfemDynamics:
    """Independent harmonic/transient reference for a continuous loss callable."""

    def __init__(self, points: np.ndarray, support_callable=None,
                 gauges: np.ndarray = GAUGES, gauge_length: float = GAUGE_LENGTH):
        points = np.asarray(points, dtype=float)
        mesh = MeshLine(points)
        basis = Basis(mesh, ElementLineHermite())
        coords = basis.global_coordinates()[0].T
        if support_callable is None:
            modulus = np.full((basis.nelems, basis.X.shape[-1]), K0)
        else:
            d = np.asarray(support_callable(coords.ravel()), dtype=float)
            modulus = K0 * (1.0 - d.reshape(coords.shape).T)
        self.K = _beam_form.assemble(basis, ei=EI, foundation=modulus).tocsr()
        self.M = _density_form.assemble(basis, density=M_LINE).tocsr()
        self.C = _density_form.assemble(basis, density=C_LINE).tocsr()
        self.b_force = _force_load.assemble(basis)
        self.g_k = _ground_k_load.assemble(basis, modulus=modulus)
        self.g_c = _ground_c_load.assemble(basis)
        half = gauge_length / 2.0
        self.G = (-RADIUS / gauge_length
                  * (_rotation_matrix(mesh, basis, gauges + half)
                     - _rotation_matrix(mesh, basis, gauges - half))).tocsr()
        self.free = np.arange(2, basis.N - 2)  # clamped ends
        self.N = basis.N

    def solve_frequency(self, omega: float) -> dict[str, np.ndarray]:
        free = self.free
        A = (self.K + 1j * omega * self.C - omega**2 * self.M)[free][:, free].tocsc()
        lu = splu(A)
        out = {}
        for name, b in (("force", self.b_force),
                        ("ground", self.g_k + 1j * omega * self.g_c)):
            u = np.zeros(self.N, dtype=complex)
            u[free] = lu.solve(b[free])
            out[name] = {"state": u, "strain": self.G @ u}
        return out

    def newmark(self, load: np.ndarray, factor, dt: float, n_steps: int):
        """Shared recurrence on independently assembled matrices."""
        return newmark_average(self.M, self.C, self.K, self.G, self.free,
                               load, factor, dt, n_steps)


__all__ = ["SkfemDynamics"]
