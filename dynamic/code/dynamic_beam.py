"""Dynamic transfer beam: production harmonic/transient dynamics + references.

Physics (protocol step 1): 100 m clamped Euler-Bernoulli pipe on bilateral
springs k(x)=k0*(1-d(x)), line mass m=400 kg/m, distributed viscous
c=2*zeta*sqrt(m*k0) with zeta=0.02, complex convention exp(+i omega t),
consistent mass, 0.2 m rotation-difference gauges. Channel "force": 1 kN over
4.3 m footprint centred at +6 m. Channel "ground": imposed foundation motion
u_g = 1 mm cos^2(pi x/40) on |x|<20 m (zero outside) applied as
(k(x)+i omega c) u_g; pipe ends stay clamped (finite fixture, not infinite).

Production assembly is a hand-rolled Hermite beam (mirrors beam_inverse
quadrature); the independent reference uses scikit-fem (vendor) assembly.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

from beam_inverse import (
    EI,
    K0,
    RADIUS,
    SupportBasis,
    _hermite_shapes,
    _hermite_derivatives,
)

# Protocol-fixed dynamic parameters (numerical test inputs, not field data).
M_LINE = 400.0                      # kg/m line mass
ZETA = 0.02                         # damping ratio target for k0 mode
C_LINE = 2.0 * ZETA * np.sqrt(M_LINE * K0)  # N s / m^2 distributed viscous
FORCE_TOTAL = 1000.0                # N, force channel total
FOOTPRINT = 4.3                     # m
FORCE_CENTER = 6.0                  # m
UG_AMPLITUDE = 1.0e-3               # m
UG_HALFWIDTH = 20.0                 # m
GAUGE_LENGTH = 0.2                  # m
GAUGES = np.linspace(-12.0, 12.0, 61)
DOMAIN = (-50.0, 50.0)
OMEGA0 = np.sqrt(K0 / M_LINE)       # rad/s
F0 = OMEGA0 / (2.0 * np.pi)         # Hz
RATIOS = (0.25, 0.75, 1.00, 1.15)

_GQ, _GW = np.polynomial.legendre.leggauss(4)
_T = (_GQ + 1.0) / 2.0
_WQ = _GW / 2.0


def ground_profile(x: np.ndarray) -> np.ndarray:
    """Imposed foundation displacement profile u_g(x) in metres."""
    x = np.asarray(x, dtype=float)
    return np.where(
        np.abs(x) < UG_HALFWIDTH,
        UG_AMPLITUDE * np.cos(np.pi * x / (2.0 * UG_HALFWIDTH)) ** 2,
        0.0,
    )


def aligned_mesh(h: float, extra_edges: tuple[float, ...] = ()) -> np.ndarray:
    """Uniform mesh on DOMAIN plus force-footprint and ground edges."""
    half = FOOTPRINT / 2.0
    edges = [DOMAIN[0], DOMAIN[1], FORCE_CENTER - half, FORCE_CENTER + half,
             -UG_HALFWIDTH, UG_HALFWIDTH, *extra_edges]
    pts = np.r_[np.arange(DOMAIN[0], DOMAIN[1] + h / 2.0, h), edges]
    pts = np.sort(np.clip(pts, DOMAIN[0], DOMAIN[1]))
    keep = np.ones(pts.size, bool)
    keep[1:] = np.diff(pts) > h / 100.0
    return pts[keep]


def _support_values(support, x: np.ndarray, basis: SupportBasis) -> np.ndarray:
    """Loss d(x) in [0,1] from None / callable / 29 interior B-spline coeffs."""
    if support is None:
        return np.zeros_like(x)
    if callable(support):
        return np.asarray(support(x), dtype=float)
    coeffs = np.asarray(support, dtype=float)
    if coeffs.size == basis.n_variables:
        coeffs = basis.full_coefficients(coeffs)
    return basis.values(x) @ coeffs


_MESH_CACHE: dict[bytes, dict] = {}


def _mesh_matrices(points: np.ndarray, basis: SupportBasis) -> dict:
    """Field-independent element matrices, cached by mesh."""
    key = points.tobytes()
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    n = points.size
    ne = n - 1
    h = np.diff(points)
    dofs = np.column_stack([2 * np.arange(ne), 2 * np.arange(ne) + 1,
                            2 * np.arange(1, ne + 1), 2 * np.arange(1, ne + 1) + 1])
    rows = np.repeat(dofs, 4, axis=1).ravel()
    cols = np.tile(dofs, (1, 4)).ravel()

    bend = np.empty((ne, 4, 4))
    mass = np.empty((ne, 4, 4))
    found = np.empty((ne, 4, 4))
    kbasis = np.empty((ne, basis.n_coefficients, 4, 4))
    b_force = np.zeros((ne, 4))
    g_c = np.zeros((ne, 4))
    g_k_intact = np.zeros((ne, 4))
    g_k_basis = np.empty((ne, basis.n_coefficients, 4))
    ug = ground_profile(points[:-1, None] + h[:, None] * _T[None, :])
    bvals = basis.values((points[:-1, None] + h[:, None] * _T[None, :]).ravel()
                         ).reshape(ne, _T.size, basis.n_coefficients)
    lo_f, hi_f = FORCE_CENTER - FOOTPRINT / 2.0, FORCE_CENTER + FOOTPRINT / 2.0
    for e in range(ne):
        he = h[e]
        sh = _hermite_shapes(_T, he)
        bend[e] = EI / he**3 * np.array(
            [[12, 6 * he, -12, 6 * he], [6 * he, 4 * he * he, -6 * he, 2 * he * he],
             [-12, -6 * he, 12, -6 * he], [6 * he, 2 * he * he, -6 * he, 4 * he * he]])
        nn = np.einsum("q,qi,qj->ij", _WQ, sh, sh, optimize=True)
        mass[e] = M_LINE * he * nn
        found[e] = K0 * he * nn
        kbasis[e] = K0 * he * np.einsum("q,qc,qi,qj->cij", _WQ, bvals[e], sh, sh,
                                        optimize=True)
        xc = points[e] + he * _T
        inside = ((xc >= lo_f) & (xc <= hi_f)).astype(float)  # edges mesh-aligned
        b_force[e] = FORCE_TOTAL / FOOTPRINT * he * (_WQ * inside) @ sh
        g_c[e] = C_LINE * he * (_WQ * ug[e]) @ sh
        g_k_intact[e] = K0 * he * (_WQ * ug[e]) @ sh
        g_k_basis[e] = K0 * he * np.einsum("q,qc,qi->ci", _WQ * ug[e], bvals[e], sh)

    def csr(local):
        return coo_matrix((local.ravel(), (rows, cols)), shape=(2 * n, 2 * n)).tocsr()

    out = {
        "K_intact": csr(bend + found), "M": csr(mass), "C": csr(mass) * (C_LINE / M_LINE),
        "K_basis": [csr(kbasis[:, c]) for c in range(basis.n_coefficients)],
        "b_force": np.bincount(dofs.ravel(), weights=b_force.ravel(), minlength=2 * n),
        "g_c": np.bincount(dofs.ravel(), weights=g_c.ravel(), minlength=2 * n),
        "g_k_intact": np.bincount(dofs.ravel(), weights=g_k_intact.ravel(), minlength=2 * n),
        "g_k_basis_dense": np.array(
            [np.bincount(dofs.ravel(), weights=g_k_basis[:, c].ravel(), minlength=2 * n)
             for c in range(basis.n_coefficients)]),
        "points": points,
    }
    _MESH_CACHE[key] = out
    return out


def gauge_matrix(points: np.ndarray, gauges: np.ndarray = GAUGES,
                 gauge_length: float = GAUGE_LENGTH, radius: float = RADIUS
                 ) -> csr_matrix:
    """Real 0.2 m rotation-difference gauge matrix G (n_gauges x ndof)."""
    n = points.size
    half = gauge_length / 2.0

    def rot_rows(x_eval):
        e = np.clip(np.searchsorted(points, x_eval) - 1, 0, n - 2)
        t = (x_eval - points[e]) / (points[e + 1] - points[e])
        dsh = _hermite_derivatives(t, points[e + 1] - points[e])
        r, c, v = [], [], []
        for row in range(x_eval.size):
            ee = e[row]
            for j in range(4):
                r.append(row)
                c.append(2 * ee + j if j < 2 else 2 * (ee + 1) + (j - 2))
                v.append(dsh[row, j])
        return coo_matrix((v, (r, c)), shape=(x_eval.size, 2 * n)).tocsr()

    left = rot_rows(gauges - half)
    right = rot_rows(gauges + half)
    return (-radius / gauge_length * (right - left)).tocsr()


class BeamDynamics:
    """Production harmonic/transient solver for one support field."""

    def __init__(self, points: np.ndarray, support=None, *, bc: str = "clamp",
                 basis: SupportBasis | None = None):
        self.points = np.asarray(points, dtype=float)
        self.basis = basis or SupportBasis()
        mats = _mesh_matrices(self.points, self.basis)
        self._mats = mats
        if support is None:
            self.K = mats["K_intact"]
            self.g_k = mats["g_k_intact"]
        elif callable(support):
            self.K, self.g_k = self._assemble_callable(support)
        else:
            coeffs = np.asarray(support, dtype=float)
            full = self.basis.full_coefficients(coeffs) if coeffs.size == self.basis.n_variables else coeffs
            K = mats["K_intact"].copy()
            g_k = mats["g_k_intact"].copy()
            for value, Kb, gb in zip(full, mats["K_basis"], mats["g_k_basis_dense"], strict=True):
                if value:
                    K = K - value * Kb
                    g_k = g_k - value * gb
            self.K, self.g_k = K, g_k
        self.M, self.C = mats["M"], mats["C"]
        self.b_force, self.g_c = mats["b_force"], mats["g_c"]
        ndof = 2 * self.points.size
        if bc == "clamp":
            self.free = np.arange(2, ndof - 2)
        elif bc == "ss":
            self.free = np.setdiff1d(np.arange(ndof), [0, ndof - 2])  # w fixed, theta free
        else:
            raise ValueError("bc must be 'clamp' or 'ss'")
        self.G = gauge_matrix(self.points)

    def _assemble_callable(self, support):
        """Direct quadrature assembly for a continuous loss callable."""
        mats, points = self._mats, self.points
        ne = points.size - 1
        h = np.diff(points)
        xq = (points[:-1, None] + h[:, None] * _T[None, :]).ravel()
        d = _support_values(support, xq, self.basis).reshape(ne, _T.size)
        loss = np.empty((ne, 4, 4))
        g_loss = np.zeros((ne, 4))
        ug = ground_profile(points[:-1, None] + h[:, None] * _T[None, :])
        for e in range(ne):
            sh = _hermite_shapes(_T, h[e])
            loss[e] = K0 * h[e] * np.einsum("q,qi,qj->ij", _WQ * d[e], sh, sh,
                                            optimize=True)
            g_loss[e] = K0 * h[e] * (_WQ * d[e] * ug[e]) @ sh
        dofs = np.column_stack([2 * np.arange(ne), 2 * np.arange(ne) + 1,
                                2 * np.arange(1, ne + 1), 2 * np.arange(1, ne + 1) + 1])
        rows = np.repeat(dofs, 4, axis=1).ravel()
        cols = np.tile(dofs, (1, 4)).ravel()
        K_loss = coo_matrix((loss.ravel(), (rows, cols)),
                            shape=(2 * points.size, 2 * points.size)).tocsr()
        g_loss_v = np.bincount(dofs.ravel(), weights=g_loss.ravel(), minlength=2 * points.size)
        return mats["K_intact"] - K_loss, mats["g_k_intact"] - g_loss_v

    def sine_load(self, p0: float) -> np.ndarray:
        """Consistent load vector for p(x)=p0 sin(pi (x-x_min)/L) (verification)."""
        points = self.points
        L = points[-1] - points[0]
        ne = points.size - 1
        h = np.diff(points)
        vec = np.zeros(2 * points.size)
        for e in range(ne):
            sh = _hermite_shapes(_T, h[e])
            xq = points[e] + h[e] * _T
            fe = p0 * np.sin(np.pi * (xq - points[0]) / L)
            vec_e = h[e] * (_WQ * fe) @ sh
            for j in range(4):
                vec[2 * e + j if j < 2 else 2 * (e + 1) + (j - 2)] += vec_e[j]
        return vec

    def solve_frequency(self, omega: float) -> dict[str, np.ndarray]:
        """Complex gauge strains for both channels at circular frequency omega."""
        free = self.free
        A = (self.K + 1j * omega * self.C - omega**2 * self.M)[free][:, free].tocsc()
        lu = splu(A)
        out = {}
        for name, b in (("force", self.b_force),
                        ("ground", self.g_k + 1j * omega * self.g_c)):
            u = np.zeros(2 * self.points.size, dtype=complex)
            u[free] = lu.solve(b[free])
            out[name] = {"state": u, "strain": self.G @ u}
        return out

    def newmark(self, load: np.ndarray, factor, dt: float, n_steps: int):
        """Average-acceleration Newmark for load(t)=factor(t)*load, zero IC."""
        return newmark_average(self.M, self.C, self.K, self.G, self.free,
                               load, factor, dt, n_steps)


def newmark_average(M, C, K, G, free, load, factor, dt: float, n_steps: int):
    """Average-acceleration (beta=1/4, gamma=1/2) recurrence, zero IC.

    load(t) = factor(t) * load; gauge strains G @ u recorded every step.
    Shared stepping for production and independently assembled matrices.
    """
    A = (K + (2.0 / dt) * C + (4.0 / dt**2) * M)[free][:, free].tocsc()
    lu = splu(A)
    u = np.zeros(M.shape[0])
    v = np.zeros_like(u)
    a = np.zeros_like(u)
    f0 = factor(0.0) * load
    a[free] = splu(M[free][:, free].tocsc()).solve(f0[free])
    times = dt * np.arange(n_steps + 1)
    record = np.empty((n_steps + 1, G.shape[0]))
    record[0] = G @ u
    for n in range(n_steps):
        rhs = (factor(times[n + 1]) * load + M @ (4 * u / dt**2 + 4 * v / dt + a)
               + C @ (2 * u / dt + v))
        u_new = np.zeros_like(u)
        u_new[free] = lu.solve(rhs[free])
        a_new = 4 * (u_new - u) / dt**2 - 4 * v / dt - a
        v_new = v + dt * (a + a_new) / 2.0
        u, v, a = u_new, v_new, a_new
        record[n + 1] = G @ u
    return times, record


# ---------------------------------------------------------------- references


def ss_sine_exact(x: np.ndarray, omega: float, p0: float,
                  L: float = 100.0, ei: float = EI, k0: float = K0,
                  mass: float = M_LINE, damp: float = C_LINE,
                  radius: float = RADIUS, gauge_length: float = GAUGE_LENGTH):
    """Exact complex finite-gauge strain, uniform SS beam, sine load, exp(+iwt)."""
    xi = np.asarray(x, dtype=float) - (-L / 2.0)
    kb = np.pi / L
    D = ei * kb**4 + k0 + 1j * omega * damp - omega**2 * mass
    half = gauge_length / 2.0
    dtheta = p0 * kb / D * (np.cos(kb * (xi + half)) - np.cos(kb * (xi - half)))
    return -radius / gauge_length * dtheta


def ss_modal_oscillator(omega_load: float | None, p0: float, times: np.ndarray,
                        pulse_T: float | None = None, L: float = 100.0):
    """Exact modal coordinate q(t) of the uniform SS sine-mode via solve_ivp.

    w(x,t)=q(t) sin(pi xi/L); m q'' + c q' + (EI kb^4 + k0) q = p0 f(t).
    """
    from scipy.integrate import solve_ivp

    kb = np.pi / L
    wn2 = (EI * kb**4 + K0) / M_LINE
    if pulse_T is not None:
        def f(t):
            return np.where((t > 0.0) & (t < pulse_T), np.sin(np.pi * t / pulse_T) ** 2, 0.0)
    else:
        def f(t):
            return np.cos(omega_load * t)

    def rhs(t, y):
        return [y[1], p0 / M_LINE * f(t) - C_LINE / M_LINE * y[1] - wn2 * y[0]]

    sol = solve_ivp(rhs, [times[0], times[-1]], [0.0, 0.0], t_eval=times,
                    rtol=1e-11, atol=1e-14)
    return sol.y[0]


def ss_gauge_from_q(x: np.ndarray, q: np.ndarray, L: float = 100.0,
                    radius: float = RADIUS, gauge_length: float = GAUGE_LENGTH):
    """Finite-gauge strain field for w=q sin(pi xi/L), real q(t)."""
    xi = np.asarray(x, dtype=float) + L / 2.0
    kb = np.pi / L
    half = gauge_length / 2.0
    dtheta = kb * (np.cos(kb * (xi + half)) - np.cos(kb * (xi - half)))
    return -radius / gauge_length * np.asarray(q)[:, None] * dtheta[None, :]
