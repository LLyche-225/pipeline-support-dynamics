"""Coordinator checks of coupled mechanics using rigid motion and force balance."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from continuum_reference import continuum_forward, _assemble_interface
from skfem_reference import mesh_with_load_edges, skfem_forward

points = mesh_with_load_edges(-12, 12, 0.4, np.array([-2., 0., 2.]))
depth = np.linspace(0, 5, 11)
parts = _assemble_interface(points, depth, lambda x: np.zeros_like(x), 36e6)
sampled = []
def graded_support(x):
    sampled.append(np.asarray(x).copy())
    return .4 * np.exp(-((x - 1) ** 2) / 2)
_assemble_interface(points, depth, graded_support, 36e6)
gauss, _ = np.polynomial.legendre.leggauss(4)
expected_points = points[:-1][None, :] + .5 * (1 + gauss[:, None]) * np.diff(points)[None, :]
assert len(sampled) == 1
assert np.allclose(sampled[0], expected_points, rtol=0, atol=1e-12)
bb, bs, ss = parts[:3]
interface = sparse.bmat([[bb, bs], [bs.T, ss]], format="csr")
translation = np.zeros(interface.shape[0])
translation[:bb.shape[0]:2] = 1.0
translation[bb.shape[0] + 1::2] = 1.0
rigid_error = np.linalg.norm(interface @ translation) / np.max(np.abs(interface.data))
assert rigid_error < 1e-10, ("interface violates equal vertical translation", rigid_error)

evaluation = np.linspace(-8, 8, 41)
options = dict(points=points, depth_nodes=depth, load_centers=np.array([-2., 0., 2.]),
               support_callable=lambda x: np.zeros_like(x), evaluation_points=evaluation,
               gauge_length=.2, ei=1.6232123e8, interface_stiffness=36e6,
               soil_E=100e6, soil_nu=.3, width=.6, force=750000., load_width=4.3, radius=.3)
result = continuum_forward(**options)
diag = result["diagnostics"]
reaction_error = np.max(np.abs(diag["vertical_reaction_error"]) / 750000.)
assert reaction_error < 1e-8, ("force balance", reaction_error)
assert np.max(diag["relative_energy_balance_error"]) < 1e-8
nodes = result["soil_nodes"]
surface = np.flatnonzero(np.isclose(nodes[:, 1], 0))
surface = surface[np.argsort(nodes[surface, 0])]
interpolated = np.column_stack([np.interp(evaluation, nodes[surface, 0], result["soil_uz"][surface, j]) for j in range(3)])
surface_error = np.linalg.norm(interpolated - result["soil_surface_vertical_displacement"])
assert surface_error < 1e-12, ("multi-point vector probe component ordering", surface_error)
assert np.max(result["displacement"]) > 0
symmetry = np.linalg.norm(result["strain"][:, 0] - result["strain"][::-1, 2]) / np.linalg.norm(result["strain"][:, 0])
assert symmetry < 1e-7, ("reflection symmetry", symmetry)
stiff = continuum_forward(**(options | {"soil_E": 1e12}))
_, winkler = skfem_forward(points, options["load_centers"], support_callable=options["support_callable"],
                          evaluation_points=evaluation, gauge_length=.2, foundation=36e6,
                          force=750000., load_width=4.3, ei=1.6232123e8, radius=.3)
limit_error = np.linalg.norm(stiff["strain"] - winkler) / np.linalg.norm(winkler)
assert limit_error < .002, ("rigid soil Winkler limit", limit_error)
print(json.dumps({"interface_rigid_error": float(rigid_error), "relative_force_error": float(reaction_error),
                  "surface_probe_error": float(surface_error), "reflection_relative_error": float(symmetry),
                  "stiff_soil_winkler_relative_error": float(limit_error), "mesh": result["mesh"]}, indent=2))
