"""Verification suite for numerical correctness.

Corrected verification module addressing:
- Uses interior coefficients for derivative checks (not clipped at bounds)
- Matches evaluation locations for cross-assembly comparison
- Uses actual direct truth for fine-reference checks
- Includes prescribed 5-point mesh sequence
- Preserves numerical arrays for plotting
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from beam_inverse import HermiteBeamFE, SupportBasis, fit_support
from skfem_reference import skfem_forward, mesh_with_load_edges
from green_verification import BeamParameters, green_gauge_strain
from shape_factory import pilot_gaussian_shape


def verify_load_conservation(
    load_centers: list[float],
    force: float = 750_000.0,
    domain: tuple = (-50.0, 50.0),
    n_elements: int = 400,
    rtol: float = 1e-13,
) -> dict:
    """Verify total load conservation in RHS assembly."""
    fe = HermiteBeamFE(
        load_centers,
        x_min=domain[0],
        x_max=domain[1],
        n_elements=n_elements,
    )
    displacement_dofs = np.arange(0, 2 * fe.x.size, 2)

    results = {}
    for idx in range(len(load_centers)):
        total = float(np.sum(fe.rhs[displacement_dofs, idx]))
        error = abs(total - force) / force
        results[f"load_{idx}"] = {
            "expected": force,
            "actual": total,
            "relative_error": error,
            "passes": error < rtol,
        }

    return {
        "test": "load_conservation",
        "all_pass": all(r["passes"] for r in results.values()),
        "details": results,
    }


def verify_symmetry(
    n_elements: int = 400,
    n_eval: int = 41,
    atol: float = 1e-12,
) -> dict:
    """Verify symmetry of intact response under centered load."""
    fe = HermiteBeamFE([0.0], n_elements=n_elements)
    symmetric = np.linspace(-20, 20, n_eval)
    strain, _ = fe.strain_and_jacobian(np.zeros(fe.basis.n_variables), symmetric)

    max_asymmetry = float(np.max(np.abs(strain - strain[::-1])))

    return {
        "test": "symmetry",
        "max_asymmetry": max_asymmetry,
        "threshold": atol,
        "passes": max_asymmetry < atol,
    }


def verify_finite_difference_jacobian(
    shape_name: str = "pilot_gaussian",
    n_coefficients: int = 9,
    n_elements: int = 120,
    n_eval: int = 41,
    h: float = 1e-6,
    rtol: float = 2e-4,
) -> dict:
    """Verify analytic Jacobian against finite differences using three load centers and interior coefficients."""
    from shape_factory import get_shape_function

    shape_func = get_shape_function(shape_name)
    basis = SupportBasis(n_coefficients)
    control_x = np.linspace(*basis.support, basis.n_coefficients)

    # Use genuinely interior coefficients: 0.25 + 0.05 * cos(arange(nvars))
    nvars = n_coefficients - 2
    variables = 0.25 + 0.05 * np.cos(np.arange(nvars))

    # Three load centers [-4, 0, 4] with three matching sensor arrays
    load_centers = [-4.0, 0.0, 4.0]
    fe = HermiteBeamFE(load_centers, n_elements=n_elements, n_coefficients=n_coefficients)

    # Three sensor arrays, one per load
    sensor_arrays = [
        np.linspace(-20, -4, n_eval),
        np.linspace(-10, 10, n_eval),
        np.linspace(4, 20, n_eval),
    ]

    # Call with all three sensor arrays at once
    strain, jacobian = fe.strain_and_jacobian(variables, sensor_arrays)

    # Finite difference in a random direction using interior variables
    rng = np.random.default_rng(20261101)
    direction = rng.normal(size=variables.size)
    direction = direction / np.linalg.norm(direction)

    # FD without clipping (variables are interior)
    plus = fe.strain_and_jacobian(variables + h * direction, sensor_arrays)[0]
    minus = fe.strain_and_jacobian(variables - h * direction, sensor_arrays)[0]
    finite_directional = (plus - minus) / (2 * h)
    analytic_directional = jacobian @ direction

    error = np.linalg.norm(finite_directional - analytic_directional) / np.linalg.norm(analytic_directional)

    return {
        "test": "finite_difference_jacobian",
        "relative_error": float(error),
        "threshold": rtol,
        "passes": error < rtol,
        "arrays": {
            "load_centers": np.array(load_centers),
            "sensor_arrays": [s for s in sensor_arrays],
            "finite_directional": finite_directional,
            "analytic_directional": analytic_directional,
        }
    }


def verify_mesh_convergence(
    shape_name: str = "pilot_gaussian",
    n_coefficients: int = 9,
    h_sequence: list = None,
    n_eval: int = 33,
    target_rms: float = 1e-6,
) -> dict:
    """Verify mesh convergence with prescribed 5-point sequence."""
    if h_sequence is None:
        # Prescribed 5-point sequence as specified
        h_sequence = [0.5, 0.25, 0.125, 0.0625, 0.03125]

    from shape_factory import get_shape_function
    shape_func = get_shape_function(shape_name)
    basis = SupportBasis(n_coefficients)
    control_x = np.linspace(*basis.support, basis.n_coefficients)
    variables = shape_func(control_x)[1:-1]

    sensors = np.linspace(-20, 20, n_eval)
    meshes = [
        HermiteBeamFE([0.0], n_elements=int(100 / h), n_coefficients=n_coefficients)
        for h in h_sequence
    ]

    strains = [mesh.strain_and_jacobian(variables, sensors)[0] for mesh in meshes]

    # Compute errors relative to finest
    finest = strains[-1]
    errors = [
        np.linalg.norm(strains[i] - finest) / np.linalg.norm(finest)
        for i in range(len(h_sequence) - 1)
    ]

    # Check monotonic decrease (minimum 4 non-reference points on convergence curve)
    monotonic = all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))

    # Production mesh (h=0.25) vs finest: check below 1 microstrain RMS
    production_idx = h_sequence.index(0.25) if 0.25 in h_sequence else 1
    production_strain = strains[production_idx]
    rms_diff = np.sqrt(np.mean((production_strain - finest) ** 2))

    return {
        "test": "mesh_convergence",
        "h_sequence": h_sequence,
        "relative_errors": errors,
        "monotonic_decrease": monotonic,
        "production_rms_vs_finest_microstrain": float(rms_diff * 1e6),
        "production_threshold_microstrain": target_rms * 1e6,
        "passes": monotonic and rms_diff < target_rms,
        "arrays": {
            "sensors": sensors,
            "strains": np.array(strains),
        }
    }


def verify_scikit_fem_independence(
    scenario: str = "intact",
    h: float = 0.215,
    rtol: float = 1e-8,
) -> dict:
    """Verify custom FE matches scikit-fem at matching evaluation locations."""
    centers = np.array([0.0])
    x_min, x_max = -43.0, 43.0
    n_elements = int(np.round((x_max - x_min) / h))
    points = np.linspace(x_min, x_max, n_elements + 1)

    if scenario == "intact":
        support_coefficients = None
        variables = np.zeros(29)  # n_coefficients - 2
    else:
        basis = SupportBasis(n_coefficients=31)
        control_x = np.linspace(*basis.support, 31)
        d_values = pilot_gaussian_shape(control_x)
        support_coefficients = np.zeros(31)
        support_coefficients[1:-1] = d_values[1:-1]
        variables = d_values[1:-1]

    # Common evaluation points for both implementations
    evaluation = np.linspace(-20, 20, 81)

    # scikit-fem reference at exact evaluation points
    reference_w, reference_strain = skfem_forward(
        points, centers, support_coefficients,
        evaluation_points=evaluation,
        gauge_length=0.2,
    )

    # Custom implementation at same evaluation points
    custom = HermiteBeamFE(centers, x_min=x_min, x_max=x_max, n_elements=n_elements, n_coefficients=31)
    displacement, _, _ = custom._state(variables, with_jacobian=False)
    custom_w, _ = custom._evaluate(evaluation, displacement[:, 0], derivative=False)
    custom_strain = custom.strain_and_jacobian(variables, evaluation)[0]

    ref_w = reference_w.ravel() if reference_w.ndim > 1 else reference_w
    ref_strain = reference_strain.ravel() if reference_strain.ndim > 1 else reference_strain

    w_error = np.linalg.norm(custom_w - ref_w) / np.linalg.norm(ref_w)
    strain_error = np.linalg.norm(custom_strain - ref_strain) / np.linalg.norm(ref_strain)

    return {
        "test": "scikit_fem_independence",
        "scenario": scenario,
        "h": h,
        "displacement_relative_error": float(w_error),
        "strain_relative_error": float(strain_error),
        "threshold": rtol,
        "passes": w_error < rtol and strain_error < rtol,
        "arrays": {
            "evaluation": evaluation,
            "custom_w": custom_w,
            "reference_w": ref_w,
            "custom_strain": custom_strain,
            "reference_strain": ref_strain,
        }
    }


def verify_green_intact_reference(
    ei: float = 162321230.0,
    k0: float = 18e6,
    radius: float = 0.3,
    force: float = 750e3,
    source_dx: float = 0.001,
    half_source_dx: float = 0.0005,
    rtol_integration: float = 1e-6,
    rms_threshold: float = 1e-6,
    relative_threshold: float = 1e-4,
) -> dict:
    """Verify intact Green solution with prescribed 5-point mesh sequence using load-aligned meshes."""
    params = BeamParameters(ei=ei, k0=k0, outer_fiber_radius=radius)

    evaluation = np.linspace(-20, 20, 81)

    # Compute gauge errors at prescribed mesh sequence
    h_sequence = [0.5, 0.25, 0.125, 0.0625, 0.03125]

    # Green reference strain
    strain_nominal = green_gauge_strain(
        evaluation,
        parameters=params,
        center=0.0,
        width=4.3,
        total_force=force,
        gauge_length=0.2,
        source_dx=source_dx,
    )

    # Compute with half spacing to check integration error
    strain_refined = green_gauge_strain(
        evaluation,
        parameters=params,
        center=0.0,
        width=4.3,
        total_force=force,
        gauge_length=0.2,
        source_dx=half_source_dx,
    )

    integration_error = np.linalg.norm(strain_nominal - strain_refined) / np.linalg.norm(strain_refined)

    peak_strain = float(np.max(np.abs(strain_nominal)))

    # Now compare against FE at each mesh resolution using mesh_with_load_edges
    fe_errors = []
    fe_strains = []
    mesh_coordinates = []

    for h in h_sequence:
        # Use mesh_with_load_edges to align mesh with load edges
        points = mesh_with_load_edges(-50, 50, h, np.array([0.0]), load_width=4.3)
        mesh_coordinates.append(points)

        # FE solution at same evaluation points
        _, fe_strain = skfem_forward(
            points,
            np.array([0.0]),
            support_coefficients=None,  # Intact
            evaluation_points=evaluation,
            gauge_length=0.2,
            ei=ei,
            foundation=k0,
            force=force,
            load_width=4.3,
            radius=radius,
        )

        fe_strain_vec = fe_strain.ravel() if fe_strain.ndim > 1 else fe_strain
        fe_strains.append(fe_strain_vec)
        error = np.linalg.norm(fe_strain_vec - strain_refined) / np.linalg.norm(strain_refined)
        fe_errors.append(float(error))

    # Assert h=0.25 RMS below 1e-6 strain and finest relative error below 1e-4
    production_idx = h_sequence.index(0.25) if 0.25 in h_sequence else 1
    production_strain = fe_strains[production_idx]
    rms_diff = np.sqrt(np.mean((production_strain - strain_refined) ** 2))

    finest_relative_error = fe_errors[-1]

    production_passes = rms_diff < rms_threshold
    finest_passes = finest_relative_error < relative_threshold

    return {
        "test": "green_intact_reference",
        "beta": float(params.beta),
        "peak_strain_microstrain": peak_strain * 1e6,
        "integration_error": float(integration_error),
        "integration_threshold": rtol_integration,
        "integration_passes": integration_error < rtol_integration,
        "h_sequence": h_sequence,
        "fe_vs_green_errors": fe_errors,
        "production_h025_rms_strain": float(rms_diff),
        "production_rms_threshold": rms_threshold,
        "production_passes": production_passes,
        "finest_relative_error": finest_relative_error,
        "finest_relative_threshold": relative_threshold,
        "finest_passes": finest_passes,
        "passes": integration_error < rtol_integration and production_passes and finest_passes,
        "arrays": {
            "evaluation": evaluation,
            "green_strain": strain_refined,
            "fe_strains": np.array(fe_strains),
            "fe_errors": np.array(fe_errors),
            "mesh_coordinates": mesh_coordinates,
        }
    }


def verify_domain_truncation(
    shape_name: str = "pilot_gaussian",
    n_coefficients: int = 9,
    h: float = 0.25,
    domains: list = None,
    rms_threshold: float = 1e-6,
) -> dict:
    """Verify domain truncation using same physical support at different domains."""
    if domains is None:
        domains = [(-40, 40), (-50, 50), (-60, 60)]

    from shape_factory import get_shape_function
    shape_func = get_shape_function(shape_name)
    basis = SupportBasis(n_coefficients)
    control_x = np.linspace(*basis.support, basis.n_coefficients)
    variables = shape_func(control_x)[1:-1]

    # Fixed evaluation points
    evaluation = np.linspace(-20, 20, 81)

    # Run FE for each domain
    strains = []
    domain_labels = []
    for x_min, x_max in domains:
        n_elements = int((x_max - x_min) / h)
        fe = HermiteBeamFE(
            [0.0],
            x_min=x_min,
            x_max=x_max,
            n_elements=n_elements,
            n_coefficients=n_coefficients,
        )
        strain = fe.strain_and_jacobian(variables, evaluation)[0]
        strains.append(strain)
        domain_labels.append(f"[{x_min},{x_max}]")

    # Compare production ±50 to ±60
    production_idx = 1  # ±50
    extended_idx = 2    # ±60

    rms_diff = np.sqrt(np.mean((strains[production_idx] - strains[extended_idx]) ** 2))

    return {
        "test": "domain_truncation",
        "h": h,
        "domains": domain_labels,
        "production_vs_extended_rms_strain": float(rms_diff),
        "rms_threshold": rms_threshold,
        "passes": rms_diff < rms_threshold,
        "arrays": {
            "evaluation": evaluation,
            "strains": np.array(strains),
            "domain_labels": domain_labels,
        }
    }


def run_verification_suite(output_path: Path) -> dict:
    """Run complete verification suite and save results."""
    results = {
        "load_conservation": verify_load_conservation([0.0, -4.0, 4.0]),
        "symmetry": verify_symmetry(),
        "finite_difference_jacobian": verify_finite_difference_jacobian(),
        "mesh_convergence": verify_mesh_convergence(),
        "scikit_fem_intact": verify_scikit_fem_independence("intact", h=0.215),
        "scikit_fem_graded": verify_scikit_fem_independence("graded", h=0.215),
        "green_intact": verify_green_intact_reference(),
        "domain_truncation": verify_domain_truncation(),
    }

    # Count checks BEFORE adding summary
    checks = [v for k, v in results.items() if k != "summary"]
    total_tests = len(checks)

    all_pass = all(r.get("passes", r.get("all_pass", False)) for r in results.values())
    results["summary"] = {
        "all_tests_pass": bool(all_pass),
        "total_tests": total_tests,
        "passed": sum(1 for r in results.values() if r.get("passes", r.get("all_pass", False))),
    }

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        elif obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        else:
            # Keep other objects as-is (will fail if not JSON serializable)
            return obj

    results_serializable = convert_for_json(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2, allow_nan=False)

    return results


__all__ = ["run_verification_suite"]
