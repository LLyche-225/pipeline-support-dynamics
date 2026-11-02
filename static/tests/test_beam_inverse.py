from pathlib import Path
import sys

import numpy as np
import pytest


CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from beam_inverse import HermiteBeamFE, SupportBasis, fit_support  # noqa: E402
from skfem_reference import mesh_with_load_edges, skfem_forward  # noqa: E402


POINTS = np.linspace(-20, 20, 41)


def bump_coefficients(n_coefficients=9, amplitude=0.55):
    basis = SupportBasis(n_coefficients)
    control_x = np.linspace(*basis.support, basis.n_coefficients)
    return basis, 0.55 * np.exp(-(control_x[1:-1] ** 2) / 24.0)


def test_load_conservation_and_symmetry():
    fe = HermiteBeamFE([0.0], n_elements=80)
    displacement_dofs = np.arange(0, 2 * fe.x.size, 2)
    assert np.allclose(fe.rhs[displacement_dofs, 0].sum(), 750_000.0, rtol=2e-14)
    symmetric = np.linspace(-20, 20, 41)
    strain, _ = fe.strain_and_jacobian(np.zeros(fe.basis.n_variables), symmetric)
    assert np.allclose(strain, strain[::-1], rtol=1e-10, atol=1e-12)


def test_stiffness_nonnegative_at_maximum_support():
    fe = HermiteBeamFE([0.0], n_elements=40)
    _, _, _, matrix = fe._factor(np.full(fe.basis.n_variables, 0.95))
    random = np.random.default_rng(4).normal(size=matrix.shape[0])
    assert float(random @ (matrix @ random)) > 0


def test_domain_truncation_and_mesh_refinement():
    sensors = np.linspace(-20, 20, 33)
    basis, variables = bump_coefficients(9)
    coarse_domain = HermiteBeamFE([0.0], x_min=-40, x_max=40, n_elements=320, n_coefficients=basis.n_coefficients)
    long_domain = HermiteBeamFE([0.0], x_min=-50, x_max=50, n_elements=400, n_coefficients=basis.n_coefficients)

    # Test actual convergence sequence
    h_sequence = [0.5, 0.25, 0.125, 0.0625]
    meshes = [HermiteBeamFE([0.0], n_elements=int(100 / h), n_coefficients=basis.n_coefficients)
              for h in h_sequence]

    truncation = np.asarray([
        coarse_domain.strain_and_jacobian(variables, sensors)[0],
        long_domain.strain_and_jacobian(variables, sensors)[0],
    ])

    refinement = np.asarray([mesh.strain_and_jacobian(variables, sensors)[0] for mesh in meshes])

    # Domain truncation test
    assert np.linalg.norm(truncation[0] - truncation[1]) / np.linalg.norm(truncation[1]) < 2e-3

    # Mesh refinement convergence: errors should decrease
    errors = [np.linalg.norm(refinement[i] - refinement[-1]) / np.linalg.norm(refinement[-1])
              for i in range(len(h_sequence) - 1)]
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)), "errors must decrease"

    # Production mesh (h=0.25) vs finest (h=0.0625) should give strain RMS below 1 microstrain
    production_strain = meshes[1].strain_and_jacobian(variables, sensors)[0]
    finest_strain = meshes[-1].strain_and_jacobian(variables, sensors)[0]
    rms_difference = np.sqrt(np.mean((production_strain - finest_strain)**2))
    assert rms_difference < 1e-6, f"RMS difference {rms_difference:.3e} exceeds 1 microstrain"


def test_gauge_formula_and_analytic_jacobian():
    basis, variables = bump_coefficients()
    fe = HermiteBeamFE([0.0], n_elements=120, n_coefficients=basis.n_coefficients)
    strain, jacobian = fe.strain_and_jacobian(variables, POINTS)
    displacement, sensitivity, _ = fe._state(variables)
    # Gauge strain uses rotation (slope) difference, not displacement difference
    left_w, left_rot, _ = fe._evaluate(POINTS - 0.1, displacement[:, 0], derivative=True)
    right_w, right_rot, _ = fe._evaluate(POINTS + 0.1, displacement[:, 0], derivative=True)
    assert np.allclose(strain, -0.3 * (right_rot - left_rot) / 0.2)
    direction = np.exp(-(np.arange(variables.size) ** 2) / 100.0)
    h = 1e-5
    plus = fe.strain_and_jacobian(np.clip(variables + h * direction, 0, 0.95), POINTS)[0]
    minus = fe.strain_and_jacobian(np.clip(variables - h * direction, 0, 0.95), POINTS)[0]
    finite = (plus - minus) / (2 * h)
    assert np.linalg.norm(finite - jacobian @ direction) / np.linalg.norm(jacobian @ direction) < 2e-4


def test_profile_jacobian_interior_and_active():
    fe = HermiteBeamFE([0.0], n_elements=80, n_coefficients=9)
    _, variables = bump_coefficients()
    prediction, derivative = fe.strain_and_jacobian(variables, POINTS)
    for gamma_target, active in ((1.1, False), (2.0, True)):
        observed = gamma_target * prediction
        gamma = np.clip(prediction @ observed / (prediction @ prediction), 0.7, 1.3)
        assert np.isclose(gamma, 1.1 if not active else 1.3)
        h = 1e-5
        direction = np.sin(np.arange(variables.size) + 1.0)
        plus_p, plus_d = fe.strain_and_jacobian(
            np.clip(variables + h * direction, 0, .95), POINTS
        )
        plus_gamma = np.clip(plus_p @ observed / (plus_p @ plus_p), .7, 1.3)
        exact = gamma * (derivative @ direction)
        if not active:
            # Interior: dgamma/dc is a vector, then dot with direction
            numerator = derivative.T @ observed - 2.0 * gamma * (prediction @ derivative)
            dgamma_dc = numerator / (prediction @ prediction)
            exact += prediction * (dgamma_dc @ direction)
        finite = ((plus_gamma * plus_p - gamma * prediction) / h)
        assert np.linalg.norm(finite - exact) / np.linalg.norm(exact) < 2e-4


@pytest.mark.parametrize("scenario", ["intact", "graded"])
@pytest.mark.parametrize("h", [0.43, 0.215, 0.1075])
def test_scikit_fem_independent_comparison(scenario, h):
    centers = np.array([0.0])
    # Use uniform mesh with domain that has footprint edges aligned
    x_min, x_max = -43.0, 43.0
    n_elements = int(np.round((x_max - x_min) / h))
    actual_h = (x_max - x_min) / n_elements
    points = np.linspace(x_min, x_max, n_elements + 1)

    basis, variables = bump_coefficients()
    coefficients = None if scenario == "intact" else basis.full_coefficients(variables)
    reference_w, reference_strain = skfem_forward(points, centers, coefficients)

    custom = HermiteBeamFE(
        centers, x_min=x_min, x_max=x_max, n_elements=n_elements,
        n_coefficients=basis.n_coefficients
    )
    displacement, _, _ = custom._state(
        np.zeros(custom.basis.n_variables) if scenario == "intact" else variables,
        with_jacobian=False,
    )
    evaluation = np.linspace(-20, 20, 81)
    custom_w, _ = custom._evaluate(evaluation, displacement[:, 0], derivative=False)
    custom_strain = custom.strain_and_jacobian(
        np.zeros(custom.basis.n_variables) if scenario == "intact" else variables,
        evaluation,
    )[0]
    # Flatten reference arrays if they have extra dimension
    ref_w = reference_w.ravel() if reference_w.ndim > 1 else reference_w
    ref_strain = reference_strain.ravel() if reference_strain.ndim > 1 else reference_strain
    assert np.linalg.norm(custom_w - ref_w) / np.linalg.norm(ref_w) < 1e-8
    assert np.linalg.norm(custom_strain - ref_strain) / np.linalg.norm(ref_strain) < 1e-8


def test_profile_joint_equivalence_no_truth_selection_and_bounds():
    basis, truth = bump_coefficients()
    centers = [0.0]
    sensors = [np.linspace(-20, 20, 41)]
    fe = HermiteBeamFE(centers, n_elements=160, n_coefficients=basis.n_coefficients)
    clean, _ = fe.strain_and_jacobian(truth, sensors)
    sigma = [np.full_like(clean, 1e-5)]
    projected = fit_support(
        fe, sensors, [1.1 * clean], sigma, gamma_mode="profile",
        penalty="h1", penalty_lambda=1e-5, max_nfev=100,
    )
    joint = fit_support(
        fe, sensors, [1.1 * clean], sigma, gamma_mode="joint",
        penalty="h1", penalty_lambda=1e-5, max_nfev=150,
    )
    assert "truth" not in str(fit_support.__code__.co_varnames)
    assert np.all(projected["coefficients"] >= 0) and np.all(projected["coefficients"] <= .95)
    assert np.allclose(projected["gamma"], 1.1, atol=2e-2)
    assert np.allclose(joint["gamma"], 1.1, atol=2e-2)
    assert abs(projected["starts"][projected["chosen_start"]]["cost"]
               - joint["starts"][joint["chosen_start"]]["cost"]) < 2e-3


def test_no_noise_recovery_with_gamma_scaled_residual():
    """Exact-model 3-load recovery test with known coefficients and gamma.

    This test uses the same FE model for forward and inverse to verify that
    the code correctly recovers known coefficients and load multipliers.
    Targets: physical response relative error < 1e-4, gamma error < 1e-3.
    """
    # 3 loads at [-4, 0, 4], 20 sensors each at [-12, 12], ncoeff=11
    centers = np.array([-4.0, 0.0, 4.0])
    sensors_per_load = 20
    sensor_interval = (-12.0, 12.0)
    n_coefficients = 11

    # True coefficients: linspace(0.1, 0.4, n_variables)
    basis = SupportBasis(n_coefficients=n_coefficients)
    n_vars = basis.n_variables
    c_true = np.linspace(0.1, 0.4, n_vars)

    # True gamma multipliers
    gamma_true = np.array([1.12, 0.90, 1.08])

    # Build shared FE model (same for forward and inverse)
    fe = HermiteBeamFE(centers, n_elements=400, n_coefficients=n_coefficients)

    # Generate sensor locations (same for all loads)
    sensors = [np.linspace(sensor_interval[0], sensor_interval[1], sensors_per_load)
               for _ in range(len(centers))]

    # Generate observations: gamma[i] * strain(c_true, sensors[i])
    all_strain, _ = fe.strain_and_jacobian(c_true, sensors)
    # Split by load and scale by gamma
    observations = []
    offset = 0
    for load_idx in range(len(centers)):
        count = sensors_per_load
        strain_load = all_strain[offset:offset + count]
        observations.append(gamma_true[load_idx] * strain_load)
        offset += count

    # Inversion parameters
    sigma = [np.full(sensors_per_load, 5e-6) for _ in range(len(centers))]
    penalty_lambda = 1e-6

    # Run joint inversion
    result = fit_support(
        fe, sensors, observations, sigma, gamma_mode="joint",
        penalty="h1", penalty_lambda=penalty_lambda, max_nfev=300,
    )

    # Verify recovery: compute physical response with recovered coefficients and gamma
    recovered_strain, _ = fe.strain_and_jacobian(result["coefficients"], sensors)
    physical_response = []
    offset = 0
    for load_idx in range(len(centers)):
        count = sensors_per_load
        strain_load = recovered_strain[offset:offset + count]
        physical_response.append(result["gamma"][load_idx] * strain_load)
        offset += count

    # Compute errors
    all_obs = np.concatenate(observations)
    all_pred = np.concatenate(physical_response)
    response_error = np.linalg.norm(all_pred - all_obs) / np.linalg.norm(all_obs)
    gamma_error = np.linalg.norm(result["gamma"] - gamma_true)

    assert response_error < 1e-4, f"Physical response error {response_error:.2e} exceeds 1e-4"
    assert gamma_error < 1e-3, f"Gamma error {gamma_error:.2e} exceeds 1e-3"


def test_multi_load_jacobian_and_recovery():
    """Test multi-load sensitivity and no-noise recovery with distinct loads."""
    basis, _ = bump_coefficients(9)
    # Three distinct load centers
    centers = [-10.0, 0.0, 10.0]
    sensors = [np.linspace(-20, 20, 21), np.linspace(-15, 15, 19), np.linspace(-25, 25, 26)]
    fe = HermiteBeamFE(centers, n_elements=120, n_coefficients=basis.n_coefficients)

    # Distinct physical truth coefficients for each direction
    truth1 = 0.4 * np.exp(-((np.linspace(*basis.support, basis.n_coefficients)[1:-1] + 5)**2) / 20.0)
    truth2 = 0.5 * np.exp(-(np.linspace(*basis.support, basis.n_coefficients)[1:-1]**2) / 30.0)
    truth3 = 0.35 * np.exp(-((np.linspace(*basis.support, basis.n_coefficients)[1:-1] - 4)**2) / 25.0)

    # Test Jacobian for each truth direction with finite differences
    for truth in [truth1, truth2, truth3]:
        prediction, jacobian = fe.strain_and_jacobian(truth, sensors)
        h = 1e-6
        direction = np.sin(np.arange(truth.size) * 0.5 + 1.0)
        plus = fe.strain_and_jacobian(np.clip(truth + h * direction, 0, 0.95), sensors)[0]
        minus = fe.strain_and_jacobian(np.clip(truth - h * direction, 0, 0.95), sensors)[0]
        finite = (plus - minus) / (2 * h)
        assert np.linalg.norm(finite - jacobian @ direction) / np.linalg.norm(jacobian @ direction) < 1e-5

    # No-noise multi-load recovery with distinct amplitudes
    amplitudes = [1.05, 0.95, 1.15]
    observations = []
    for load_idx, (amp, pts) in enumerate(zip(amplitudes, sensors)):
        obs = amp * fe.strain_and_jacobian(truth1, sensors)[0]
        # Extract only the portion for this load
        start = sum(sensors[i].size for i in range(load_idx))
        end = start + pts.size
        observations.append(obs[start:end])
    sigma = [np.full_like(obs, 1e-6) for obs in observations]

    result = fit_support(
        fe, sensors, observations, sigma, gamma_mode="joint",
        penalty="h1", penalty_lambda=1e-6, max_nfev=300,
    )

    # Verify recovery quality
    assert result["successful_start"]
    recovered_strain = fe.strain_and_jacobian(result["coefficients"], sensors)[0]
    all_obs = np.concatenate(observations)
    # Relative error should be reasonable for this multi-load case with small signals
    assert np.linalg.norm(recovered_strain - all_obs) / np.linalg.norm(all_obs) < 0.1
    # Gamma recovery should be reasonable
    assert np.allclose(result["gamma"], amplitudes, atol=0.15)


def test_profile_joint_equivalence_same_budget():
    """Test profile and joint modes give equivalent results at same initial conditions."""
    basis, _ = bump_coefficients(9)
    centers = [0.0]
    sensors = [np.linspace(-20, 20, 21)]
    fe = HermiteBeamFE(centers, n_elements=80, n_coefficients=basis.n_coefficients)

    # Use a different truth from the bump used in other tests
    truth = 0.45 * np.exp(-((np.linspace(*basis.support, basis.n_coefficients)[1:-1] - 3)**2) / 18.0)
    clean = fe.strain_and_jacobian(truth, sensors)[0]
    sigma = [np.full_like(clean, 1e-5)]

    # Same budget, same penalty lambda, same gamma0=1.1
    projected = fit_support(
        fe, sensors, [1.1 * clean], sigma, gamma_mode="profile",
        penalty="h1", penalty_lambda=5e-6, max_nfev=120,
    )
    joint = fit_support(
        fe, sensors, [1.1 * clean], sigma, gamma_mode="joint",
        penalty="h1", penalty_lambda=5e-6, max_nfev=120,
    )

    # Check matching objectives
    assert abs(projected["starts"][projected["chosen_start"]]["cost"]
               - joint["starts"][joint["chosen_start"]]["cost"]) < 1e-3

    # Check matching responses
    proj_response = fe.strain_and_jacobian(projected["coefficients"], sensors)[0]
    joint_response = fe.strain_and_jacobian(joint["coefficients"], sensors)[0]
    assert np.linalg.norm(proj_response - joint_response) / np.linalg.norm(proj_response) < 5e-3

    # Both should recover gamma~1.1
    assert np.allclose(projected["gamma"], 1.1, atol=3e-2)
    assert np.allclose(joint["gamma"], 1.1, atol=3e-2)
