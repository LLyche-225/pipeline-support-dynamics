
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import minimize_scalar


CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from beam_inverse import HermiteBeamFE, SupportBasis, fit_support, profile_load_amplitude  # noqa: E402


F = np.array([2.0, -1.0, 0.5])
DF = np.array([[1.0, 0.2], [-0.3, 0.1], [0.4, -0.6]])
TARGET = np.array([2.1, -1.2, 0.7])
MEAN = 1.05


def test_profile_scalar_matches_independent_bounded_minimization():
    tau = 0.23
    gamma, _, prior_residual, _ = profile_load_amplitude(F, DF, TARGET, MEAN, tau)
    objective = lambda value: np.sum((value * F - TARGET) ** 2) + ((value - MEAN) / tau) ** 2
    independent = minimize_scalar(
        objective, bounds=(0.7, 1.3), method="bounded", options={"xatol": 1e-12}
    )
    assert np.isclose(gamma, independent.x, rtol=0.0, atol=2e-9)
    assert np.isclose(prior_residual, (gamma - MEAN) / tau, rtol=0.0, atol=1e-14)


def test_profile_directional_derivative_and_augmented_jacobian():
    tau = 0.23
    direction = np.array([0.13, -0.27])
    coefficients = np.array([0.11, -0.07])

    def evaluate(value):
        response = F + DF @ (coefficients + value * direction)
        gamma, dgamma, prior_residual, prior_jacobian = profile_load_amplitude(
            response, DF, TARGET, MEAN, tau
        )
        residual = np.r_[gamma * response - TARGET, prior_residual]
        jacobian = np.vstack([
            gamma * DF + np.outer(response, dgamma),
            prior_jacobian,
        ])
        return gamma, dgamma, residual, jacobian

    gamma, dgamma, residual, jacobian = evaluate(0.0)
    step = 1e-5
    plus = evaluate(step)
    minus = evaluate(-step)
    finite_gamma = (plus[0] - minus[0]) / (2.0 * step)
    finite_residual = (plus[2] - minus[2]) / (2.0 * step)
    assert np.isclose(dgamma @ direction, finite_gamma, rtol=2e-7, atol=2e-9)
    assert np.linalg.norm(jacobian @ direction - finite_residual) < 2e-6


@pytest.mark.parametrize("target_gamma,expected_gamma", [(0.2, 0.7), (2.0, 1.3)])
def test_profile_active_bounds(target_gamma, expected_gamma):
    target = target_gamma * F
    gamma, dgamma, prior_residual, prior_jacobian = profile_load_amplitude(
        F, DF, target, 1.0, 0.4
    )
    assert gamma == expected_gamma
    assert np.all(dgamma == 0.0)
    assert np.all(prior_jacobian == 0.0)
    assert np.isfinite(prior_residual)


def test_tau_zero_tiny_and_infinite_limits():
    exact_gamma, dgamma, prior_residual, prior_jacobian = profile_load_amplitude(
        F, DF, TARGET, 1.08, 0.0
    )
    assert exact_gamma == 1.08
    assert np.all(dgamma == 0.0) and prior_residual == 0.0
    assert np.all(prior_jacobian == 0.0)

    tau = 1e-8
    gamma, dgamma, prior_residual, prior_jacobian = profile_load_amplitude(
        F, DF, TARGET, MEAN, tau
    )
    norm = float(F @ F)
    cross = float(F @ TARGET)
    closed_form = (MEAN + tau**2 * cross) / (1.0 + tau**2 * norm)
    closed_residual = tau * (cross - MEAN * norm) / (1.0 + tau**2 * norm)
    assert np.isclose(gamma, closed_form, rtol=2e-15, atol=0.0)
    assert np.isclose(prior_residual, closed_residual, rtol=2e-13, atol=0.0)
    assert np.all(np.isfinite(dgamma)) and np.all(np.isfinite(prior_jacobian))

    free_gamma, free_dgamma, free_residual, free_jacobian = profile_load_amplitude(
        F, DF, TARGET, MEAN, np.inf
    )
    assert np.isclose(free_gamma, float(F @ TARGET) / float(F @ F), rtol=0.0, atol=0.0)
    assert free_residual == 0.0
    assert np.all(free_jacobian == 0.0)
    assert np.all(np.isfinite(free_dgamma))


def test_informative_prior_identifies_zero_response():
    zero = np.zeros_like(F)
    gamma, dgamma, prior_residual, prior_jacobian = profile_load_amplitude(
        zero, np.zeros_like(DF), TARGET, 1.12, 0.2
    )
    assert gamma == 1.12
    assert np.all(dgamma == 0.0)
    assert prior_residual == 0.0
    assert np.all(prior_jacobian == 0.0)


def test_input_rejection_for_lengths_means_and_standard_deviations():
    fe = HermiteBeamFE([0.0], n_elements=20, n_coefficients=5)
    sensors = [np.linspace(-10.0, 10.0, 5)]
    clean, _ = fe.strain_and_jacobian(np.zeros(fe.basis.n_variables), sensors)
    sigma = [np.full_like(clean, 2e-5)]

    invalid_calls = [
        {"load_mean": [1.0, 1.0]},
        {"load_mean": np.full((1, 1), 1.0)},
        {"load_mean": [np.nan]},
        {"load_mean": [0.0]},
        {"load_std": [np.inf, np.inf]},
        {"load_std": [-1.0]},
        {"load_std": [np.nan]},
        {"load_mean": [1.5], "gamma_mode": "fixed"},
        {"load_mean": [1.5], "load_std": [0.0]},
    ]
    for keywords in invalid_calls:
        with pytest.raises(ValueError):
            fit_support(fe, sensors, [clean], sigma, **keywords)

    with pytest.raises(ValueError, match="positive finite whitened response"):
        profile_load_amplitude(np.zeros(3), np.zeros((3, 2)), TARGET, 1.0, np.inf)
    with pytest.raises(ValueError):
        profile_load_amplitude(F, np.zeros((2, 2)), TARGET, 1.0, np.inf)

    # A finite noisy mean is admissible even when it lies outside gamma bounds.
    noisy, noisy_dgamma, noisy_residual, noisy_jacobian = profile_load_amplitude(
        F, DF, TARGET, 1.5, 0.2
    )
    assert 0.7 <= noisy <= 1.3
    assert np.all(np.isfinite(noisy_dgamma))
    assert np.isfinite(noisy_residual)
    assert np.all(np.isfinite(noisy_jacobian))


def test_noisy_out_of_bounds_mean_is_permitted_in_mixed_load_information():
    fe = HermiteBeamFE([-5.0, 5.0], n_elements=40, n_coefficients=5)
    sensors = [np.linspace(-12.0, 12.0, 5), np.linspace(-12.0, 12.0, 5)]
    clean, _ = fe.strain_and_jacobian(np.zeros(fe.basis.n_variables), sensors)
    result = fit_support(
        fe,
        sensors,
        [clean[: sensors[0].size], clean[sensors[0].size :]],
        [np.full(points.size, 2e-5) for points in sensors],
        gamma_mode="profile",
        max_nfev=50,
        load_mean=[1.5, 0.9],
        load_std=[0.2, 0.0],
    )
    assert 0.7 <= result["gamma"][0] <= 1.3
    assert result["gamma"][1] == 0.9
    assert np.array_equal(result["load_std"], [0.2, 0.0])


def test_three_load_profile_joint_mixed_priors_and_heteroscedastic_noise():
    centers = np.array([-8.0, 0.0, 8.0])
    basis = SupportBasis(n_coefficients=9)
    control_x = np.linspace(*basis.support, basis.n_coefficients)[1:-1]
    truth = 0.38 * np.exp(-((control_x + 2.0) ** 2) / 20.0)
    fe = HermiteBeamFE(centers, n_elements=100, n_coefficients=9)
    sensors = [
        np.linspace(-20.0, 20.0, 11),
        np.linspace(-16.0, 16.0, 9),
        np.linspace(-22.0, 22.0, 11),
    ]
    clean, _ = fe.strain_and_jacobian(truth, sensors)
    gamma_true = np.array([1.0, 1.08, 0.85])
    observations = []
    offset = 0
    for points, gamma in zip(sensors, gamma_true):
        count = points.size
        observations.append(gamma * clean[offset:offset + count])
        offset += count
    sigma = [np.linspace(3e-6, 7e-6, points.size) for points in sensors]
    means = gamma_true.copy()
    std = np.array([np.inf, 0.02, 0.0])

    common = {
        "penalty": "h1",
        "penalty_lambda": 1e-7,
        "max_nfev": 1000,
        "load_mean": means,
        "load_std": std,
    }
    profile = fit_support(fe, sensors, observations, sigma, gamma_mode="profile", **common)
    joint = fit_support(fe, sensors, observations, sigma, gamma_mode="joint", **common)

    profile_record = profile["starts"][profile["chosen_start"]]
    joint_record = joint["starts"][joint["chosen_start"]]
    assert profile_record["success"] and joint_record["success"]
    assert abs(profile_record["cost"] - joint_record["cost"]) < 1e-12
    assert np.linalg.norm(profile["coefficients"] - joint["coefficients"]) < 1e-8
    profile_response = fe.strain_and_jacobian(profile["coefficients"], sensors)[0]
    joint_response = fe.strain_and_jacobian(joint["coefficients"], sensors)[0]
    response_scale = np.linalg.norm(profile_response)
    assert np.linalg.norm(profile_response - joint_response) / response_scale < 1e-8
    assert np.allclose(profile["gamma"], gamma_true, rtol=0.0, atol=1e-7)
    assert np.allclose(joint["gamma"], gamma_true, rtol=0.0, atol=1e-7)

    assert len(profile["starts"]) == len(joint["starts"]) == 2
    assert joint_record["x"].size == fe.basis.n_variables + 2
    assert all(np.isfinite(record["projected_first_order"]) for record in profile["starts"])
    assert all(record["wall_time_s"] >= 0.0 for record in joint["starts"])
    assert all(np.isfinite(record["calibration_residual_norm"]) for record in profile["starts"])
    assert all(np.isfinite(record["strain_residual_norm"]) for record in joint["starts"])
    assert np.array_equal(profile["load_mean"], means)
    assert np.array_equal(profile["load_std"], std)
