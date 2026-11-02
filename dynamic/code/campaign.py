"""Campaign execution for support inversion comparison."""

from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

from beam_inverse import HermiteBeamFE, SupportBasis, fit_support
from skfem_reference import skfem_forward, mesh_with_load_edges
from shape_factory import get_shape_function


@dataclass
class PhysicalModel:
    """Physical parameters from campaign plan."""
    EI_N_m2: float = 162321230.0
    k0_N_m2: float = 18000000.0
    radius_m: float = 0.3
    nominal_force_N: float = 750000.0
    footprint_m: float = 4.3
    domain_m: tuple = (-50.0, 50.0)
    gauge_m: float = 0.2
    inverse_element_m: float = 0.25
    truth_element_m: float = 0.125
    spline_count: int = 31


@dataclass
class DataConfig:
    """Data generation configuration."""
    train_load_centers_m: list = None
    unused_load_center_m: float = 6.0
    unused_load_multiplier: float = 1.0
    sensors_per_load: int = 20
    sensor_interval_m: tuple = (-12.0, 12.0)
    sigma_strain: float = 5e-6

    def __post_init__(self):
        if self.train_load_centers_m is None:
            self.train_load_centers_m = [-4.0, 0.0, 4.0]


def generate_dataset(
    shape_name: str,
    regime: str,
    seed: int,
    physical: PhysicalModel,
    data_config: DataConfig,
    output_dir: Path,
) -> dict:
    """Generate one dataset with truth, observations, and metadata.

    Uses support_callable to generate truth without B-spline projection.
    Evaluates strain at exact sensor locations using scikit-fem basis functions.

    Returns metadata dict with paths to saved NPZ files.
    """
    rng = np.random.default_rng(seed)

    # Get shape function - this is the true support field
    shape_func = get_shape_function(shape_name)

    # Determine regime parameters
    if regime == "calibrated":
        multipliers = [1.0, 1.0, 1.0]
        sigma = data_config.sigma_strain
        train_centers_actual = np.array(data_config.train_load_centers_m)
        train_centers_nominal = train_centers_actual.copy()
        shear = 0.0
    elif regime == "amplitude_bias":
        multipliers = [1.12, 0.90, 1.08]
        sigma = data_config.sigma_strain
        train_centers_actual = np.array(data_config.train_load_centers_m)
        train_centers_nominal = train_centers_actual.copy()
        shear = 0.0
    elif regime == "pilot":
        multipliers = [1.1, 0.92, 1.06]
        sigma = 5e-6
        train_centers_actual = np.array(data_config.train_load_centers_m)
        train_centers_nominal = train_centers_actual.copy()
        shear = 0.0
        if shape_name != "pilot_gaussian":
            raise ValueError("pilot regime requires pilot_gaussian shape")
    elif regime == "amplitude_bias_sigma20microstrain":
        multipliers = [1.12, 0.90, 1.08]
        sigma = 20e-6
        train_centers_actual = np.array(data_config.train_load_centers_m)
        train_centers_nominal = train_centers_actual.copy()
        shear = 0.0
    elif regime == "amplitude_bias_pasternak_kG0.2k0ell2":
        multipliers = [1.12, 0.90, 1.08]
        sigma = data_config.sigma_strain
        train_centers_actual = np.array(data_config.train_load_centers_m)
        train_centers_nominal = train_centers_actual.copy()
        # kG = 0.2 * k0 * ell_squared, where ell_squared = sqrt(EI/k0)
        ell_squared = np.sqrt(physical.EI_N_m2 / physical.k0_N_m2)
        shear = 0.2 * physical.k0_N_m2 * ell_squared
    elif regime == "amplitude_bias_load_position_plus0.25m":
        multipliers = [1.12, 0.90, 1.08]
        sigma = data_config.sigma_strain
        train_centers_nominal = np.array(data_config.train_load_centers_m)
        train_centers_actual = train_centers_nominal + 0.25  # Perturb training positions
        shear = 0.0
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Validate that multiplier count matches train_load_centers_m count
    # Fixed multiplier regimes have hardcoded THREE multipliers
    if len(multipliers) != len(data_config.train_load_centers_m):
        raise ValueError(
            f"Regime {regime} uses {len(multipliers)} fixed multipliers, "
            f"but got {len(data_config.train_load_centers_m)} train_load_centers_m. "
            f"This regime requires exactly {len(multipliers)} training loads."
        )

    # Generate truth mesh with all load footprint edges (training + unused)
    all_centers = np.r_[train_centers_actual, data_config.unused_load_center_m]

    # Add rectangular boundaries if shape is rectangle
    extra_edges = None
    if shape_name == "rectangle":
        # Rectangle at x=0.8, width=1.6, so edges at 0.8-0.8=0.0 and 0.8+0.8=1.6
        extra_edges = np.array([0.0, 1.6])
    elif shape_name == "asymmetric":
        # Asymmetric has width change at x=1.2
        extra_edges = np.array([1.2])

    truth_points = mesh_with_load_edges(
        physical.domain_m[0],
        physical.domain_m[1],
        physical.truth_element_m,
        all_centers,
        load_width=physical.footprint_m,
        extra_edges=extra_edges,
    )

    # Generate sensor locations for each load
    sensor_locations = []
    for _ in train_centers_actual:
        sensors = np.linspace(
            data_config.sensor_interval_m[0],
            data_config.sensor_interval_m[1],
            data_config.sensors_per_load
        )
        sensor_locations.append(sensors)

    # Generate truth strain using scikit-fem with support_callable at exact sensor points
    truth_strain_all = []
    observations = []

    for idx, (center_actual, mult, sensors) in enumerate(zip(train_centers_actual, multipliers, sensor_locations)):
        _, strain = skfem_forward(
            truth_points,
            np.array([center_actual]),
            support_callable=shape_func,
            evaluation_points=sensors,
            gauge_length=physical.gauge_m,
            ei=physical.EI_N_m2,
            foundation=physical.k0_N_m2,
            shear=shear,
            force=physical.nominal_force_N * mult,
            load_width=physical.footprint_m,
            radius=physical.radius_m,
        )
        truth_strain = strain[:, 0]  # Extract single load column
        truth_strain_all.append(truth_strain)

        # Add noise
        noise = rng.normal(0, sigma, size=truth_strain.shape)
        observations.append(truth_strain + noise)

    # Generate unused load response at evaluation grid
    unused_eval_grid = np.linspace(-12, 12, 241)
    _, unused_strain = skfem_forward(
        truth_points,
        np.array([data_config.unused_load_center_m]),
        support_callable=shape_func,
        evaluation_points=unused_eval_grid,
        gauge_length=physical.gauge_m,
        ei=physical.EI_N_m2,
        foundation=physical.k0_N_m2,
        shear=shear,
        force=physical.nominal_force_N * data_config.unused_load_multiplier,
        load_width=physical.footprint_m,
        radius=physical.radius_m,
    )
    unused_truth_strain = unused_strain[:, 0]

    # Evaluate true support on a fine grid for scoring
    score_grid = np.linspace(-12, 12, 1201)
    true_d = shape_func(score_grid)

    # Save dataset with proper float arrays (no dtype=object)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = f"{shape_name}_{regime}_seed{seed}"
    npz_path = output_dir / f"{dataset_name}.npz"

    # Convert lists to 2D arrays (n_load, n_sensor)
    sensor_array = np.array([s for s in sensor_locations])
    obs_array = np.array([o for o in observations])
    truth_strain_array = np.array([t for t in truth_strain_all])

    np.savez(
        npz_path,
        # Observations and metadata for inversion
        observations=obs_array,
        sensor_locations=sensor_array,
        sigma=sigma,
        train_centers_nominal=train_centers_nominal,
        train_centers_actual=train_centers_actual,
        multipliers=np.array(multipliers),
        # Truth fields for scoring only
        truth_strain=truth_strain_array,
        score_grid=score_grid,
        true_d=true_d,
        # Unused load data
        unused_center=data_config.unused_load_center_m,
        unused_eval_grid=unused_eval_grid,
        unused_truth_strain=unused_truth_strain,
        # Physical and schema metadata
        shape_name=shape_name,
        regime=regime,
        seed=seed,
        shear=shear,
    )

    # Add JSON metadata file
    metadata = {
        "schema_version": "1.0",
        "npz_path": str(npz_path),
        "npz_file": npz_path.name,
        "shape": shape_name,
        "regime": regime,
        "seed": seed,
        "multipliers": [float(m) for m in multipliers],
        "n_sensors_per_load": [len(s) for s in sensor_locations],
        "sigma": float(sigma),
        "shear": float(shear),
        "train_centers_nominal": train_centers_nominal.tolist(),
        "train_centers_actual": train_centers_actual.tolist(),
        "physical": asdict(physical),
        "data_config": asdict(data_config),
    }

    json_path = output_dir / f"{dataset_name}.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def run_inversion(
    dataset_path: Path,
    method: str,
    penalty_lambda: float,
    physical: PhysicalModel,
    starts: tuple = (0.05, 0.30),
    max_nfev: int = 200,
    oracle_multipliers: list | None = None,
) -> dict:
    """Run one inversion method on a dataset.

    Args:
        dataset_path: Path to NPZ dataset
        method: One of H1_fixed, H1_profile, TV_fixed, TV_profile, H1_joint, H1_oracle
        penalty_lambda: Regularization parameter
        physical: Physical model parameters
        starts: Coefficient start values tuple
        max_nfev: Maximum function evaluations
        oracle_multipliers: For H1_oracle only, the true force multipliers

    Returns:
        Result dictionary with coefficients, gamma, timing, train_prediction, train_norm
    """
    data = np.load(dataset_path, allow_pickle=False)

    # Read nominal centers for inversion (inverse always uses nominal positions)
    train_centers = data["train_centers_nominal"]
    sensor_locations = data["sensor_locations"]
    observations = data["observations"]
    sigma = float(data["sigma"])

    # Parse method
    if method.startswith("H1_"):
        penalty_type = "h1"
        gamma_mode = method.split("_")[1]
    elif method.startswith("TV_"):
        penalty_type = "tv"
        gamma_mode = method.split("_")[1]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build forward model with assembly timing
    assembly_start = time.perf_counter()
    fe = HermiteBeamFE(
        train_centers,
        x_min=physical.domain_m[0],
        x_max=physical.domain_m[1],
        n_elements=int((physical.domain_m[1] - physical.domain_m[0]) / physical.inverse_element_m),
        n_coefficients=physical.spline_count,
        gauge_length=physical.gauge_m,
        foundation=physical.k0_N_m2,
        force=physical.nominal_force_N,
        load_width=physical.footprint_m,
        ei=physical.EI_N_m2,
        radius=physical.radius_m,
    )
    assembly_time = time.perf_counter() - assembly_start

    # Prepare observations for fit_support
    obs_list = [observations[i] for i in range(len(train_centers))]
    sigma_list = [np.full_like(obs_list[i], sigma) for i in range(len(train_centers))]
    sensor_list = [sensor_locations[i] for i in range(len(train_centers))]

    # For oracle mode, multiply RHS by known multipliers instead of dividing y
    if gamma_mode == "oracle":
        if oracle_multipliers is None:
            raise ValueError("H1_oracle requires oracle_multipliers")

        # Oracle: scale RHS columns by actual multipliers, fit with gamma_mode=fixed
        # This keeps y and sigma unchanged (fair weighted objective)
        oracle_gammas = np.array(oracle_multipliers)

        # Scale RHS columns in place for oracle assembly
        fe_oracle_rhs = fe.rhs.copy()
        for load_idx in range(len(train_centers)):
            fe_oracle_rhs[:, load_idx] *= oracle_gammas[load_idx]

        # Temporarily replace FE rhs for fitting
        original_rhs = fe.rhs
        fe.rhs = fe_oracle_rhs

        # Oracle reuses existing FE assembly, so fit time is separate
        started = time.perf_counter()
        result = fit_support(
            fe,
            sensor_list,
            obs_list,
            sigma_list,
            gamma_mode="fixed",  # Fit with gamma=1 on scaled RHS
            penalty=penalty_type,
            penalty_lambda=penalty_lambda,
            starts=starts,
            max_nfev=max_nfev,
        )
        fit_time = time.perf_counter() - started

        # Restore original RHS
        fe.rhs = original_rhs

        # Compute physical predictions: use original (unscaled) FE with oracle gamma scaling
        all_predictions = fe.strain_and_jacobian(result["coefficients"], sensor_list)[0]
        n_sensors_per_load = [s.size for s in sensor_list]
        train_prediction_per_load = []
        offset = 0
        for load_idx, count in enumerate(n_sensors_per_load):
            pred_unit = all_predictions[offset:offset + count]
            train_prediction_per_load.append(oracle_gammas[load_idx] * pred_unit)
            offset += count

        # Override reported gamma with actual oracle multipliers
        result["gamma"] = oracle_gammas
        result["assembly_time_s"] = assembly_time

    else:
        # Normal modes: fixed, profile, or joint
        started = time.perf_counter()
        result = fit_support(
            fe,
            sensor_list,
            obs_list,
            sigma_list,
            gamma_mode=gamma_mode,
            penalty=penalty_type,
            penalty_lambda=penalty_lambda,
            starts=starts,
            max_nfev=max_nfev,
        )
        fit_time = time.perf_counter() - started
        result["assembly_time_s"] = assembly_time

        # Compute physical predictions: fitted_gamma * nominal_response
        # Call strain_and_jacobian once with all sensor arrays, then split by load
        all_predictions = fe.strain_and_jacobian(result["coefficients"], sensor_list)[0]
        # Split predictions in load-major order: reshape and scale each row by gamma
        n_sensors_per_load = [s.size for s in sensor_list]
        train_prediction_per_load = []
        offset = 0
        for load_idx, count in enumerate(n_sensors_per_load):
            pred_unit = all_predictions[offset:offset + count]
            train_prediction_per_load.append(result["gamma"][load_idx] * pred_unit)
            offset += count

    # Compute train_norm: ||gamma * prediction - observations|| / sigma
    train_residuals = []
    for load_idx in range(len(train_centers)):
        residual = (train_prediction_per_load[load_idx] - obs_list[load_idx]) / sigma
        train_residuals.append(residual)

    train_norm = float(np.linalg.norm(np.concatenate(train_residuals)))

    # Store results
    result["train_prediction"] = np.array(train_prediction_per_load)
    result["train_norm"] = train_norm
    result["fit_time_s"] = fit_time
    result["method"] = method
    result["penalty_lambda"] = penalty_lambda
    result["starts_used"] = starts

    return result


__all__ = ["PhysicalModel", "DataConfig", "generate_dataset", "run_inversion"]
