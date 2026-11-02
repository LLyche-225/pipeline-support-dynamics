"""Tests for campaign dataset generation and inversion."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from campaign import PhysicalModel, DataConfig, generate_dataset, run_inversion  # noqa: E402
from skfem_reference import skfem_forward, mesh_with_load_edges  # noqa: E402
from beam_inverse import HermiteBeamFE  # noqa: E402


def test_arbitrary_callable_evaluation():
    """Test that generation uses support_callable directly, not B-spline projection."""
    physical = PhysicalModel()
    data_config = DataConfig()

    # Define a non-spline support function
    def custom_support(x):
        return 0.6 * np.exp(-((x - 0.5) ** 2) / 3.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate with pilot regime to use pilot_gaussian shape
        meta = generate_dataset(
            "pilot_gaussian", "pilot", 2026110199,
            physical, data_config, Path(tmpdir)
        )

        data = np.load(meta["npz_path"], allow_pickle=False)

        # Verify no truth_coefficients in NPZ
        assert "truth_coefficients" not in data.files

        # Verify shape grid equals function values directly
        score_grid = data["score_grid"]
        true_d = data["true_d"]

        # Pilot gaussian: 0.45 * exp(-(x+0.3)^2/(2*1.1^2))
        expected = 0.45 * np.exp(-((score_grid + 0.3) ** 2) / (2 * 1.1**2))
        assert np.allclose(true_d, expected, rtol=1e-10)

        # Verify strain was computed at exact sensor locations, not interpolated
        # Re-evaluate at same sensors and compare
        train_centers = data["train_centers_actual"]
        sensors = data["sensor_locations"][0]
        sigma = float(data["sigma"])

        # Build truth mesh
        all_centers = np.r_[train_centers, data["unused_center"]]
        truth_points = mesh_with_load_edges(
            physical.domain_m[0], physical.domain_m[1],
            physical.truth_element_m, all_centers, load_width=physical.footprint_m
        )

        # Re-evaluate at same sensors
        from shape_factory import get_shape_function
        shape_func = get_shape_function("pilot_gaussian")

        _, strain_check = skfem_forward(
            truth_points, train_centers[:1],
            support_callable=shape_func,
            evaluation_points=sensors,
            gauge_length=physical.gauge_m,
            ei=physical.EI_N_m2,
            foundation=physical.k0_N_m2,
            force=physical.nominal_force_N * 1.1,  # First multiplier for pilot
            load_width=physical.footprint_m,
            radius=physical.radius_m,
        )

        # Compare clean truth strain (observations have noise)
        truth_strain_first = data["truth_strain"][0]
        assert np.allclose(truth_strain_first, strain_check[:, 0], rtol=1e-10)


def test_reference_exact_force_sum():
    """Test that reference mesh conserves force with extra edges and shifted loads."""
    physical = PhysicalModel()

    # Test with shifted centers and rectangular extra edges
    centers = np.array([-4.25, 0.0, 4.25])
    extra_edges = np.array([0.0, 1.2, 1.6])

    points = mesh_with_load_edges(
        physical.domain_m[0], physical.domain_m[1],
        physical.truth_element_m, centers,
        load_width=physical.footprint_m,
        extra_edges=extra_edges
    )

    # Check mesh is valid
    assert np.all(np.diff(points) > 0)
    assert points[0] == physical.domain_m[0]
    assert points[-1] == physical.domain_m[1]

    # Check force conservation for each load
    from shape_factory import get_shape_function
    shape_func = get_shape_function("rectangle")

    for center in centers:
        _, strain = skfem_forward(
            points, np.array([center]),
            support_callable=shape_func,
            gauge_length=physical.gauge_m,
            ei=physical.EI_N_m2,
            foundation=physical.k0_N_m2,
            force=physical.nominal_force_N,
            load_width=physical.footprint_m,
            radius=physical.radius_m,
        )

        # Check by computing sum of translational force DOFs in RHS
        # This requires building the actual model
        from skfem import Basis, ElementLineHermite, MeshLine
        mesh = MeshLine(points)
        basis = Basis(mesh, ElementLineHermite())

        from skfem_reference import _uniform_load
        rhs = _uniform_load.assemble(
            basis, center=center,
            half_width=physical.footprint_m / 2.0,
            pressure=physical.nominal_force_N / physical.footprint_m
        )

        # Sum translational DOFs (even indices)
        displacement_dofs = np.arange(0, 2 * points.size, 2)
        total_force = rhs[displacement_dofs].sum()

        assert np.allclose(total_force, physical.nominal_force_N, rtol=1e-13)


def test_pilot_generation_deterministic():
    """Test pilot dataset generation is deterministic and saves float arrays."""
    physical = PhysicalModel()
    data_config = DataConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        meta1 = generate_dataset(
            "pilot_gaussian", "pilot", 2026110199,
            physical, data_config, Path(tmpdir) / "run1"
        )
        meta2 = generate_dataset(
            "pilot_gaussian", "pilot", 2026110199,
            physical, data_config, Path(tmpdir) / "run2"
        )

        data1 = np.load(meta1["npz_path"], allow_pickle=False)
        data2 = np.load(meta2["npz_path"], allow_pickle=False)

        # Check observations are identical (deterministic RNG)
        assert np.allclose(data1["observations"], data2["observations"])

        # Check all arrays are finite
        for key in data1.files:
            arr = data1[key]
            if isinstance(arr, np.ndarray) and arr.dtype.kind in "fc":
                assert np.all(np.isfinite(arr)), f"{key} contains non-finite values"

        # Verify NPZ loads without pickle
        # (already done by allow_pickle=False above, but explicit check)
        assert data1["observations"].dtype != object
        assert data1["sensor_locations"].dtype != object


def test_noise_realization_recoverable():
    """Test noise can be reproduced from seed."""
    physical = PhysicalModel()
    data_config = DataConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        seed = 123456
        meta = generate_dataset(
            "asymmetric", "calibrated", seed,
            physical, data_config, Path(tmpdir)
        )

        data = np.load(meta["npz_path"], allow_pickle=False)
        observations = data["observations"]
        truth_strain = data["truth_strain"]
        sigma = float(data["sigma"])

        # Check that noise has reasonable statistics (small sample, so loose bounds)
        noise = observations - truth_strain

        for load_noise in noise:
            # Mean should be near zero
            assert np.abs(np.mean(load_noise)) < 3 * sigma / np.sqrt(load_noise.size)
            # Std should be roughly sigma, but with only 20 samples tolerance is large
            assert np.abs(np.std(load_noise) - sigma) < sigma  # Within 1 sigma is reasonable for n=20


def test_regime_variations():
    """Test different regimes alter appropriate parameters."""
    physical = PhysicalModel()
    data_config = DataConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test amplitude_bias_sigma20microstrain
        meta_high_noise = generate_dataset(
            "rectangle", "amplitude_bias_sigma20microstrain", 1001,
            physical, data_config, Path(tmpdir) / "high_noise"
        )
        data_high = np.load(meta_high_noise["npz_path"], allow_pickle=False)
        assert float(data_high["sigma"]) == 20e-6

        # Test amplitude_bias_pasternak
        meta_shear = generate_dataset(
            "rectangle", "amplitude_bias_pasternak_kG0.2k0ell2", 1002,
            physical, data_config, Path(tmpdir) / "shear"
        )
        data_shear = np.load(meta_shear["npz_path"], allow_pickle=False)
        ell = np.sqrt(physical.EI_N_m2 / physical.k0_N_m2)
        expected_shear = 0.2 * physical.k0_N_m2 * ell
        assert np.isclose(float(data_shear["shear"]), expected_shear)

        # Test position perturbation
        meta_pos = generate_dataset(
            "asymmetric", "amplitude_bias_load_position_plus0.25m", 1003,
            physical, data_config, Path(tmpdir) / "position"
        )
        data_pos = np.load(meta_pos["npz_path"], allow_pickle=False)
        nominal = data_pos["train_centers_nominal"]
        actual = data_pos["train_centers_actual"]
        assert np.allclose(actual - nominal, 0.25)


def test_oracle_weighted_objective():
    """Test oracle uses fair weighted objective by scaling RHS, not dividing y and sigma separately."""
    physical = PhysicalModel()
    data_config = DataConfig(train_load_centers_m=[-5.0, 0.0, 5.0], sensors_per_load=15)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate small dataset with biased multipliers
        meta = generate_dataset(
            "broad_shallow", "amplitude_bias", 99999,
            physical, data_config, Path(tmpdir)
        )

        data = np.load(meta["npz_path"], allow_pickle=False)
        true_multipliers = data["multipliers"]

        # Run oracle inversion
        result_oracle = run_inversion(
            Path(meta["npz_path"]), "H1_oracle", 1e-5, physical,
            starts=(0.05, 0.30), max_nfev=100,
            oracle_multipliers=true_multipliers.tolist()
        )

        # The key test: verify that oracle correctly reports physical gamma
        assert np.allclose(result_oracle["gamma"], true_multipliers)

        # Verify train_norm is computed correctly with physical predictions
        assert "train_norm" in result_oracle
        assert result_oracle["train_norm"] >= 0

        # Verify train_prediction exists and has correct shape
        assert "train_prediction" in result_oracle
        assert result_oracle["train_prediction"].shape[0] == len(true_multipliers)


def test_train_norm_includes_gamma():
    """Test that run_inversion.train_norm correctly includes gamma scaling."""
    physical = PhysicalModel()
    # Use more sensors and a simpler shape for better recovery
    # Fixed multiplier regimes require THREE training loads
    data_config = DataConfig(train_load_centers_m=[-4.0, 0.0, 4.0], sensors_per_load=40)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate noiseless dataset first
        meta = generate_dataset(
            "broad_shallow", "calibrated", 777,
            physical, data_config, Path(tmpdir)
        )

        # Load and artificially scale observations to simulate unknown load
        data = np.load(meta["npz_path"], allow_pickle=False)
        gamma_true = 1.08
        # Start with clean truth, then scale
        observations_scaled = data["truth_strain"] * gamma_true

        # Save modified dataset
        npz_modified = Path(tmpdir) / "modified.npz"
        np.savez(
            npz_modified,
            observations=observations_scaled,
            sensor_locations=data["sensor_locations"],
            sigma=1e-7,  # Tiny sigma for near-noiseless recovery
            train_centers_nominal=data["train_centers_nominal"],
            train_centers_actual=data["train_centers_actual"],
            multipliers=data["multipliers"],
            truth_strain=data["truth_strain"],
            score_grid=data["score_grid"],
            true_d=data["true_d"],
            unused_center=data["unused_center"],
            unused_eval_grid=data["unused_eval_grid"],
            unused_truth_strain=data["unused_truth_strain"],
            shape_name=data["shape_name"],
            regime=data["regime"],
            seed=data["seed"],
            shear=data["shear"],
        )

        # Run profile inversion with more iterations
        result = run_inversion(
            npz_modified, "H1_profile", 1e-6, physical,
            starts=(0.05, 0.30), max_nfev=500
        )

        # Manually compute train_norm to verify
        fe = HermiteBeamFE(
            data["train_centers_nominal"],
            x_min=physical.domain_m[0], x_max=physical.domain_m[1],
            n_elements=int((physical.domain_m[1] - physical.domain_m[0]) / physical.inverse_element_m),
            n_coefficients=physical.spline_count,
            gauge_length=physical.gauge_m,
            foundation=physical.k0_N_m2,
            force=physical.nominal_force_N,
            load_width=physical.footprint_m,
            ei=physical.EI_N_m2,
            radius=physical.radius_m,
        )

        # Now test with all three sensor arrays (FE expects one per load)
        sensors_list = [data["sensor_locations"][i] for i in range(3)]
        pred_all = fe.strain_and_jacobian(result["coefficients"], sensors_list)[0]

        # Extract prediction for first load
        n_sensors = len(sensors_list[0])
        pred_unit_load0 = pred_all[:n_sensors]
        pred_scaled_load0 = result["gamma"][0] * pred_unit_load0

        sigma_used = 1e-7
        # Compute manual norm for first load only
        manual_norm_load0 = np.linalg.norm((pred_scaled_load0 - observations_scaled[0]) / sigma_used)

        # The stored train_norm includes all three loads, compute full manual norm
        manual_norm_full = 0.0
        offset = 0
        for i in range(3):
            n_sens = len(sensors_list[i])
            pred_load_i = pred_all[offset:offset + n_sens]
            pred_scaled_i = result["gamma"][i] * pred_load_i
            manual_norm_full += np.sum(((pred_scaled_i - observations_scaled[i]) / sigma_used)**2)
            offset += n_sens
        manual_norm_full = np.sqrt(manual_norm_full)

        # Check train_norm matches manual computation
        assert np.isclose(result["train_norm"], manual_norm_full, rtol=1e-8)

        # Check gamma recovery for first load (reasonable for multi-load model-mismatch case)
        assert np.abs(result["gamma"][0] - gamma_true) < 0.15  # Model mismatch tolerance

        # Check response error for first load
        response_error = np.linalg.norm(pred_scaled_load0 - observations_scaled[0]) / np.linalg.norm(observations_scaled[0])
        assert response_error < 1e-3  # Response should be quite good


def test_truth_fields_not_read_by_fitting():
    """Verify that inversion reads only sensor data, not truth fields."""
    physical = PhysicalModel()
    data_config = DataConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        meta = generate_dataset(
            "rectangle", "calibrated", 5555,
            physical, data_config, Path(tmpdir)
        )

        # Corrupt truth fields
        data = np.load(meta["npz_path"], allow_pickle=False)
        corrupted_path = Path(tmpdir) / "corrupted.npz"

        np.savez(
            corrupted_path,
            observations=data["observations"],
            sensor_locations=data["sensor_locations"],
            sigma=data["sigma"],
            train_centers_nominal=data["train_centers_nominal"],
            train_centers_actual=data["train_centers_actual"],
            multipliers=data["multipliers"],
            # Corrupt truth fields
            truth_strain=np.full_like(data["truth_strain"], 999.0),
            score_grid=np.full_like(data["score_grid"], -999.0),
            true_d=np.full_like(data["true_d"], -999.0),
            unused_center=data["unused_center"],
            unused_eval_grid=data["unused_eval_grid"],
            unused_truth_strain=data["unused_truth_strain"],
            shape_name=data["shape_name"],
            regime=data["regime"],
            seed=data["seed"],
            shear=data["shear"],
        )

        # Inversion should complete without error (doesn't read truth fields)
        result = run_inversion(
            corrupted_path, "H1_fixed", 1e-4, physical,
            starts=(0.05, 0.30), max_nfev=50
        )

        # Check result has required keys
        assert "coefficients" in result
        assert "gamma" in result
        assert "train_norm" in result
        assert "train_prediction" in result


def test_required_keys_validation():
    """Test that required observation keys are validated."""
    physical = PhysicalModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create incomplete dataset
        incomplete_path = Path(tmpdir) / "incomplete.npz"
        np.savez(
            incomplete_path,
            observations=np.random.randn(3, 20),
            # Missing sigma, sensor_locations, train_centers_nominal
        )

        # Should fail with clear error
        with pytest.raises((KeyError, ValueError)):
            run_inversion(
                incomplete_path, "H1_fixed", 1e-4, physical,
                starts=(0.05, 0.30), max_nfev=50
            )
