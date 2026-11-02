"""Tests for campaign runner and selection logic."""

from pathlib import Path
import sys
import tempfile
import json

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from run_campaign import select_candidate, score_selected, hash_protocol
from campaign import PhysicalModel, DataConfig, generate_dataset


def test_selection_largest_feasible_lambda():
    """Test selection of largest lambda among feasible candidates."""
    N = 60
    candidates = [
        {"penalty_lambda": 0.1, "train_norm": 8.0, "successful_start": True},
        {"penalty_lambda": 1.0, "train_norm": 9.0, "successful_start": True},
        {"penalty_lambda": 10.0, "train_norm": 9.2, "successful_start": True},
        {"penalty_lambda": 100.0, "train_norm": 11.0, "successful_start": True},
    ]

    selection = select_candidate(candidates, N)

    # Threshold = 1.2 * sqrt(60) = 9.29...
    # Feasible: 0.1, 1.0, 10.0 (all have norm <= 9.29)
    # Should select largest feasible: 10.0
    assert selection["selected_lambda"] == 10.0
    assert selection["selection_reason"] == "largest_feasible_lambda"
    assert not selection["optimizer_failed"]
    assert not selection["no_feasible_candidate"]


def test_selection_no_feasible_candidate():
    """Test selection when no feasible candidate exists."""
    N = 60
    candidates = [
        {"penalty_lambda": 0.1, "train_norm": 10.0, "successful_start": True},
        {"penalty_lambda": 1.0, "train_norm": 11.0, "successful_start": True},
        {"penalty_lambda": 10.0, "train_norm": 12.0, "successful_start": True},
    ]

    selection = select_candidate(candidates, N)

    # All exceed threshold 9.29, so no feasible candidate
    # Should select minimum norm among successful
    assert selection["selected_lambda"] == 0.1
    assert selection["selection_reason"] == "no_feasible_min_successful_norm"
    assert not selection["optimizer_failed"]
    assert selection["no_feasible_candidate"]


def test_selection_optimizer_failed():
    """Test selection when all candidates failed."""
    N = 60
    candidates = [
        {"penalty_lambda": 0.1, "train_norm": 8.0, "successful_start": False},
        {"penalty_lambda": 1.0, "train_norm": 9.0, "successful_start": False},
        {"penalty_lambda": 10.0, "train_norm": 11.0, "successful_start": False},
    ]

    selection = select_candidate(candidates, N)

    # No successful candidates
    assert selection["selection_reason"] == "optimizer_failed"
    assert selection["optimizer_failed"]
    assert not selection["no_feasible_candidate"]
    # Should return best failed as diagnostic
    assert selection["selected_lambda"] == 0.1


def test_selection_path_bracketed():
    """Test bracketing flag when higher successful candidate violates threshold."""
    N = 60
    candidates = [
        {"penalty_lambda": 0.1, "train_norm": 8.0, "successful_start": True},
        {"penalty_lambda": 1.0, "train_norm": 9.0, "successful_start": True},
        {"penalty_lambda": 10.0, "train_norm": 11.0, "successful_start": True},
    ]

    selection = select_candidate(candidates, N)

    # Selected: 1.0 (largest feasible)
    # Higher successful: 10.0 with norm > threshold
    # So path is bracketed
    assert selection["selected_lambda"] == 1.0
    assert selection["path_bracketed"]


def test_selection_unbracketed_high():
    """Test unbracketed_high when selected is at largest lambda."""
    N = 60
    candidates = [
        {"penalty_lambda": 0.1, "train_norm": 8.0, "successful_start": True},
        {"penalty_lambda": 1.0, "train_norm": 9.0, "successful_start": True},
        {"penalty_lambda": 10.0, "train_norm": 9.2, "successful_start": True},
    ]

    selection = select_candidate(candidates, N)

    # Selected: 10.0 (largest feasible and at largest tested)
    assert selection["selected_lambda"] == 10.0
    assert selection["unbracketed_high"]


def test_physical_score_zero_when_estimates_equal_references():
    """Test that scoring gives zero error when estimate equals reference using exact model.

    Uses zero coefficients (intact) with the SAME FE for both generation and scoring
    to test scoring arithmetic without discretization differences.
    """
    physical = PhysicalModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Manually construct a perfect test case using the scoring FE directly
        # Build FE with unused load center
        from beam_inverse import HermiteBeamFE

        unused_center = 6.0
        fe = HermiteBeamFE(
            [unused_center],
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

        # Zero coefficients (intact)
        coefficients = np.zeros(physical.spline_count - 2)

        # Evaluate at unused grid with the SAME FE
        unused_eval_grid = np.linspace(-12, 12, 241)
        unused_truth_strain = fe.strain_and_jacobian(coefficients, [unused_eval_grid])[0]

        # Create mock dataset with this exact truth
        from shape_factory import intact_shape
        score_grid = np.linspace(-12, 12, 1201)
        true_d = intact_shape(score_grid)

        npz_path = Path(tmpdir) / "test.npz"
        np.savez(
            npz_path,
            shape_name="intact",
            multipliers=np.array([1.0, 1.0, 1.0]),
            unused_center=unused_center,
            unused_eval_grid=unused_eval_grid,
            unused_truth_strain=unused_truth_strain,
            score_grid=score_grid,
            true_d=true_d,
        )

        perfect_result = {
            "coefficients": coefficients,
            "gamma": np.array([1.0, 1.0, 1.0]),
        }

        metrics, arrays = score_selected(npz_path, perfect_result, physical)

        # Should have near-zero support error (intact is zero everywhere)
        assert metrics["absolute_support_L2"] < 1e-10
        assert metrics["gamma_RMSE"] < 1e-14

        # Unused load prediction should match saved truth EXACTLY (same FE, same points)
        unused_rel_error = metrics["unused_load_strain_relative_L2"]
        assert unused_rel_error < 1e-12
        assert metrics["gamma_RMSE"] < 1e-14


def test_intact_no_zero_division():
    """Test that intact shape doesn't cause division by zero in relative L2."""
    physical = PhysicalModel()
    data_config = DataConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate intact dataset
        meta = generate_dataset(
            "intact", "calibrated", 123,
            physical, data_config, Path(tmpdir)
        )

        # Create arbitrary result
        dummy_result = {
            "coefficients": np.zeros(physical.spline_count - 2),
            "gamma": np.ones(3),
        }

        metrics, arrays = score_selected(Path(meta["npz_path"]), dummy_result, physical)

        # Relative L2 should be None for intact
        assert metrics["support_rel_L2_except_intact"] is None
        # Max false loss should be present
        assert metrics["max_false_loss_for_intact"] is not None
        assert isinstance(metrics["max_false_loss_for_intact"], float)


def test_json_strict_serialization():
    """Test that all metric values are serializable as finite floats or null."""
    physical = PhysicalModel()
    data_config = DataConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        meta = generate_dataset(
            "asymmetric", "calibrated", 456,
            physical, data_config, Path(tmpdir)
        )

        dummy_result = {
            "coefficients": np.linspace(0.1, 0.4, physical.spline_count - 2),
            "gamma": np.array([1.0, 1.0, 1.0]),
        }

        metrics, arrays = score_selected(Path(meta["npz_path"]), dummy_result, physical)

        # Try to serialize
        json_str = json.dumps(metrics)

        # Verify all values are finite or None
        for key, value in metrics.items():
            if value is not None:
                assert np.isfinite(value), f"{key} = {value} is not finite"
                assert not isinstance(value, np.ndarray), f"{key} is an array, not a scalar"


def test_protocol_hash_deterministic():
    """Test that protocol hash is deterministic."""
    plan = {
        "version": "test",
        "data": {"seed": 123, "value": 3.14},
        "nested": {"a": 1, "b": 2},
    }

    hash1 = hash_protocol(plan)
    hash2 = hash_protocol(plan)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex length

    # Different plan should give different hash
    plan2 = plan.copy()
    plan2["data"]["seed"] = 124
    hash3 = hash_protocol(plan2)
    assert hash3 != hash1


def test_candidate_count_expansion():
    """Test that candidate list includes all lambda values."""
    N = 60
    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    # Simulate candidates
    candidates = [
        {"penalty_lambda": lam, "train_norm": 8.0 + i, "successful_start": True}
        for i, lam in enumerate(lambdas)
    ]

    assert len(candidates) == len(lambdas)

    selection = select_candidate(candidates, N)

    # Selection should not reduce candidate count
    assert len(candidates) == len(lambdas)
