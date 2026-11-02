
from pathlib import Path
import json
import sys
import tempfile

import numpy as np
import pytest


CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_load_information as runner  # noqa: E402
from beam_inverse import SupportBasis  # noqa: E402


def test_expansion_counts_deduplication_and_memberships():
    design = runner.load_design()
    datasets, paths, counts = runner.expand_study(design)

    assert counts == {
        "datasets": 200,
        "paths": 848,
        "candidates": 5088,
        "starts": 10176,
        "primary_datasets": 72,
        "resolution_datasets": 144,
        "overlap_datasets": 16,
    }
    assert len(datasets) == len({dataset["key"] for dataset in datasets})
    assert len(paths) == len({path["key"] for path in paths})
    assert sum(dataset["memberships"] == ["primary"] for dataset in datasets) == 56
    assert sum(dataset["memberships"] == ["resolution"] for dataset in datasets) == 128
    assert sum(
        set(dataset["memberships"]) == {"primary", "resolution"} for dataset in datasets
    ) == 16
    assert {path["method"] for path in paths} == {
        "fixed", "free", "calibrated", "oracle", "overconfident", "conservative"
    }
    assert sum(path["method"] == "overconfident" for path in paths) == 24
    assert sum(path["method"] == "conservative" for path in paths) == 24
    assert all(path["key"].startswith(path["dataset_key"]) for path in paths)


def test_paired_calibration_vector_has_exact_uniform_standard_deviation():
    means_05, z_05 = runner.calibration_measurement(1101, 0.05)
    means_10, z_10 = runner.calibration_measurement(1101, 0.10)

    expected_z = np.random.default_rng(
        np.random.SeedSequence([1101, 831])
    ).uniform(-np.sqrt(3.0), np.sqrt(3.0), 3)
    np.testing.assert_array_equal(z_05, expected_z)
    np.testing.assert_array_equal(z_05, z_10)
    np.testing.assert_allclose(means_05, 1.0 + 0.05 * z_05, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(means_10, 1.0 + 0.10 * z_05, rtol=0.0, atol=0.0)
    assert np.all(np.abs(z_05) <= np.sqrt(3.0))
    assert np.sqrt((2.0 * np.sqrt(3.0)) ** 2 / 12.0) == pytest.approx(1.0)


def test_frozen_broad_label_maps_to_legacy_generator_shape():
    config = {"shape": "broad", "seed": 1101}
    assert runner._generator_shape("broad") == "broad_shallow"
    assert runner._dataset_generated_names(config)[0] == "broad_shallow_calibrated_seed1101.npz"


def test_method_mapping_uses_observed_mean_except_oracle():
    observed = np.array([0.91, 1.02, 1.11])
    mappings = {
        method: runner.method_inputs(method, observed, 0.05)
        for method in ("fixed", "free", "calibrated", "overconfident", "conservative", "oracle")
    }

    assert mappings["fixed"]["gamma_mode"] == "fixed"
    np.testing.assert_array_equal(mappings["fixed"]["load_mean"], observed)
    assert mappings["fixed"]["load_std"] == 0.0
    assert mappings["fixed"]["finite_positive_prior"] is False
    assert mappings["free"]["gamma_mode"] == "profile"
    assert mappings["free"]["load_std"] == np.inf
    assert mappings["free"]["finite_positive_prior"] is False
    assert mappings["calibrated"]["load_std"] == 0.05
    assert mappings["overconfident"]["load_std"] == pytest.approx(0.01)
    assert mappings["conservative"]["load_std"] == pytest.approx(0.15)
    assert all(
        np.array_equal(mapping["load_mean"], observed)
        for method, mapping in mappings.items()
        if method != "oracle"
    )
    assert mappings["oracle"]["gamma_mode"] == "oracle"
    np.testing.assert_array_equal(mappings["oracle"]["load_mean"], np.ones(3))


def test_selection_counts_three_calibration_observations_only_for_finite_priors():
    def candidate(lam, norm, successful=True, prior=False):
        return {
            "penalty_lambda": lam,
            "successful_start": successful,
            "finite_positive_prior": prior,
            "norms": {
                "selection_norm": norm,
                "data_norm": norm,
                "calibration_norm": 0.0,
            },
        }

    free = [candidate(0.03, 9.0, prior=False), candidate(3000.0, 9.2, prior=False)]
    calibrated = [candidate(0.03, 9.0, prior=True), candidate(3000.0, 9.5, prior=True)]
    free_selection = runner._selection_details(free, sensors=20)
    prior_selection = runner._selection_details(calibrated, sensors=20)

    assert free_selection["N_strain_observations"] == 60
    assert free_selection["M_calibration_observations"] == 0
    assert free_selection["threshold"] == pytest.approx(1.2 * np.sqrt(60))
    assert prior_selection["N_strain_observations"] == 60
    assert prior_selection["M_calibration_observations"] == 3
    assert prior_selection["threshold"] == pytest.approx(1.2 * np.sqrt(63))
    assert free_selection["selected_lambda"] == 3000.0
    assert prior_selection["selected_lambda"] == 3000.0


def test_strict_json_serialization_represents_infinity_as_null():
    value = {
        "infinite": np.inf,
        "negative_infinite": -np.inf,
        "finite": np.float64(1.25),
        "array": np.array([1.0, 2.0]),
    }
    encoded = runner.canonical_json(runner.safe_json(value))
    assert json.loads(encoded) == {
        "infinite": None,
        "negative_infinite": None,
        "finite": 1.25,
        "array": [1.0, 2.0],
    }
    with pytest.raises(runner.ProtocolError):
        runner.safe_json({"object": np.array([object()], dtype=object)})
    with pytest.raises(runner.ProtocolError):
        runner.safe_json({"invalid": np.nan})


class TinyForward:
    """Small linear forward model exercising the real fit_support API."""

    def __init__(self, centers, **kwargs):
        del kwargs
        self.centers = np.asarray(centers, dtype=float)
        self.basis = SupportBasis(n_coefficients=5)
        self.gauge_length = 0.0
        self.x = np.array([self.centers[0] - 2.0, self.centers[-1] + 2.0])
        self.rhs = np.ones((4, self.centers.size))

    def strain_and_jacobian(self, coefficients, points):
        all_points = np.concatenate([np.asarray(point) for point in points])
        values = self.basis.values(all_points)
        variables = np.asarray(coefficients, dtype=float)
        prediction = values[:, 1:-1] @ variables
        jacobian = values[:, 1:-1]
        return prediction, jacobian


def _smoke_design():
    return {
        "schema": "load-information-v3.0",
        "primary": {
            "shapes": ["intact"],
            "load_standard_uncertainty": [0.05],
            "sensors": [2],
            "strain_sigma": [0.003],
            "seeds": [831],
        },
        "resolution": {
            "shapes": [],
            "load_standard_uncertainty": [],
            "sensors": [],
            "strain_sigma": [],
            "seeds": [],
        },
        "methods": ["fixed", "free", "calibrated", "oracle"],
        "confidence_sensitivity": {
            "primary_load_standard_uncertainty": 0.05,
            "methods": {"overconfident": 0.2, "conservative": 3.0},
        },
        "regularization": {
            "penalty": "h1",
            "lambda_path": [1.0e-6],
            "starts": [0.10, 0.20],
            "max_nfev": 50,
            "selection": "smoke",
        },
        "resource_limit": {"blas_threads": 1, "parallel_paths": 1, "wall_time_limit_hours": 2},
    }


def _contains_absolute_string(value):
    if isinstance(value, dict):
        return any(_contains_absolute_string(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_string(child) for child in value)
    return isinstance(value, str) and value.startswith("/")


def _install_tiny_model(monkeypatch, output_dir, null_metric=False):
    true_coefficients = np.array([0.35, 0.25, 0.15])
    points = [np.array([-1.0, 0.5]), np.array([-0.4, 0.8]), np.array([-0.2, 1.0])]
    forward = TinyForward([-4.0, 0.0, 4.0])
    unit_prediction = forward.strain_and_jacobian(true_coefficients, points)[0]
    gamma = runner.calibration_measurement(831, 0.05)[0]
    noise = np.array([0.002, -0.001, 0.0015, -0.002, 0.001, -0.0015])
    observations = gamma[:, None] * unit_prediction.reshape(3, 2) + noise.reshape(3, 2)

    def fake_generate(shape, regime, seed, physical, data_config, directory):
        del shape, regime, seed, physical, data_config
        directory.mkdir(parents=True, exist_ok=True)
        score_grid = np.linspace(-1.5, 1.5, 21)
        true_d = forward.basis.values(score_grid) @ forward.basis.full_coefficients(true_coefficients)
        unused_grid = np.linspace(-1.0, 1.0, 11)
        unused_truth = forward.strain_and_jacobian(
            true_coefficients, [unused_grid]
        )[0]
        npz_name = "intact_calibrated_seed831.npz"
        npz_path = directory / npz_name
        np.savez(
            npz_path,
            observations=observations,
            sensor_locations=np.asarray(points),
            sigma=0.003,
            train_centers_nominal=np.array([-4.0, 0.0, 4.0]),
            train_centers_actual=np.array([-4.0, 0.0, 4.0]),
            multipliers=np.ones(3),
            truth_strain=unit_prediction.reshape(3, 2),
            score_grid=score_grid,
            true_d=true_d,
            unused_center=6.0,
            unused_eval_grid=unused_grid,
            unused_truth_strain=unused_truth,
            shape_name="intact",
            regime="calibrated",
            seed=831,
            shear=0.0,
        )
        metadata = {
            "npz_path": str(npz_path),
            "shape": "intact",
            "regime": "calibrated",
            "seed": 831,
            "n_sensors_per_load": [2, 2, 2],
            "sigma": 0.003,
        }
        (directory / "intact_calibrated_seed831.json").write_text(
            json.dumps(metadata, allow_nan=False), encoding="utf-8"
        )
        return metadata

    def fake_score(dataset_path, result, physical):
        del dataset_path, physical
        coefficients = np.asarray(result["coefficients"], dtype=float)
        score_grid = np.linspace(-1.5, 1.5, 21)
        estimated = forward.basis.values(score_grid) @ forward.basis.full_coefficients(coefficients)
        unused_grid = np.linspace(-1.0, 1.0, 11)
        predicted = forward.strain_and_jacobian(coefficients, [unused_grid])[0]
        metrics = {
            "absolute_support_L2": float(np.linalg.norm(estimated)),
            "gamma_RMSE": float(np.linalg.norm(result["gamma"] - 1.0) / np.sqrt(3.0)),
            "unused_load_strain_relative_L2": float(np.linalg.norm(predicted) / 10.0),
            "unused_load_peak_strain_error": float(abs(np.max(np.abs(predicted)) - 1.0)),
        }
        if null_metric:
            # e.g. the intact shape has no support defect to localize.
            metrics["unused_load_peak_strain_error"] = None
        arrays = {
            "score_x": score_grid,
            "estimated_d": estimated,
            "unused_x": unused_grid,
            "predicted_strain": predicted,
        }
        return metrics, arrays

    monkeypatch.setattr(runner, "generate_dataset", fake_generate)
    monkeypatch.setattr(runner, "HermiteBeamFE", TinyForward)
    monkeypatch.setattr(runner, "score_selected", fake_score)
    return output_dir


def test_numerical_smoke_all_methods_checkpoint_and_verify(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        output = _install_tiny_model(monkeypatch, Path(temporary) / "smoke")
        design = _smoke_design()
        lock = runner.freeze_protocol(output, design)
        status = runner.execute_run(output, design, resume=False)

        assert status["status"] == "completed"
        assert status["completed_paths"] == 6
        candidate_files = sorted(output.glob("paths/*/*.json"))
        assert len(candidate_files) == 12  # six candidates and six completed-path records
        methods = set()
        for path_file in output.glob("paths/*/path.json"):
            record = runner.read_json(path_file)
            methods.add(record["method"])
            assert record["status"] == "selected"
            assert len(record["candidate_files"]) == 1
            assert record["selection"]["threshold"] > 0.0
        assert methods == {"fixed", "free", "calibrated", "oracle", "overconfident", "conservative"}
        assert lock["protocol_hash"]
        for json_file in output.rglob("*.json"):
            value = runner.read_json(json_file)
            assert not _contains_absolute_string(value)

        report = runner.verify_protocol(output, design)
        assert report["passed"] is True
        assert report["incomplete"] is False
        assert report["checked_candidates"] == 6


def test_verify_detects_tampered_objective_and_array_hash(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        output = _install_tiny_model(monkeypatch, Path(temporary) / "smoke")
        design = _smoke_design()
        runner.freeze_protocol(output, design)
        runner.execute_run(output, design, resume=False)
        candidate_path = next(output.glob("paths/*/candidate_lambda_*.json"))
        original = runner.read_json(candidate_path)

        tampered = dict(original)
        tampered["objective"] = float(original["objective"] + 1.0)
        candidate_path.write_text(json.dumps(tampered, allow_nan=False), encoding="utf-8")
        with pytest.raises(runner.ProtocolError, match="existing candidate failed replay"):
            runner.execute_run(output, design, resume=True)
        report = runner.verify_protocol(output, design)
        assert report["passed"] is False
        assert any("objective mismatch" in error for errors in report["errors"].values() for error in errors)

        candidate_path.write_text(json.dumps(original, allow_nan=False), encoding="utf-8")
        assert runner.verify_protocol(output, design)["passed"] is True

        tampered = dict(original)
        tampered["coefficients"] = [value + 0.1 for value in original["coefficients"]]
        candidate_path.write_text(json.dumps(tampered, allow_nan=False), encoding="utf-8")
        report = runner.verify_protocol(output, design)
        assert report["passed"] is False
        assert any(
            "array hash mismatch" in error or "objective mismatch" in error
            for errors in report["errors"].values()
            for error in errors
        )


def test_resume_reuses_verified_candidates_and_orphan_selected_arrays(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        output = _install_tiny_model(monkeypatch, Path(temporary) / "smoke")
        design = _smoke_design()
        runner.freeze_protocol(output, design)
        runner.execute_run(output, design, resume=False)
        for path_file in output.glob("paths/*/path.json"):
            path_file.unlink()

        status = runner.execute_run(output, design, resume=True)
        assert status["status"] == "completed"
        assert status["reused_completed_paths"] == 0
        for path_file in output.glob("paths/*/path.json"):
            record = runner.read_json(path_file)
            assert record["reused_candidates"] == [True]
        assert runner.verify_protocol(output, design)["passed"] is True


def test_resume_refuses_source_mismatch(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "locked"
        design = _smoke_design()
        runner.freeze_protocol(output, design)
        original_hashes = runner.mapped_hashes

        def changed_hashes(names):
            values = original_hashes(names)
            return {name: ("0" * 64 if index == 0 else value) for index, (name, value) in enumerate(values.items())}

        monkeypatch.setattr(runner, "mapped_hashes", changed_hashes)
        with pytest.raises(runner.ProtocolError, match="source lock mismatch"):
            runner.execute_run(output, design, resume=True)


def test_fresh_run_ignores_macos_system_metadata(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        output = _install_tiny_model(monkeypatch, Path(temporary) / "meta")
        design = _smoke_design()
        runner.freeze_protocol(output, design)
        (output / "._load_information_lock.json").write_text("AppleDouble", encoding="utf-8")
        (output / ".DS_Store").write_text("Finder", encoding="utf-8")

        status = runner.execute_run(output, design, resume=False)
        assert status["status"] == "completed"
        assert status["completed_paths"] == 6

        # Metadata inside result directories does not break verify or resume.
        path_dir = next((output / "paths").iterdir())
        (path_dir / "._path.json").write_text("AppleDouble", encoding="utf-8")
        dataset_dir = next((output / "datasets").iterdir())
        (dataset_dir / ".DS_Store").write_text("Finder", encoding="utf-8")
        report = runner.verify_protocol(output, design)
        assert report["passed"] is True
        assert report["incomplete"] is False
        resumed = runner.execute_run(output, design, resume=True)
        assert resumed["reused_completed_paths"] == 6

        # A real unexpected file is still rejected.
        (path_dir / "stray.txt").write_text("unexpected", encoding="utf-8")
        assert runner.verify_protocol(output, design)["passed"] is False
        with pytest.raises(runner.ProtocolError, match="unexpected files"):
            runner.execute_run(output, design, resume=True)


def test_timeout_mid_path_is_not_counted_as_completed(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "timeout"
        design = _smoke_design()
        runner.freeze_protocol(output, design)

        def expiring_path(path_config, datasets, design, output_dir, lock, deadline, forward_cache):
            return {"path_key": path_config["key"], "status": "timeout"}, False

        monkeypatch.setattr(runner, "_run_one_path", expiring_path)
        status = runner.execute_run(output, design, resume=False)
        assert status["status"] == "timeout"
        assert status["completed_paths"] == 0
        assert status["reused_completed_paths"] == 0
        saved = runner.read_json(output / "run_status.json")
        assert saved["status"] == "timeout"
        assert saved["completed_paths"] == 0


def test_verify_marks_incomplete_when_path_record_missing(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        output = _install_tiny_model(monkeypatch, Path(temporary) / "incomplete")
        design = _smoke_design()
        runner.freeze_protocol(output, design)
        runner.execute_run(output, design, resume=False)
        removed = next(output.glob("paths/*/path.json"))
        removed_key = removed.parent.name
        removed.unlink()

        report = runner.verify_protocol(output, design)
        assert report["passed"] is True  # integrity of existing files still holds
        assert report["incomplete"] is True
        assert removed_key in report["incomplete_paths"]


def test_fixed_and_oracle_pass_load_mean_without_rhs_scaling(monkeypatch, tmp_path):
    true_coefficients = np.array([0.35, 0.25, 0.15])
    points = [np.array([-1.0, 0.5]), np.array([-0.4, 0.8]), np.array([-0.2, 1.0])]
    forward = TinyForward([-4.0, 0.0, 4.0])
    unit_prediction = forward.strain_and_jacobian(true_coefficients, points)[0]
    nonunit_mean = np.array([0.95, 1.05, 1.10])
    observations = nonunit_mean[:, None] * unit_prediction.reshape(3, 2)
    dataset_path = tmp_path / "tiny.npz"
    np.savez(
        dataset_path,
        observations=observations,
        sensor_locations=np.asarray(points),
        sigma=0.003,
    )
    design = _smoke_design()
    original_rhs = forward.rhs.copy()
    seen_unmodified = []
    real_fit = runner.fit_support

    def spy(model, *args, **kwargs):
        seen_unmodified.append(np.array_equal(model.rhs, original_rhs))
        return real_fit(model, *args, **kwargs)

    monkeypatch.setattr(runner, "fit_support", spy)

    fixed = runner._fit_one_candidate(
        forward, dataset_path, "fixed", 1.0e-6, design, nonunit_mean.copy(), 0.05
    )
    np.testing.assert_allclose(fixed["gamma"], nonunit_mean, rtol=0.0, atol=1.0e-12)
    assert fixed["runner_inputs"]["gamma_mode"] == "fixed"

    oracle = runner._fit_one_candidate(
        forward, dataset_path, "oracle", 1.0e-6, design, nonunit_mean.copy(), 0.05
    )
    np.testing.assert_allclose(oracle["gamma"], np.ones(3), rtol=0.0, atol=1.0e-12)
    assert oracle["runner_inputs"]["gamma_mode"] == "oracle"

    assert seen_unmodified == [True, True]
    np.testing.assert_array_equal(forward.rhs, original_rhs)


def test_null_metrics_compare_as_null_not_float(monkeypatch):
    with tempfile.TemporaryDirectory() as temporary:
        output = _install_tiny_model(
            monkeypatch, Path(temporary) / "nullmetrics", null_metric=True
        )
        design = _smoke_design()
        runner.freeze_protocol(output, design)
        status = runner.execute_run(output, design, resume=False)
        assert status["status"] == "completed"
        report = runner.verify_protocol(output, design)
        assert report["passed"] is True
        assert report["incomplete"] is False
        for path_file in output.glob("paths/*/path.json"):
            record = runner.read_json(path_file)
            assert record["metrics"]["unused_load_peak_strain_error"] is None

        # A null metric replaced by a number is a replay mismatch.
        target = next(output.glob("paths/*/path.json"))
        record = runner.read_json(target)
        record["metrics"]["unused_load_peak_strain_error"] = 1.0
        target.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
        report = runner.verify_protocol(output, design)
        assert report["passed"] is False
        assert any(
            "selected metric mismatch" in error
            for errors in report["errors"].values()
            for error in errors
        )
