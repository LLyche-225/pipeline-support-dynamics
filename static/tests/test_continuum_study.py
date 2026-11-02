
from pathlib import Path
import sys

import numpy as np
import pytest


CODE = Path(__file__).resolve().parents[1] / "code"
VENDOR = Path(__file__).resolve().parents[1] / "vendor"
for directory in (CODE, VENDOR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from campaign import PhysicalModel  # noqa: E402
from continuum_reference import continuum_forward  # noqa: E402
from run_continuum_study import (  # noqa: E402
    ProtocolError,
    _build_inverse_forward,
    _candidate_record,
    _check_lock,
    _create_lock,
    _fit_one_candidate,
    _ensure_dataset,
    _interface_metrics,
    _is_platform_metadata,
    _new_lock,
    _path_inventory,
    _run_one_path,
    _run_is_completed,
    _same_optional_metric,
    _score_selected,
    expand_continuum_study,
    load_designs,
)
import run_continuum_study  # noqa: E402
from skfem_reference import mesh_with_load_edges  # noqa: E402


def test_real_small_continuum_and_small_inverse_forward():
    points = mesh_with_load_edges(-3.0, 3.0, 0.5, np.array([0.0]), load_width=1.2)
    result = continuum_forward(
        points,
        np.linspace(0.0, 1.0, 3),
        np.array([0.0]),
        lambda x: np.zeros_like(x),
        np.array([-1.0, 0.0, 1.0]),
        0.2,
        ei=1.0e4,
        interface_stiffness=1.0e5,
        soil_E=1.0e6,
        soil_nu=0.3,
        width=0.1,
        force=1.0,
        load_width=1.2,
        radius=0.1,
    )
    assert result["strain"].shape == (3, 1)
    assert result["mesh"]["vector_probe_order_verified"] is True
    assert np.allclose(
        np.asarray(result["diagnostics"]["total_vertical_reaction"]),
        -np.asarray(result["diagnostics"]["applied_force"]),
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    physical = PhysicalModel(domain_m=(-6.0, 6.0), inverse_element_m=0.25)
    forward = _build_inverse_forward([0.0], 1.7e7, physical)
    assert forward.foundation == 1.7e7
    assert forward.force == physical.nominal_force_N
    assert np.allclose(forward.centers, [0.0])


def test_expansion_counts_match_frozen_continuum_design():
    continuum, study = load_designs()
    datasets, paths, counts = expand_continuum_study(continuum, study)
    assert (len(datasets), len(paths)) == (36, 144)
    assert counts == {"datasets": 36, "paths": 144, "candidates": 864, "starts": 1728}


def test_interface_metrics_keep_physical_parameters_separate_and_null_centroid():
    x = np.linspace(-1.0, 1.0, 21)
    reference = 0.2 * np.exp(-(x**2) / 0.2)
    metrics = _interface_metrics(np.zeros_like(x), reference, x)
    assert metrics["inferred_effective_winkler_loss_area_m"] == pytest.approx(0.0)
    assert metrics["imposed_interface_loss_centroid_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["inferred_effective_winkler_loss_centroid_m"] is None
    assert not any(
        "support" in key and ("relative_L2" in key or "rel_L2" in key)
        for key in metrics
    )

    zero = _interface_metrics(np.zeros_like(x), np.zeros_like(x), x)
    assert zero["imposed_interface_loss_centroid_m"] is None
    assert zero["inferred_effective_winkler_loss_centroid_m"] is None


def test_prediction_rejects_incorrect_soil_specific_k0(tmp_path):
    candidate = {"calibrated_k0_N_m2": 1.8e7}
    with pytest.raises(ProtocolError, match="soil-specific calibrated k0"):
        _score_selected(tmp_path / "unused.npz", candidate, expected_k0=1.7e7)


def test_small_inverse_candidate_replays_all_starts(tmp_path):
    physical = PhysicalModel(domain_m=(-6.0, 6.0), inverse_element_m=0.25)
    centres = np.array([-2.0, 0.0, 2.0])
    forward = _build_inverse_forward(centres, 1.7e7, physical)
    points = [np.array([-2.0, 2.0])] * 3
    observations = forward.strain_and_jacobian(
        np.zeros(forward.basis.n_variables), points
    )[0].reshape(3, 2)
    dataset_path = tmp_path / "small.npz"
    np.savez(
        dataset_path,
        sensor_locations=np.asarray(points),
        observations=observations,
        sigma=np.asarray(5.0e-6),
        train_centers_nominal=centres,
    )
    lock = _new_lock()
    dataset = {"key": "small", "tau": 0.01, "soil_E_Pa": 5.0e7}
    dataset_record = {
        "npz_file": "datasets/small/data.npz",
        "npz_sha256": "unused-in-record-construction",
        "measured_load_mean": [1.0, 1.0, 1.0],
        "physical": {
            "calibrated_inverse_k0_N_m2": 1.7e7,
            "domain_m": (-6.0, 6.0),
            "inverse_h_m": 0.25,
        },
    }
    path_config = {"key": "small__method-calibrated", "method": "calibrated"}
    study = {"regularization": {"lambda_path": [0.3], "starts": [0.05, 0.3], "max_nfev": 10}}
    result = _fit_one_candidate(
        forward,
        dataset_path,
        "calibrated",
        0.3,
        study,
        np.ones(3),
        0.01,
    )
    record = _candidate_record(
        result=result,
        dataset=dataset,
        dataset_record=dataset_record,
        path_config=path_config,
        penalty_lambda=0.3,
        study=study,
        lock=lock,
    )
    replay = run_continuum_study._verify_candidate_record(
        record,
        dataset_path,
        dataset=dataset,
        dataset_record=dataset_record,
        study=study,
        lock=lock,
    )
    assert replay["passed"], replay["errors"]


def test_dataset_noise_and_sensor_layout_are_exact(monkeypatch, tmp_path):
    continuum, _ = load_designs()
    sensors = np.linspace(-12.0, 12.0, 20)
    evaluation = np.unique(np.concatenate([np.linspace(-12.0, 12.0, 241), sensors]))
    reference_path = tmp_path / "reference.npz"
    np.savez(
        reference_path,
        evaluation_points=evaluation,
        strain=np.arange(4 * evaluation.size, dtype=float).reshape(evaluation.size, 4),
        sensor_indices=np.searchsorted(evaluation, sensors),
        unused_indices=np.searchsorted(evaluation, np.linspace(-12.0, 12.0, 241)),
        soil_nodes=np.zeros((2, 4)),
        soil_quads=np.zeros((1, 4), dtype=int),
        soil_ux=np.zeros((2, 4)),
        soil_uz=np.zeros((2, 4)),
    )
    reference = {
        "arrays_file": "reference.npz",
        "arrays_sha256": run_continuum_study.hash_file(reference_path),
    }
    monkeypatch.setattr(run_continuum_study, "_shared_reference", lambda *args, **kwargs: reference)
    lock = _new_lock()
    dataset = {
        "key": "noise-check",
        "shape": "intact",
        "soil_E_Pa": 5.0e7,
        "tau": 0.1,
        "sigma": 5.0e-6,
        "sensors": 20,
        "seed": 1201,
    }
    record = _ensure_dataset(
        tmp_path,
        dataset,
        continuum=continuum,
        lock=lock,
        mesh_report={"runs": []},
        calibrated_k0={5.0e7: 4.9e6},
    )
    path = run_continuum_study.output_file(tmp_path, record["npz_file"])
    with np.load(path, allow_pickle=False) as data:
        assert data["sensor_locations"].shape == (3, 20)
        assert data["observations"].shape == (3, 20)
        expected = np.random.default_rng(1201).normal(0.0, 5.0e-6, (3, 20))
        np.testing.assert_allclose(
            data["observations"] - data["noiseless_train_strain"],
            expected,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_intact_selected_scores_are_physical_and_nullable(tmp_path):
    unused_x = np.linspace(-1.0, 1.0, 3)
    path = tmp_path / "intact.npz"
    np.savez(
        path,
        shape_name=np.asarray("intact"),
        score_grid=np.linspace(-2.0, 2.0, 5),
        reference_interface_d=np.zeros(5),
        unused_eval_grid=unused_x,
        unused_truth_strain=np.array([-2.0e-5, -3.0e-5, -2.0e-5]),
        observations=np.zeros((3, 2)),
        sensor_locations=np.array([[-1.0, 1.0]] * 3),
    )
    candidate = {
        "calibrated_k0_N_m2": 1.7e7,
        "coefficients": np.zeros(29),
        "gamma": np.ones(3),
        "train_prediction": np.zeros((3, 2)),
        "norms": {"data_norm": 0.0, "calibration_norm": 0.0, "selection_norm": 0.0},
    }
    metrics, arrays = _score_selected(path, candidate, expected_k0=1.7e7)
    assert np.isfinite(metrics["unused_load_strain_relative_L2"])
    assert metrics["load_RMSE"] == 0.0
    assert metrics["inferred_effective_winkler_loss_centroid_m"] is None
    assert metrics["imposed_interface_loss_centroid_m"] is None
    assert metrics["intact_spurious_effective_winkler_loss_area_m"] == 0.0
    assert arrays["predicted_strain"].shape == unused_x.shape


def test_source_lock_rejects_a_changed_hash():
    lock = _new_lock()
    assert {"code/campaign.py", "code/run_campaign.py"} <= set(lock["source_files"])
    first = next(iter(lock["source_files"]))
    lock["source_files"][first] = "0" * 64
    with pytest.raises(ProtocolError, match="source/design lock mismatch"):
        _check_lock(lock)


def test_missing_path_and_timeout_are_not_completion():
    paths = [{"key": path} for path in ("a", "b")]
    candidate_files = {
        "a": {f"candidate-{index}" for index in range(6)},
        "b": {f"candidate-{index}" for index in range(6)},
    }
    inventory = _path_inventory(paths, candidate_files, {"b"})
    assert inventory["missing_paths"] == ["a"]
    assert inventory["all_candidates_without_path"] == ["a"]
    assert inventory["complete"] is False
    assert _run_is_completed(
        {"status": "timeout", "completed_paths": 2}, inventory, expected_paths=2
    ) is False


def test_same_optional_metric_compares_strings_nulls_bools_and_numbers():
    reason = "no_feasible_min_successful_norm"
    assert _same_optional_metric(reason, reason) is True
    assert _same_optional_metric("selected_min_augmented_norm", reason) is False
    assert _same_optional_metric(reason, 1.0) is False
    assert _same_optional_metric(1.0, "1.0") is False
    assert _same_optional_metric(None, None) is True
    assert _same_optional_metric(None, 0.0) is False
    assert _same_optional_metric(0.0, None) is False
    assert _same_optional_metric(True, True) is True
    assert _same_optional_metric(True, False) is False
    assert _same_optional_metric(True, 1) is False
    assert _same_optional_metric(1.0, 1.0 + 1.0e-9) is True
    assert _same_optional_metric(1.0, 1.1) is False


def test_platform_metadata_is_ignored_but_real_files_are_not(tmp_path):
    assert _is_platform_metadata(".DS_Store")
    assert _is_platform_metadata("._candidate-lam-0.3.json")
    assert not _is_platform_metadata("candidate-lam-0.3.json")
    assert not _is_platform_metadata("unexpected.txt")
    (tmp_path / ".DS_Store").write_bytes(b"finder")
    (tmp_path / "._continuum_source_lock.json").write_bytes(b"appledouble")
    lock = _create_lock(tmp_path)
    assert lock["schema_version"] == "continuum-source-lock-v3.0"
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unexpected.txt").write_text("material", encoding="utf-8")
    with pytest.raises(ProtocolError, match="not empty"):
        _create_lock(occupied)


def test_partial_path_resume_reuses_valid_checkpoints_and_ignores_metadata(tmp_path):
    centres = run_continuum_study.TRAIN_CENTERS
    forward = _build_inverse_forward(centres, 1.7e7)
    points = [np.array([-4.0, 4.0])] * 3
    observations = forward.strain_and_jacobian(
        np.zeros(forward.basis.n_variables), points
    )[0].reshape(3, 2)
    dataset_path = tmp_path / "data.npz"
    np.savez(
        dataset_path,
        shape_name=np.asarray("intact"),
        score_grid=np.linspace(-12.0, 12.0, 21),
        reference_interface_d=np.zeros(21),
        unused_eval_grid=np.linspace(-12.0, 12.0, 5),
        unused_truth_strain=np.array([-2.0e-5, -3.0e-5, -4.0e-5, -3.0e-5, -2.0e-5]),
        sensor_locations=np.asarray(points),
        observations=observations,
        sigma=np.asarray(5.0e-6),
        train_centers_nominal=centres,
    )
    lock = _new_lock()
    dataset = {"key": "small", "tau": 0.01, "soil_E_Pa": 5.0e7, "sensors": 2}
    dataset_record = {
        "npz_file": "data.npz",
        "npz_sha256": run_continuum_study.hash_file(dataset_path),
        "measured_load_mean": [1.0, 1.0, 1.0],
        "physical": {
            "calibrated_inverse_k0_N_m2": 1.7e7,
        },
    }
    study = {"regularization": {"lambda_path": [0.3, 0.6], "starts": [0.05, 0.3], "max_nfev": 10}}
    path_config = {"key": "small__method-calibrated", "method": "calibrated", "dataset_key": "small"}
    path_dir = tmp_path / "paths" / path_config["key"]
    path_dir.mkdir(parents=True)
    result = _fit_one_candidate(
        forward, dataset_path, "calibrated", 0.3, study, np.ones(3), 0.01
    )
    checkpoint = _candidate_record(
        result=result,
        dataset=dataset,
        dataset_record=dataset_record,
        path_config=path_config,
        penalty_lambda=0.3,
        study=study,
        lock=lock,
    )
    run_continuum_study.write_new_json(path_dir / checkpoint["candidate_file"], checkpoint)
    apple_double = path_dir / "._candidate-lam-0.6.json"
    apple_double.write_bytes(b"appledouble")
    ds_store = path_dir / ".DS_Store"
    ds_store.write_bytes(b"finder")
    deadline = run_continuum_study.time.monotonic() + 3600.0
    completed, reused = _run_one_path(
        tmp_path,
        path_config,
        dataset=dataset,
        dataset_record=dataset_record,
        continuum={},
        study=study,
        lock=lock,
        deadline=deadline,
    )
    assert (completed, reused) == (True, True)
    assert (path_dir / run_continuum_study._candidate_filename(0.6)).is_file()
    assert (path_dir / "path.json").is_file()
    assert apple_double.read_bytes() == b"appledouble"
    assert ds_store.read_bytes() == b"finder"
    completed_again, reused_again = _run_one_path(
        tmp_path,
        path_config,
        dataset=dataset,
        dataset_record=dataset_record,
        continuum={},
        study=study,
        lock=lock,
        deadline=deadline,
    )
    assert (completed_again, reused_again) == (True, True)
    json_names = {
        item.name
        for item in path_dir.glob("*.json")
        if not _is_platform_metadata(item.name)
    }
    assert json_names == {
        "path.json",
        run_continuum_study._candidate_filename(0.3),
        run_continuum_study._candidate_filename(0.6),
    }
    stray_config = {"key": "stray", "method": "calibrated", "dataset_key": "small"}
    stray_dir = tmp_path / "paths" / "stray"
    stray_dir.mkdir()
    (stray_dir / "unexpected.txt").write_text("material", encoding="utf-8")
    with pytest.raises(ProtocolError, match="unexpected files"):
        _run_one_path(
            tmp_path,
            stray_config,
            dataset=dataset,
            dataset_record=dataset_record,
            continuum={},
            study=study,
            lock=lock,
            deadline=deadline,
        )
