"""Focused tests for the bounded regularization-endpoint diagnostic.

The tests run against the accepted B1 output read-only and root's independent
null reference (audit/null_independent.json).  All diagnostic writes go to
pytest temporary directories; no source or result file is modified.
"""
from pathlib import Path
import json
import math
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "vendor", ROOT / "code"):
    _absolute = str(_path.resolve())
    if _absolute not in sys.path:
        sys.path.insert(0, _absolute)

import run_load_information as runner  # noqa: E402
import check_regularization_endpoint as endpoint  # noqa: E402
from campaign import PhysicalModel  # noqa: E402


SOURCE = ROOT / "results_load_information"
REFERENCE = ROOT / "audit" / "null_independent.json"

pytestmark = pytest.mark.skipif(
    not (SOURCE / "verification_report.json").is_file() or not REFERENCE.is_file(),
    reason="accepted B1 source or root independent null reference is unavailable",
)

COMPAT_UPPER_KEY = (
    "shape-intact_tau-0.01_sigma-5.0000000000000004e-06_sensors-20_seed-1101__method-fixed"
)
INCOMPAT_UPPER_KEY = (
    "shape-broad_shallow_tau-0.050000000000000003_sigma-5.0000000000000004e-06"
    "_sensors-20_seed-1105__method-calibrated"
)
NONUPPER_KEY = (
    "shape-asymmetric_tau-0.050000000000000003_sigma-5.0000000000000004e-06"
    "_sensors-20_seed-1101__method-free"
)
ALL_METHODS = ("fixed", "free", "calibrated", "oracle", "overconfident", "conservative")


@pytest.fixture(scope="module")
def design():
    return runner.load_design()


@pytest.fixture(scope="module")
def study(design):
    datasets, paths, counts = runner.expand_study(design)
    return datasets, paths, counts


@pytest.fixture(scope="module")
def reference_rows():
    report = json.loads(REFERENCE.read_text(encoding="utf-8"))
    return {row["path_key"]: row for row in report["rows"]}


def _path_config(study, key):
    return next(path for path in study[1] if path["key"] == key)


def _dataset_config(study, dataset_key):
    return next(dataset for dataset in study[0] if dataset["key"] == dataset_key)


def _dataset_record(dataset_key):
    return runner.read_json(SOURCE / "datasets" / dataset_key / "dataset_record.json")


def _dataset_npz(dataset_key):
    return SOURCE / _dataset_record(dataset_key)["npz_file"]


def _fake_lock(design):
    return {
        "source_output": {"path": "results_load_information", "protocol_hash": "0" * 64},
        "design_hash": runner.hash_json(design),
        "protocol_hash": "1" * 64,
        "inherited_lambda_path": design["regularization"]["lambda_path"],
    }


def _closed_form_null(forward, dataset_path, method, observed_mean, tau):
    """Direct weighted-least-squares null formula, independent of the runner."""
    points, observations, sigmas, _ = runner._observation_inputs(dataset_path)
    unit = forward.strain_and_jacobian(np.zeros(forward.basis.n_variables), points)[0]
    inputs = runner.method_inputs(method, np.asarray(observed_mean, dtype=float), float(tau))
    mean = np.asarray(inputs["load_mean"], dtype=float)
    std = float(inputs["load_std"])
    gamma = np.empty(len(points))
    offset = 0
    for index, point in enumerate(points):
        block = slice(offset, offset + point.size)
        whitened_response = unit[block] / sigmas[index]
        whitened_target = observations[index] / sigmas[index]
        if std == 0.0:
            gamma[index] = mean[index]
        else:
            precision = 0.0 if np.isinf(std) else 1.0 / std**2
            gamma[index] = np.clip(
                (whitened_response @ whitened_target + precision * mean[index])
                / (whitened_response @ whitened_response + precision),
                0.7,
                1.3,
            )
        offset += point.size
    prediction = runner._physical_prediction(unit, points, gamma)
    residual = np.concatenate(
        [(prediction[index] - observations[index]) / sigmas[index] for index in range(len(points))]
    )
    data_squared = float(residual @ residual)
    if inputs["finite_positive_prior"]:
        calibration_squared = float(np.sum(((gamma - mean) / std) ** 2))
        prior_count = 3
    else:
        calibration_squared = 0.0
        prior_count = 0
    observation_count = int(sum(point.size for point in points))
    threshold = 1.2 * math.sqrt(observation_count + prior_count)
    augmented = math.sqrt(data_squared + calibration_squared)
    return {
        "gamma": gamma,
        "prediction": prediction,
        "data_squared": data_squared,
        "calibration_squared": calibration_squared,
        "augmented": augmented,
        "threshold": threshold,
        "compatible": augmented <= threshold,
        "prior_count": prior_count,
        "observation_count": observation_count,
    }


def test_null_endpoint_physical_prediction_all_methods_actual_fe(study, reference_rows):
    """Zero support loss keeps a nonzero elastic response under the actual loads."""
    dataset_key = _path_config(study, NONUPPER_KEY)["dataset_key"].replace(
        "shape-asymmetric", "shape-intact"
    )
    record = _dataset_record(dataset_key)
    dataset_path = SOURCE / record["npz_file"]
    forward = runner.build_forward(dataset_path, PhysicalModel())
    observed_mean = np.asarray(record["measured_load_mean"], dtype=float)
    tau = float(_dataset_config(study, dataset_key)["tau"])
    for method in ALL_METHODS:
        path_key = f"{dataset_key}__method-{method}"
        result = endpoint.null_endpoint(forward, dataset_path, method, observed_mean, tau)
        expected = _closed_form_null(forward, dataset_path, method, observed_mean, tau)

        assert result["coefficients"].shape == (endpoint.N_COEFFICIENTS,)
        assert np.all(result["coefficients"] == 0.0)
        # Coefficients are zero but the physical response is not.
        assert np.linalg.norm(result["predicted_strain"]) > 0.0
        np.testing.assert_allclose(result["predicted_strain"], expected["prediction"], rtol=1.0e-12, atol=1.0e-15)
        np.testing.assert_allclose(result["gamma"], expected["gamma"], rtol=1.0e-12, atol=1.0e-12)
        assert result["data_squared_norm"] == pytest.approx(expected["data_squared"], rel=1.0e-12)
        assert result["calibration_squared_norm"] == pytest.approx(expected["calibration_squared"], rel=1.0e-12)
        assert result["augmented_norm"] == pytest.approx(expected["augmented"], rel=1.0e-12)
        assert result["null_compatible"] == expected["compatible"]

        # The threshold counts N strain plus M calibration observations.
        assert result["N_strain_observations"] == expected["observation_count"] == 60
        assert result["M_calibration_observations"] == expected["prior_count"]
        assert result["threshold"] == pytest.approx(1.2 * math.sqrt(60 + expected["prior_count"]))

        # Root's independent direct formula agrees on amplitude, norm, class.
        row = reference_rows[path_key]
        np.testing.assert_allclose(result["gamma"], row["gamma"], rtol=1.0e-12, atol=1.0e-12)
        assert result["augmented_norm"] == pytest.approx(row["selection_norm"], rel=1.0e-12)
        assert result["null_compatible"] == row["null_compatible"]


def test_null_counts_match_root_independent_reference(study, reference_rows):
    """All 848 paths reproduce root's independent amplitudes/norms/classes."""
    datasets, paths, counts = study
    assert counts["paths"] == 848
    datasets_by_key = {dataset["key"]: dataset for dataset in datasets}
    cache = {}
    upper = 0
    null_compatible = 0
    upper_null_compatible = 0
    upper_needing_extension = 0
    compatible_shapes = set()
    for path_config in paths:
        dataset = datasets_by_key[path_config["dataset_key"]]
        if dataset["key"] not in cache:
            dataset_path = _dataset_npz(dataset["key"])
            forward = runner.build_forward(dataset_path, PhysicalModel())
            mean = np.asarray(_dataset_record(dataset["key"])["measured_load_mean"], dtype=float)
            cache[dataset["key"]] = (forward, dataset_path, mean)
        forward, dataset_path, mean = cache[dataset["key"]]
        result = endpoint.null_endpoint(
            forward, dataset_path, path_config["method"], mean, float(dataset["tau"])
        )
        row = reference_rows[path_config["key"]]
        np.testing.assert_allclose(result["gamma"], row["gamma"], rtol=1.0e-12, atol=1.0e-12)
        assert result["augmented_norm"] == pytest.approx(row["selection_norm"], rel=1.0e-12)
        assert result["data_squared_norm"] == pytest.approx(row["data_norm"] ** 2, rel=1.0e-12)
        assert result["calibration_squared_norm"] == pytest.approx(row["calibration_norm"] ** 2, rel=1.0e-9)
        assert result["null_compatible"] == row["null_compatible"]

        source_record = runner.read_json(SOURCE / "paths" / path_config["key"] / "path.json")
        assert source_record["selection"]["unbracketed_high"] == row["original_upper"]
        upper += int(row["original_upper"])
        null_compatible += int(result["null_compatible"])
        upper_null_compatible += int(row["original_upper"] and result["null_compatible"])
        upper_needing_extension += int(row["original_upper"] and not result["null_compatible"])
        if result["null_compatible"]:
            compatible_shapes.add(dataset["shape"])

    assert len(reference_rows) == 848
    assert upper == 361
    assert null_compatible == 284
    assert upper_null_compatible == 284
    assert upper_needing_extension == 77
    assert compatible_shapes == {"intact"}


def test_endpoint_decision_uses_only_observable_selection_and_null():
    """Classification reads the source selection flag and null result only."""
    base_source = {"selection": {"unbracketed_high": True}}
    base_null = {"null_compatible": True}
    decision = endpoint.endpoint_decision(base_source, base_null)
    assert decision["action"] == "lambda_infinity"
    assert decision["requires_endpoint_treatment"] is True

    assert endpoint.endpoint_decision(
        {"selection": {"unbracketed_high": True}}, {"null_compatible": False}
    )["action"] == "finite_extensions"
    assert endpoint.endpoint_decision(
        {"selection": {"unbracketed_high": False}}, {"null_compatible": True}
    )["action"] == "null_check_only"
    assert endpoint.endpoint_decision(
        {"selection": {"unbracketed_high": False}}, {"null_compatible": False}
    )["requires_endpoint_treatment"] is False

    # Extra truth/scoring fields must not influence the classification.
    decorated = endpoint.endpoint_decision(
        {"selection": {"unbracketed_high": True}, "reference_field": [1.0], "metrics": {}},
        {"null_compatible": True, "unused_response": [2.0]},
    )
    assert decorated == decision

    # Boolean protocol fields reject lookalike types.
    for bad in (None, 0, 1, "true", "false"):
        with pytest.raises(endpoint.EndpointProtocolError):
            endpoint.endpoint_decision({"selection": {"unbracketed_high": bad}}, base_null)
        with pytest.raises(endpoint.EndpointProtocolError):
            endpoint.endpoint_decision(base_source, {"null_compatible": bad})


def _lambda_infinity_case(study, design, output_dir):
    """One real lambda-infinity endpoint record written under output_dir."""
    path_config = _path_config(study, COMPAT_UPPER_KEY)
    dataset = _dataset_config(study, path_config["dataset_key"])
    dataset_path = _dataset_npz(dataset["key"])
    dataset_record = _dataset_record(dataset["key"])
    path_record = runner.read_json(SOURCE / "paths" / path_config["key"] / "path.json")
    forward = runner.build_forward(dataset_path, PhysicalModel())
    lock = _fake_lock(design)

    result = endpoint.null_endpoint(
        forward,
        dataset_path,
        path_config["method"],
        np.asarray(dataset_record["measured_load_mean"], dtype=float),
        float(dataset["tau"]),
    )
    assert result["null_compatible"] is True
    null_record = endpoint._null_record(
        result,
        dataset=dataset,
        path_config=path_config,
        dataset_record=dataset_record,
        path_record=path_record,
        lock=lock,
    )
    record = endpoint._endpoint_record(
        null_record=null_record,
        source_candidates=endpoint._load_source_candidates(SOURCE, path_config, design),
        extension_candidates=[],
        path_config=path_config,
        dataset=dataset,
        dataset_path=dataset_path,
        output_dir=output_dir,
        lock=lock,
    )
    return record, result, null_record, dataset, path_config


def test_lambda_infinity_endpoint_scores_zero_field_with_fitted_gamma(study, design, tmp_path):
    """The admissible H1 endpoint keeps zero coefficients and a nonzero response."""
    from run_campaign import hash_file

    record, result, null_record, dataset, path_config = _lambda_infinity_case(study, design, tmp_path)
    lock = _fake_lock(design)

    assert record["action"] == "lambda_infinity"
    assert record["selection"]["lambda_infinity_endpoint"] is True
    assert record["selection"]["selected_lambda"] == "infinity"
    assert record["selection"]["still_unbracketed_high"] is False
    assert record["selected_start"] == "null_endpoint"
    assert np.all(np.asarray(record["selected_coefficients"]) == 0.0)
    np.testing.assert_allclose(record["selected_gamma"], result["gamma"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        record["selected_train_prediction"], result["predicted_strain"], rtol=0.0, atol=0.0
    )
    assert np.linalg.norm(record["selected_train_prediction"]) > 0.0

    # Scoring is post hoc: zero support-loss field on an intact reference.  The
    # unused-load prediction is the intact elastic response (nonzero), so the
    # relative unused error is the noiseless discretization difference between
    # the fitted gamma response and the unused response, not one.
    assert record["posthoc_scoring_only"] is True
    assert record["metrics"]["loss_area_m"] == pytest.approx(0.0, abs=1.0e-15)
    assert record["metrics"]["loss_area_absolute_error_m"] == pytest.approx(0.0, abs=1.0e-15)
    assert record["metrics"]["support_rel_L2_except_intact"] is None
    assert 0.0 < record["metrics"]["unused_load_strain_relative_L2"] < 0.01

    arrays_path = tmp_path / record["arrays_file"]
    assert arrays_path.is_file()
    assert hash_file(arrays_path) == record["arrays_sha256"]
    with np.load(arrays_path, allow_pickle=False) as saved:
        assert set(saved.files) == {
            "score_x", "reference_d", "estimated_d", "unused_x",
            "reference_strain", "predicted_strain", "coefficients",
            "gamma", "train_prediction",
        }
        assert np.all(saved["estimated_d"] == 0.0)
        assert np.all(saved["coefficients"] == 0.0)
        assert np.linalg.norm(saved["predicted_strain"]) > 0.0
        np.testing.assert_allclose(saved["gamma"], result["gamma"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(saved["train_prediction"], result["predicted_strain"])

    errors = endpoint._verify_endpoint_record(
        record,
        source_dir=SOURCE,
        output_dir=tmp_path,
        dataset=dataset,
        path_config=path_config,
        null_record=null_record,
        extension_candidates=[],
        design=design,
        lock={key: value for key, value in lock.items() if key != "inherited_lambda_path"},
    )
    assert errors == []


def test_verify_rejects_missing_endpoint_arrays_without_recreation(study, design, tmp_path):
    """Missing endpoint evidence fails verification and is never recreated."""
    record, _, null_record, dataset, path_config = _lambda_infinity_case(study, design, tmp_path)
    verify_lock = {
        key: value for key, value in _fake_lock(design).items() if key != "inherited_lambda_path"
    }

    def verify():
        return endpoint._verify_endpoint_record(
            record,
            source_dir=SOURCE,
            output_dir=tmp_path,
            dataset=dataset,
            path_config=path_config,
            null_record=null_record,
            extension_candidates=[],
            design=design,
            lock=verify_lock,
        )

    arrays_path = tmp_path / record["arrays_file"]
    assert arrays_path.is_file()
    assert verify() == []  # valid branch: recorded evidence replays cleanly

    # Deleting the canonical arrays file must fail verification, and the
    # verifier must not recreate the missing evidence.
    original = arrays_path.read_bytes()
    arrays_path.unlink()
    errors = verify()
    assert errors and any("arrays" in error for error in errors)
    assert not arrays_path.exists()

    # A hash-tampered arrays file is rejected and left byte-identical.
    arrays_path.write_bytes(original)
    tampered = original[:-1] + bytes([original[-1] ^ 0xFF])
    arrays_path.write_bytes(tampered)
    errors = verify()
    assert errors and any("hash" in error for error in errors)
    assert arrays_path.read_bytes() == tampered


def test_verify_rejects_noncanonical_arrays_alias_without_recreation(study, design, tmp_path):
    """A record pointing to another existing file with a valid recorded hash is rejected.

    The alias file exists and its bytes match the recorded arrays hash, but the
    record does not name the canonical endpoints/<key>/endpoint_arrays.npz path.
    Verification must reject the record before any access or replay, must not
    recreate the missing canonical file, and must leave the alias untouched.
    """
    record, _, null_record, dataset, path_config = _lambda_infinity_case(study, design, tmp_path)
    verify_lock = {
        key: value for key, value in _fake_lock(design).items() if key != "inherited_lambda_path"
    }

    canonical_path = tmp_path / record["arrays_file"]
    original = canonical_path.read_bytes()
    alias = tmp_path / "alias_arrays.npz"
    alias.write_bytes(original)
    alias_record = dict(record)
    alias_record["arrays_file"] = alias.relative_to(tmp_path).as_posix()
    canonical_path.unlink()

    errors = endpoint._verify_endpoint_record(
        alias_record,
        source_dir=SOURCE,
        output_dir=tmp_path,
        dataset=dataset,
        path_config=path_config,
        null_record=null_record,
        extension_candidates=[],
        design=design,
        lock=verify_lock,
    )
    assert errors and any("canonical" in error for error in errors)
    assert not canonical_path.exists()
    assert alias.read_bytes() == original


def test_source_manifest_stable_under_reverification_timestamp(design, monkeypatch):
    """A B1 re-verification timestamp update keeps the B2 source identity."""
    baseline = endpoint._source_manifest(SOURCE, design)
    assert "verification_report_semantic_sha256" in baseline

    report_path = SOURCE / "verification_report.json"
    real_read_json = endpoint.read_json

    def install(**changes):
        def fake_read_json(path):
            record = real_read_json(path)
            if Path(path) == report_path:
                record = {**record, **changes}
            return record

        monkeypatch.setattr(endpoint, "read_json", fake_read_json)

    # Only verified_utc changes after a valid B1 re-verification: the source
    # identity and every recorded hash stay exactly the same.
    install(verified_utc="2099-01-01T00:00:00.000000Z")
    assert endpoint._source_manifest(SOURCE, design) == baseline

    # A changed verification status or count is still rejected outright.
    for changes in (
        {"passed": False},
        {"incomplete": True},
        {"checked_candidates": 5087},
        {"checked_completed_paths": 847},
    ):
        install(**changes)
        with pytest.raises(endpoint.EndpointProtocolError, match="not complete and passed"):
            endpoint._source_manifest(SOURCE, design)

    # The excluded timestamp field itself must still be a JSON string.
    install(verified_utc=None)
    with pytest.raises(endpoint.EndpointProtocolError, match="verified_utc"):
        endpoint._source_manifest(SOURCE, design)


@pytest.fixture(scope="module")
def real_extension_fit(study, design):
    """One real extension fit reused by the candidate-record tests."""
    path_config = _path_config(study, INCOMPAT_UPPER_KEY)
    dataset = _dataset_config(study, path_config["dataset_key"])
    dataset_path = _dataset_npz(dataset["key"])
    dataset_record = _dataset_record(dataset["key"])
    forward = runner.build_forward(dataset_path, PhysicalModel())
    penalty_lambda = endpoint.EXTENSION_LAMBDAS[0]
    result = runner._fit_one_candidate(
        forward,
        dataset_path,
        path_config["method"],
        penalty_lambda,
        design,
        np.asarray(dataset_record["measured_load_mean"], dtype=float),
        float(dataset["tau"]),
    )
    record = endpoint._extension_candidate_record(
        result,
        dataset=dataset,
        path_config=path_config,
        dataset_record=dataset_record,
        path_record=runner.read_json(SOURCE / "paths" / path_config["key"] / "path.json"),
        penalty_lambda=penalty_lambda,
        design=design,
        lock=_fake_lock(design),
    )
    return path_config, dataset, penalty_lambda, result, record


def test_extension_candidate_replays_objective_on_real_forward(real_extension_fit, design):
    path_config, dataset, penalty_lambda, result, record = real_extension_fit
    assert record["candidate_origin"] == "b2_finite_extension"
    assert record["penalty_lambda"] == pytest.approx(3.0e4)
    assert penalty_lambda not in design["regularization"]["lambda_path"]
    assert len(record["starts"]) == 2
    assert type(record["successful_start"]) is bool
    norms = record["norms"]
    assert record["objective"] == pytest.approx(
        norms["selection_norm"] ** 2 + norms["penalty_norm"] ** 2, rel=1.0e-12
    )
    # The stored train prediction is gamma_j times the fitted unit strain.
    assert np.linalg.norm(record["train_prediction"]) > 0.0
    verification = endpoint._verify_extension_candidate(
        record,
        source_dir=SOURCE,
        dataset=dataset,
        path_config=path_config,
        penalty_lambda=penalty_lambda,
        design=design,
        lock=_fake_lock(design),
    )
    assert verification["passed"] is True
    assert verification["errors"] == []
    assert verification["recomputed"]["objective"] == pytest.approx(record["objective"], rel=1.0e-8)


def test_extension_candidate_rejects_missing_or_modified_inputs(real_extension_fit, design):
    path_config, dataset, penalty_lambda, _, record = real_extension_fit

    def roundtrip(candidate_record):
        return json.loads(runner.canonical_json(runner.safe_json(candidate_record)))

    def verify(candidate_record, lam=penalty_lambda):
        return endpoint._verify_extension_candidate(
            candidate_record,
            source_dir=SOURCE,
            dataset=dataset,
            path_config=path_config,
            penalty_lambda=lam,
            design=design,
            lock=_fake_lock(design),
        )

    tampered = roundtrip(record)
    tampered["load_mean"] = [float(value) + 0.01 for value in record["load_mean"]]
    assert any("identity mismatch: load_mean" in error for error in verify(tampered)["errors"])

    tampered = roundtrip(record)
    tampered["coefficients"][0] += 0.1
    assert any("array hash mismatch" in error for error in verify(tampered)["errors"])

    tampered = roundtrip(record)
    tampered["train_prediction"][0][0] += 1.0e-9
    outcome = verify(tampered)
    assert outcome["passed"] is False

    # An inherited-path lambda is not a legitimate extension candidate.
    inherited = roundtrip(record)
    outcome = verify(inherited, lam=float(design["regularization"]["lambda_path"][-1]))
    assert any("falsely belongs to the inherited path" in error for error in outcome["errors"])

    # Wrong scalar types fail instead of being coerced.
    tampered = roundtrip(record)
    tampered["successful_start"] = "yes"
    assert verify(tampered)["passed"] is False


def test_combined_selection_ignores_failed_candidates():
    def candidate(lam, norm, successful=True):
        return {
            "penalty_lambda": lam,
            "successful_start": successful,
            "finite_positive_prior": True,
            "norms": {"selection_norm": norm, "data_norm": norm, "calibration_norm": 0.0},
        }

    source = [candidate(lam, 9.0) for lam in (0.03, 0.3, 3.0, 30.0, 300.0, 3000.0)]
    threshold = 1.2 * math.sqrt(63)

    # A failed extension with a tiny norm must never be selected.
    extensions = [
        candidate(3.0e4, 9.1),
        candidate(3.0e5, 9.3),
        candidate(3.0e6, 9.4),
        candidate(3.0e7, 0.1, successful=False),
    ]
    selection = endpoint._combined_selection(source, extensions, sensors=20)
    assert selection["threshold"] == pytest.approx(threshold)
    assert selection["selected_lambda"] == 3.0e6
    assert selection["selected_origin"] == "extension"
    assert selection["still_unbracketed_high"] is False
    assert selection["optimizer_failed"] is False

    # A successful compatible 3e7 still reports the upper end as unbracketed.
    extensions[-1] = candidate(3.0e7, 9.5)
    selection = endpoint._combined_selection(source, extensions, sensors=20)
    assert selection["selected_lambda"] == 3.0e7
    assert selection["still_unbracketed_high"] is True

    # Without a compatible candidate the smallest successful norm is kept.
    hard_source = [candidate(lam, 20.0 + index) for index, lam in enumerate((0.03, 0.3, 3.0, 30.0, 300.0, 3000.0))]
    hard_extensions = [candidate(lam, 30.0) for lam in endpoint.EXTENSION_LAMBDAS]
    selection = endpoint._combined_selection(hard_source, hard_extensions, sensors=20)
    assert selection["no_feasible_candidate"] is True
    assert selection["selected_lambda"] == 0.03

    # Candidates from different prior regimes are never mixed.
    mixed = [candidate(0.03, 9.0), {**candidate(3.0e4, 9.1), "finite_positive_prior": False}]
    with pytest.raises(runner.ProtocolError):
        endpoint._combined_selection(mixed, [], sensors=20)


def _contains_absolute_string(value):
    if isinstance(value, dict):
        return any(_contains_absolute_string(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_string(child) for child in value)
    return isinstance(value, str) and value.startswith("/")


def _file_digests(output_dir):
    from run_campaign import hash_file

    return {
        path.relative_to(output_dir).as_posix(): hash_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and not endpoint._is_system_metadata(path.name)
    }


def test_freeze_run_verify_resume_bounded_subset(study, design, tmp_path):
    output = tmp_path / "endpoint_bounded"
    keys = [COMPAT_UPPER_KEY, INCOMPAT_UPPER_KEY, NONUPPER_KEY]

    lock = endpoint.freeze_protocol(output, SOURCE, design)
    assert lock["schema_version"] == "regularization-endpoint-lock-v1.0"
    assert set(lock["new_sources"]) == {"code/check_regularization_endpoint.py"}
    assert set(lock["new_tests"]) == {"tests/test_regularization_endpoint.py"}
    assert lock["addon_rule"]["extension_lambda_path_m"] == [3.0e4, 3.0e5, 3.0e6, 3.0e7]
    assert lock["source_output"]["path"] == "results_load_information"

    # A second freeze on a nonempty output refuses.
    with pytest.raises(endpoint.EndpointProtocolError, match="not empty"):
        endpoint.freeze_protocol(output, SOURCE, design)

    status = endpoint.execute_run(output, SOURCE, design, resume=False, path_keys=keys)
    assert status["status"] == "completed"
    assert status["completed_paths"] == 3
    # The recorded scope follows study order, not request order.
    expected_order = [path["key"] for path in study[1] if path["key"] in set(keys)]
    assert status["selected_path_keys"] == expected_order

    with pytest.raises(endpoint.EndpointProtocolError, match="use --resume"):
        endpoint.execute_run(output, SOURCE, design, resume=False, path_keys=keys)

    null_dir = output / "null_checks"
    assert {path.parent.name for path in null_dir.glob("*/null_check.json")} == set(keys)
    compatible_null = runner.read_json(null_dir / COMPAT_UPPER_KEY / "null_check.json")
    incompatible_null = runner.read_json(null_dir / INCOMPAT_UPPER_KEY / "null_check.json")
    nonupper_null = runner.read_json(null_dir / NONUPPER_KEY / "null_check.json")
    assert compatible_null["null_compatible"] is True
    assert incompatible_null["null_compatible"] is False
    assert nonupper_null["null_compatible"] is False
    assert type(compatible_null["null_compatible"]) is bool
    assert nonupper_null["load_std"] is None  # free method: infinite std as JSON null
    assert type(incompatible_null["load_std"]) is float

    endpoint_dir = output / "endpoints"
    assert not (endpoint_dir / NONUPPER_KEY).exists()
    compatible_endpoint = runner.read_json(endpoint_dir / COMPAT_UPPER_KEY / "endpoint.json")
    assert compatible_endpoint["action"] == "lambda_infinity"
    assert np.all(np.asarray(compatible_endpoint["selected_coefficients"]) == 0.0)
    assert np.linalg.norm(compatible_endpoint["selected_train_prediction"]) > 0.0
    np.testing.assert_allclose(
        compatible_endpoint["selected_train_prediction"], compatible_null["predicted_strain"]
    )
    extension_files = sorted((endpoint_dir / INCOMPAT_UPPER_KEY).glob("candidate_lambda_*.json"))
    assert len(extension_files) == 4
    incompatible_endpoint = runner.read_json(endpoint_dir / INCOMPAT_UPPER_KEY / "endpoint.json")
    assert incompatible_endpoint["action"] == "finite_extensions"
    selected_lambda = incompatible_endpoint["selection"]["selected_lambda"]
    all_lambdas = list(design["regularization"]["lambda_path"]) + list(endpoint.EXTENSION_LAMBDAS)
    assert selected_lambda in all_lambdas
    selected_candidates = {
        record["penalty_lambda"]: record
        for record in endpoint._load_source_candidates(
            SOURCE, _path_config(study, INCOMPAT_UPPER_KEY), design
        )
    }
    for path in extension_files:
        record = runner.read_json(path)
        selected_candidates[record["penalty_lambda"]] = record
    chosen = selected_candidates[selected_lambda]
    assert chosen["successful_start"] is True
    np.testing.assert_allclose(
        incompatible_endpoint["selected_train_prediction"], chosen["train_prediction"]
    )
    np.testing.assert_allclose(incompatible_endpoint["selected_gamma"], chosen["gamma"])

    summary = runner.read_json(output / "summary.json")
    assert summary["scope"] == "limited"
    assert summary["counts"]["null_checks"] == 3
    assert summary["counts"]["null_compatible"] == 1
    assert summary["counts"]["nonupper_null_compatible"] == 0
    assert summary["counts"]["originally_unbracketed_high"] == 2
    assert summary["counts"]["finite_extensions"] == 1
    assert summary["counts"]["null_compatible_posthoc_by_shape"] == {"intact": 1}

    report = endpoint.verify_protocol(output, SOURCE, design)
    assert report["passed"] is True
    assert report["incomplete"] is True  # bounded subset of the 848 paths
    assert report["checked_null_checks"] == 3
    assert report["checked_extension_candidates"] == 4
    assert report["checked_endpoint_treatments"] == 2
    assert report["errors"] == {}

    for json_file in output.rglob("*.json"):
        assert not _contains_absolute_string(runner.read_json(json_file))

    # Resume reuses verified records and leaves every byte untouched.
    before = _file_digests(output)
    resumed = endpoint.execute_run(output, SOURCE, design, resume=True, path_keys=keys)
    assert resumed["status"] == "completed"
    assert resumed["completed_paths"] == 3
    assert resumed["reused_paths"] == 5  # three null checks and two endpoints
    after = _file_digests(output)
    changed = {
        name
        for name in before
        if before[name] != after.get(name) and name != "run_status.json"
    }
    assert changed == set()
    assert set(after) == set(before)

    # A same-count but different key set is a scope change, not a resume.
    other = NONUPPER_KEY.replace("seed-1101", "seed-1102")
    with pytest.raises(endpoint.EndpointProtocolError, match="keys differ"):
        endpoint.execute_run(
            output, SOURCE, design, resume=True, path_keys=[COMPAT_UPPER_KEY, INCOMPAT_UPPER_KEY, other]
        )
    with pytest.raises(endpoint.EndpointProtocolError, match="unknown --path-keys"):
        endpoint.execute_run(output, SOURCE, design, resume=True, path_keys=["no-such-path"])

    # A tampered null response is caught by the independent physical replay.
    null_path = null_dir / COMPAT_UPPER_KEY / "null_check.json"
    original_bytes = null_path.read_bytes()
    tampered = json.loads(original_bytes)
    tampered["predicted_strain"][0][0] += 1.0e-9
    null_path.write_text(json.dumps(tampered, allow_nan=False) + "\n", encoding="utf-8")
    report = endpoint.verify_protocol(output, SOURCE, design)
    assert report["passed"] is False
    flat = [error for errors in report["errors"].values() for error in errors]
    assert any("predicted strain" in error or "array_hash" in error for error in flat)
    with pytest.raises(endpoint.EndpointProtocolError, match="failed"):
        endpoint.execute_run(output, SOURCE, design, resume=True, path_keys=keys)
    null_path.write_bytes(original_bytes)
    # The rejected attempt marked the run status as error; a clean resume
    # reverifies every record and restores the completed status.
    recovered = endpoint.execute_run(output, SOURCE, design, resume=True, path_keys=keys)
    assert recovered["status"] == "completed"
    assert recovered["reused_paths"] == 5
    assert endpoint.verify_protocol(output, SOURCE, design)["passed"] is True

    # Missing endpoint evidence fails verification and is never recreated.
    arrays_path = endpoint_dir / COMPAT_UPPER_KEY / "endpoint_arrays.npz"
    original_arrays = arrays_path.read_bytes()
    arrays_path.unlink()
    report = endpoint.verify_protocol(output, SOURCE, design)
    assert report["passed"] is False
    assert not arrays_path.exists()
    flat = [error for errors in report["errors"].values() for error in errors]
    assert any("arrays" in error for error in flat)
    arrays_path.write_bytes(original_arrays)
    assert endpoint.verify_protocol(output, SOURCE, design)["passed"] is True


def test_cli_parser_accepts_bounded_path_keys():
    args = endpoint._parser().parse_args(
        ["run", "--output", "results_regularization_endpoint", "--path-keys", "a", "b"]
    )
    assert args.path_keys == ["a", "b"]
    args = endpoint._parser().parse_args(["verify", "--output", "results_regularization_endpoint"])
    assert args.command == "verify"
