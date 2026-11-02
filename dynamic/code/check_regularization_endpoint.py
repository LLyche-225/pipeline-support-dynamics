"""Bounded diagnostic for the finite upper end of the load-information path.

The runner consumes the accepted B1 output read-only.  It never adds the new
penalties to the frozen main path; they are stored as an explicitly inherited
extension used only for originally upper-selected paths.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "vendor", ROOT / "code"):
    _absolute = str(_path.resolve())
    if _absolute not in sys.path:
        sys.path.insert(0, _absolute)

from beam_inverse import profile_load_amplitude  # noqa: E402
from campaign import PhysicalModel  # noqa: E402
from run_campaign import hash_file, score_selected  # noqa: E402
from run_load_information import (  # noqa: E402
    OBJECTIVE_ATOL,
    OBJECTIVE_RTOL,
    ProtocolError,
    SOURCE_FILES,
    TEST_FILES,
    VENDOR_FILES,
    _candidate_filename,
    _fit_one_candidate,
    _is_system_metadata,
    _load_arrays,
    _numeric_array_hash,
    _observation_inputs,
    _physical_prediction,
    _selection_details,
    _verify_candidate_record,
    _write_or_reuse_arrays,
    build_forward,
    canonical_json,
    current_environment,
    expand_study,
    hash_json,
    load_and_check_lock,
    load_design,
    mapped_hashes,
    method_inputs,
    output_file,
    read_json,
    safe_json,
    validate_relative_output,
    write_new_json,
    write_replace_json,
)


LOCK_FILE = "regularization_endpoint_lock.json"
DEFAULT_SOURCE = "results_load_information"
NEW_SOURCE_FILES = ("code/check_regularization_endpoint.py",)
NEW_TEST_FILES = ("tests/test_regularization_endpoint.py",)
EXTENSION_LAMBDAS = (3.0e4, 3.0e5, 3.0e6, 3.0e7)
EXTENSION_STARTS = (0.05, 0.30)
EXTENSION_MAX_NFEV = 200
WALL_BUDGET_SECONDS = 30.0 * 60.0
N_COEFFICIENTS = 29

ADDON_RULE = {
    "all_source_paths_null_checked": True,
    "null_coefficients": [0.0] * N_COEFFICIENTS,
    "null_physical_prediction": (
        "gamma_j times the intact unit strain; zero support loss still carries "
        "nonzero elastic response under the actual loads"
    ),
    "null_selection": "method amplitudes only; exact fixed/oracle, profiled free/finite prior",
    "null_compatibility": "augmented_norm <= 1.2*sqrt(N+M)",
    "additional_fits": "only originally unbracketed_high and null-incompatible paths",
    "extension_lambda_path_m": list(EXTENSION_LAMBDAS),
    "extension_starts": list(EXTENSION_STARTS),
    "extension_max_nfev": EXTENSION_MAX_NFEV,
    "selection": "admissible lambda-infinity zero field, else largest successful compatible lambda among old six plus four extensions; otherwise minimum successful augmented norm",
    "truth_use": "original and endpoint reference fields, unused response, and gamma are post hoc scoring only",
}


class EndpointProtocolError(RuntimeError):
    """A source, output, or diagnostic record violates the frozen rule."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise EndpointProtocolError(f"{name} must be a JSON boolean")
    return value


def _require_string(value: Any, name: str) -> str:
    if type(value) is not str:
        raise EndpointProtocolError(f"{name} must be a JSON string")
    return value


def _require_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise EndpointProtocolError(f"{name} must be a JSON integer")
    return value


def _same_number(replay: float, reported: Any, name: str, rtol: float, atol: float) -> bool:
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        return False
    return math.isclose(float(replay), float(reported), rel_tol=rtol, abs_tol=atol)


def _source_path_record(source_output: Path, path_key: str) -> Path:
    return source_output / "paths" / path_key / "path.json"


def _source_dataset_record(source_output: Path, dataset_key: str) -> Path:
    return source_output / "datasets" / dataset_key / "dataset_record.json"


def _source_manifest(source_output: Path, design: dict) -> dict:
    datasets, paths, counts = expand_study(design)
    if counts["datasets"] != 200 or counts["paths"] != 848:
        raise EndpointProtocolError("source design does not expand to the accepted B1 study")
    source_lock = load_and_check_lock(source_output, design)
    report_path = source_output / "verification_report.json"
    if not report_path.is_file():
        raise EndpointProtocolError("accepted source verification report is missing")
    report = read_json(report_path)
    if (
        report.get("schema_version") != "load-information-verification-v3.0"
        or report.get("passed") is not True
        or report.get("incomplete") is not False
        or _require_int(report.get("checked_candidates"), "source checked_candidates") != 5088
        or _require_int(report.get("checked_completed_paths"), "source checked_completed_paths") != 848
    ):
        raise EndpointProtocolError("accepted source verification report is not complete and passed")
    _require_string(report.get("verified_utc"), "source verified_utc")
    # Re-running the accepted B1 verifier refreshes only verified_utc; the
    # source identity must stay stable under that timestamp-only update, so the
    # manifest hashes the report's semantic content excluding verified_utc.
    verification_report_identity = hash_json(
        {key: value for key, value in report.items() if key != "verified_utc"}
    )

    dataset_hashes = {}
    for dataset in datasets:
        record_path = _source_dataset_record(source_output, dataset["key"])
        if not record_path.is_file():
            raise EndpointProtocolError(f"source dataset record is missing: {dataset['key']}")
        record = read_json(record_path)
        if record.get("dataset_key") != dataset["key"] or record.get("config_hash") != dataset["config_hash"]:
            raise EndpointProtocolError(f"source dataset identity mismatch: {dataset['key']}")
        npz_path = source_output / _require_string(record.get("npz_file"), "source npz_file")
        if not npz_path.is_file() or hash_file(npz_path) != record.get("npz_sha256"):
            raise EndpointProtocolError(f"source dataset hash mismatch: {dataset['key']}")
        dataset_hashes[dataset["key"]] = hash_file(record_path)

    path_hashes = {}
    candidate_hashes = {}
    for path_config in paths:
        record_path = _source_path_record(source_output, path_config["key"])
        if not record_path.is_file():
            raise EndpointProtocolError(f"source path record is missing: {path_config['key']}")
        record = read_json(record_path)
        if (
            record.get("path_key") != path_config["key"]
            or record.get("dataset_key") != path_config["dataset_key"]
            or record.get("method") != path_config["method"]
            or record.get("selected") is not True
        ):
            raise EndpointProtocolError(f"source path identity mismatch: {path_config['key']}")
        path_hashes[path_config["key"]] = hash_file(record_path)
        for penalty_lambda in design["regularization"]["lambda_path"]:
            candidate_path = record_path.parent / _candidate_filename(penalty_lambda)
            if not candidate_path.is_file():
                raise EndpointProtocolError(
                    f"source candidate is missing: {path_config['key']} lambda={penalty_lambda}"
                )
            candidate_hashes[candidate_path.relative_to(source_output).as_posix()] = hash_file(candidate_path)

    return {
        "path": source_output.relative_to(ROOT).as_posix(),
        "lock_sha256": hash_file(source_output / "load_information_lock.json"),
        "verification_report_semantic_sha256": verification_report_identity,
        "design_hash": source_lock["design_hash"],
        "protocol_hash": source_lock["protocol_hash"],
        "dataset_record_hashes": dataset_hashes,
        "path_record_hashes": path_hashes,
        "candidate_record_hashes": candidate_hashes,
    }


def _protocol_fingerprint(lock: dict) -> str:
    return hash_json(
        {
            "schema_version": lock["schema_version"],
            "source_output": lock["source_output"],
            "design_hash": lock["design_hash"],
            "inherited_sources": lock["inherited_sources"],
            "new_sources": lock["new_sources"],
            "new_tests": lock["new_tests"],
            "vendor": lock["vendor"],
            "environment_provenance": lock["environment_provenance"],
            "thread_settings": lock["thread_settings"],
            "addon_rule": lock["addon_rule"],
        }
    )


def freeze_protocol(output_dir: Path, source_dir: Path, design: dict) -> dict:
    source_manifest = _source_manifest(source_dir, design)
    inherited = mapped_hashes(SOURCE_FILES)
    inherited_tests = mapped_hashes(TEST_FILES)
    new_sources = mapped_hashes(NEW_SOURCE_FILES)
    new_tests = mapped_hashes(NEW_TEST_FILES)
    vendor = {
        "version": current_environment()["skfem_version"],
        "files": mapped_hashes(VENDOR_FILES),
    }
    environment = current_environment()
    import os

    lock = {
        "schema_version": "regularization-endpoint-lock-v1.0",
        "created_utc": utc_now(),
        "source_output": source_manifest,
        "design_file": "study_design_v3.json",
        "design_hash": hash_json(design),
        "inherited_sources": inherited,
        "inherited_tests": inherited_tests,
        "new_sources": new_sources,
        "new_tests": new_tests,
        "vendor": vendor,
        "environment_provenance": environment,
        "thread_settings": {
            name: os.environ.get(name, "unset")
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "addon_rule": ADDON_RULE,
    }
    lock["protocol_hash"] = _protocol_fingerprint(lock)
    if output_dir.exists() and any(
        not _is_system_metadata(item.name) for item in output_dir.iterdir()
    ):
        raise EndpointProtocolError("freeze output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_new_json(output_dir / LOCK_FILE, lock)
    return lock


def load_and_check_endpoint_lock(output_dir: Path, source_dir: Path, design: dict) -> dict:
    path = output_dir / LOCK_FILE
    if not path.is_file():
        raise EndpointProtocolError(f"missing diagnostic lock: {path.relative_to(ROOT).as_posix()}")
    lock = read_json(path)
    if lock.get("schema_version") != "regularization-endpoint-lock-v1.0":
        raise EndpointProtocolError("unsupported regularization-endpoint lock schema")
    expected = {
        "design_hash": hash_json(design),
        "inherited_sources": mapped_hashes(SOURCE_FILES),
        "inherited_tests": mapped_hashes(TEST_FILES),
        "new_sources": mapped_hashes(NEW_SOURCE_FILES),
        "new_tests": mapped_hashes(NEW_TEST_FILES),
        "vendor": {
            "version": current_environment()["skfem_version"],
            "files": mapped_hashes(VENDOR_FILES),
        },
        "environment_provenance": current_environment(),
        "addon_rule": ADDON_RULE,
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise EndpointProtocolError(f"diagnostic lock mismatch for {key}")
    # This also checks that every inherited numerical source still matches B1.
    source_manifest = _source_manifest(source_dir, design)
    if lock.get("source_output") != source_manifest:
        raise EndpointProtocolError("source output manifest mismatch")
    if lock.get("protocol_hash") != _protocol_fingerprint(lock):
        raise EndpointProtocolError("diagnostic protocol hash mismatch")
    return lock


def _inputs_for_path(
    source_dir: Path,
    dataset: dict,
    path_config: dict,
) -> tuple[dict, dict, Path, dict]:
    dataset_record = read_json(_source_dataset_record(source_dir, dataset["key"]))
    path_record = read_json(_source_path_record(source_dir, path_config["key"]))
    dataset_path = source_dir / _require_string(dataset_record.get("npz_file"), "dataset npz_file")
    inputs = method_inputs(
        path_config["method"],
        np.asarray(dataset_record["measured_load_mean"], dtype=float),
        float(dataset["tau"]),
    )
    return dataset_record, path_record, dataset_path, inputs


def null_endpoint(
    forward,
    dataset_path: Path,
    method: str,
    observed_mean: np.ndarray,
    tau: float,
) -> dict:
    """Evaluate the zero-loss endpoint with its nonzero physical response.

    Zero coefficients mean an intact foundation, not zero strain: under the
    actual loads the train prediction is gamma_j times the intact unit strain.
    """
    points, observations, sigmas, _ = _observation_inputs(dataset_path)
    inputs = method_inputs(method, observed_mean, tau)
    coefficients = np.zeros(forward.basis.n_variables, dtype=float)
    unit_prediction = forward.strain_and_jacobian(coefficients, points)[0]
    gamma = []
    calibration_squared = 0.0
    offset = 0
    for index, point in enumerate(points):
        count = point.size
        response = unit_prediction[offset : offset + count] / sigmas[index]
        target = observations[index] / sigmas[index]
        if inputs["gamma_mode"] == "profile":
            value, _, prior_residual, _ = profile_load_amplitude(
                response,
                np.zeros((count, forward.basis.n_variables), dtype=float),
                target,
                float(inputs["load_mean"][index]),
                float(inputs["load_std"]),
            )
            gamma.append(float(value))
            if inputs["finite_positive_prior"]:
                calibration_squared += float(prior_residual**2)
        else:
            gamma.append(float(inputs["load_mean"][index]))
        offset += count
    gamma = np.asarray(gamma, dtype=float)
    prediction = _physical_prediction(unit_prediction, points, gamma)
    data_residual = np.concatenate(
        [(prediction[index] - observations[index]) / sigmas[index] for index in range(len(points))]
    )
    data_squared = float(data_residual @ data_residual)
    observation_count = int(sum(point.size for point in points))
    prior_count = 3 if inputs["finite_positive_prior"] else 0
    threshold = 1.2 * math.sqrt(observation_count + prior_count)
    augmented_squared = data_squared + calibration_squared
    augmented_norm = math.sqrt(augmented_squared)
    return {
        "coefficients": coefficients,
        "gamma": gamma,
        "predicted_strain": prediction,
        "inputs": inputs,
        "data_squared_norm": data_squared,
        "calibration_squared_norm": float(calibration_squared),
        "augmented_squared_norm": augmented_squared,
        "augmented_norm": augmented_norm,
        "threshold": threshold,
        "null_compatible": bool(augmented_norm <= threshold),
        "N_strain_observations": observation_count,
        "M_calibration_observations": prior_count,
    }


def _null_record(
    result: dict,
    *,
    dataset: dict,
    path_config: dict,
    dataset_record: dict,
    path_record: dict,
    lock: dict,
) -> dict:
    inputs = result["inputs"]
    arrays = {
        "coefficients": result["coefficients"],
        "gamma": result["gamma"],
        "predicted_strain": result["predicted_strain"],
    }
    return {
        "schema_version": "regularization-endpoint-null-v1.0",
        "path_key": path_config["key"],
        "dataset_key": dataset["key"],
        "method": path_config["method"],
        "originally_unbracketed_high": _require_bool(
            path_record["selection"]["unbracketed_high"], "source unbracketed_high"
        ),
        "coefficients": arrays["coefficients"].tolist(),
        "gamma": arrays["gamma"].tolist(),
        "predicted_strain": arrays["predicted_strain"].tolist(),
        "gamma_mode": inputs["gamma_mode"],
        "load_mean": inputs["load_mean"].tolist(),
        "load_std": None if np.isinf(inputs["load_std"]) else float(inputs["load_std"]),
        "finite_positive_prior": inputs["finite_positive_prior"],
        "data_squared_norm": result["data_squared_norm"],
        "calibration_squared_norm": result["calibration_squared_norm"],
        "augmented_squared_norm": result["augmented_squared_norm"],
        "augmented_norm": result["augmented_norm"],
        "N_strain_observations": result["N_strain_observations"],
        "M_calibration_observations": result["M_calibration_observations"],
        "threshold": result["threshold"],
        "null_compatible": result["null_compatible"],
        "array_hash": _numeric_array_hash(arrays),
        "source_dataset_file": dataset_record["npz_file"],
        "source_dataset_hash": dataset_record["npz_sha256"],
        "source_path_record_sha256": hash_file(_source_path_record(Path(ROOT / lock["source_output"]["path"]), path_config["key"])),
        "design_hash": lock["design_hash"],
        "source_protocol_hash": lock["source_output"]["protocol_hash"],
        "protocol_hash": lock["protocol_hash"],
    }


def _verify_null_record(
    record: dict,
    *,
    source_dir: Path,
    dataset: dict,
    path_config: dict,
    lock: dict,
    forward,
) -> list[str]:
    errors: list[str] = []
    try:
        dataset_record, path_record, dataset_path, inputs = _inputs_for_path(
            source_dir, dataset, path_config
        )
        replay = null_endpoint(
            forward, dataset_path, path_config["method"], np.asarray(dataset_record["measured_load_mean"]), float(dataset["tau"])
        )
        expected = _null_record(
            replay,
            dataset=dataset,
            path_config=path_config,
            dataset_record=dataset_record,
            path_record=path_record,
            lock=lock,
        )
        scalar_names = (
            "data_squared_norm",
            "calibration_squared_norm",
            "augmented_squared_norm",
            "augmented_norm",
            "threshold",
        )
        for name in scalar_names:
            if not _same_number(expected[name], record.get(name), name, 1.0e-12, 1.0e-12):
                errors.append(f"{name} mismatch")
        for name in (
            "schema_version",
            "path_key",
            "dataset_key",
            "method",
            "originally_unbracketed_high",
            "gamma_mode",
            "load_mean",
            "load_std",
            "finite_positive_prior",
            "N_strain_observations",
            "M_calibration_observations",
            "null_compatible",
            "array_hash",
            "source_dataset_file",
            "source_dataset_hash",
            "source_path_record_sha256",
            "design_hash",
            "source_protocol_hash",
            "protocol_hash",
        ):
            if record.get(name) != expected[name]:
                errors.append(f"identity mismatch: {name}")
        arrays = {
            "coefficients": np.asarray(record.get("coefficients", []), dtype=float),
            "gamma": np.asarray(record.get("gamma", []), dtype=float),
            "predicted_strain": np.asarray(record.get("predicted_strain", []), dtype=float),
        }
        if arrays["coefficients"].shape != (N_COEFFICIENTS,) or not np.all(arrays["coefficients"] == 0.0):
            errors.append("null coefficients are not 29 zeros")
        # Independent physical re-evaluation: the recorded response must be
        # gamma_j times the intact unit strain, and the recorded norms must
        # replay from the recorded arrays alone (not from null_endpoint).
        obs_points, obs_values, obs_sigmas, _ = _observation_inputs(dataset_path)
        unit_prediction = forward.strain_and_jacobian(arrays["coefficients"], obs_points)[0]
        physical_prediction = _physical_prediction(unit_prediction, obs_points, arrays["gamma"])
        if arrays["predicted_strain"].shape != physical_prediction.shape or not np.allclose(
            arrays["predicted_strain"], physical_prediction, rtol=1.0e-12, atol=1.0e-15
        ):
            errors.append("null predicted strain is not gamma times the intact unit strain")
        residual = np.concatenate(
            [
                (physical_prediction[index] - obs_values[index]) / obs_sigmas[index]
                for index in range(len(obs_points))
            ]
        )
        data_squared = float(residual @ residual)
        if not _same_number(data_squared, record.get("data_squared_norm"), "data_squared_norm", 1.0e-9, 1.0e-12):
            errors.append("data_squared_norm does not replay from recorded arrays")
        calibration_squared = 0.0
        if record.get("finite_positive_prior") is True:
            prior_std = float(record["load_std"])
            calibration_squared = float(
                np.sum(((arrays["gamma"] - np.asarray(record["load_mean"], dtype=float)) / prior_std) ** 2)
            )
        augmented = math.sqrt(data_squared + calibration_squared)
        if not _same_number(augmented, record.get("augmented_norm"), "augmented_norm", 1.0e-9, 1.0e-12):
            errors.append("augmented_norm does not replay from recorded arrays")
        if not np.all((arrays["gamma"] >= 0.7) & (arrays["gamma"] <= 1.3)):
            errors.append("null gamma violates physical bounds")
        if inputs["gamma_mode"] != "profile" and not np.allclose(arrays["gamma"], inputs["load_mean"]):
            errors.append("exact null gamma mismatch")
        if not np.allclose(arrays["gamma"], replay["gamma"], rtol=1.0e-12, atol=1.0e-12):
            errors.append("profile null gamma mismatch")
        del inputs
    except (EndpointProtocolError, KeyError, TypeError, ValueError, OSError) as error:
        errors.append(f"null replay failed: {type(error).__name__}: {error}")
    return errors


def _extension_candidate_record(
    result: dict,
    *,
    dataset: dict,
    path_config: dict,
    dataset_record: dict,
    path_record: dict,
    penalty_lambda: float,
    design: dict,
    lock: dict,
) -> dict:
    inputs = result["runner_inputs"]
    arrays = {
        "coefficients": np.asarray(result["coefficients"], dtype=float),
        "gamma": np.asarray(result["gamma"], dtype=float),
        "train_prediction": np.asarray(result["runner_prediction"], dtype=float),
    }
    source_dir = Path(ROOT / lock["source_output"]["path"])
    return {
        "schema_version": "regularization-endpoint-candidate-v1.0",
        "candidate_origin": "b2_finite_extension",
        "path_key": path_config["key"],
        "dataset_key": dataset["key"],
        "method": path_config["method"],
        "penalty": "H1",
        "penalty_lambda": float(penalty_lambda),
        "inherited_lambda_path": design["regularization"]["lambda_path"],
        "extension_lambda_path": list(EXTENSION_LAMBDAS),
        "gamma_mode": inputs["gamma_mode"],
        "load_mean": inputs["load_mean"].tolist(),
        "load_std": None if np.isinf(inputs["load_std"]) else float(inputs["load_std"]),
        "finite_positive_prior": inputs["finite_positive_prior"],
        "successful_start": bool(result["successful_start"]),
        "chosen_start": int(result["chosen_start"]),
        "starts": result["starts"],
        "starts_used": list(EXTENSION_STARTS),
        "max_nfev": EXTENSION_MAX_NFEV,
        "coefficients": arrays["coefficients"].tolist(),
        "gamma": arrays["gamma"].tolist(),
        "train_prediction": arrays["train_prediction"].tolist(),
        "objective": result["runner_norms"]["objective"],
        "norms": result["runner_norms"],
        "array_hash": _numeric_array_hash(arrays),
        "source_dataset_file": dataset_record["npz_file"],
        "source_dataset_hash": dataset_record["npz_sha256"],
        "source_path_record_sha256": hash_file(_source_path_record(source_dir, path_config["key"])),
        "design_hash": lock["design_hash"],
        "source_protocol_hash": lock["source_output"]["protocol_hash"],
        "protocol_hash": lock["protocol_hash"],
        "candidate_file": _candidate_filename(penalty_lambda),
    }


def _verify_extension_candidate(
    record: dict,
    *,
    source_dir: Path,
    dataset: dict,
    path_config: dict,
    penalty_lambda: float,
    design: dict,
    lock: dict,
) -> dict:
    try:
        if type(record.get("successful_start")) is not bool:
            raise EndpointProtocolError("successful_start must be a JSON boolean")
        if type(record.get("chosen_start")) is not int:
            raise EndpointProtocolError("chosen_start must be a JSON integer")
        dataset_record, path_record, dataset_path, inputs = _inputs_for_path(
            source_dir, dataset, path_config
        )
        expected = _extension_candidate_record(
            {
                "runner_inputs": inputs,
                "coefficients": np.asarray(record.get("coefficients", [])),
                "gamma": np.asarray(record.get("gamma", [])),
                "runner_prediction": np.asarray(record.get("train_prediction", [])),
                "runner_norms": record.get("norms", {}),
                "successful_start": record.get("successful_start"),
                "chosen_start": record.get("chosen_start"),
                "starts": record.get("starts", []),
            },
            dataset=dataset,
            path_config=path_config,
            dataset_record=dataset_record,
            path_record=path_record,
            penalty_lambda=penalty_lambda,
            design=design,
            lock=lock,
        )
        identity = (
            "schema_version",
            "candidate_origin",
            "path_key",
            "dataset_key",
            "method",
            "penalty",
            "penalty_lambda",
            "inherited_lambda_path",
            "extension_lambda_path",
            "gamma_mode",
            "load_mean",
            "load_std",
            "finite_positive_prior",
            "starts_used",
            "max_nfev",
            "source_dataset_file",
            "source_dataset_hash",
            "source_path_record_sha256",
            "design_hash",
            "source_protocol_hash",
            "protocol_hash",
            "candidate_file",
        )
        errors = [
            f"identity mismatch: {name}"
            for name in identity
            if record.get(name) != expected[name]
        ]
        if float(penalty_lambda) in set(design["regularization"]["lambda_path"]):
            errors.append("extension lambda falsely belongs to the inherited path")
        replay = _verify_candidate_record(record, dataset_path, PhysicalModel())
        if not replay["passed"]:
            errors.extend(replay["errors"])
        return {"passed": not errors, "errors": errors, "recomputed": replay["recomputed"]}
    except (EndpointProtocolError, KeyError, TypeError, ValueError, OSError) as error:
        return {
            "passed": False,
            "errors": [f"candidate replay failed: {type(error).__name__}: {error}"],
            "recomputed": None,
        }


def endpoint_decision(source_path_record: dict, null_record: dict) -> dict:
    """Classify using only observable source selection and null compatibility."""
    upper = _require_bool(
        source_path_record.get("selection", {}).get("unbracketed_high"),
        "source unbracketed_high",
    )
    compatible = _require_bool(null_record.get("null_compatible"), "null_compatible")
    return {
        "originally_unbracketed_high": upper,
        "requires_endpoint_treatment": upper,
        "null_compatible": compatible,
        "action": "lambda_infinity" if upper and compatible else ("finite_extensions" if upper else "null_check_only"),
    }


def _combined_selection(
    source_candidates: list[dict],
    extension_candidates: list[dict],
    sensors: int,
) -> dict:
    if not source_candidates and not extension_candidates:
        raise EndpointProtocolError("no candidates are available for endpoint selection")
    selection = _selection_details(source_candidates + extension_candidates, sensors)
    selected_lambda = selection["selected_lambda"]
    extension_lambdas = set(EXTENSION_LAMBDAS)
    selection = dict(selection)
    selection["selected_origin"] = "extension" if selected_lambda in extension_lambdas else "inherited"
    selection["lambda_infinity_endpoint"] = False
    selection["still_unbracketed_high"] = selected_lambda == EXTENSION_LAMBDAS[-1]
    return selection


def _score_and_arrays(dataset_path: Path, selected: dict) -> tuple[dict, dict]:
    metrics, score_arrays = score_selected(
        dataset_path,
        {
            "coefficients": np.asarray(selected["coefficients"], dtype=float),
            "gamma": np.asarray(selected["gamma"], dtype=float),
        },
        PhysicalModel(),
    )
    arrays = dict(score_arrays)
    arrays["coefficients"] = np.asarray(selected["coefficients"], dtype=float)
    arrays["gamma"] = np.asarray(selected["gamma"], dtype=float)
    arrays["train_prediction"] = np.asarray(selected["train_prediction"], dtype=float)
    return metrics, arrays


def _endpoint_record(
    *,
    null_record: dict,
    source_candidates: list[dict],
    extension_candidates: list[dict],
    path_config: dict,
    dataset: dict,
    dataset_path: Path,
    output_dir: Path,
    lock: dict,
) -> dict:
    decision = endpoint_decision({"selection": {"unbracketed_high": null_record["originally_unbracketed_high"]}}, null_record)
    if not decision["requires_endpoint_treatment"]:
        raise EndpointProtocolError("endpoint treatment requested for a non-upper source path")
    if decision["null_compatible"]:
        selection = {
            "selected_lambda": "infinity",
            "selected_origin": "null_zero_field",
            "lambda_infinity_endpoint": True,
            "still_unbracketed_high": False,
            "selection_reason": "null_endpoint_admissible",
            "optimizer_failed": False,
            "no_feasible_candidate": False,
            "path_bracketed": True,
            "threshold": null_record["threshold"],
            "N_strain_observations": null_record["N_strain_observations"],
            "M_calibration_observations": null_record["M_calibration_observations"],
            "candidate_count": len(source_candidates) + 1,
        }
        selected = {
            "coefficients": np.asarray(null_record["coefficients"], dtype=float),
            "gamma": np.asarray(null_record["gamma"], dtype=float),
            "train_prediction": np.asarray(null_record["predicted_strain"], dtype=float),
        }
    else:
        if [record["penalty_lambda"] for record in extension_candidates] != list(EXTENSION_LAMBDAS):
            raise EndpointProtocolError("all four extension candidates are required before endpoint completion")
        selection = _combined_selection(
            source_candidates, extension_candidates, int(dataset["sensors"])
        )
        all_candidates = source_candidates + extension_candidates
        selected = next(record for record in all_candidates if record["penalty_lambda"] == selection["selected_lambda"])
    metrics, arrays = _score_and_arrays(dataset_path, selected)
    arrays_path = output_dir / "endpoints" / path_config["key"] / "endpoint_arrays.npz"
    _write_or_reuse_arrays(arrays_path, arrays)
    record = {
        "schema_version": "regularization-endpoint-v1.0",
        "path_key": path_config["key"],
        "dataset_key": dataset["key"],
        "method": path_config["method"],
        "original_selection": {
            "selected_lambda": _source_original_lambda(lock, path_config["key"]),
            "unbracketed_high": null_record["originally_unbracketed_high"],
        },
        "decision": decision,
        "action": decision["action"],
        "inherited_candidate_files": [
            Path("paths").joinpath(path_config["key"], _candidate_filename(lam)).as_posix()
            for lam in lock["inherited_lambda_path"]
        ],
        "extension_candidate_files": [
            Path("endpoints").joinpath(path_config["key"], _candidate_filename(lam)).as_posix()
            for lam in EXTENSION_LAMBDAS
        ],
        "selection": selection,
        "selected_start": "null_endpoint" if decision["null_compatible"] else int(selected["chosen_start"]),
        "selected_coefficients": np.asarray(selected["coefficients"], dtype=float).tolist(),
        "selected_gamma": np.asarray(selected["gamma"], dtype=float).tolist(),
        "selected_train_prediction": np.asarray(selected["train_prediction"], dtype=float).tolist(),
        "metrics": metrics,
        "arrays_file": Path("endpoints").joinpath(path_config["key"], "endpoint_arrays.npz").as_posix(),
        "arrays_sha256": hash_file(arrays_path),
        "design_hash": lock["design_hash"],
        "source_protocol_hash": lock["source_output"]["protocol_hash"],
        "protocol_hash": lock["protocol_hash"],
    }
    # Selection is complete before any post-hoc reference/unused scoring is read.
    record["posthoc_scoring_only"] = True
    return record


def _source_original_lambda(lock: dict, path_key: str) -> float:
    source_dir = Path(ROOT / lock["source_output"]["path"])
    record = read_json(_source_path_record(source_dir, path_key))
    value = record["selection"]["selected_lambda"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EndpointProtocolError("source selected lambda is not numeric")
    return float(value)


def _load_source_candidates(
    source_dir: Path,
    path_config: dict,
    design: dict,
) -> list[dict]:
    path_dir = source_dir / "paths" / path_config["key"]
    return [
        read_json(path_dir / _candidate_filename(penalty_lambda))
        for penalty_lambda in design["regularization"]["lambda_path"]
    ]


def _verify_endpoint_record(
    record: dict,
    *,
    source_dir: Path,
    output_dir: Path,
    dataset: dict,
    path_config: dict,
    null_record: dict,
    extension_candidates: list[dict],
    design: dict,
    lock: dict,
) -> list[str]:
    errors: list[str] = []
    # Verification is strictly read-only.  The expected-record replay below
    # reuses the canonical arrays file through _write_or_reuse_arrays, so its
    # presence and recorded hash are required BEFORE any replay; missing or
    # tampered evidence is rejected here and is never recreated or repaired.
    arrays_file = record.get("arrays_file")
    arrays_sha256 = record.get("arrays_sha256")
    if type(arrays_file) is not str or type(arrays_sha256) is not str:
        return ["endpoint arrays file or recorded hash is missing or not a string"]
    # Only the canonical endpoint arrays path is evidence.  A record pointing
    # at any other file (even an existing one with a matching recorded hash)
    # is rejected here, before any access, replay, or scoring, so verification
    # can never recreate or mutate canonical evidence through an alias.
    canonical_arrays_file = Path("endpoints").joinpath(
        path_config["key"], "endpoint_arrays.npz"
    ).as_posix()
    if arrays_file != canonical_arrays_file:
        return ["endpoint arrays file is not the canonical endpoint arrays path"]
    arrays_path = output_dir / canonical_arrays_file
    if not arrays_path.is_file():
        return ["endpoint arrays file is missing"]
    if hash_file(arrays_path) != arrays_sha256:
        return ["endpoint arrays file/hash mismatch"]
    try:
        dataset_record, _, dataset_path, _ = _inputs_for_path(source_dir, dataset, path_config)
        source_candidates = _load_source_candidates(source_dir, path_config, design)
        expected = _endpoint_record(
            null_record=null_record,
            source_candidates=source_candidates,
            extension_candidates=extension_candidates,
            path_config=path_config,
            dataset=dataset,
            dataset_path=dataset_path,
            output_dir=output_dir,
            lock={
                **lock,
                "inherited_lambda_path": design["regularization"]["lambda_path"],
            },
        )
        del dataset_record
        for name in (
            "schema_version",
            "path_key",
            "dataset_key",
            "method",
            "original_selection",
            "decision",
            "action",
            "inherited_candidate_files",
            "extension_candidate_files",
            "selection",
            "selected_start",
            "selected_coefficients",
            "selected_gamma",
            "selected_train_prediction",
            "metrics",
            "arrays_file",
            "arrays_sha256",
            "design_hash",
            "source_protocol_hash",
            "protocol_hash",
            "posthoc_scoring_only",
        ):
            if record.get(name) != expected[name]:
                errors.append(f"endpoint field mismatch: {name}")
        with np.load(arrays_path, allow_pickle=False) as saved:
            if set(saved.files) != {
                "score_x", "reference_d", "estimated_d", "unused_x",
                "reference_strain", "predicted_strain", "coefficients",
                "gamma", "train_prediction",
            }:
                errors.append("endpoint arrays schema mismatch")
    except (EndpointProtocolError, ProtocolError, KeyError, TypeError, ValueError, OSError, StopIteration) as error:
        errors.append(f"endpoint replay failed: {type(error).__name__}: {error}")
    return errors


def _check_run_inventory(output_dir: Path, *, resume: bool) -> None:
    if not output_dir.exists():
        return
    allowed = {LOCK_FILE, "null_checks", "endpoints", "run_status.json", "summary.json", "verification_report.json"}
    entries = [item for item in output_dir.iterdir() if not _is_system_metadata(item.name)]
    if not resume and any(item.name != LOCK_FILE for item in entries):
        raise EndpointProtocolError("output contains diagnostic results; use --resume")
    unexpected = [item.name for item in entries if item.name not in allowed]
    if unexpected:
        raise EndpointProtocolError("unexpected output entries: " + ", ".join(sorted(unexpected)))


def execute_run(
    output_dir: Path,
    source_dir: Path,
    design: dict,
    *,
    resume: bool,
    limit_paths: int | None = None,
    path_keys: list[str] | None = None,
) -> dict:
    lock = load_and_check_endpoint_lock(output_dir, source_dir, design)
    _check_run_inventory(output_dir, resume=resume)
    datasets, paths, counts = expand_study(design)
    if limit_paths is not None and limit_paths < 0:
        raise EndpointProtocolError("--limit-paths must be nonnegative")
    if limit_paths is not None and path_keys is not None:
        raise EndpointProtocolError("--limit-paths and --path-keys are mutually exclusive")
    if path_keys is not None:
        if (
            not path_keys
            or len(set(path_keys)) != len(path_keys)
            or not all(type(key) is str for key in path_keys)
        ):
            raise EndpointProtocolError("--path-keys must be distinct nonempty strings")
        known = {path_config["key"] for path_config in paths}
        unknown = [key for key in path_keys if key not in known]
        if unknown:
            raise EndpointProtocolError("unknown --path-keys: " + ", ".join(sorted(unknown)))
        requested = set(path_keys)
        selected_paths = [path_config for path_config in paths if path_config["key"] in requested]
    else:
        selected_paths = paths if limit_paths is None else paths[:limit_paths]
    datasets_by_key = {dataset["key"]: dataset for dataset in datasets}
    prior_status_path = output_dir / "run_status.json"
    started_utc = None
    elapsed_seconds = 0.0
    selected_keys = [path_config["key"] for path_config in selected_paths]
    if prior_status_path.is_file():
        prior = read_json(prior_status_path)
        started_utc = prior.get("wall_time_started_utc")
        if started_utc is not None:
            if _require_int(prior.get("selected_path_count"), "selected_path_count") != len(selected_paths):
                raise EndpointProtocolError("resume selected-path scope differs")
            if prior.get("selected_path_keys") != selected_keys:
                raise EndpointProtocolError("resume selected-path keys differ")
    if started_utc is None:
        started_utc = utc_now()
    else:
        started_time = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
        elapsed_seconds = max(0.0, (datetime.now(timezone.utc) - started_time).total_seconds())
    remaining = max(0.0, WALL_BUDGET_SECONDS - elapsed_seconds)
    deadline = time.monotonic() + remaining
    completed = 0
    reused = 0
    timeout = False
    forward_cache: dict[str, Any] = {}
    base_status = {
        "schema_version": "regularization-endpoint-run-status-v1.0",
        "updated_utc": utc_now(),
        "wall_time_started_utc": started_utc,
        "wall_time_budget_seconds": WALL_BUDGET_SECONDS,
        "selected_path_count": len(selected_paths),
        "selected_path_keys": selected_keys,
        "expected_full_counts": counts,
        "limit_paths": limit_paths,
        "path_keys": path_keys,
    }
    write_replace_json(prior_status_path, {**base_status, "status": "running", "completed_paths": completed, "reused_paths": reused})
    try:
        for path_config in selected_paths:
            dataset = datasets_by_key[path_config["dataset_key"]]
            dataset_record, source_path_record, dataset_path, _ = _inputs_for_path(source_dir, dataset, path_config)
            null_path = output_dir / "null_checks" / path_config["key"] / "null_check.json"
            if null_path.is_file():
                null_record = read_json(null_path)
                forward = forward_cache.setdefault(
                    dataset["key"], build_forward(dataset_path, PhysicalModel())
                )
                errors = _verify_null_record(
                    null_record,
                    source_dir=source_dir,
                    dataset=dataset,
                    path_config=path_config,
                    lock=lock,
                    forward=forward,
                )
                if errors:
                    raise EndpointProtocolError(f"existing null check failed: {path_config['key']}: " + "; ".join(errors))
                reused += 1
            else:
                forward = forward_cache.setdefault(
                    dataset["key"], build_forward(dataset_path, PhysicalModel())
                )
                result = null_endpoint(
                    forward,
                    dataset_path,
                    path_config["method"],
                    np.asarray(dataset_record["measured_load_mean"], dtype=float),
                    float(dataset["tau"]),
                )
                null_record = _null_record(
                    result,
                    dataset=dataset,
                    path_config=path_config,
                    dataset_record=dataset_record,
                    path_record=source_path_record,
                    lock=lock,
                )
                errors = _verify_null_record(
                    null_record,
                    source_dir=source_dir,
                    dataset=dataset,
                    path_config=path_config,
                    lock=lock,
                    forward=forward,
                )
                if errors:
                    raise EndpointProtocolError(f"new null check failed replay: {path_config['key']}")
                write_new_json(null_path, null_record)

            decision = endpoint_decision(source_path_record, null_record)
            if decision["requires_endpoint_treatment"]:
                # The extension fits reuse the design's optimizer settings; the
                # frozen addon rule pins them to these exact values.
                if (
                    tuple(design["regularization"]["starts"]) != EXTENSION_STARTS
                    or int(design["regularization"]["max_nfev"]) != EXTENSION_MAX_NFEV
                ):
                    raise EndpointProtocolError("design optimizer settings differ from the frozen extension rule")
                endpoint_dir = output_dir / "endpoints" / path_config["key"]
                endpoint_path = endpoint_dir / "endpoint.json"
                if endpoint_path.is_file():
                    if decision["action"] == "finite_extensions":
                        extension_candidates = _read_extension_candidates(
                            endpoint_dir, path_config, dataset, design, lock, source_dir
                        )
                    else:
                        extension_candidates = []
                    errors = _verify_endpoint_record(
                        read_json(endpoint_path),
                        source_dir=source_dir,
                        output_dir=output_dir,
                        dataset=dataset,
                        path_config=path_config,
                        null_record=null_record,
                        extension_candidates=extension_candidates,
                        design=design,
                        lock=lock,
                    )
                    if errors:
                        raise EndpointProtocolError(f"existing endpoint failed replay: {path_config['key']}: " + "; ".join(errors))
                    reused += 1
                else:
                    extension_candidates = []
                    if not decision["null_compatible"]:
                        for penalty_lambda in EXTENSION_LAMBDAS:
                            if time.monotonic() >= deadline:
                                timeout = True
                                break
                            candidate_path = endpoint_dir / _candidate_filename(penalty_lambda)
                            if candidate_path.is_file():
                                record = read_json(candidate_path)
                                verification = _verify_extension_candidate(
                                    record,
                                    source_dir=source_dir,
                                    dataset=dataset,
                                    path_config=path_config,
                                    penalty_lambda=penalty_lambda,
                                    design=design,
                                    lock=lock,
                                )
                                if not verification["passed"]:
                                    raise EndpointProtocolError(
                                        f"existing extension candidate failed replay: {path_config['key']}: "
                                        + "; ".join(verification["errors"])
                                    )
                            else:
                                result = _fit_one_candidate(
                                    forward,
                                    dataset_path,
                                    path_config["method"],
                                    penalty_lambda,
                                    design,
                                    np.asarray(dataset_record["measured_load_mean"], dtype=float),
                                    float(dataset["tau"]),
                                )
                                record = _extension_candidate_record(
                                    result,
                                    dataset=dataset,
                                    path_config=path_config,
                                    dataset_record=dataset_record,
                                    path_record=source_path_record,
                                    penalty_lambda=penalty_lambda,
                                    design=design,
                                    lock=lock,
                                )
                                verification = _verify_extension_candidate(
                                    record,
                                    source_dir=source_dir,
                                    dataset=dataset,
                                    path_config=path_config,
                                    penalty_lambda=penalty_lambda,
                                    design=design,
                                    lock=lock,
                                )
                                if not verification["passed"]:
                                    raise EndpointProtocolError(
                                        f"new extension candidate failed replay: {path_config['key']}: "
                                        + "; ".join(verification["errors"])
                                    )
                                write_new_json(candidate_path, record)
                            extension_candidates.append(record)
                    if timeout:
                        break
                    endpoint_record = _endpoint_record(
                        null_record=null_record,
                        source_candidates=_load_source_candidates(source_dir, path_config, design),
                        extension_candidates=extension_candidates,
                        path_config=path_config,
                        dataset=dataset,
                        dataset_path=dataset_path,
                        output_dir=output_dir,
                        lock={**lock, "inherited_lambda_path": design["regularization"]["lambda_path"]},
                    )
                    errors = _verify_endpoint_record(
                        endpoint_record,
                        source_dir=source_dir,
                        output_dir=output_dir,
                        dataset=dataset,
                        path_config=path_config,
                        null_record=null_record,
                        extension_candidates=extension_candidates,
                        design=design,
                        lock=lock,
                    )
                    if errors:
                        raise EndpointProtocolError(f"new endpoint failed replay: {path_config['key']}: " + "; ".join(errors))
                    write_new_json(endpoint_path, endpoint_record)
            completed += 1
            write_replace_json(
                prior_status_path,
                {**base_status, "status": "running", "completed_paths": completed, "reused_paths": reused},
            )
    except Exception as error:
        write_replace_json(
            prior_status_path,
            {**base_status, "status": "error", "completed_paths": completed, "reused_paths": reused, "error": f"{type(error).__name__}: {error}"},
        )
        raise
    status_value = "timeout" if timeout else "completed"
    status = {**base_status, "status": status_value, "completed_paths": completed, "reused_paths": reused}
    write_replace_json(prior_status_path, status)
    if not timeout:
        summary = build_summary(output_dir, source_dir, design, selected_paths, counts)
        if not (output_dir / "summary.json").exists():
            write_new_json(output_dir / "summary.json", summary)
    return status


def _read_extension_candidates(
    endpoint_dir: Path,
    path_config: dict,
    dataset: dict,
    design: dict,
    lock: dict,
    source_dir: Path,
) -> list[dict]:
    records = []
    for penalty_lambda in EXTENSION_LAMBDAS:
        path = endpoint_dir / _candidate_filename(penalty_lambda)
        if not path.is_file():
            raise EndpointProtocolError(
                f"completed endpoint is missing extension lambda {penalty_lambda}: {path_config['key']}"
            )
        record = read_json(path)
        verification = _verify_extension_candidate(
            record,
            source_dir=source_dir,
            dataset=dataset,
            path_config=path_config,
            penalty_lambda=penalty_lambda,
            design=design,
            lock=lock,
        )
        if not verification["passed"]:
            raise EndpointProtocolError(
                f"existing extension candidate failed replay: {path_config['key']}: "
                + "; ".join(verification["errors"])
            )
        records.append(record)
    return records


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_summary(rows: list[dict]) -> dict:
    result: dict[str, Any] = {"count": len(rows)}
    for name in (
        "loss_area_m",
        "loss_area_absolute_error_m",
        "unused_load_strain_relative_L2",
    ):
        original = [row["original_metrics"][name] for row in rows]
        endpoint = [row["endpoint_metrics"][name] for row in rows]
        deltas = [end - start for start, end in zip(original, endpoint)]
        result[name] = {
            "original_median": median(original) if original else None,
            "endpoint_median": median(endpoint) if endpoint else None,
            "original_quartiles": [_quantile(original, 0.25), _quantile(original, 0.75)],
            "endpoint_quartiles": [_quantile(endpoint, 0.25), _quantile(endpoint, 0.75)],
            "median_original_to_endpoint_delta": median(deltas) if deltas else None,
        }
    return result


def build_summary(
    output_dir: Path,
    source_dir: Path,
    design: dict,
    selected_paths: list[dict],
    counts: dict,
) -> dict:
    datasets_by_key = {dataset["key"]: dataset for dataset in expand_study(design)[0]}
    rows = []
    nonupper_compatible = 0
    null_compatible = 0
    upper_count = 0
    finite_extensions = 0
    still_upper = 0
    all_extension_failures = 0
    posthoc_shapes: dict[str, int] = {}
    for path_config in selected_paths:
        dataset = datasets_by_key[path_config["dataset_key"]]
        null = read_json(output_dir / "null_checks" / path_config["key"] / "null_check.json")
        null_compatible += int(null["null_compatible"])
        if not null["originally_unbracketed_high"]:
            nonupper_compatible += int(null["null_compatible"])
            continue
        upper_count += 1
        endpoint = read_json(output_dir / "endpoints" / path_config["key"] / "endpoint.json")
        if endpoint["action"] == "finite_extensions":
            finite_extensions += 1
            still_upper += int(endpoint["selection"]["still_unbracketed_high"])
            all_extension_failures += int(
                all(
                    not read_json(output_dir / name)["successful_start"]
                    for name in endpoint["extension_candidate_files"]
                )
            )
        if null["null_compatible"]:
            posthoc_shapes[dataset["shape"]] = posthoc_shapes.get(dataset["shape"], 0) + 1
        if (
            dataset["sensors"] == 20
            and math.isclose(dataset["sigma"], 5.0e-6, rel_tol=0.0, abs_tol=1.0e-15)
            and 1101 <= dataset["seed"] <= 1106
            and "primary" in dataset["memberships"]
        ):
            source_path = read_json(_source_path_record(source_dir, path_config["key"]))
            rows.append(
                {
                    "path_key": path_config["key"],
                    "shape": dataset["shape"],
                    "tau": dataset["tau"],
                    "method": path_config["method"],
                    "original_metrics": {
                        name: source_path["metrics"][name]
                        for name in (
                            "loss_area_m",
                            "loss_area_absolute_error_m",
                            "unused_load_strain_relative_L2",
                        )
                    },
                    "endpoint_metrics": {
                        name: endpoint["metrics"][name]
                        for name in (
                            "loss_area_m",
                            "loss_area_absolute_error_m",
                            "unused_load_strain_relative_L2",
                        )
                    },
                }
            )
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(f"{row['shape']}|tau-{row['tau']:g}|{row['method']}", []).append(row)
    return {
        "schema_version": "regularization-endpoint-summary-v1.0",
        "scope": "full" if len(selected_paths) == counts["paths"] else "limited",
        "selected_path_count": len(selected_paths),
        "counts": {
            "null_checks": len(selected_paths),
            "null_compatible": null_compatible,
            "nonupper_null_compatible": nonupper_compatible,
            "originally_unbracketed_high": upper_count,
            "null_compatible_posthoc_by_shape": posthoc_shapes,
            "finite_extensions": finite_extensions,
            "all_extension_candidates_failed": all_extension_failures,
            "still_unbracketed_at_3e7": still_upper,
        },
        "baseline_definition": {
            "sensors": 20,
            "strain_sigma_microstrain": 5.0,
            "seeds": [1101, 1102, 1103, 1104, 1105, 1106],
            "study": "primary",
        },
        "paired_baseline_paths": rows,
        "groups": {name: _metric_summary(group) for name, group in sorted(groups.items())},
        "interpretation": "Deterministic intact-compatibility diagnostic; no confidence or significance claim; reference and unused-response metrics are post hoc only.",
    }


def _inventory_errors(output_dir: Path, expected_files: set[str]) -> list[str]:
    errors = []
    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and not _is_system_metadata(path.name)
    }
    for name in sorted(actual_files - expected_files - {"verification_report.json"}):
        errors.append(f"unexpected file: {name}")
    for name in sorted(expected_files - actual_files):
        errors.append(f"missing file: {name}")
    return errors


def verify_protocol(output_dir: Path, source_dir: Path, design: dict) -> dict:
    lock = load_and_check_endpoint_lock(output_dir, source_dir, design)
    status_path = output_dir / "run_status.json"
    if not status_path.is_file():
        raise EndpointProtocolError("run status is missing")
    status = read_json(status_path)
    selected_keys = status.get("selected_path_keys")
    if not isinstance(selected_keys, list) or not all(type(key) is str for key in selected_keys):
        raise EndpointProtocolError("selected path keys must be JSON strings")
    _, all_paths, counts = expand_study(design)
    paths = [path_config for path_config in all_paths if path_config["key"] in set(selected_keys)]
    if len(paths) != len(selected_keys):
        raise EndpointProtocolError("run-status path keys do not match the source study")
    datasets_by_key = {dataset["key"]: dataset for dataset in expand_study(design)[0]}
    expected_files = {LOCK_FILE, "run_status.json", "summary.json"}
    errors_by_path: dict[str, list[str]] = {}
    checked_nulls = 0
    checked_candidates = 0
    checked_endpoints = 0
    forward_cache: dict[str, Any] = {}
    for path_config in paths:
        dataset = datasets_by_key[path_config["dataset_key"]]
        null_relative = Path("null_checks").joinpath(path_config["key"], "null_check.json").as_posix()
        expected_files.add(null_relative)
        null_path = output_dir / null_relative
        if not null_path.is_file():
            errors_by_path.setdefault(path_config["key"], []).append("missing null check")
            continue
        null_record = read_json(null_path)
        dataset_record, source_path_record, dataset_path, _ = _inputs_for_path(source_dir, dataset, path_config)
        forward = forward_cache.setdefault(dataset["key"], build_forward(dataset_path, PhysicalModel()))
        errors = _verify_null_record(
            null_record,
            source_dir=source_dir,
            dataset=dataset,
            path_config=path_config,
            lock=lock,
            forward=forward,
        )
        checked_nulls += 1
        decision = endpoint_decision(source_path_record, null_record)
        if not decision["requires_endpoint_treatment"]:
            if errors:
                errors_by_path[path_config["key"]] = errors
            continue
        endpoint_relative = Path("endpoints").joinpath(path_config["key"], "endpoint.json").as_posix()
        expected_files.add(endpoint_relative)
        expected_files.add(Path("endpoints").joinpath(path_config["key"], "endpoint_arrays.npz").as_posix())
        endpoint_path = output_dir / endpoint_relative
        if not endpoint_path.is_file():
            errors_by_path.setdefault(path_config["key"], []).append("missing endpoint record")
            continue
        extension_candidates = []
        if decision["action"] == "finite_extensions":
            for penalty_lambda in EXTENSION_LAMBDAS:
                relative = Path("endpoints").joinpath(
                    path_config["key"], _candidate_filename(penalty_lambda)
                ).as_posix()
                expected_files.add(relative)
                candidate_path = output_dir / relative
                if not candidate_path.is_file():
                    errors_by_path.setdefault(path_config["key"], []).append(f"missing extension lambda {penalty_lambda}")
                    continue
                candidate = read_json(candidate_path)
                verification = _verify_extension_candidate(
                    candidate,
                    source_dir=source_dir,
                    dataset=dataset,
                    path_config=path_config,
                    penalty_lambda=penalty_lambda,
                    design=design,
                    lock=lock,
                )
                checked_candidates += 1
                if not verification["passed"]:
                    errors_by_path.setdefault(path_config["key"], []).extend(verification["errors"])
                extension_candidates.append(candidate)
        endpoint_errors = _verify_endpoint_record(
            read_json(endpoint_path),
            source_dir=source_dir,
            output_dir=output_dir,
            dataset=dataset,
            path_config=path_config,
            null_record=null_record,
            extension_candidates=extension_candidates,
            design=design,
            lock=lock,
        )
        checked_endpoints += 1
        errors.extend(endpoint_errors)
        if errors:
            errors_by_path[path_config["key"]] = errors
        del dataset_record
    inventory_errors = _inventory_errors(output_dir, expected_files)
    if inventory_errors:
        errors_by_path.setdefault("__inventory__", []).extend(inventory_errors)
    incomplete = (
        len(paths) != counts["paths"]
        or status.get("status") != "completed"
        or checked_nulls != len(paths)
        or checked_endpoints != sum(
            (output_dir / "null_checks" / p["key"] / "null_check.json").is_file()
            and endpoint_decision(
                read_json(_source_path_record(source_dir, p["key"])),
                read_json(output_dir / "null_checks" / p["key"] / "null_check.json"),
            )["requires_endpoint_treatment"]
            for p in paths
        )
    )
    passed = not errors_by_path and status.get("status") == "completed"
    report = {
        "schema_version": "regularization-endpoint-verification-v1.0",
        "verified_utc": utc_now(),
        "passed": passed,
        "incomplete": bool(incomplete),
        "selected_path_count": len(paths),
        "checked_null_checks": checked_nulls,
        "checked_extension_candidates": checked_candidates,
        "checked_endpoint_treatments": checked_endpoints,
        "expected_full_counts": counts,
        "errors": errors_by_path,
    }
    write_replace_json(output_dir / "verification_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "run", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output", required=True, help="relative diagnostic output directory")
        subparser.add_argument("--source", default=DEFAULT_SOURCE, help="relative accepted B1 output")
        if command == "run":
            subparser.add_argument("--resume", action="store_true")
            subparser.add_argument("--limit-paths", type=int)
            subparser.add_argument(
                "--path-keys",
                nargs="+",
                help="explicit bounded subset of source path keys (study order is kept)",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = validate_relative_output(args.output)
    source_dir = validate_relative_output(args.source)
    design = load_design()
    if args.command == "freeze":
        lock = freeze_protocol(output_dir, source_dir, design)
        report = {"command": "freeze", "output": args.output, "protocol_hash": lock["protocol_hash"]}
    elif args.command == "run":
        report = execute_run(
            output_dir,
            source_dir,
            design,
            resume=args.resume,
            limit_paths=args.limit_paths,
            path_keys=args.path_keys,
        )
    else:
        report = verify_protocol(output_dir, source_dir, design)
    print(canonical_json(safe_json(report)))
    if args.command == "verify":
        return 0 if report["passed"] and not report["incomplete"] else 2
    if args.command == "run":
        return 0 if report["status"] == "completed" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
