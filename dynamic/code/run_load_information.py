"""Serial runner for the v3 load-information study.

The runner is deliberately checkpoint-oriented: each dataset and each lambda
candidate is a separate file, and no completed artifact is replaced.
"""

from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "vendor", ROOT / "code"):
    absolute = str(path.resolve())
    if absolute not in sys.path:
        sys.path.insert(0, absolute)

from beam_inverse import HermiteBeamFE, SupportBasis, fit_support  # noqa: E402
from campaign import DataConfig, PhysicalModel, generate_dataset  # noqa: E402
from run_campaign import hash_file, score_selected, select_candidate  # noqa: E402


LOCK_FILE = "load_information_lock.json"
SOURCE_FILES = (
    "code/beam_inverse.py",
    "code/campaign.py",
    "code/shape_factory.py",
    "code/skfem_reference.py",
    "code/run_campaign.py",
    "code/run_load_information.py",
)
TEST_FILES = (
    "tests/test_beam_inverse.py",
    "tests/test_campaign.py",
    "tests/test_load_information.py",
    "tests/test_runner.py",
    "tests/test_load_information_runner.py",
)
VENDOR_FILES = (
    "vendor/skfem/__init__.py",
    "vendor/scikit_fem-12.0.2.dist-info/METADATA",
    "vendor/scikit_fem-12.0.2.dist-info/WHEEL",
)
OBJECTIVE_RTOL = 1.0e-8
OBJECTIVE_ATOL = 1.0e-8


class ProtocolError(RuntimeError):
    """A lock, output, or result violates the frozen protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def safe_json(value: Any) -> Any:
    """Convert NumPy values and represent positive infinity as JSON null."""
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ProtocolError("object arrays cannot be serialized")
        return safe_json(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        if math.isnan(number):
            raise ProtocolError("NaN is not permitted in JSON output")
        return None if math.isinf(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ProtocolError("NaN is not permitted in JSON output")
        return None if math.isinf(value) else value
    raise TypeError(f"unsupported JSON value {type(value)!r}")


def _reject_json_constant(value: str) -> None:
    raise ProtocolError(f"nonfinite JSON constant {value!r} is not permitted")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_json_constant)


def write_new_json(path: Path, value: dict) -> None:
    if path.exists():
        raise ProtocolError(f"refusing to overwrite existing file: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(safe_json(value)) + "\n", encoding="utf-8")


def write_replace_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(canonical_json(safe_json(value)) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_new_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise ProtocolError(f"refusing to overwrite existing NPZ: {path.name}")
    checked = {}
    for name, array in arrays.items():
        value = np.asarray(array)
        if value.dtype.hasobject:
            raise ProtocolError(f"object array {name!r} cannot be saved")
        checked[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **checked)


def validate_relative_output(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts or not raw or raw in {".", "./"}:
        raise ProtocolError("--output must be a nonempty path relative to the project root")
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ProtocolError("output escapes the project root") from error
    return resolved


def output_file(output_dir: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ProtocolError("output filenames must be nonempty relative strings")
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ProtocolError(f"output filename escapes the output directory: {relative!r}")
    resolved = (output_dir / name).resolve()
    try:
        resolved.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ProtocolError(f"output filename escapes the output directory: {relative!r}") from error
    return resolved


def mapped_hashes(names: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for name in names:
        path = ROOT / name
        if not path.is_file():
            raise ProtocolError(f"required source file is missing: {name}")
        result[name] = hash_file(path)
    return result


def _is_system_metadata(name: str) -> bool:
    """Recognized macOS system metadata (.DS_Store, AppleDouble ._*) on USB/ExFAT.

    Only these names are ignored when checking output files; any other
    unexpected file still aborts the run.
    """
    return name == ".DS_Store" or name.startswith("._")


def current_environment() -> dict[str, str]:
    import scipy
    import skfem

    return {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "skfem_version": skfem.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def lock_fingerprint(
    source_hashes: dict[str, str],
    test_hashes: dict[str, str],
    vendor: dict,
    environment: dict,
    design: dict,
) -> str:
    return hash_json(
        {
            "source_hashes": source_hashes,
            "test_hashes": test_hashes,
            "vendor": vendor,
            "environment_provenance": environment,
            "design": design,
        }
    )


def freeze_protocol(output_dir: Path, design: dict) -> dict:
    if not (ROOT / "study_design_v3.json").is_file():
        raise ProtocolError("study_design_v3.json is missing")
    source_hashes = mapped_hashes(SOURCE_FILES)
    test_hashes = mapped_hashes(TEST_FILES)
    vendor_hashes = mapped_hashes(VENDOR_FILES)
    environment = current_environment()
    lock = {
        "schema_version": "load-information-lock-v3.0",
        "created_utc": utc_now(),
        "design_file": "study_design_v3.json",
        "design_hash": hash_json(design),
        "source_files": source_hashes,
        "test_files": test_hashes,
        "vendor": {
            "version": environment["skfem_version"],
            "files": vendor_hashes,
        },
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
    }
    lock["protocol_hash"] = lock_fingerprint(
        lock["source_files"],
        lock["test_files"],
        lock["vendor"],
        {**lock["environment_provenance"], "thread_settings": lock["thread_settings"]},
        design,
    )
    if output_dir.exists() and any(
        not _is_system_metadata(item.name) for item in output_dir.iterdir()
    ):
        raise ProtocolError("freeze output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILE
    write_new_json(lock_path, lock)
    return lock


def load_and_check_lock(output_dir: Path, design: dict) -> dict:
    lock_path = output_dir / LOCK_FILE
    if not lock_path.is_file():
        raise ProtocolError(f"missing source lock: {lock_path.relative_to(ROOT).as_posix()}")
    lock = read_json(lock_path)
    if lock.get("schema_version") != "load-information-lock-v3.0":
        raise ProtocolError("unsupported lock schema")
    expected = {
        "design_hash": hash_json(design),
        "source_files": mapped_hashes(SOURCE_FILES),
        "test_files": mapped_hashes(TEST_FILES),
        "vendor": {
            "version": current_environment()["skfem_version"],
            "files": mapped_hashes(VENDOR_FILES),
        },
        "environment_provenance": current_environment(),
        "thread_settings": {
            name: os.environ.get(name, "unset")
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ProtocolError(f"source lock mismatch for {key}")
    if lock.get("protocol_hash") != lock_fingerprint(
        lock["source_files"],
        lock["test_files"],
        lock["vendor"],
        {**lock["environment_provenance"], "thread_settings": lock["thread_settings"]},
        design,
    ):
        raise ProtocolError("source lock protocol hash mismatch")
    return lock


def load_design() -> dict:
    design = read_json(ROOT / "study_design_v3.json")
    if design.get("schema") != "load-information-v3.0":
        raise ProtocolError("unexpected study design schema")
    return design


def _format_number(value: float) -> str:
    return format(float(value), ".17g").replace("+", "")


def dataset_key(config: dict) -> str:
    return (
        f"shape-{config['shape']}_tau-{_format_number(config['tau'])}"
        f"_sigma-{_format_number(config['sigma'])}"
        f"_sensors-{int(config['sensors'])}_seed-{int(config['seed'])}"
    )


def path_key(dataset: dict, method: str) -> str:
    return f"{dataset['key']}__method-{method}"


def _generator_shape(shape: str) -> str:
    # The frozen design uses the short label "broad"; the legacy generator's
    # registered shape name is "broad_shallow".
    return "broad_shallow" if shape == "broad" else shape


def expand_study(design: dict) -> tuple[list[dict], list[dict], dict[str, int]]:
    datasets_by_key: dict[str, dict] = {}
    order = []

    def add(study: str, shape: str, tau: float, sigma: float, sensors: int, seed: int) -> None:
        config = {
            "shape": shape,
            "tau": float(tau),
            "sigma": float(sigma),
            "sensors": int(sensors),
            "seed": int(seed),
        }
        key = dataset_key(config)
        if key not in datasets_by_key:
            config["key"] = key
            config["memberships"] = []
            config["config_hash"] = hash_json(config)
            datasets_by_key[key] = config
            order.append(key)
        if study not in datasets_by_key[key]["memberships"]:
            datasets_by_key[key]["memberships"].append(study)

    for shape in design["primary"]["shapes"]:
        for tau in design["primary"]["load_standard_uncertainty"]:
            for sensors in design["primary"]["sensors"]:
                for sigma in design["primary"]["strain_sigma"]:
                    for seed in design["primary"]["seeds"]:
                        add("primary", shape, tau, sigma, sensors, seed)
    for shape in design["resolution"]["shapes"]:
        for tau in design["resolution"]["load_standard_uncertainty"]:
            for sensors in design["resolution"]["sensors"]:
                for sigma in design["resolution"]["strain_sigma"]:
                    for seed in design["resolution"]["seeds"]:
                        add("resolution", shape, tau, sigma, sensors, seed)

    datasets = [datasets_by_key[key] for key in order]
    paths = []
    for dataset in datasets:
        for method in design["methods"]:
            paths.append(
                {
                    "key": path_key(dataset, method),
                    "dataset_key": dataset["key"],
                    "dataset": dataset["key"],
                    "method": method,
                    "method_family": "primary_or_resolution",
                }
            )

    sensitivity_tau = float(design["confidence_sensitivity"]["primary_load_standard_uncertainty"])
    sensitivity_datasets = [
        dataset
        for dataset in datasets
        if "primary" in dataset["memberships"] and dataset["tau"] == sensitivity_tau
    ]
    for dataset in sensitivity_datasets:
        for method, factor in design["confidence_sensitivity"]["methods"].items():
            paths.append(
                {
                    "key": path_key(dataset, method),
                    "dataset_key": dataset["key"],
                    "dataset": dataset["key"],
                    "method": method,
                    "method_family": "confidence_sensitivity",
                    "tau_factor": float(factor),
                }
            )

    lambda_path = design["regularization"]["lambda_path"]
    counts = {
        "datasets": len(datasets),
        "paths": len(paths),
        "candidates": len(paths) * len(lambda_path),
        "starts": len(paths) * len(lambda_path) * len(design["regularization"]["starts"]),
        "primary_datasets": sum("primary" in d["memberships"] for d in datasets),
        "resolution_datasets": sum("resolution" in d["memberships"] for d in datasets),
        "overlap_datasets": sum(
            "primary" in d["memberships"] and "resolution" in d["memberships"] for d in datasets
        ),
    }
    return datasets, paths, counts


def calibration_measurement(seed: int, tau: float) -> tuple[np.ndarray, np.ndarray]:
    seed_sequence = np.random.SeedSequence([int(seed), 831])
    z = np.random.default_rng(seed_sequence).uniform(-np.sqrt(3.0), np.sqrt(3.0), 3)
    return 1.0 + float(tau) * z, z


def method_inputs(method: str, observed_mean: np.ndarray, tau: float) -> dict:
    observed = np.asarray(observed_mean, dtype=float)
    true_gamma = np.ones(3)
    if method == "fixed":
        return {
            "gamma_mode": "fixed",
            "load_mean": observed.copy(),
            "load_std": 0.0,
            "finite_positive_prior": False,
        }
    if method == "free":
        return {
            "gamma_mode": "profile",
            "load_mean": observed.copy(),
            "load_std": np.inf,
            "finite_positive_prior": False,
        }
    if method == "calibrated":
        std = float(tau)
    elif method == "overconfident":
        std = 0.2 * float(tau)
    elif method == "conservative":
        std = 3.0 * float(tau)
    else:
        if method != "oracle":
            raise ProtocolError(f"unknown load-information method: {method}")
        return {
            "gamma_mode": "oracle",
            "load_mean": true_gamma.copy(),
            "load_std": 0.0,
            "finite_positive_prior": False,
        }
    return {
        "gamma_mode": "profile",
        "load_mean": observed.copy(),
        "load_std": std,
        "finite_positive_prior": True,
    }


def _dataset_generated_names(config: dict) -> tuple[str, str]:
    stem = f"{_generator_shape(config['shape'])}_calibrated_seed{config['seed']}"
    return f"{stem}.npz", f"{stem}.json"


def ensure_dataset(
    config: dict,
    design: dict,
    output_dir: Path,
    lock: dict,
) -> dict:
    key = config["key"]
    dataset_dir = output_dir / "datasets" / key
    npz_name, metadata_name = _dataset_generated_names(config)
    npz_path = dataset_dir / npz_name
    metadata_path = dataset_dir / metadata_name
    record_path = dataset_dir / "dataset_record.json"
    if record_path.exists():
        record = read_json(record_path)
        if (
            record.get("dataset_key") != key
            or record.get("config_hash") != config["config_hash"]
            or record.get("design_hash") != lock["design_hash"]
            or record.get("protocol_hash") != lock["protocol_hash"]
            or record.get("source_hash") != hash_json(lock["source_files"])
            or record.get("npz_file") != Path("datasets").joinpath(key, npz_name).as_posix()
            or not npz_path.is_file()
            or hash_file(npz_path) != record.get("npz_sha256")
        ):
            raise ProtocolError(f"existing dataset record mismatch: {key}")
        generated_metadata = read_json(metadata_path)
        if generated_metadata.get("npz_path") != npz_name:
            raise ProtocolError(f"generated dataset metadata is not relative: {key}")
        return record

    existing = (
        {item.name for item in dataset_dir.iterdir() if not _is_system_metadata(item.name)}
        if dataset_dir.exists() and dataset_dir.is_dir()
        else set()
    )
    if dataset_dir.exists() and not dataset_dir.is_dir():
        raise ProtocolError(f"dataset path is not a directory: {key}")
    if existing and existing != {npz_name, metadata_name}:
        raise ProtocolError(f"refusing to overwrite partial dataset directory: {key}")

    means, z = calibration_measurement(config["seed"], config["tau"])
    if not existing:
        physical = PhysicalModel()
        data_config = DataConfig(
            sensors_per_load=int(config["sensors"]),
            sigma_strain=float(config["sigma"]),
        )
        generated = generate_dataset(
            _generator_shape(config["shape"]),
            "calibrated",
            int(config["seed"]),
            physical,
            data_config,
            dataset_dir,
        )
        if Path(generated["npz_path"]).resolve() != npz_path.resolve():
            raise ProtocolError(f"generator returned an unexpected path for dataset {key}")
        # The legacy generator writes an absolute diagnostic path. Replace it in the
        # newly generated intermediate before any dataset checkpoint exists.
        generated["npz_path"] = npz_name
        generated["npz_file"] = npz_name
        write_replace_json(metadata_path, generated)
    elif not (npz_path.is_file() and metadata_path.is_file()):
        raise ProtocolError(f"partial generated dataset cannot be resumed: {key}")

    generated_metadata = read_json(metadata_path)
    if (
        generated_metadata.get("shape") != _generator_shape(config["shape"])
        or generated_metadata.get("regime") != "calibrated"
        or generated_metadata.get("seed") != config["seed"]
        or generated_metadata.get("n_sensors_per_load") != [int(config["sensors"])] * 3
        or generated_metadata.get("sigma") != config["sigma"]
        or generated_metadata.get("npz_path") != npz_name
    ):
        raise ProtocolError(f"generated dataset does not match configuration: {key}")

    record = {
        "schema_version": "load-information-dataset-v3.0",
        "dataset_key": key,
        "config": config,
        "config_hash": config["config_hash"],
        "generator_shape": _generator_shape(config["shape"]),
        "memberships": config["memberships"],
        "design_hash": lock["design_hash"],
        "protocol_hash": lock["protocol_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "npz_file": Path("datasets").joinpath(key, npz_name).as_posix(),
        "metadata_file": Path("datasets").joinpath(key, metadata_name).as_posix(),
        "npz_sha256": hash_file(npz_path),
        "measured_load_mean": means.tolist(),
        "measured_load_error_z": z.tolist(),
        "true_load_standard_uncertainty": config["tau"],
        "true_gamma": [1.0, 1.0, 1.0],
    }
    write_new_json(record_path, record)
    return record


def build_forward(dataset_path: Path, physical: PhysicalModel) -> HermiteBeamFE:
    with np.load(dataset_path, allow_pickle=False) as data:
        centers = data["train_centers_nominal"]
    return HermiteBeamFE(
        centers,
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


def _observation_inputs(dataset_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], float]:
    with np.load(dataset_path, allow_pickle=False) as data:
        points = [np.asarray(row, dtype=float).copy() for row in data["sensor_locations"]]
        observations = [np.asarray(row, dtype=float).copy() for row in data["observations"]]
        sigma = float(data["sigma"])
    sigmas = [np.full_like(values, sigma, dtype=float) for values in observations]
    return points, observations, sigmas, sigma


def _h1_penalty(forward: HermiteBeamFE, coefficients: np.ndarray, penalty_lambda: float) -> np.ndarray:
    if penalty_lambda == 0.0:
        return np.empty(0, dtype=float)
    quadrature_points, weights, _, derivatives = forward.basis.penalty_quadrature()
    del quadrature_points
    full_derivative = derivatives[:, 1:-1] @ np.asarray(coefficients, dtype=float)
    return np.sqrt(penalty_lambda * weights) * full_derivative


def _physical_prediction(
    unit_prediction: np.ndarray,
    points: list[np.ndarray],
    gamma: np.ndarray,
) -> np.ndarray:
    rows = []
    offset = 0
    for index, point in enumerate(points):
        count = point.size
        rows.append(gamma[index] * unit_prediction[offset : offset + count])
        offset += count
    return np.asarray(rows, dtype=float)


def _numeric_array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name], dtype=float))
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _fit_one_candidate(
    forward: HermiteBeamFE,
    dataset_path: Path,
    method: str,
    penalty_lambda: float,
    design: dict,
    observed_mean: np.ndarray,
    tau: float,
) -> dict:
    points, observations, sigmas, _ = _observation_inputs(dataset_path)
    inputs = method_inputs(method, observed_mean, tau)
    started = time.perf_counter()
    # The oracle's privileged truth enters once, through the fixed load mean;
    # scaling forward.rhs as well would double-apply non-unit amplitudes.
    fit_gamma_mode = "fixed" if inputs["gamma_mode"] == "oracle" else inputs["gamma_mode"]
    result = fit_support(
        forward,
        points,
        observations,
        sigmas,
        gamma_mode=fit_gamma_mode,
        penalty="h1",
        penalty_lambda=penalty_lambda,
        starts=tuple(design["regularization"]["starts"]),
        max_nfev=int(design["regularization"]["max_nfev"]),
        load_mean=inputs["load_mean"],
        load_std=inputs["load_std"],
    )

    coefficients = np.asarray(result["coefficients"], dtype=float)
    gamma = np.asarray(result["gamma"], dtype=float)
    unit_prediction = forward.strain_and_jacobian(coefficients, points)[0]
    prediction = _physical_prediction(unit_prediction, points, gamma)
    residual_rows = []
    for index, values in enumerate(observations):
        residual_rows.append((prediction[index] - values) / sigmas[index])
    data_norm = float(np.linalg.norm(np.concatenate(residual_rows)))
    if inputs["finite_positive_prior"]:
        calibration_norm = float(np.linalg.norm((gamma - inputs["load_mean"]) / inputs["load_std"]))
    else:
        calibration_norm = 0.0
    selection_norm = float(math.hypot(data_norm, calibration_norm))
    penalty = _h1_penalty(forward, coefficients, penalty_lambda)
    penalty_norm = float(np.linalg.norm(penalty))
    independent_objective = float(selection_norm**2 + penalty_norm**2)
    reported_objective = float(result["starts"][result["chosen_start"]]["cost"])
    if not math.isclose(
        independent_objective,
        reported_objective,
        rel_tol=OBJECTIVE_RTOL,
        abs_tol=OBJECTIVE_ATOL,
    ):
        raise ProtocolError(
            f"{method} lambda={penalty_lambda}: objective replay differs by "
            f"{abs(independent_objective - reported_objective):.3e}"
        )
    result["elapsed_s"] = time.perf_counter() - started
    result["runner_inputs"] = inputs
    result["runner_prediction"] = prediction
    result["runner_norms"] = {
        "data_norm": data_norm,
        "calibration_norm": calibration_norm,
        "selection_norm": selection_norm,
        "penalty_norm": penalty_norm,
        "objective_norm": float(math.sqrt(independent_objective)),
        "objective": independent_objective,
    }
    return result


def _candidate_filename(penalty_lambda: float) -> str:
    return f"candidate_lambda_{_format_number(penalty_lambda)}.json"


def _candidate_record(
    *,
    result: dict,
    dataset: dict,
    dataset_record: dict,
    path_config: dict,
    method: str,
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
    source_hash = hash_json(lock["source_files"])
    return {
        "schema_version": "load-information-candidate-v3.0",
        "path_key": path_config["key"],
        "dataset_key": dataset["key"],
        "method": method,
        "penalty": "H1",
        "penalty_lambda": penalty_lambda,
        "gamma_mode": inputs["gamma_mode"],
        "load_mean": inputs["load_mean"].tolist(),
        "load_std": None if np.isinf(inputs["load_std"]) else float(inputs["load_std"]),
        "finite_positive_prior": inputs["finite_positive_prior"],
        "successful_start": bool(result["successful_start"]),
        "chosen_start": int(result["chosen_start"]),
        "starts": result["starts"],
        "coefficients": arrays["coefficients"].tolist(),
        "gamma": arrays["gamma"].tolist(),
        "train_prediction": arrays["train_prediction"].tolist(),
        "objective": result["runner_norms"]["objective"],
        "norms": result["runner_norms"],
        "array_hash": _numeric_array_hash(arrays),
        "dataset_hash": dataset_record["npz_sha256"],
        "dataset_file": dataset_record["npz_file"],
        "design_hash": lock["design_hash"],
        "source_hash": source_hash,
        "protocol_hash": lock["protocol_hash"],
        "design_file": "study_design_v3.json",
        "lambda_path": design["regularization"]["lambda_path"],
        "starts_used": design["regularization"]["starts"],
        "max_nfev": design["regularization"]["max_nfev"],
        "elapsed_s": result.get("elapsed_s", 0.0),
        "verification": {
            "objective_recomputed": result["runner_norms"]["objective"],
            "objective_absolute_difference": 0.0,
            "relative_tolerance": OBJECTIVE_RTOL,
            "absolute_tolerance": OBJECTIVE_ATOL,
            "passed": True,
        },
        "candidate_file": _candidate_filename(penalty_lambda),
    }


def _load_arrays(record: dict) -> dict[str, np.ndarray]:
    return {
        "coefficients": np.asarray(record["coefficients"], dtype=float),
        "gamma": np.asarray(record["gamma"], dtype=float),
        "train_prediction": np.asarray(record["train_prediction"], dtype=float),
    }


def _verify_candidate_record(
    record: dict,
    dataset_path: Path,
    physical: PhysicalModel,
) -> dict:
    errors = []
    arrays = _load_arrays(record)
    if _numeric_array_hash(arrays) != record.get("array_hash"):
        errors.append("array hash mismatch")
    mean = np.asarray(record["load_mean"], dtype=float)
    gamma = arrays["gamma"]
    if record["gamma_mode"] == "fixed" and not np.allclose(gamma, mean):
        errors.append("fixed gamma does not match its exact load mean")
    if record["gamma_mode"] == "oracle" and not np.allclose(arrays["gamma"], mean):
        errors.append("oracle gamma does not match privileged truth")
    forward = build_forward(dataset_path, physical)
    points, observations, sigmas, _ = _observation_inputs(dataset_path)
    coefficients = arrays["coefficients"]
    unit_prediction = forward.strain_and_jacobian(coefficients, points)[0]
    prediction = _physical_prediction(unit_prediction, points, gamma)
    residual = np.concatenate(
        [(prediction[index] - observations[index]) / sigmas[index] for index in range(len(points))]
    )
    data_norm = float(np.linalg.norm(residual))
    std = np.inf if record["load_std"] is None else float(record["load_std"])
    if record["finite_positive_prior"]:
        calibration_norm = float(np.linalg.norm((gamma - mean) / std))
    else:
        calibration_norm = 0.0
    selection_norm = float(math.hypot(data_norm, calibration_norm))
    penalty = _h1_penalty(forward, coefficients, float(record["penalty_lambda"]))
    penalty_norm = float(np.linalg.norm(penalty))
    objective = float(selection_norm**2 + penalty_norm**2)
    if not math.isclose(objective, float(record["objective"]), rel_tol=OBJECTIVE_RTOL, abs_tol=OBJECTIVE_ATOL):
        errors.append(
            f"objective mismatch: replay={objective:.17g}, reported={record['objective']:.17g}"
        )
    if not np.allclose(prediction, arrays["train_prediction"], rtol=1.0e-9, atol=1.0e-12):
        errors.append("train prediction mismatch")
    reported_norms = record["norms"]
    for name, replay in (
        ("data_norm", data_norm),
        ("calibration_norm", calibration_norm),
        ("selection_norm", selection_norm),
        ("penalty_norm", penalty_norm),
        ("objective_norm", float(math.sqrt(objective))),
    ):
        if not math.isclose(replay, float(reported_norms[name]), rel_tol=1.0e-9, abs_tol=1.0e-10):
            errors.append(f"{name} mismatch")
    chosen = int(record["chosen_start"])
    if chosen not in range(len(record["starts"])):
        errors.append("chosen start is out of range")
    else:
        chosen_cost = float(record["starts"][chosen]["cost"])
        if not math.isclose(chosen_cost, objective, rel_tol=OBJECTIVE_RTOL, abs_tol=OBJECTIVE_ATOL):
            errors.append("chosen start cost does not replay")
    if len(record["starts"]) != len(record["starts_used"]):
        errors.append("start count mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "recomputed": {
            "data_norm": data_norm,
            "calibration_norm": calibration_norm,
            "selection_norm": selection_norm,
            "penalty_norm": penalty_norm,
            "objective": objective,
        },
    }


def _selection_details(candidates: list[dict], sensors: int) -> dict:
    finite_prior = [candidate for candidate in candidates if candidate["finite_positive_prior"]]
    if len({candidate["finite_positive_prior"] for candidate in candidates}) != 1:
        raise ProtocolError("mixed prior status within one path")
    prior_count = 3 if finite_prior else 0
    observation_count = 3 * sensors
    temporary = [
        {
            "penalty_lambda": candidate["penalty_lambda"],
            "train_norm": candidate["norms"]["selection_norm"],
            "successful_start": candidate["successful_start"],
        }
        for candidate in candidates
    ]
    selection = select_candidate(temporary, observation_count + prior_count)
    selected_lambda = selection["selected_lambda"]
    selected = next(candidate for candidate in candidates if candidate["penalty_lambda"] == selected_lambda)
    selection["selection_norm"] = selection.pop("train_norm")
    selection["data_norm"] = selected["norms"].get("data_norm")
    selection["calibration_norm"] = selected["norms"].get("calibration_norm")
    selection["N_strain_observations"] = observation_count
    selection["M_calibration_observations"] = prior_count
    selection["threshold"] = 1.2 * math.sqrt(observation_count + prior_count)
    selection["candidate_count"] = len(candidates)
    return selection


def _existing_candidate(
    path_dir: Path,
    penalty_lambda: float,
    *,
    dataset: dict,
    dataset_record: dict,
    path_config: dict,
    method: str,
    lock: dict,
    output_dir: Path,
) -> dict | None:
    filename = _candidate_filename(penalty_lambda)
    path = path_dir / filename
    if not path.exists():
        return None
    record = read_json(path)
    source_hash = hash_json(lock["source_files"])
    expected_inputs = method_inputs(
        method,
        np.asarray(dataset_record["measured_load_mean"], dtype=float),
        float(dataset["tau"]),
    )
    expected_identity = {
        "schema_version": "load-information-candidate-v3.0",
        "path_key": path_config["key"],
        "dataset_key": dataset["key"],
        "method": method,
        "penalty": "H1",
        "penalty_lambda": penalty_lambda,
        "design_hash": lock["design_hash"],
        "source_hash": source_hash,
        "protocol_hash": lock["protocol_hash"],
        "dataset_hash": dataset_record["npz_sha256"],
        "candidate_file": filename,
        "gamma_mode": expected_inputs["gamma_mode"],
        "load_mean": expected_inputs["load_mean"].tolist(),
        "load_std": (
            None
            if np.isinf(expected_inputs["load_std"])
            else float(expected_inputs["load_std"])
        ),
        "finite_positive_prior": expected_inputs["finite_positive_prior"],
    }
    for key, value in expected_identity.items():
        if record.get(key) != value:
            raise ProtocolError(f"existing candidate mismatch for {path_config['key']} at {key}")
    verification = _verify_candidate_record(
        record, output_file(output_dir, dataset_record["npz_file"]), PhysicalModel()
    )
    if not verification["passed"]:
        raise ProtocolError(
            f"existing candidate failed replay: {path_config['key']} lambda={penalty_lambda}: "
            + "; ".join(verification["errors"])
        )
    return record


def _expected_candidate_names(design: dict, completed: bool, selected: bool | None = None) -> set[str]:
    names = {_candidate_filename(lam) for lam in design["regularization"]["lambda_path"]}
    if not completed:
        # A process can be killed after writing arrays but before path.json.
        names.add("selected_arrays.npz")
    if completed:
        names.add("path.json")
    if completed and selected:
        names.add("selected_arrays.npz")
    return names


def _validate_path_directory(path_dir: Path, design: dict, completed: bool, selected: bool | None = None) -> None:
    if not path_dir.exists():
        return
    if not path_dir.is_dir():
        raise ProtocolError(f"candidate path is not a directory: {path_dir.name}")
    actual = {item.name for item in path_dir.iterdir() if not _is_system_metadata(item.name)}
    expected = _expected_candidate_names(design, completed, selected)
    if not actual.issubset(expected):
        raise ProtocolError(
            f"unexpected files in path directory {path_dir.name}: "
            + ", ".join(sorted(actual - expected))
        )


def _write_or_reuse_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if not path.exists():
        write_new_npz(path, arrays)
        return
    try:
        with np.load(path, allow_pickle=False) as saved:
            matches = set(saved.files) == set(arrays) and all(
                not saved[name].dtype.hasobject
                and np.allclose(saved[name], arrays[name], rtol=1.0e-10, atol=1.0e-12)
                for name in arrays
            )
    except (OSError, ValueError):
        matches = False
    if not matches:
        raise ProtocolError(f"orphan selected arrays do not match replay: {path.name}")


def _run_one_path(
    path_config: dict,
    datasets: dict[str, dict],
    design: dict,
    output_dir: Path,
    lock: dict,
    deadline: float | None,
    forward_cache: dict[str, HermiteBeamFE] | None = None,
) -> tuple[dict, bool]:
    dataset = datasets[path_config["dataset_key"]]
    dataset_record = ensure_dataset(dataset, design, output_dir, lock)
    dataset_path = output_file(output_dir, dataset_record["npz_file"])
    path_dir = output_dir / "paths" / path_config["key"]
    path_file = path_dir / "path.json"
    if path_file.exists():
        existing = read_json(path_file)
        identity = {
            "path_key": path_config["key"],
            "dataset_key": dataset["key"],
            "method": path_config["method"],
            "design_hash": lock["design_hash"],
            "protocol_hash": lock["protocol_hash"],
            "source_hash": hash_json(lock["source_files"]),
            "dataset_hash": dataset_record["npz_sha256"],
        }
        for key, value in identity.items():
            if existing.get(key) != value:
                raise ProtocolError(f"existing completed path mismatch at {key}: {path_config['key']}")
        candidates = []
        for penalty_lambda in design["regularization"]["lambda_path"]:
            existing_candidate = _existing_candidate(
                path_dir,
                penalty_lambda,
                dataset=dataset,
                dataset_record=dataset_record,
                path_config=path_config,
                method=path_config["method"],
                lock=lock,
                output_dir=output_dir,
            )
            if existing_candidate is None:
                raise ProtocolError(
                    f"completed path is missing lambda {penalty_lambda}: {path_config['key']}"
                )
            candidates.append(existing_candidate)
        expected_candidate_files = [
            Path("paths").joinpath(path_config["key"], _candidate_filename(lam)).as_posix()
            for lam in design["regularization"]["lambda_path"]
        ]
        if (
            existing.get("candidate_files") != expected_candidate_files
            or existing.get("lambda_path") != design["regularization"]["lambda_path"]
            or existing.get("starts") != design["regularization"]["starts"]
        ):
            raise ProtocolError(f"existing completed path protocol mismatch: {path_config['key']}")
        selected_errors = _validate_selected_path(
            existing,
            candidates,
            dataset_path,
            PhysicalModel(),
            output_dir,
        )
        if selected_errors:
            raise ProtocolError(
                f"existing completed path failed replay: {path_config['key']}: "
                + "; ".join(selected_errors)
            )
        _validate_path_directory(
            path_dir,
            design,
            completed=True,
            selected=bool(existing.get("selected")),
        )
        return existing, True

    physical = PhysicalModel()
    if forward_cache is None:
        forward_cache = {}
    forward = forward_cache.get(dataset["key"])
    if forward is None:
        forward = build_forward(dataset_path, physical)
        forward_cache[dataset["key"]] = forward
    _validate_path_directory(path_dir, design, completed=False)
    candidates = []
    reuse = []
    for penalty_lambda in design["regularization"]["lambda_path"]:
        if deadline is not None and time.monotonic() >= deadline:
            return {
                "path_key": path_config["key"],
                "status": "timeout",
            }, False
        existing = _existing_candidate(
            path_dir,
            penalty_lambda,
            dataset=dataset,
            dataset_record=dataset_record,
            path_config=path_config,
            method=path_config["method"],
            lock=lock,
            output_dir=output_dir,
        )
        if existing is None:
            result = _fit_one_candidate(
                forward,
                dataset_path,
                path_config["method"],
                penalty_lambda,
                design,
                np.asarray(dataset_record["measured_load_mean"], dtype=float),
                float(dataset["tau"]),
            )
            record = _candidate_record(
                result=result,
                dataset=dataset,
                dataset_record=dataset_record,
                path_config=path_config,
                method=path_config["method"],
                penalty_lambda=penalty_lambda,
                design=design,
                lock=lock,
            )
            verification = _verify_candidate_record(record, dataset_path, physical)
            if not verification["passed"]:
                raise ProtocolError(
                    f"new candidate failed replay: {path_config['key']}: {verification['errors']}"
                )
            write_new_json(path_dir / _candidate_filename(penalty_lambda), record)
            reuse.append(False)
        else:
            record = existing
            reuse.append(True)
        candidates.append(record)

    selection = _selection_details(candidates, int(dataset["sensors"]))
    any_success = any(candidate["successful_start"] for candidate in candidates)
    path_record = {
        "schema_version": "load-information-path-v3.0",
        "path_key": path_config["key"],
        "dataset_key": dataset["key"],
        "method": path_config["method"],
        "method_family": path_config["method_family"],
        "penalty": "H1",
        "lambda_path": design["regularization"]["lambda_path"],
        "starts": design["regularization"]["starts"],
        "candidate_files": [
            Path("paths").joinpath(path_config["key"], _candidate_filename(lam)).as_posix()
            for lam in design["regularization"]["lambda_path"]
        ],
        "reused_candidates": reuse,
        "any_successful_start": any_success,
        "status": "selected" if any_success else "optimizer_failed",
        "selected": any_success,
        "selection": selection,
        "dataset_file": dataset_record["npz_file"],
        "dataset_hash": dataset_record["npz_sha256"],
        "design_hash": lock["design_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
        "design_file": "study_design_v3.json",
        "memberships": dataset["memberships"],
    }
    if any_success:
        selected_lambda = selection["selected_lambda"]
        selected = next(c for c in candidates if c["penalty_lambda"] == selected_lambda)
        metrics, arrays = score_selected(
            dataset_path,
            {"coefficients": np.asarray(selected["coefficients"]), "gamma": np.asarray(selected["gamma"])},
            physical,
        )
        arrays = dict(arrays)
        arrays["coefficients"] = np.asarray(selected["coefficients"], dtype=float)
        arrays["gamma"] = np.asarray(selected["gamma"], dtype=float)
        arrays["train_prediction"] = np.asarray(selected["train_prediction"], dtype=float)
        arrays_path = path_dir / "selected_arrays.npz"
        _write_or_reuse_arrays(arrays_path, arrays)
        path_record["metrics"] = metrics
        path_record["selected_start"] = selected["chosen_start"]
        path_record["selected_coefficients"] = selected["coefficients"]
        path_record["selected_gamma"] = selected["gamma"]
        path_record["selected_train_prediction"] = selected["train_prediction"]
        path_record["arrays_file"] = Path("paths").joinpath(path_config["key"], "selected_arrays.npz").as_posix()
        path_record["arrays_sha256"] = hash_file(arrays_path)
    write_new_json(path_file, path_record)
    _validate_path_directory(path_dir, design, completed=True, selected=any_success)
    return path_record, False


def _validate_selected_path(
    path_record: dict,
    candidates: list[dict],
    dataset_path: Path,
    physical: PhysicalModel,
    output_dir: Path,
) -> list[str]:
    errors: list[str] = []
    actual_success = any(candidate["successful_start"] for candidate in candidates)
    if (
        path_record.get("any_successful_start") != actual_success
        or path_record.get("selected") != actual_success
        or path_record.get("status") != ("selected" if actual_success else "optimizer_failed")
    ):
        errors.append("path success status mismatch")
        return errors
    selection = _selection_details(candidates, sensors=int(len(candidates[0]["train_prediction"][0])))
    stored_selection = path_record["selection"]
    for key in ("selected_lambda", "selection_reason", "optimizer_failed", "no_feasible_candidate"):
        if selection[key] != stored_selection.get(key):
            errors.append(f"selection field mismatch: {key}")
    if not path_record["any_successful_start"]:
        if path_record.get("selected") or "metrics" in path_record or "arrays_file" in path_record:
            errors.append("failed path reports a selected solution")
        return errors
    selected_lambda = selection["selected_lambda"]
    selected = next(c for c in candidates if c["penalty_lambda"] == selected_lambda)
    if not selected["successful_start"]:
        errors.append("selection uses a failed candidate")
    if int(path_record.get("selected_start", -1)) != int(selected["chosen_start"]):
        errors.append("selected start mismatch")
    metrics, arrays = score_selected(
        dataset_path,
        {"coefficients": np.asarray(selected["coefficients"]), "gamma": np.asarray(selected["gamma"])},
        physical,
    )
    for name, replay in metrics.items():
        reported = path_record.get("metrics", {}).get(name)
        if replay is None:
            if reported is not None:
                errors.append(f"selected metric mismatch: {name}")
        elif reported is None or not math.isclose(
            float(replay), float(reported), rel_tol=1.0e-8, abs_tol=1.0e-8
        ):
            errors.append(f"selected metric mismatch: {name}")
    arrays = dict(arrays)
    arrays["coefficients"] = np.asarray(selected["coefficients"], dtype=float)
    arrays["gamma"] = np.asarray(selected["gamma"], dtype=float)
    arrays["train_prediction"] = np.asarray(selected["train_prediction"], dtype=float)
    try:
        arrays_path = output_file(output_dir, path_record.get("arrays_file", ""))
    except ProtocolError:
        arrays_path = None
        errors.append("selected arrays filename is not relative")
    if arrays_path is None or not arrays_path.is_file():
        errors.append("selected arrays file is missing")
    else:
        if hash_file(arrays_path) != path_record.get("arrays_sha256"):
            errors.append("selected arrays hash mismatch")
        with np.load(arrays_path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(saved[name].dtype.hasobject for name in saved.files):
                errors.append("selected arrays schema mismatch")
            else:
                for name in arrays:
                    if not np.allclose(saved[name], arrays[name], rtol=1.0e-8, atol=1.0e-10):
                        errors.append(f"selected array mismatch: {name}")
    return errors


def _iter_expected_candidate_files(output_dir: Path, paths: list[dict], design: dict):
    for path_config in paths:
        path_dir = output_dir / "paths" / path_config["key"]
        for lam in design["regularization"]["lambda_path"]:
            yield path_config, path_dir / _candidate_filename(lam)


def verify_protocol(output_dir: Path, design: dict) -> dict:
    errors_by_path: dict[str, list[str]] = {}
    incomplete_paths: list[str] = []
    try:
        lock = load_and_check_lock(output_dir, design)
    except ProtocolError as error:
        return {
            "passed": False,
            "incomplete": False,
            "checked_candidates": 0,
            "checked_paths": 0,
            "errors": {"lock": [str(error)]},
        }
    datasets, paths, _ = expand_study(design)
    datasets_by_key = {dataset["key"]: dataset for dataset in datasets}
    dataset_records = {}
    for dataset in datasets:
        dataset_dir = output_dir / "datasets" / dataset["key"]
        record_path = dataset_dir / "dataset_record.json"
        if not record_path.exists():
            continue
        record = read_json(record_path)
        dataset_records[dataset["key"]] = record
        expected_npz = Path("datasets").joinpath(
            dataset["key"], _dataset_generated_names(dataset)[0]
        ).as_posix()
        dataset_errors = []
        try:
            npz_path = output_file(output_dir, record.get("npz_file", ""))
        except ProtocolError:
            npz_path = None
            dataset_errors.append("dataset filename is not relative")
        if (
            record.get("dataset_key") != dataset["key"]
            or record.get("config") != dataset
            or record.get("config_hash") != dataset["config_hash"]
            or record.get("design_hash") != lock["design_hash"]
            or record.get("protocol_hash") != lock["protocol_hash"]
            or record.get("source_hash") != hash_json(lock["source_files"])
            or record.get("npz_file") != expected_npz
        ):
            dataset_errors.append("dataset record identity mismatch")
        if npz_path is None or not npz_path.is_file() or hash_file(npz_path) != record.get("npz_sha256"):
            errors_by_path.setdefault(dataset["key"], []).append("dataset hash mismatch")
        if dataset_errors:
            errors_by_path.setdefault(dataset["key"], []).extend(dataset_errors)

    physical = PhysicalModel()
    checked_candidates = 0
    for path_config, candidate_path in _iter_expected_candidate_files(output_dir, paths, design):
        if not candidate_path.exists():
            incomplete_paths.append(path_config["key"])
            continue
        dataset_record = dataset_records.get(path_config["dataset_key"])
        if dataset_record is None:
            errors_by_path.setdefault(path_config["key"], []).append("candidate has no dataset record")
            continue
        record = read_json(candidate_path)
        identity_errors = []
        if record.get("path_key") != path_config["key"]:
            identity_errors.append("path key mismatch")
        if record.get("method") != path_config["method"]:
            identity_errors.append("method mismatch")
        if record.get("dataset_hash") != dataset_record["npz_sha256"]:
            identity_errors.append("dataset hash mismatch")
        if record.get("design_hash") != lock["design_hash"]:
            identity_errors.append("design hash mismatch")
        if record.get("source_hash") != hash_json(lock["source_files"]):
            identity_errors.append("source hash mismatch")
        expected_file = Path("datasets").joinpath(
            path_config["dataset_key"], _dataset_generated_names(datasets_by_key[path_config["dataset_key"]])[0]
        ).as_posix()
        if record.get("dataset_file") != expected_file:
            identity_errors.append("dataset filename mismatch")
        expected_inputs = method_inputs(
            path_config["method"],
            np.asarray(dataset_record["measured_load_mean"], dtype=float),
            float(datasets_by_key[path_config["dataset_key"]]["tau"]),
        )
        if (
            record.get("gamma_mode") != expected_inputs["gamma_mode"]
            or record.get("load_mean") != expected_inputs["load_mean"].tolist()
            or record.get("load_std")
            != (None if np.isinf(expected_inputs["load_std"]) else float(expected_inputs["load_std"]))
            or record.get("finite_positive_prior") != expected_inputs["finite_positive_prior"]
        ):
            identity_errors.append("load-information mapping mismatch")
        try:
            replay = _verify_candidate_record(
                record, output_file(output_dir, dataset_record["npz_file"]), physical
            )
        except ProtocolError as error:
            replay = {"passed": False, "errors": [str(error)]}
        identity_errors.extend(replay["errors"])
        checked_candidates += 1
        if identity_errors:
            errors_by_path.setdefault(path_config["key"], []).extend(identity_errors)

    checked_paths = 0
    optimizer_failed_paths = []
    for path_config in paths:
        path_file = output_dir / "paths" / path_config["key"] / "path.json"
        if not path_file.exists():
            # Missing path.json means incomplete even when every candidate exists.
            incomplete_paths.append(path_config["key"])
            try:
                _validate_path_directory(
                    output_dir / "paths" / path_config["key"], design, completed=False
                )
            except ProtocolError as error:
                errors_by_path.setdefault(path_config["key"], []).append(str(error))
            continue
        checked_paths += 1
        path_record = read_json(path_file)
        path_errors = []
        dataset_record = dataset_records.get(path_config["dataset_key"])
        if dataset_record is None:
            path_errors.append("completed path has no dataset record")
            errors_by_path.setdefault(path_config["key"], []).extend(path_errors)
            continue
        candidates = []
        for lam in design["regularization"]["lambda_path"]:
            candidate_path = output_dir / "paths" / path_config["key"] / _candidate_filename(lam)
            if not candidate_path.exists():
                path_errors.append("completed path is missing a candidate")
                continue
            candidates.append(read_json(candidate_path))
        if len(candidates) == len(design["regularization"]["lambda_path"]):
            expected_candidate_files = [
                Path("paths").joinpath(path_config["key"], _candidate_filename(lam)).as_posix()
                for lam in design["regularization"]["lambda_path"]
            ]
            if (
                path_record.get("path_key") != path_config["key"]
                or path_record.get("dataset_key") != path_config["dataset_key"]
                or path_record.get("method") != path_config["method"]
                or path_record.get("method_family") != path_config.get("method_family")
                or path_record.get("candidate_files") != expected_candidate_files
                or path_record.get("lambda_path") != design["regularization"]["lambda_path"]
                or path_record.get("starts") != design["regularization"]["starts"]
                or path_record.get("design_hash") != lock["design_hash"]
                or path_record.get("source_hash") != hash_json(lock["source_files"])
                or path_record.get("protocol_hash") != lock["protocol_hash"]
                or path_record.get("dataset_hash") != dataset_record["npz_sha256"]
            ):
                path_errors.append("completed path identity mismatch")
            try:
                dataset_path = output_file(output_dir, dataset_record["npz_file"])
                path_errors.extend(
                    _validate_selected_path(
                        path_record,
                        candidates,
                        dataset_path,
                        physical,
                        output_dir,
                    )
                )
            except ProtocolError as error:
                path_errors.append(str(error))
            if not any(candidate["successful_start"] for candidate in candidates):
                optimizer_failed_paths.append(path_config["key"])
        if path_errors:
            errors_by_path.setdefault(path_config["key"], []).extend(path_errors)
        try:
            _validate_path_directory(
                output_dir / "paths" / path_config["key"],
                design,
                completed=True,
                selected=bool(path_record.get("selected")),
            )
        except ProtocolError as error:
            errors_by_path.setdefault(path_config["key"], []).append(str(error))

    actual_path_dirs = {
        item.name
        for item in (output_dir / "paths").glob("*")
        if item.is_dir() and not _is_system_metadata(item.name)
    } if (output_dir / "paths").exists() else set()
    expected_path_dirs = {path_config["key"] for path_config in paths}
    for extra in sorted(actual_path_dirs - expected_path_dirs):
        errors_by_path.setdefault(extra, []).append("unexpected path directory")
    if (output_dir / "datasets").exists():
        actual_dataset_dirs = {
            item.name
            for item in (output_dir / "datasets").glob("*")
            if item.is_dir() and not _is_system_metadata(item.name)
        }
        expected_dataset_dirs = {dataset["key"] for dataset in datasets}
        for extra in sorted(actual_dataset_dirs - expected_dataset_dirs):
            errors_by_path.setdefault(extra, []).append("unexpected dataset directory")

    passed = not errors_by_path
    report = {
        "schema_version": "load-information-verification-v3.0",
        "verified_utc": utc_now(),
        "passed": passed,
        "incomplete": bool(incomplete_paths) or checked_paths < len(paths),
        "incomplete_paths": sorted(set(incomplete_paths)),
        "checked_candidates": checked_candidates,
        "checked_completed_paths": checked_paths,
        "optimizer_failed_paths": optimizer_failed_paths,
        "expected_candidates": len(paths) * len(design["regularization"]["lambda_path"]),
        "expected_paths": len(paths),
        "errors": errors_by_path,
    }
    write_replace_json(output_dir / "verification_report.json", report)
    return report


def _quick_status(output_dir: Path, design: dict) -> dict:
    try:
        lock = load_and_check_lock(output_dir, design)
        lock_error = None
    except ProtocolError as error:
        lock = None
        lock_error = str(error)
    datasets, paths, counts = expand_study(design)
    dataset_count = 0
    path_count = 0
    candidate_count = 0
    if lock is not None:
        for dataset in datasets:
            if (output_dir / "datasets" / dataset["key"] / "dataset_record.json").is_file():
                dataset_count += 1
        for path_config in paths:
            if (output_dir / "paths" / path_config["key"] / "path.json").is_file():
                path_count += 1
            candidate_count += sum(
                (output_dir / "paths" / path_config["key"] / _candidate_filename(lam)).is_file()
                for lam in design["regularization"]["lambda_path"]
            )
    return {
        "schema_version": "load-information-status-v3.0",
        "status_utc": utc_now(),
        "lock_valid": lock_error is None,
        "lock_error": lock_error,
        "counts": counts,
        "existing": {
            "datasets": dataset_count,
            "completed_paths": path_count,
            "candidates": candidate_count,
        },
    }


def execute_run(
    output_dir: Path,
    design: dict,
    *,
    resume: bool,
    limit_paths: int | None = None,
) -> dict:
    lock = load_and_check_lock(output_dir, design)
    datasets, paths, counts = expand_study(design)
    if limit_paths is not None and limit_paths < 0:
        raise ProtocolError("--limit-paths must be nonnegative")
    selected_paths = paths if limit_paths is None else paths[:limit_paths]
    existing_entries = [
        item
        for item in output_dir.iterdir()
        if item.name != LOCK_FILE and not _is_system_metadata(item.name)
    ] if output_dir.exists() else []
    if existing_entries and not resume:
        raise ProtocolError("output contains results; use --resume to continue them")
    datasets_by_key = {dataset["key"]: dataset for dataset in datasets}
    budget_seconds = float(design["resource_limit"]["wall_time_limit_hours"]) * 3600.0
    prior_status_path = output_dir / "run_status.json"
    if prior_status_path.is_file():
        prior_status = read_json(prior_status_path)
        started_utc = prior_status.get("wall_time_started_utc")
    else:
        started_utc = None
    if started_utc is None:
        started_utc = utc_now()
        elapsed_seconds = 0.0
    else:
        started_time = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
        elapsed_seconds = max(0.0, (datetime.now(timezone.utc) - started_time).total_seconds())
    remaining_seconds = max(0.0, budget_seconds - elapsed_seconds)
    deadline = time.monotonic() + remaining_seconds
    completed = 0
    reused_paths = 0
    timeout = False
    last_error = None
    forward_cache = {}
    initial_status = {
        "schema_version": "load-information-run-status-v3.0",
        "updated_utc": utc_now(),
        "wall_time_started_utc": started_utc,
        "status": "running",
        "completed_paths": completed,
        "reused_completed_paths": reused_paths,
        "selected_path_count": len(selected_paths),
        "expected_full_counts": counts,
        "limit_paths": limit_paths,
    }
    write_replace_json(output_dir / "run_status.json", initial_status)
    try:
        for path_config in selected_paths:
            if time.monotonic() >= deadline:
                timeout = True
                break
            path_record, was_reused = _run_one_path(
                path_config,
                datasets_by_key,
                design,
                output_dir,
                lock,
                deadline,
                forward_cache,
            )
            if path_record.get("status") == "timeout":
                # An unfinished path is not completed; preserve the checkpoint.
                timeout = True
                break
            completed += 1
            reused_paths += int(was_reused)
            status = {
                "schema_version": "load-information-run-status-v3.0",
                "updated_utc": utc_now(),
                "wall_time_started_utc": started_utc,
                "status": "running",
                "completed_paths": completed,
                "reused_completed_paths": reused_paths,
                "selected_path_count": len(selected_paths),
                "expected_full_counts": counts,
                "limit_paths": limit_paths,
            }
            write_replace_json(output_dir / "run_status.json", status)
    except Exception as error:  # preserve checkpoint evidence before CLI exits
        last_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        status = {
            "schema_version": "load-information-run-status-v3.0",
            "updated_utc": utc_now(),
            "wall_time_started_utc": started_utc,
            "wall_time_budget_seconds": budget_seconds,
            "wall_time_elapsed_seconds": elapsed_seconds + (time.monotonic() - (deadline - remaining_seconds)),
            "status": "error" if last_error else ("timeout" if timeout else "completed"),
            "completed_paths": completed,
            "reused_completed_paths": reused_paths,
            "selected_path_count": len(selected_paths),
            "expected_full_counts": counts,
            "limit_paths": limit_paths,
            "error": last_error,
        }
        write_replace_json(output_dir / "run_status.json", status)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "run", "status", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output", required=True, help="relative output directory")
        if command == "run":
            subparser.add_argument("--resume", action="store_true")
            subparser.add_argument("--limit-paths", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = validate_relative_output(args.output)
    design = load_design()
    if args.command == "freeze":
        lock = freeze_protocol(output_dir, design)
        print(canonical_json({"command": "freeze", "output": args.output, "protocol_hash": lock["protocol_hash"]}))
        return 0
    if args.command == "run":
        report = execute_run(
            output_dir,
            design,
            resume=args.resume,
            limit_paths=args.limit_paths,
        )
    elif args.command == "status":
        report = _quick_status(output_dir, design)
        write_replace_json(output_dir / "status_report.json", report)
    else:
        report = verify_protocol(output_dir, design)
    print(canonical_json(safe_json(report)))
    if args.command == "verify":
        # passed covers integrity; formal use also requires completeness.
        return 0 if report["passed"] and not report["incomplete"] else 2
    if args.command == "run":
        return 0 if report["status"] == "completed" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
