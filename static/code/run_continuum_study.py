"""Serial continuum mesh, calibration, inversion, and verification runner."""

from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import hashlib
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "vendor", ROOT / "code"):
    absolute = str(path.resolve())
    if absolute not in sys.path:
        sys.path.insert(0, absolute)

from beam_inverse import HermiteBeamFE, SupportBasis, profile_load_amplitude  # noqa: E402
from campaign import PhysicalModel  # noqa: E402
from continuum_reference import continuum_forward  # noqa: E402
from run_load_information import (  # noqa: E402
    ProtocolError,
    _candidate_filename,
    _fit_one_candidate,
    _selection_details,
    calibration_measurement,
    canonical_json,
    hash_file,
    hash_json,
    method_inputs,
    output_file,
    read_json,
    safe_json,
    validate_relative_output,
    write_new_json,
    write_new_npz,
    write_replace_json,
)
from shape_factory import get_shape_function  # noqa: E402
from skfem_reference import mesh_with_load_edges  # noqa: E402


LOCK_FILE = "continuum_source_lock.json"
MESH_REPORT = "mesh_check/mesh_report.json"
CALIBRATION_REPORT = "calibration/calibration_report.json"
SOURCE_FILES = (
    "code/beam_inverse.py",
    "code/campaign.py",
    "code/continuum_reference.py",
    "code/run_campaign.py",
    "code/run_continuum_study.py",
    "code/run_load_information.py",
    "code/shape_factory.py",
    "code/skfem_reference.py",
    "continuum_design_v3.json",
    "study_design_v3.json",
    "vendor/skfem/__init__.py",
    "vendor/scikit_fem-12.0.2.dist-info/METADATA",
    "vendor/scikit_fem-12.0.2.dist-info/WHEEL",
)
TRAIN_CENTERS = np.array([-4.0, 0.0, 4.0])
ALL_CENTERS = np.array([-4.0, 0.0, 4.0, 6.0])
UNUSED_CENTER = 6.0
SCORING_GRID = np.linspace(-12.0, 12.0, 1201)
UNUSED_GRID = np.linspace(-12.0, 12.0, 241)
CENTROID_ZERO_AREA_M = 1.0e-8
OBJECTIVE_RTOL = 1.0e-8
OBJECTIVE_ATOL = 1.0e-8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def mapped_hashes(names: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = ROOT / name
        if not path.is_file():
            raise ProtocolError(f"required source file is missing: {name}")
        result[name] = hash_file(path)
    return result


def current_environment() -> dict[str, str]:
    import scipy
    import skfem

    return {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "skfem_version": skfem.__version__,
        "platform": sys.platform,
        "machine": sys.platform,
    }


def load_designs() -> tuple[dict, dict]:
    continuum = read_json(ROOT / "continuum_design_v3.json")
    study = read_json(ROOT / "study_design_v3.json")
    if study.get("schema") != "load-information-v3.0":
        raise ProtocolError("unexpected study design schema")
    expected = continuum["inversion"]
    counts = (
        len(expected["shapes"]) * len(continuum["soil_E_Pa"])
        * len(expected["tau"]) * len(expected["seeds"])
    )
    if (
        counts != expected["data_count"]
        or counts * len(expected["methods"]) != expected["path_count"]
        or counts * len(expected["methods"]) * len(study["regularization"]["lambda_path"])
        != expected["candidate_count"]
    ):
        raise ProtocolError("continuum design counts are internally inconsistent")
    if (
        study["methods"] != expected["methods"]
        or study["force_generation"]["true_multipliers"] != expected["force_multipliers"]
        or study["regularization"]["penalty"] != "h1"
        or study["regularization"]["max_nfev"] != 200
    ):
        raise ProtocolError("study design does not match the frozen continuum inversion protocol")
    return continuum, study


def _lock_fingerprint(
    source_hashes: dict[str, str],
    design_hashes: dict[str, str],
    environment: dict[str, str],
) -> str:
    return hash_json(
        {
            "source_hashes": source_hashes,
            "design_hashes": design_hashes,
            "environment": environment,
            "thread_settings": {
                name: os.environ.get(name, "unset")
                for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        }
    )


def _new_lock() -> dict:
    continuum, study = load_designs()
    design_hashes = mapped_hashes(("continuum_design_v3.json", "study_design_v3.json"))
    environment = current_environment()
    lock = {
        "schema_version": "continuum-source-lock-v3.0",
        "created_utc": utc_now(),
        "source_files": mapped_hashes(SOURCE_FILES),
        "design_hashes": design_hashes,
        "design_hash": hash_json({"continuum": continuum, "study": study}),
        "environment": environment,
        "thread_settings": {
            name: os.environ.get(name, "unset")
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "interpretation": continuum["interpretation"],
    }
    lock["protocol_hash"] = _lock_fingerprint(
        lock["source_files"], lock["design_hashes"], environment
    )
    return lock


def _check_lock(lock: dict) -> dict:
    if lock.get("schema_version") != "continuum-source-lock-v3.0":
        raise ProtocolError("unsupported continuum source lock")
    expected = _new_lock()
    for key in ("source_files", "design_hashes", "design_hash", "environment", "thread_settings"):
        if lock.get(key) != expected[key]:
            raise ProtocolError(f"source/design lock mismatch for {key}")
    if lock.get("protocol_hash") != expected["protocol_hash"]:
        raise ProtocolError("continuum protocol hash mismatch")
    return lock


def _create_lock(output_dir: Path) -> dict:
    path = output_dir / LOCK_FILE
    if path.exists():
        return _check_lock(read_json(path))
    if output_dir.exists() and any(
        not _is_platform_metadata(item.name) for item in output_dir.iterdir()
    ):
        raise ProtocolError("check-mesh output directory is not empty and has no lock")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = _new_lock()
    write_new_json(path, lock)
    return lock


def _load_lock(output_dir: Path) -> dict:
    path = output_dir / LOCK_FILE
    if not path.is_file():
        raise ProtocolError(f"missing source/design lock: {path.relative_to(ROOT).as_posix()}")
    return _check_lock(read_json(path))


def _number(value: float) -> str:
    return format(float(value), ".17g").replace("+", "")


def _is_platform_metadata(name: str) -> bool:
    """Known macOS/ExFAT sidecar names that are never study material."""
    return name == ".DS_Store" or name.startswith("._")


def _soil_key(soil_e: float) -> str:
    return f"E{int(round(float(soil_e) / 1.0e6))}MPa"


def _dataset_key(shape: str, soil_e: float, tau: float, seed: int) -> str:
    return f"shape-{shape}_{_soil_key(soil_e)}_tau-{_number(tau)}_seed-{int(seed)}"


def expand_continuum_study(
    continuum: dict, study: dict
) -> tuple[list[dict], list[dict], dict[str, int]]:
    datasets: list[dict] = []
    inversion = continuum["inversion"]
    for shape in inversion["shapes"]:
        for soil_e in continuum["soil_E_Pa"]:
            for tau in inversion["tau"]:
                for seed in inversion["seeds"]:
                    key = _dataset_key(shape, soil_e, tau, seed)
                    datasets.append(
                        {
                            "key": key,
                            "shape": shape,
                            "soil_E_Pa": float(soil_e),
                            "tau": float(tau),
                            "sigma": float(inversion["strain_sigma"]),
                            "sensors": int(inversion["sensors_per_load"]),
                            "seed": int(seed),
                        }
                    )
    methods = inversion["methods"]
    paths = [
        {"key": f"{dataset['key']}__method-{method}", "dataset_key": dataset["key"], "method": method}
        for dataset in datasets
        for method in methods
    ]
    lambdas = study["regularization"]["lambda_path"]
    starts = study["regularization"]["starts"]
    full_lambda_count = inversion["candidate_count"] // inversion["path_count"]
    counts = {
        "datasets": len(datasets),
        "paths": len(paths),
        "candidates": len(paths) * len(lambdas),
        "starts": len(paths) * len(lambdas) * len(starts),
    }
    if len(lambdas) == full_lambda_count:
        expected = {
            "datasets": inversion["data_count"],
            "paths": inversion["path_count"],
            "candidates": inversion["candidate_count"],
            "starts": 1728,
        }
        if counts != expected:
            raise ProtocolError(f"expanded counts do not match frozen design: {counts} != {expected}")
    elif counts["datasets"] != inversion["data_count"] or counts["paths"] != inversion["path_count"]:
        raise ProtocolError("dataset or path expansion does not match the frozen continuum design")
    return datasets, paths, counts


def _sensor_points(count: int) -> np.ndarray:
    return np.linspace(-12.0, 12.0, int(count))


def _mesh_for(h: float, centres: np.ndarray, half_domain: float, *, asymmetric: bool) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    reflected = -centres
    all_centres = np.unique(np.concatenate([centres, reflected]))
    return mesh_with_load_edges(
        -float(half_domain),
        float(half_domain),
        float(h),
        all_centres,
        load_width=PhysicalModel().footprint_m,
        extra_edges=np.array([1.2]) if asymmetric else None,
    )


def _field_arrays(result: dict) -> dict[str, np.ndarray]:
    names = (
        "evaluation_points",
        "displacement",
        "strain",
        "beam_nodes",
        "beam_displacements",
        "beam_rotations",
        "soil_nodes",
        "soil_quads",
        "soil_ux",
        "soil_uz",
        "soil_surface_vertical_displacement",
    )
    return {name: np.asarray(result[name]) for name in names}


def _solve_continuum(
    *,
    soil_e: float,
    shape: str,
    h: float,
    dz: float,
    half_domain: float,
    load_centres: np.ndarray,
    evaluation_points: np.ndarray,
    design: dict,
) -> tuple[dict, np.ndarray, float]:
    common = PhysicalModel()
    points = _mesh_for(h, load_centres, half_domain, asymmetric=shape == "asymmetric")
    depth = np.linspace(0.0, float(design["layer_depth_m"]), round(float(design["layer_depth_m"]) / dz + 1))
    started = time.perf_counter()
    result = continuum_forward(
        points,
        depth,
        np.asarray(load_centres, dtype=float),
        get_shape_function(shape),
        np.asarray(evaluation_points, dtype=float),
        common.gauge_m,
        common.EI_N_m2,
        float(design["interface_stiffness_N_m2"]),
        float(soil_e),
        float(design["soil_nu"]),
        float(design["out_of_plane_width_m"]),
        common.nominal_force_N,
        common.footprint_m,
        common.radius_m,
        support_discontinuity_points=np.array([1.2]) if shape == "asymmetric" else None,
    )
    elapsed = time.perf_counter() - started
    return result, points, elapsed


def _physics_errors(diagnostics: dict) -> dict[str, float]:
    applied = np.asarray(diagnostics["applied_force"], dtype=float)
    reaction_error = np.asarray(diagnostics["vertical_reaction_error"], dtype=float)
    energy_error = np.asarray(diagnostics["relative_energy_balance_error"], dtype=float)
    resultant = np.asarray(diagnostics["load_resultant"], dtype=float)
    return {
        "max_relative_reaction_error": float(np.max(np.abs(reaction_error / applied))),
        "max_relative_energy_error": float(np.max(np.abs(energy_error))),
        "max_relative_load_resultant_error": float(np.max(np.abs((resultant - applied) / applied))),
    }


def _write_continuum_run(
    output_dir: Path,
    *,
    parent: str,
    soil_e: float,
    shape: str,
    h: float,
    dz: float,
    half_domain: float,
    load_centres: np.ndarray,
    evaluation_points: np.ndarray,
    design: dict,
    lock: dict,
) -> tuple[dict, Path]:
    run_name = f"{_soil_key(soil_e)}_{shape}_L{int(half_domain)}_h{_number(h)}_dz{_number(dz)}"
    run_dir = output_dir / parent / "runs" / run_name
    fields_path = run_dir / "fields.npz"
    record_path = run_dir / "diagnostics.json"
    if record_path.exists():
        if not fields_path.is_file() or hash_file(fields_path) != read_json(record_path).get("fields_sha256"):
            raise ProtocolError(f"partial continuum run cannot be resumed: {run_name}")
        return read_json(record_path), fields_path
    if run_dir.exists() and any(
        not _is_platform_metadata(item.name) for item in run_dir.iterdir()
    ):
        raise ProtocolError(f"refusing to overwrite partial continuum run: {run_name}")
    result, points, elapsed = _solve_continuum(
        soil_e=soil_e,
        shape=shape,
        h=h,
        dz=dz,
        half_domain=half_domain,
        load_centres=load_centres,
        evaluation_points=evaluation_points,
        design=design,
    )
    arrays = _field_arrays(result)
    write_new_npz(fields_path, arrays)
    record = {
        "schema_version": "continuum-mesh-run-v3.0",
        "run_name": run_name,
        "soil_E_Pa": float(soil_e),
        "shape": shape,
        "half_domain_m": float(half_domain),
        "h_m": float(h),
        "dz_m": float(dz),
        "load_centres_m": np.asarray(load_centres).tolist(),
        "evaluation_points_m": arrays["evaluation_points"].tolist(),
        "mesh_points_hash": hash_json({"points": points.tolist()}),
        "depth_nodes_hash": hash_json(
            {"depth": np.linspace(0.0, design["layer_depth_m"], round(design["layer_depth_m"] / dz + 1)).tolist()}
        ),
        "diagnostics": result["diagnostics"],
        "mesh": result["mesh"],
        "solver": result["solver"],
        "runner_elapsed_s": float(elapsed),
        "physics_relative_errors": _physics_errors(result["diagnostics"]),
        "fields_file": Path(parent).joinpath("runs", run_name, "fields.npz").as_posix(),
        "fields_sha256": hash_file(fields_path),
        "design_hash": lock["design_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
    }
    write_new_json(record_path, record)
    return record, fields_path


def _change_metrics(coarse: np.ndarray, fine: np.ndarray) -> dict[str, float]:
    difference = np.asarray(fine, dtype=float) - np.asarray(coarse, dtype=float)
    return {
        "rms_change_microstrain": float(np.sqrt(np.mean(difference**2)) * 1.0e6),
        "relative_L2_change": float(np.linalg.norm(difference) / np.linalg.norm(fine)),
    }


def _load_run_strain(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["strain"], dtype=float).copy()


def _check_mesh(output_dir: Path, continuum: dict, study: dict) -> dict:
    lock = _create_lock(output_dir)
    existing_report_path = output_dir / MESH_REPORT
    if existing_report_path.is_file():
        existing = read_json(existing_report_path)
        if (
            existing.get("schema_version") != "continuum-mesh-report-v3.0"
            or existing.get("design_hash") != lock["design_hash"]
            or existing.get("source_hash") != hash_json(lock["source_files"])
            or existing.get("protocol_hash") != lock["protocol_hash"]
        ):
            raise ProtocolError("existing mesh report does not match the source/design lock")
        for run in existing["runs"]:
            fields = output_file(output_dir, run["fields_file"])
            if not fields.is_file() or hash_file(fields) != run["fields_sha256"]:
                raise ProtocolError(f"existing mesh fields hash mismatch: {run['run_name']}")
        return existing
    source_hash = hash_json(lock["source_files"])
    sensor_points = _sensor_points(continuum["inversion"]["sensors_per_load"])
    evaluation = np.unique(np.concatenate([UNUSED_GRID, sensor_points]))
    records: dict[tuple[float, str, float], dict] = {}
    extent_records: dict[tuple[float, str], dict] = {}
    errors: list[str] = []
    for soil_e in continuum["soil_E_Pa"]:
        for shape in continuum["mesh_check_shapes"]:
            for size in continuum["mesh_check_sizes_m"]:
                try:
                    record, path = _write_continuum_run(
                        output_dir,
                        parent="mesh_check",
                        soil_e=soil_e,
                        shape=shape,
                        h=size,
                        dz=size,
                        half_domain=continuum["half_domain_m"],
                        load_centres=ALL_CENTERS,
                        evaluation_points=evaluation,
                        design=continuum,
                        lock=lock,
                    )
                    records[(float(soil_e), shape, float(size))] = record
                    if max(record["physics_relative_errors"].values()) > continuum["mesh_acceptance"]["force_and_energy_relative_max"]:
                        errors.append(f"physics balance failed for {record['run_name']}")
                except Exception as error:
                    errors.append(f"{_soil_key(soil_e)} {shape} h={size}: {type(error).__name__}: {error}")
            try:
                record, path = _write_continuum_run(
                    output_dir,
                    parent="mesh_check",
                    soil_e=soil_e,
                    shape=shape,
                    h=continuum["mesh_acceptance"]["extent_check_h_m"],
                    dz=continuum["mesh_acceptance"]["extent_check_h_m"],
                    half_domain=continuum["mesh_acceptance"]["extent_check_half_domain_m"],
                    load_centres=ALL_CENTERS,
                    evaluation_points=evaluation,
                    design=continuum,
                    lock=lock,
                )
                extent_records[(float(soil_e), shape)] = record
                if max(record["physics_relative_errors"].values()) > continuum["mesh_acceptance"]["force_and_energy_relative_max"]:
                    errors.append(f"extent physics balance failed for {record['run_name']}")
            except Exception as error:
                errors.append(f"extent {_soil_key(soil_e)} {shape}: {type(error).__name__}: {error}")

    changes: dict[str, dict] = {}
    for (soil_e, shape, size), record in records.items():
        if size != continuum["reference_h_m"]:
            continue
        coarse_key = (soil_e, shape, 0.25)
        if coarse_key not in records:
            errors.append(f"missing coarse comparison for {_soil_key(soil_e)} {shape}")
            continue
        fine = _load_run_strain(output_file(output_dir, record["fields_file"]))
        coarse = _load_run_strain(output_file(output_dir, records[coarse_key]["fields_file"]))
        per_load = {
            f"load_x_{_number(centre)}": _change_metrics(coarse[:, index], fine[:, index])
            for index, centre in enumerate(ALL_CENTERS)
        }
        changes[f"{_soil_key(soil_e)}_{shape}_0.25_to_0.125"] = per_load
        for name, values in per_load.items():
            if values["rms_change_microstrain"] > continuum["mesh_acceptance"]["last_strain_rms_change_microstrain_max"]:
                errors.append(f"last mesh RMS change failed: {_soil_key(soil_e)} {shape} {name}")
            if values["relative_L2_change"] > continuum["mesh_acceptance"]["last_strain_relative_L2_change_max"]:
                errors.append(f"last mesh relative L2 failed: {_soil_key(soil_e)} {shape} {name}")
    extent_changes: dict[str, dict] = {}
    for (soil_e, shape), record in extent_records.items():
        baseline_key = (soil_e, shape, continuum["mesh_acceptance"]["extent_check_h_m"])
        if baseline_key not in records:
            errors.append(f"missing extent baseline for {_soil_key(soil_e)} {shape}")
            continue
        wide = _load_run_strain(output_file(output_dir, record["fields_file"]))
        base = _load_run_strain(output_file(output_dir, records[baseline_key]["fields_file"]))
        per_load = {
            f"load_x_{_number(centre)}": _change_metrics(base[:, index], wide[:, index])
            for index, centre in enumerate(ALL_CENTERS)
        }
        extent_changes[f"{_soil_key(soil_e)}_{shape}_L50_to_L60"] = per_load
        for name, values in per_load.items():
            if values["rms_change_microstrain"] > continuum["mesh_acceptance"]["extent_strain_rms_change_microstrain_max"]:
                errors.append(f"extent RMS change failed: {_soil_key(soil_e)} {shape} {name}")
    report = {
        "schema_version": "continuum-mesh-report-v3.0",
        "created_utc": utc_now(),
        "interpretation": continuum["interpretation"],
        "experimental_validation_claimed": False,
        "passed": not errors and len(changes) == 4 and len(extent_changes) == 4,
        "criteria": continuum["mesh_acceptance"],
        "last_changes_0.25_to_0.125": changes,
        "extent_changes_L50_to_L60": extent_changes,
        "runs": [record for record in records.values()] + [record for record in extent_records.values()],
        "errors": errors,
        "design_hash": lock["design_hash"],
        "source_hash": source_hash,
        "protocol_hash": lock["protocol_hash"],
    }
    write_replace_json(output_dir / MESH_REPORT, report)
    return report


def _accepted_mesh(output_dir: Path, lock: dict) -> dict:
    path = output_dir / MESH_REPORT
    if not path.is_file():
        raise ProtocolError("mesh check has not been accepted because its report is missing")
    report = read_json(path)
    if (
        report.get("schema_version") != "continuum-mesh-report-v3.0"
        or report.get("design_hash") != lock["design_hash"]
        or report.get("source_hash") != hash_json(lock["source_files"])
        or report.get("protocol_hash") != lock["protocol_hash"]
        or report.get("passed") is not True
    ):
        raise ProtocolError("mesh check is missing, not accepted, or does not match the lock")
    for run in report["runs"]:
        fields = output_file(output_dir, run.get("fields_file", ""))
        if not fields.is_file() or hash_file(fields) != run.get("fields_sha256"):
            raise ProtocolError(f"accepted mesh fields hash mismatch: {run.get('run_name')}")
    return report


def _calibration_inverse(k0: float, evaluation: np.ndarray) -> np.ndarray:
    common = PhysicalModel()
    forward = HermiteBeamFE(
        [0.0],
        x_min=common.domain_m[0],
        x_max=common.domain_m[1],
        n_elements=int((common.domain_m[1] - common.domain_m[0]) / common.inverse_element_m),
        n_coefficients=common.spline_count,
        gauge_length=common.gauge_m,
        foundation=float(k0),
        force=common.nominal_force_N,
        load_width=common.footprint_m,
        ei=common.EI_N_m2,
        radius=common.radius_m,
    )
    return forward.strain_and_jacobian(np.zeros(forward.basis.n_variables), [evaluation])[0]


def _calibrate_one(
    output_dir: Path,
    soil_e: float,
    continuum: dict,
    lock: dict,
) -> dict:
    settings = continuum["equivalent_winkler_calibration"]
    record_path = output_dir / "calibration" / f"{_soil_key(soil_e)}.json"
    if record_path.is_file():
        existing = read_json(record_path)
        if (
            existing.get("schema_version") != "continuum-calibration-v3.0"
            or existing.get("soil_E_Pa") != float(soil_e)
            or existing.get("design_hash") != lock["design_hash"]
            or existing.get("source_hash") != hash_json(lock["source_files"])
            or existing.get("protocol_hash") != lock["protocol_hash"]
        ):
            raise ProtocolError(f"existing calibration mismatch: {_soil_key(soil_e)}")
        return existing
    sensors = np.linspace(settings["interval_m"][0], settings["interval_m"][1], int(settings["sensors"]))
    reference, _, elapsed = _solve_continuum(
        soil_e=soil_e,
        shape="intact",
        h=continuum["reference_h_m"],
        dz=continuum["reference_dz_m"],
        half_domain=continuum["half_domain_m"],
        load_centres=np.array([0.0]),
        evaluation_points=sensors,
        design=continuum,
    )
    observed = np.asarray(reference["strain"][:, 0], dtype=float)
    samples: list[dict[str, float]] = []

    def objective(log_k0: float) -> float:
        k0 = float(np.exp(log_k0))
        prediction = _calibration_inverse(k0, sensors)
        value = float(np.sum((prediction - observed) ** 2))
        samples.append({"log_k0": float(log_k0), "k0_N_m2": k0, "objective": value})
        return value

    lower, upper = (float(np.log(value)) for value in settings["bounds_N_m2"])
    grid_logs = np.linspace(lower, upper, int(settings["sensors"]))
    grid_objectives = np.asarray([objective(value) for value in grid_logs], dtype=float)
    optimization = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-8},
    )
    optimum_log = float(optimization.x)
    optimum_k0 = float(np.exp(optimum_log))
    optimum_objective = float(objective(optimum_log))
    grid_minimum_index = int(np.argmin(grid_objectives))
    local_minima = [
        index
        for index in range(1, grid_logs.size - 1)
        if grid_objectives[index] < grid_objectives[index - 1]
        and grid_objectives[index] < grid_objectives[index + 1]
    ]
    common = PhysicalModel()
    record = {
        "schema_version": "continuum-calibration-v3.0",
        "soil_E_Pa": float(soil_e),
        "passed": bool(optimization.success and np.isfinite(optimum_objective)),
        "optimizer_success": bool(optimization.success),
        "optimizer_message": str(optimization.message),
        "log_parameter": "natural log(k0_N_m2)",
        "xatol": 1.0e-8,
        "optimum_log_k0": optimum_log,
        "k0_N_m2": optimum_k0,
        "objective": optimum_objective,
        "grid_log_k0": grid_logs.tolist(),
        "grid_k0_N_m2": np.exp(grid_logs).tolist(),
        "grid_objective": grid_objectives.tolist(),
        "grid_minimum": {
            "index": grid_minimum_index,
            "log_k0": float(grid_logs[grid_minimum_index]),
            "k0_N_m2": float(np.exp(grid_logs[grid_minimum_index])),
            "objective": float(grid_objectives[grid_minimum_index]),
        },
        "optimum_minus_grid_minimum": {
            "log_k0": optimum_log - float(grid_logs[grid_minimum_index]),
            "objective": optimum_objective - float(grid_objectives[grid_minimum_index]),
        },
        "objective_evaluation_samples": samples,
        "local_grid_minima_indices": local_minima,
        "multiwell_grid_indicator": len(local_minima) > 1,
        "bound_active": bool(
            abs(optimum_log - lower) <= 4.0e-8 or abs(optimum_log - upper) <= 4.0e-8
        ),
        "forward": {
            "model": "HermiteBeamFE",
            "h_m": common.inverse_element_m,
            "n_coefficients": common.spline_count,
            "coefficients": "all zero (intact)",
            "gauge_length_m": common.gauge_m,
        },
        "observations": {
            "definition": "noiseless continuum finite-gauge strain, intact interface only",
            "sensor_points_m": sensors.tolist(),
            "continuum_strain": observed.tolist(),
            "reference_elapsed_s": float(elapsed),
        },
        "defect_response_used": False,
        "unused_response_used": False,
        "design_hash": lock["design_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
    }
    write_new_json(record_path, record)
    return record


def _calibrate(output_dir: Path, continuum: dict) -> dict:
    lock = _load_lock(output_dir)
    _accepted_mesh(output_dir, lock)
    existing_report_path = output_dir / CALIBRATION_REPORT
    if existing_report_path.is_file():
        existing = read_json(existing_report_path)
        if (
            existing.get("schema_version") != "continuum-calibration-report-v3.0"
            or existing.get("design_hash") != lock["design_hash"]
            or existing.get("source_hash") != hash_json(lock["source_files"])
            or existing.get("protocol_hash") != lock["protocol_hash"]
        ):
            raise ProtocolError("existing calibration report does not match the lock")
        if existing.get("passed") is True:
            return _accepted_calibration(output_dir, lock, continuum)
        return existing
    records = []
    errors: list[str] = []
    for soil_e in continuum["soil_E_Pa"]:
        try:
            records.append(_calibrate_one(output_dir, soil_e, continuum, lock))
        except Exception as error:
            errors.append(f"{_soil_key(soil_e)}: {type(error).__name__}: {error}")
    report = {
        "schema_version": "continuum-calibration-report-v3.0",
        "created_utc": utc_now(),
        "interpretation": continuum["interpretation"],
        "experimental_validation_claimed": False,
        "passed": not errors and len(records) == len(continuum["soil_E_Pa"]) and all(record["passed"] for record in records),
        "records": records,
        "record_files": [f"calibration/{_soil_key(record['soil_E_Pa'])}.json" for record in records],
        "errors": errors,
        "state": continuum["equivalent_winkler_calibration"]["state"],
        "forbidden": continuum["equivalent_winkler_calibration"]["forbidden"],
        "design_hash": lock["design_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
    }
    write_replace_json(output_dir / CALIBRATION_REPORT, report)
    return report


def _accepted_calibration(output_dir: Path, lock: dict, continuum: dict) -> dict:
    path = output_dir / CALIBRATION_REPORT
    if not path.is_file():
        raise ProtocolError("accepted calibration report is missing")
    report = read_json(path)
    if (
        report.get("schema_version") != "continuum-calibration-report-v3.0"
        or report.get("design_hash") != lock["design_hash"]
        or report.get("source_hash") != hash_json(lock["source_files"])
        or report.get("protocol_hash") != lock["protocol_hash"]
        or report.get("passed") is not True
    ):
        raise ProtocolError("calibration is missing, failed, or does not match the lock")
    calibrated = {float(record["soil_E_Pa"]): float(record["k0_N_m2"]) for record in report["records"]}
    expected = {float(value): None for value in continuum["soil_E_Pa"]}
    if set(calibrated) != set(expected):
        raise ProtocolError("calibration report does not contain exactly both soil moduli")
    actual_names = {
        item.name for item in (output_dir / "calibration").glob("E*.json")
    } if (output_dir / "calibration").exists() else set()
    expected_names = {Path(name).name for name in report["record_files"]}
    if actual_names != expected_names:
        raise ProtocolError("calibration artifact directory has missing or extra records")
    for relative in report["record_files"]:
        stored = read_json(output_file(output_dir, relative))
        if stored != next(record for record in report["records"] if record["soil_E_Pa"] == stored["soil_E_Pa"]):
            raise ProtocolError(f"calibration record mismatch: {relative}")
    return calibrated


def _build_inverse_forward(
    centres: np.ndarray, k0: float, physical: PhysicalModel = PhysicalModel()
) -> HermiteBeamFE:
    k0 = float(k0)
    if not np.isfinite(k0) or k0 <= 0.0:
        raise ProtocolError("calibrated inverse k0 must be finite and positive")
    return HermiteBeamFE(
        centres,
        x_min=physical.domain_m[0],
        x_max=physical.domain_m[1],
        n_elements=int((physical.domain_m[1] - physical.domain_m[0]) / physical.inverse_element_m),
        n_coefficients=physical.spline_count,
        gauge_length=physical.gauge_m,
        foundation=k0,
        force=physical.nominal_force_N,
        load_width=physical.footprint_m,
        ei=physical.EI_N_m2,
        radius=physical.radius_m,
    )


def _reference_from_mesh(
    output_dir: Path, mesh_report: dict, soil_e: float, shape: str
) -> dict[str, np.ndarray] | None:
    wanted = f"{_soil_key(soil_e)}_{shape}_L{50}_h{_number(0.125)}_dz{_number(0.125)}"
    for run in mesh_report["runs"]:
        if run["run_name"] != wanted:
            continue
        path = output_file(output_dir, run["fields_file"])
        if hash_file(path) != run["fields_sha256"]:
            raise ProtocolError("cached mesh fields hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            return {name: np.asarray(data[name]).copy() for name in data.files}
    return None


def _shared_reference(
    output_dir: Path,
    *,
    soil_e: float,
    shape: str,
    continuum: dict,
    lock: dict,
    mesh_report: dict,
) -> dict:
    key = f"shape-{shape}_{_soil_key(soil_e)}"
    record_path = output_dir / "references" / key / "reference_record.json"
    arrays_name = "reference_fields.npz"
    arrays_path = output_dir / "references" / key / arrays_name
    if record_path.exists():
        record = read_json(record_path)
        if (
            record.get("reference_key") != key
            or record.get("design_hash") != lock["design_hash"]
            or record.get("source_hash") != hash_json(lock["source_files"])
            or record.get("protocol_hash") != lock["protocol_hash"]
            or not arrays_path.is_file()
            or hash_file(arrays_path) != record.get("arrays_sha256")
        ):
            raise ProtocolError(f"existing reference mismatch: {key}")
        return record

    sensors = _sensor_points(continuum["inversion"]["sensors_per_load"])
    evaluation = np.unique(np.concatenate([UNUSED_GRID, sensors]))
    wanted = (
        f"{_soil_key(soil_e)}_{shape}_L{int(continuum['half_domain_m'])}"
        f"_h{_number(continuum['reference_h_m'])}_dz{_number(continuum['reference_dz_m'])}"
    )
    cached = _reference_from_mesh(output_dir, mesh_report, soil_e, shape)
    if cached is None:
        result, points, elapsed = _solve_continuum(
            soil_e=soil_e,
            shape=shape,
            h=continuum["reference_h_m"],
            dz=continuum["reference_dz_m"],
            half_domain=continuum["half_domain_m"],
            load_centres=ALL_CENTERS,
            evaluation_points=evaluation,
            design=continuum,
        )
        arrays = _field_arrays(result)
        cache_source = "new_fine_continuum_solution"
        mesh = result["mesh"]
        diagnostics = result["diagnostics"]
        elapsed_record = elapsed
    else:
        arrays = cached
        cache_source = "accepted_mesh_check_fine_fields"
        cached_run = next(run for run in mesh_report["runs"] if run["run_name"] == wanted)
        mesh = cached_run["mesh"]
        diagnostics = cached_run["diagnostics"]
        elapsed_record = cached_run["runner_elapsed_s"]
    sensor_indices = np.searchsorted(arrays["evaluation_points"], sensors)
    unused_indices = np.searchsorted(arrays["evaluation_points"], UNUSED_GRID)
    if (
        not np.array_equal(arrays["evaluation_points"][sensor_indices], sensors)
        or not np.array_equal(arrays["evaluation_points"][unused_indices], UNUSED_GRID)
    ):
        raise ProtocolError("reference query points are not present exactly")
    write_new_npz(
        arrays_path,
        {
            **arrays,
            "sensor_indices": sensor_indices,
            "unused_indices": unused_indices,
        },
    )
    record = {
        "schema_version": "continuum-reference-v3.0",
        "reference_key": key,
        "shape": shape,
        "soil_E_Pa": float(soil_e),
        "interpretation": continuum["interpretation"],
        "experimental_validation_claimed": False,
        "load_centres_m": ALL_CENTERS.tolist(),
        "evaluation_points_m": arrays["evaluation_points"].tolist(),
        "cache_source": cache_source,
        "mesh": mesh,
        "diagnostics": diagnostics,
        "reference_elapsed_s": elapsed_record,
        "arrays_file": Path("references").joinpath(key, arrays_name).as_posix(),
        "arrays_sha256": hash_file(arrays_path),
        "design_hash": lock["design_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
    }
    write_new_json(record_path, record)
    return record


def _ensure_dataset(
    output_dir: Path,
    dataset: dict,
    *,
    continuum: dict,
    lock: dict,
    mesh_report: dict,
    calibrated_k0: dict[float, float],
) -> dict:
    dataset_dir = output_dir / "datasets" / dataset["key"]
    record_path = dataset_dir / "dataset_record.json"
    npz_name = "data.npz"
    npz_path = dataset_dir / npz_name
    reference = _shared_reference(
        output_dir,
        soil_e=dataset["soil_E_Pa"],
        shape=dataset["shape"],
        continuum=continuum,
        lock=lock,
        mesh_report=mesh_report,
    )
    reference_path = output_file(output_dir, reference["arrays_file"])
    if record_path.exists():
        record = read_json(record_path)
        expected = {
            "dataset_key": dataset["key"],
            "config": dataset,
            "npz_file": Path("datasets").joinpath(dataset["key"], npz_name).as_posix(),
            "reference_file": reference["arrays_file"],
            "reference_sha256": reference["arrays_sha256"],
            "design_hash": lock["design_hash"],
            "source_hash": hash_json(lock["source_files"]),
            "protocol_hash": lock["protocol_hash"],
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ProtocolError(f"existing dataset mismatch at {key}: {dataset['key']}")
        if not npz_path.is_file() or hash_file(npz_path) != record.get("npz_sha256"):
            raise ProtocolError(f"dataset hash mismatch: {dataset['key']}")
        return record
    if dataset_dir.exists() and any(
        not _is_platform_metadata(item.name) for item in dataset_dir.iterdir()
    ):
        raise ProtocolError(f"refusing to overwrite partial dataset: {dataset['key']}")

    means, z = calibration_measurement(dataset["seed"], dataset["tau"])
    with np.load(reference_path, allow_pickle=False) as data:
        sensor_indices = np.asarray(data["sensor_indices"])
        sensor_locations = np.tile(
            np.asarray(data["evaluation_points"])[sensor_indices], (3, 1)
        )
        all_strain = np.asarray(data["strain"])
        noiseless_train = all_strain[sensor_indices, :3].T
        unused_strain = all_strain[np.asarray(data["unused_indices"]), 3]
        soil_nodes = np.asarray(data["soil_nodes"])
        soil_quads = np.asarray(data["soil_quads"])
        soil_ux = np.asarray(data["soil_ux"])
        soil_uz = np.asarray(data["soil_uz"])
    rng = np.random.default_rng(dataset["seed"])
    observations = noiseless_train + rng.normal(
        0.0, dataset["sigma"], size=noiseless_train.shape
    )
    write_new_npz(
        npz_path,
        {
            "observations": observations,
            "sensor_locations": sensor_locations,
            "noiseless_train_strain": noiseless_train,
            "unused_eval_grid": UNUSED_GRID,
            "unused_truth_strain": unused_strain,
            "score_grid": SCORING_GRID,
            "reference_interface_d": get_shape_function(dataset["shape"])(SCORING_GRID),
            "soil_nodes": soil_nodes,
            "soil_quads": soil_quads,
            "soil_ux": soil_ux,
            "soil_uz": soil_uz,
            "sigma": np.asarray(dataset["sigma"]),
            "train_centers_nominal": TRAIN_CENTERS,
            "train_centers_actual": TRAIN_CENTERS,
            "multipliers": np.ones(3),
            "unused_center": np.asarray(UNUSED_CENTER),
            "shape_name": np.asarray(dataset["shape"]),
            "seed": np.asarray(dataset["seed"]),
        },
    )
    record = {
        "schema_version": "continuum-dataset-v3.0",
        "dataset_key": dataset["key"],
        "config": dataset,
        "npz_file": Path("datasets").joinpath(dataset["key"], npz_name).as_posix(),
        "npz_sha256": hash_file(npz_path),
        "reference_file": reference["arrays_file"],
        "reference_sha256": reference["arrays_sha256"],
        "measured_load_mean": means.tolist(),
        "measured_load_error_z": z.tolist(),
        "true_load_standard_uncertainty": dataset["tau"],
        "true_gamma": [1.0, 1.0, 1.0],
        "noise_generator": f"default_rng({dataset['seed']}).normal(0, sigma, (3, {dataset['sensors']}))",
        "physical": {
            "soil_E_Pa": dataset["soil_E_Pa"],
            "interface_stiffness_N_m2": continuum["interface_stiffness_N_m2"],
            "calibrated_inverse_k0_N_m2": calibrated_k0[dataset["soil_E_Pa"]],
            "EI_N_m2": PhysicalModel().EI_N_m2,
            "force_N": PhysicalModel().nominal_force_N,
            "footprint_m": PhysicalModel().footprint_m,
            "radius_m": PhysicalModel().radius_m,
            "gauge_m": PhysicalModel().gauge_m,
            "inverse_h_m": PhysicalModel().inverse_element_m,
            "domain_m": list(PhysicalModel().domain_m),
        },
        "interpretation": continuum["interpretation"],
        "experimental_validation_claimed": False,
        "design_hash": lock["design_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
    }
    write_new_json(record_path, record)
    return record


def _numeric_array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name], dtype=float))
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _candidate_arrays(result: dict) -> dict[str, np.ndarray]:
    return {
        "coefficients": np.asarray(result["coefficients"], dtype=float),
        "gamma": np.asarray(result["gamma"], dtype=float),
        "train_prediction": np.asarray(result["runner_prediction"], dtype=float),
    }


def _candidate_record(
    *,
    result: dict,
    dataset: dict,
    dataset_record: dict,
    path_config: dict,
    penalty_lambda: float,
    study: dict,
    lock: dict,
) -> dict:
    arrays = _candidate_arrays(result)
    inputs = result["runner_inputs"]
    return {
        "schema_version": "continuum-candidate-v3.0",
        "path_key": path_config["key"],
        "dataset_key": dataset["key"],
        "method": path_config["method"],
        "penalty": "H1",
        "penalty_lambda": float(penalty_lambda),
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
        "norms": result["runner_norms"],
        "objective": result["runner_norms"]["objective"],
        "array_hash": _numeric_array_hash(arrays),
        "dataset_file": dataset_record["npz_file"],
        "dataset_hash": dataset_record["npz_sha256"],
        "soil_E_Pa": dataset["soil_E_Pa"],
        "calibrated_k0_N_m2": dataset_record["physical"]["calibrated_inverse_k0_N_m2"],
        "experimental_validation_claimed": False,
        "design_hash": lock["design_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
        "lambda_path": study["regularization"]["lambda_path"],
        "starts_used": study["regularization"]["starts"],
        "max_nfev": study["regularization"]["max_nfev"],
        "elapsed_s": float(result.get("elapsed_s", 0.0)),
        "candidate_file": _candidate_filename(penalty_lambda),
    }


def _replay_start_objective(
    forward: HermiteBeamFE,
    variables: np.ndarray,
    points: list[np.ndarray],
    observations: list[np.ndarray],
    sigma: float,
    record: dict,
) -> float:
    prediction, jacobian = forward.strain_and_jacobian(variables, points)
    rows: list[np.ndarray] = []
    mean = np.asarray(record["load_mean"], dtype=float)
    std = np.inf if record["load_std"] is None else float(record["load_std"])
    offset = 0
    for load, point in enumerate(points):
        count = point.size
        f = prediction[offset : offset + count] / sigma
        df = jacobian[offset : offset + count] / sigma
        target = observations[load] / sigma
        if record["gamma_mode"] == "profile":
            gamma, _, prior_residual, _ = profile_load_amplitude(
                f, df, target, mean[load], std
            )
            rows.append(gamma * f - target)
            if np.isfinite(std):
                rows.append(np.asarray([prior_residual]))
        else:
            rows.append(mean[load] * f - target)
        offset += count
    weights = forward.basis.penalty_quadrature()[1]
    derivatives = forward.basis.penalty_quadrature()[3][:, 1:-1] @ variables
    penalty = np.sqrt(float(record["penalty_lambda"]) * weights) * derivatives
    vector = np.concatenate(rows + [penalty])
    return float(vector @ vector)


def _verify_candidate_record(
    record: dict,
    dataset_path: Path,
    *,
    dataset: dict,
    dataset_record: dict,
    study: dict,
    lock: dict,
) -> dict[str, Any]:
    errors: list[str] = []
    arrays = {
        "coefficients": np.asarray(record["coefficients"], dtype=float),
        "gamma": np.asarray(record["gamma"], dtype=float),
        "train_prediction": np.asarray(record["train_prediction"], dtype=float),
    }
    if _numeric_array_hash(arrays) != record.get("array_hash"):
        errors.append("array hash mismatch")
    expected_k0 = float(dataset_record["physical"]["calibrated_inverse_k0_N_m2"])
    if float(record.get("calibrated_k0_N_m2", -1.0)) != expected_k0:
        errors.append("candidate did not use the soil-specific calibrated k0")
    with np.load(dataset_path, allow_pickle=False) as data:
        points = [np.asarray(row, dtype=float).copy() for row in data["sensor_locations"]]
        observations = [np.asarray(row, dtype=float).copy() for row in data["observations"]]
        sigma = float(data["sigma"])
        centres = np.asarray(data["train_centers_nominal"], dtype=float)
    stored_physical = dataset_record["physical"]
    default_physical = PhysicalModel()
    physical = PhysicalModel(
        domain_m=tuple(stored_physical.get("domain_m", default_physical.domain_m)),
        inverse_element_m=float(stored_physical.get("inverse_h_m", default_physical.inverse_element_m)),
    )
    forward = _build_inverse_forward(centres, expected_k0, physical)
    unit_prediction = forward.strain_and_jacobian(arrays["coefficients"], points)[0]
    prediction = np.asarray(
        [
            arrays["gamma"][index] * unit_prediction[index * points[0].size : (index + 1) * points[0].size]
            for index in range(3)
        ]
    )
    residual = np.concatenate(
        [(prediction[index] - observations[index]) / sigma for index in range(3)]
    )
    data_norm = float(np.linalg.norm(residual))
    std = np.inf if record["load_std"] is None else float(record["load_std"])
    mean = np.asarray(record["load_mean"], dtype=float)
    calibration_norm = (
        float(np.linalg.norm((arrays["gamma"] - mean) / std))
        if record["finite_positive_prior"]
        else 0.0
    )
    selection_norm = float(math.hypot(data_norm, calibration_norm))
    penalty = forward.basis.penalty_quadrature()[3][:, 1:-1] @ arrays["coefficients"]
    weights = forward.basis.penalty_quadrature()[1]
    penalty_norm = float(np.linalg.norm(np.sqrt(float(record["penalty_lambda"]) * weights) * penalty))
    objective = float(selection_norm**2 + penalty_norm**2)
    if not math.isclose(objective, float(record["objective"]), rel_tol=OBJECTIVE_RTOL, abs_tol=OBJECTIVE_ATOL):
        errors.append(f"objective mismatch: replay={objective:.17g}, reported={record['objective']:.17g}")
    if not np.allclose(prediction, arrays["train_prediction"], rtol=1.0e-9, atol=1.0e-12):
        errors.append("train prediction mismatch")
    expected_inputs = method_inputs(
        record["method"], np.asarray(dataset_record["measured_load_mean"]), float(dataset["tau"])
    )
    expected_identity = {
        "path_key": f"{dataset['key']}__method-{record['method']}",
        "dataset_key": dataset["key"],
        "method": record["method"],
        "penalty_lambda": record["penalty_lambda"],
        "gamma_mode": expected_inputs["gamma_mode"],
        "load_mean": expected_inputs["load_mean"].tolist(),
        "load_std": None if np.isinf(expected_inputs["load_std"]) else float(expected_inputs["load_std"]),
        "finite_positive_prior": expected_inputs["finite_positive_prior"],
        "dataset_hash": dataset_record["npz_sha256"],
        "soil_E_Pa": dataset["soil_E_Pa"],
        "design_hash": lock["design_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
        "lambda_path": study["regularization"]["lambda_path"],
        "starts_used": study["regularization"]["starts"],
        "max_nfev": study["regularization"]["max_nfev"],
    }
    for key, value in expected_identity.items():
        if record.get(key) != value:
            errors.append(f"identity mismatch: {key}")
    if len(record["starts"]) != len(record["starts_used"]):
        errors.append("start count mismatch")
    try:
        for index, start in enumerate(record["starts"]):
            replay = _replay_start_objective(
                forward,
                np.asarray(start["x"], dtype=float),
                points,
                observations,
                sigma,
                record,
            )
            if not math.isclose(replay, float(start["cost"]), rel_tol=OBJECTIVE_RTOL, abs_tol=OBJECTIVE_ATOL):
                errors.append(f"start {index} objective replay mismatch")
    except Exception as error:
        errors.append(f"start replay failed: {type(error).__name__}: {error}")
    return {"passed": not errors, "errors": errors, "recomputed_objective": objective}


def _centroid(x: np.ndarray, values: np.ndarray, area: float) -> float | None:
    if not np.isfinite(area) or abs(area) < CENTROID_ZERO_AREA_M:
        return None
    return float(np.trapezoid(values * x, x) / area)


def _interface_metrics(estimated_d: np.ndarray, reference_d: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    estimated_area = float(np.trapezoid(estimated_d, x))
    reference_area = float(np.trapezoid(reference_d, x))
    return {
        "inferred_effective_winkler_loss_area_m": estimated_area,
        "inferred_effective_winkler_loss_max": float(np.max(estimated_d)),
        "imposed_interface_loss_area_m": reference_area,
        "imposed_interface_loss_centroid_m": _centroid(x, reference_d, reference_area),
        "inferred_effective_winkler_loss_centroid_m": _centroid(x, estimated_d, estimated_area),
        "centroid_near_zero_area_threshold_m": CENTROID_ZERO_AREA_M,
    }


def _score_selected(
    dataset_path: Path,
    candidate: dict,
    *,
    expected_k0: float,
    selection: dict | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if float(candidate.get("calibrated_k0_N_m2", -1.0)) != float(expected_k0):
        raise ProtocolError("selected prediction would not use the soil-specific calibrated k0")
    with np.load(dataset_path, allow_pickle=False) as data:
        shape_name = str(data["shape_name"])
        score_grid = np.asarray(data["score_grid"], dtype=float)
        reference_d = np.asarray(data["reference_interface_d"], dtype=float)
        unused_x = np.asarray(data["unused_eval_grid"], dtype=float)
        reference_strain = np.asarray(data["unused_truth_strain"], dtype=float)
        observed = np.asarray(data["observations"], dtype=float)
        sensor_x = np.asarray(data["sensor_locations"], dtype=float)
    basis = SupportBasis(n_coefficients=PhysicalModel().spline_count)
    estimated_d = basis.values(score_grid) @ basis.full_coefficients(
        np.asarray(candidate["coefficients"], dtype=float)
    )
    unused_forward = _build_inverse_forward([UNUSED_CENTER], expected_k0)
    predicted_strain = unused_forward.strain_and_jacobian(
        np.asarray(candidate["coefficients"], dtype=float), [unused_x]
    )[0]
    predicted_train = np.asarray(candidate["train_prediction"], dtype=float).reshape(observed.shape)
    gamma = np.asarray(candidate["gamma"], dtype=float)
    metrics = {
        "unused_load_strain_relative_L2": float(
            np.linalg.norm(predicted_strain - reference_strain) / np.linalg.norm(reference_strain)
        ),
        "unused_load_peak_strain_error": float(
            abs(np.max(np.abs(predicted_strain)) - np.max(np.abs(reference_strain)))
            / np.max(np.abs(reference_strain))
        ),
        "load_RMSE": float(np.sqrt(np.mean((gamma - np.ones(3)) ** 2))),
        "physical_train_strain_RMS": float(np.sqrt(np.mean((predicted_train - observed) ** 2))),
        "strain_residual_norm": float(candidate["norms"]["data_norm"]),
        "calibration_pull_norm": float(candidate["norms"]["calibration_norm"]),
        "augmented_norm": float(candidate["norms"]["selection_norm"]),
    }
    if selection is not None:
        metrics.update(
            {
                "selection_compatible": bool(
                    not selection["optimizer_failed"] and not selection["no_feasible_candidate"]
                ),
                "selection_reason": selection["selection_reason"],
                "path_bracketed": bool(selection["path_bracketed"]),
                "unbracketed_high": bool(selection["unbracketed_high"]),
            }
        )
    metrics.update(_interface_metrics(estimated_d, reference_d, score_grid))
    if shape_name == "intact":
        metrics["intact_spurious_effective_winkler_loss_area_m"] = metrics[
            "inferred_effective_winkler_loss_area_m"
        ]
        metrics["intact_effective_spurious_loss_includes_model_bias"] = True
    arrays = {
        "score_grid": score_grid,
        "reference_interface_d": reference_d,
        "estimated_effective_winkler_d": estimated_d,
        "unused_x": unused_x,
        "reference_strain": reference_strain,
        "predicted_strain": predicted_strain,
        "sensor_x": sensor_x,
        "observed_strain": observed,
        "predicted_train_strain": predicted_train,
        "gamma": gamma,
    }
    return metrics, arrays


def _path_inventory(
    expected_paths: list[dict],
    candidate_files: dict[str, set[str]],
    path_files: set[str],
) -> dict[str, Any]:
    missing_paths: list[str] = []
    missing_candidates: dict[str, int] = {}
    for path_config in expected_paths:
        if path_config["key"] not in path_files:
            missing_paths.append(path_config["key"])
        actual = candidate_files.get(path_config["key"], set())
        if len(actual) != 6:
            missing_candidates[path_config["key"]] = 6 - len(actual)
    return {
        "missing_paths": missing_paths,
        "missing_candidates": missing_candidates,
        "all_candidates_without_path": sorted(
            key for key in missing_paths if not missing_candidates.get(key)
        ),
        "complete": not missing_paths and not missing_candidates,
    }


def _run_is_completed(status: dict | None, inventory: dict, expected_paths: int) -> bool:
    return bool(
        status is not None
        and status.get("status") == "completed"
        and inventory["complete"]
        and int(status.get("completed_paths", -1)) == expected_paths
    )


def _existing_candidate(
    path: Path,
    output_dir: Path,
    *,
    dataset: dict,
    dataset_record: dict,
    path_config: dict,
    penalty_lambda: float,
    study: dict,
    lock: dict,
) -> dict | None:
    if not path.exists():
        return None
    record = read_json(path)
    if record.get("candidate_file") != path.name:
        raise ProtocolError(f"existing candidate filename mismatch: {path.name}")
    verification = _verify_candidate_record(
        record,
        output_file(output_dir, dataset_record["npz_file"]),
        dataset=dataset,
        dataset_record=dataset_record,
        study=study,
        lock=lock,
    )
    if not verification["passed"]:
        raise ProtocolError(
            f"existing candidate failed replay: {path_config['key']}: {verification['errors']}"
        )
    return record


def _run_one_path(
    output_dir: Path,
    path_config: dict,
    *,
    dataset: dict,
    dataset_record: dict,
    continuum: dict,
    study: dict,
    lock: dict,
    deadline: float,
) -> tuple[bool, bool]:
    path_dir = output_dir / "paths" / path_config["key"]
    path_file = path_dir / "path.json"
    dataset_path = output_file(output_dir, dataset_record["npz_file"])
    if path_file.exists():
        record = read_json(path_file)
        expected_candidate_files = [
            Path("paths").joinpath(path_config["key"], _candidate_filename(lam)).as_posix()
            for lam in study["regularization"]["lambda_path"]
        ]
        if (
            record.get("path_key") != path_config["key"]
            or record.get("dataset_key") != dataset["key"]
            or record.get("method") != path_config["method"]
            or record.get("dataset_hash") != dataset_record["npz_sha256"]
            or record.get("candidate_files") != expected_candidate_files
            or record.get("design_hash") != lock["design_hash"]
            or record.get("source_hash") != hash_json(lock["source_files"])
            or record.get("protocol_hash") != lock["protocol_hash"]
        ):
            raise ProtocolError(f"existing path identity mismatch: {path_config['key']}")
        for relative in expected_candidate_files:
            candidate = read_json(output_file(output_dir, relative))
            replay = _verify_candidate_record(
                candidate,
                dataset_path,
                dataset=dataset,
                dataset_record=dataset_record,
                study=study,
                lock=lock,
            )
            if not replay["passed"]:
                raise ProtocolError(
                    f"existing path candidate failed replay: {path_config['key']}: {replay['errors']}"
                )
        if record.get("selected") and (
            not record.get("arrays_file")
            or not output_file(output_dir, record["arrays_file"]).is_file()
            or hash_file(output_file(output_dir, record["arrays_file"])) != record.get("arrays_sha256")
        ):
            raise ProtocolError(f"existing selected arrays hash mismatch: {path_config['key']}")
        return True, True
    if path_dir.exists():
        expected_names = {
            _candidate_filename(lam) for lam in study["regularization"]["lambda_path"]
        }
        unexpected = sorted(
            item.name
            for item in path_dir.iterdir()
            if not _is_platform_metadata(item.name) and item.name not in expected_names
        )
        if unexpected:
            raise ProtocolError(
                f"refusing to overwrite partial path with unexpected files: "
                f"{path_config['key']}: {unexpected}"
            )
    forward = _build_inverse_forward(
        TRAIN_CENTERS, float(dataset_record["physical"]["calibrated_inverse_k0_N_m2"])
    )
    candidates: list[dict] = []
    reused = False
    reuse: list[bool] = []
    for penalty_lambda in study["regularization"]["lambda_path"]:
        if time.monotonic() >= deadline:
            return False, reused
        candidate_path = path_dir / _candidate_filename(penalty_lambda)
        candidate_existed = candidate_path.is_file()
        record = _existing_candidate(
            candidate_path,
            output_dir,
            dataset=dataset,
            dataset_record=dataset_record,
            path_config=path_config,
            penalty_lambda=penalty_lambda,
            study=study,
            lock=lock,
        )
        if record is None:
            result = _fit_one_candidate(
                forward,
                dataset_path,
                path_config["method"],
                float(penalty_lambda),
                study,
                np.asarray(dataset_record["measured_load_mean"], dtype=float),
                float(dataset["tau"]),
            )
            record = _candidate_record(
                result=result,
                dataset=dataset,
                dataset_record=dataset_record,
                path_config=path_config,
                penalty_lambda=penalty_lambda,
                study=study,
                lock=lock,
            )
            verification = _verify_candidate_record(
                record,
                dataset_path,
                dataset=dataset,
                dataset_record=dataset_record,
                study=study,
                lock=lock,
            )
            if not verification["passed"]:
                raise ProtocolError(f"new candidate failed replay: {path_config['key']}: {verification['errors']}")
            write_new_json(candidate_path, record)
        else:
            reused = True
        reuse.append(candidate_existed)
        candidates.append(record)
    selection = _selection_details(candidates, int(dataset["sensors"]))
    any_success = any(candidate["successful_start"] for candidate in candidates)
    path_record = {
        "schema_version": "continuum-path-v3.0",
        "path_key": path_config["key"],
        "dataset_key": dataset["key"],
        "method": path_config["method"],
        "penalty": "H1",
        "lambda_path": study["regularization"]["lambda_path"],
        "starts": study["regularization"]["starts"],
        "candidate_files": [
            Path("paths").joinpath(path_config["key"], _candidate_filename(lam)).as_posix()
            for lam in study["regularization"]["lambda_path"]
        ],
        "reused_candidates": reuse,
        "any_successful_start": any_success,
        "status": "selected" if any_success else "optimizer_failed",
        "selected": any_success,
        "selection": selection,
        "dataset_file": dataset_record["npz_file"],
        "dataset_hash": dataset_record["npz_sha256"],
        "soil_E_Pa": dataset["soil_E_Pa"],
        "calibrated_k0_N_m2": dataset_record["physical"]["calibrated_inverse_k0_N_m2"],
        "experimental_validation_claimed": False,
        "design_hash": lock["design_hash"],
        "source_hash": hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
    }
    if any_success:
        selected_lambda = selection["selected_lambda"]
        selected = next(
            candidate for candidate in candidates if candidate["penalty_lambda"] == selected_lambda
        )
        metrics, arrays = _score_selected(
            dataset_path,
            selected,
            expected_k0=float(dataset_record["physical"]["calibrated_inverse_k0_N_m2"]),
            selection=selection,
        )
        arrays_path = path_dir / "selected_arrays.npz"
        if arrays_path.exists():
            with np.load(arrays_path, allow_pickle=False) as saved:
                matches = set(saved.files) == set(arrays) and all(
                    np.array_equal(saved[name], arrays[name]) for name in arrays
                )
            if not matches:
                raise ProtocolError(f"partial selected arrays mismatch: {path_config['key']}")
        else:
            write_new_npz(arrays_path, arrays)
        path_record["metrics"] = metrics
        path_record["selected_start"] = selected["chosen_start"]
        path_record["selected_coefficients"] = selected["coefficients"]
        path_record["selected_gamma"] = selected["gamma"]
        path_record["selected_train_prediction"] = selected["train_prediction"]
        path_record["arrays_file"] = Path("paths").joinpath(path_config["key"], "selected_arrays.npz").as_posix()
        path_record["arrays_sha256"] = hash_file(arrays_path)
    write_new_json(path_file, path_record)
    return True, reused


def _run(
    output_dir: Path,
    continuum: dict,
    study: dict,
    *,
    resume: bool,
) -> dict:
    lock = _load_lock(output_dir)
    mesh_report = _accepted_mesh(output_dir, lock)
    calibrated_k0 = _accepted_calibration(output_dir, lock, continuum)
    datasets, paths, counts = expand_continuum_study(continuum, study)
    existing_run_entries = [
        item.name
        for item in output_dir.iterdir()
        if item.name in {"references", "datasets", "paths", "run_status.json"}
    ]
    if existing_run_entries and not resume:
        raise ProtocolError("output contains inversion results; use --resume after verifying their identity")
    started = time.perf_counter()
    started_utc = utc_now()
    budget = float(continuum["resource_limit"]["wall_time_limit_hours"]) * 3600.0
    deadline = time.monotonic() + budget
    completed = 0
    reused_paths = 0
    timeout = False
    datasets_by_key = {dataset["key"]: dataset for dataset in datasets}
    dataset_records: dict[str, dict] = {}
    status = {
        "schema_version": "continuum-run-status-v3.0",
        "updated_utc": started_utc,
        "wall_time_started_utc": started_utc,
        "status": "running",
        "completed_paths": completed,
        "reused_completed_paths": reused_paths,
        "expected_counts": counts,
    }
    write_replace_json(output_dir / "run_status.json", status)
    error: str | None = None
    try:
        for dataset in datasets:
            if time.monotonic() >= deadline:
                timeout = True
                break
            dataset_records[dataset["key"]] = _ensure_dataset(
                output_dir,
                dataset,
                continuum=continuum,
                lock=lock,
                mesh_report=mesh_report,
                calibrated_k0=calibrated_k0,
            )
        if not timeout:
            for path_config in paths:
                if time.monotonic() >= deadline:
                    timeout = True
                    break
                dataset = datasets_by_key[path_config["dataset_key"]]
                was_complete, was_reused = _run_one_path(
                    output_dir,
                    path_config,
                    dataset=dataset,
                    dataset_record=dataset_records[path_config["dataset_key"]],
                    continuum=continuum,
                    study=study,
                    lock=lock,
                    deadline=deadline,
                )
                if not was_complete:
                    timeout = True
                    break
                completed += 1
                reused_paths += int(was_reused)
                status = {
                    "schema_version": "continuum-run-status-v3.0",
                    "updated_utc": utc_now(),
                    "wall_time_started_utc": started_utc,
                    "status": "running",
                    "completed_paths": completed,
                    "reused_completed_paths": reused_paths,
                    "expected_counts": counts,
                }
                write_replace_json(output_dir / "run_status.json", status)
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        raise
    finally:
        elapsed = time.perf_counter() - started
        status = {
            "schema_version": "continuum-run-status-v3.0",
            "updated_utc": utc_now(),
            "wall_time_started_utc": started_utc,
            "status": "error" if error else ("timeout" if timeout else "completed"),
            "wall_time_budget_seconds": budget,
            "wall_time_elapsed_seconds": elapsed,
            "wall_time_cap_exceeded": elapsed > budget,
            "completed_paths": completed,
            "reused_completed_paths": reused_paths,
            "expected_counts": counts,
            "error": error,
        }
        write_replace_json(output_dir / "run_status.json", status)
    return status


def _same_optional_metric(replay: Any, reported: Any) -> bool:
    if replay is None or reported is None:
        return replay is None and reported is None
    if isinstance(replay, bool) or isinstance(reported, bool):
        return isinstance(replay, bool) and isinstance(reported, bool) and replay == reported
    if isinstance(replay, str) or isinstance(reported, str):
        return isinstance(replay, str) and isinstance(reported, str) and replay == reported
    try:
        return math.isclose(float(replay), float(reported), rel_tol=1.0e-8, abs_tol=1.0e-8)
    except (TypeError, ValueError):
        return False


def _verify(output_dir: Path, continuum: dict, study: dict) -> dict:
    errors: dict[str, list[str]] = {}
    try:
        lock = _load_lock(output_dir)
        mesh_report = _accepted_mesh(output_dir, lock)
        calibrated_k0 = _accepted_calibration(output_dir, lock, continuum)
    except ProtocolError as error:
        return {
            "schema_version": "continuum-verification-v3.0",
            "verified_utc": utc_now(),
            "passed": False,
            "incomplete": True,
            "errors": {"lock_or_gate": [str(error)]},
        }
    datasets, paths, counts = expand_continuum_study(continuum, study)
    datasets_by_key = {dataset["key"]: dataset for dataset in datasets}
    dataset_records: dict[str, dict] = {}
    for dataset in datasets:
        record_path = output_dir / "datasets" / dataset["key"] / "dataset_record.json"
        if not record_path.is_file():
            errors.setdefault(dataset["key"], []).append("missing dataset record")
            continue
        record = read_json(record_path)
        npz_path = output_file(output_dir, record.get("npz_file", ""))
        try:
            reference_path = output_file(output_dir, record.get("reference_file", ""))
            reference_hash_ok = (
                reference_path.is_file()
                and hash_file(reference_path) == record.get("reference_sha256")
            )
        except ProtocolError:
            reference_hash_ok = False
        if (
            record.get("dataset_key") != dataset["key"]
            or record.get("config") != dataset
            or record.get("design_hash") != lock["design_hash"]
            or record.get("source_hash") != hash_json(lock["source_files"])
            or record.get("protocol_hash") != lock["protocol_hash"]
            or not npz_path.is_file()
            or hash_file(npz_path) != record.get("npz_sha256")
            or not reference_hash_ok
        ):
            errors.setdefault(dataset["key"], []).append("dataset identity or hash mismatch")
        else:
            dataset_records[dataset["key"]] = record
    expected_reference_dirs = {
        f"shape-{dataset['shape']}_{_soil_key(dataset['soil_E_Pa'])}" for dataset in datasets
    }
    actual_reference_dirs = (
        {item.name for item in (output_dir / "references").iterdir() if item.is_dir()}
        if (output_dir / "references").exists()
        else set()
    )
    for extra in sorted(actual_reference_dirs - expected_reference_dirs):
        errors.setdefault(extra, []).append("unexpected reference directory")
    actual_dataset_dirs = (
        {item.name for item in (output_dir / "datasets").iterdir() if item.is_dir()}
        if (output_dir / "datasets").exists()
        else set()
    )
    for extra in sorted(actual_dataset_dirs - {dataset["key"] for dataset in datasets}):
        errors.setdefault(extra, []).append("unexpected dataset directory")

    candidate_files: dict[str, set[str]] = {}
    path_files = {
        item.parent.name
        for item in (output_dir / "paths").glob("*/path.json")
    } if (output_dir / "paths").exists() else set()
    checked_candidates = 0
    for path_config in paths:
        path_dir = output_dir / "paths" / path_config["key"]
        names = set()
        for penalty_lambda in study["regularization"]["lambda_path"]:
            candidate_path = path_dir / _candidate_filename(penalty_lambda)
            if not candidate_path.is_file():
                continue
            names.add(candidate_path.name)
            dataset = datasets_by_key[path_config["dataset_key"]]
            dataset_record = dataset_records.get(dataset["key"])
            if dataset_record is None:
                errors.setdefault(path_config["key"], []).append("candidate has no valid dataset")
                continue
            record = read_json(candidate_path)
            replay = _verify_candidate_record(
                record,
                output_file(output_dir, dataset_record["npz_file"]),
                dataset=dataset,
                dataset_record=dataset_record,
                study=study,
                lock=lock,
            )
            checked_candidates += 1
            if not replay["passed"]:
                errors.setdefault(path_config["key"], []).extend(replay["errors"])
        candidate_files[path_config["key"]] = names
        path_file = path_dir / "path.json"
        if not path_file.is_file():
            continue
        path_record = read_json(path_file)
        dataset = datasets_by_key[path_config["dataset_key"]]
        dataset_record = dataset_records.get(dataset["key"])
        if dataset_record is None:
            errors.setdefault(path_config["key"], []).append("path has no valid dataset")
            continue
        candidate_paths = [
            path_dir / _candidate_filename(lam) for lam in study["regularization"]["lambda_path"]
        ]
        if not all(path.is_file() for path in candidate_paths):
            errors.setdefault(path_config["key"], []).append("completed path is missing a candidate")
            continue
        candidates = [
            read_json(path) for path in candidate_paths
        ]
        expected_path_identity = {
            "path_key": path_config["key"],
            "dataset_key": dataset["key"],
            "method": path_config["method"],
            "dataset_hash": dataset_record["npz_sha256"],
            "soil_E_Pa": dataset["soil_E_Pa"],
            "design_hash": lock["design_hash"],
            "source_hash": hash_json(lock["source_files"]),
            "protocol_hash": lock["protocol_hash"],
        }
        for key, value in expected_path_identity.items():
            if path_record.get(key) != value:
                errors.setdefault(path_config["key"], []).append(f"path identity mismatch: {key}")
        selection = _selection_details(candidates, int(dataset["sensors"]))
        for key in ("selected_lambda", "selection_reason", "optimizer_failed", "no_feasible_candidate"):
            if selection[key] != path_record.get("selection", {}).get(key):
                errors.setdefault(path_config["key"], []).append(f"selection mismatch: {key}")
        if not path_record.get("any_successful_start"):
            if path_record.get("selected") or "metrics" in path_record:
                errors.setdefault(path_config["key"], []).append("failed path reports selection")
        else:
            selected_lambda = selection["selected_lambda"]
            selected = next(c for c in candidates if c["penalty_lambda"] == selected_lambda)
            replay_metrics, replay_arrays = _score_selected(
                output_file(output_dir, dataset_record["npz_file"]),
                selected,
                expected_k0=calibrated_k0[dataset["soil_E_Pa"]],
                selection=selection,
            )
            for name, replay in replay_metrics.items():
                if not _same_optional_metric(replay, path_record.get("metrics", {}).get(name)):
                    errors.setdefault(path_config["key"], []).append(f"selected metric mismatch: {name}")
            arrays_path = output_file(output_dir, path_record.get("arrays_file", ""))
            if not arrays_path.is_file() or hash_file(arrays_path) != path_record.get("arrays_sha256"):
                errors.setdefault(path_config["key"], []).append("selected arrays hash mismatch")
            else:
                with np.load(arrays_path, allow_pickle=False) as saved:
                    if set(saved.files) != set(replay_arrays) or not all(
                        np.array_equal(saved[name], replay_arrays[name]) for name in replay_arrays
                    ):
                        errors.setdefault(path_config["key"], []).append("selected arrays schema mismatch")
    inventory = _path_inventory(paths, candidate_files, path_files)
    for key in inventory["missing_paths"]:
        errors.setdefault(key, []).append("path record is missing")
    for key, count in inventory["missing_candidates"].items():
        errors.setdefault(key, []).append(f"{count} candidate records are missing")
    actual_path_dirs = (
        {item.name for item in (output_dir / "paths").iterdir() if item.is_dir()}
        if (output_dir / "paths").exists()
        else set()
    )
    for extra in sorted(actual_path_dirs - {path_config["key"] for path_config in paths}):
        errors.setdefault(extra, []).append("unexpected path directory")
    for path_config in paths:
        path_dir = output_dir / "paths" / path_config["key"]
        expected_json_names = {
            "path.json",
            *(_candidate_filename(lam) for lam in study["regularization"]["lambda_path"]),
        }
        actual_json_names = (
            {
                item.name
                for item in path_dir.glob("*.json")
                if not _is_platform_metadata(item.name)
            }
            if path_dir.exists()
            else set()
        )
        unexpected = actual_json_names - expected_json_names
        if unexpected:
            errors.setdefault(path_config["key"], []).append(
                f"unexpected path JSON files: {sorted(unexpected)}"
            )
    status_path = output_dir / "run_status.json"
    status = read_json(status_path) if status_path.is_file() else None
    completed = _run_is_completed(status, inventory, len(paths))
    if status is None:
        errors.setdefault("run_status", []).append("run status is missing")
    elif status.get("status") == "timeout":
        errors.setdefault("run_status", []).append("run timed out and is not complete")
    elif status.get("status") != "completed":
        errors.setdefault("run_status", []).append(f"run status is {status.get('status')}")
    report = {
        "schema_version": "continuum-verification-v3.0",
        "verified_utc": utc_now(),
        "passed": not errors and completed,
        "incomplete": bool(inventory["missing_paths"] or inventory["missing_candidates"] or not completed),
        "checked_candidates": checked_candidates,
        "checked_paths": len(path_files),
        "expected_counts": counts,
        "inventory": inventory,
        "mesh_passed": mesh_report["passed"],
        "calibration_passed": True,
        "errors": errors,
    }
    write_replace_json(output_dir / "verification_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check-mesh", "calibrate", "run", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output", required=True, help="relative output directory")
        if command == "run":
            subparser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = validate_relative_output(args.output)
    continuum, study = load_designs()
    if args.command == "check-mesh":
        report = _check_mesh(output_dir, continuum, study)
    elif args.command == "calibrate":
        report = _calibrate(output_dir, continuum)
    elif args.command == "run":
        report = _run(output_dir, continuum, study, resume=args.resume)
    else:
        report = _verify(output_dir, continuum, study)
    print(canonical_json(safe_json(report)))
    if args.command == "verify":
        return 0 if report["passed"] else 2
    return 0 if report.get("passed", report.get("status") == "completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
