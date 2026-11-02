"""Publication figures and grouped summary for the continuum extension.

The command reads only a source-locked, completed, independently verified
continuum result tree.  It never creates substitute numerical results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "vendor", ROOT / "code"):
    _absolute = str(_path.resolve())
    if _absolute not in sys.path:
        sys.path.insert(0, _absolute)

import run_continuum_study as runner  # noqa: E402
import make_load_information_figures as style  # noqa: E402
from run_load_information import ProtocolError, read_json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402


EXPECTED_COUNTS = {
    "datasets": 36,
    "paths": 144,
    "candidates": 864,
    "starts": 1728,
}
METHODS = style.MAIN_METHODS
TAU_LEVELS = (0.01, 0.10)
SOIL_E_PA = (50.0e6, 200.0e6)
RESPONSE_SHAPE = "asymmetric"
RESPONSE_TAU = 0.10
RESPONSE_SEED = 1201
NUMERIC_METRICS = (
    "selected_lambda",
    "selected_load_amplitude_mean",
    "load_RMSE",
    "unused_load_strain_relative_L2",
    "unused_load_peak_strain_error",
    "physical_train_strain_RMS",
    "strain_residual_norm",
    "calibration_pull_norm",
    "augmented_norm",
    "inferred_effective_winkler_loss_area_m",
    "inferred_effective_winkler_loss_max",
    "imposed_interface_loss_area_m",
    "imposed_interface_loss_centroid_m",
    "inferred_effective_winkler_loss_centroid_m",
    "intact_spurious_effective_winkler_loss_area_m",
)
FORBIDDEN_RELATIVE_METRICS = {
    "support_rel_L2",
    "support_relative_L2",
    "relative_field_error",
    "interface_loss_relative_to_winkler_loss",
}
CSV_COLUMNS = (
    "record_type",
    "soil_E_Pa",
    "soil_E_MPa",
    "shape",
    "tau",
    "tau_percent",
    "method",
    "n_paths",
    "n_selected",
    "n_optimizer_failed",
    "n_incompatible",
    "n_unbracketed_high",
    "metric_name",
    "n",
    "n_valid",
    "null_count",
    "failed_path_count",
    "median",
    "q25",
    "q75",
)

SummaryError = style.SummaryError


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise SummaryError(f"missing {description}: {path}")


def _actual_directories(parent: Path) -> set[str]:
    if not parent.is_dir():
        return set()
    return {
        item.name
        for item in parent.iterdir()
        if item.is_dir() and not item.name.startswith("._")
    }


def _validate_gates(status: dict[str, Any], verification: dict[str, Any]) -> None:
    expected_status = {
        "schema_version": "continuum-run-status-v3.0",
        "status": "completed",
        "completed_paths": EXPECTED_COUNTS["paths"],
        "expected_counts": EXPECTED_COUNTS,
    }
    problems = [
        f"run status has {key}={status.get(key)!r}; expected {value!r}"
        for key, value in expected_status.items()
        if status.get(key) != value
    ]
    expected_verification = {
        "schema_version": "continuum-verification-v3.0",
        "passed": True,
        "incomplete": False,
        "checked_candidates": EXPECTED_COUNTS["candidates"],
        "checked_paths": EXPECTED_COUNTS["paths"],
        "expected_counts": EXPECTED_COUNTS,
        "mesh_passed": True,
        "calibration_passed": True,
    }
    problems.extend(
        f"verification has {key}={verification.get(key)!r}; expected {value!r}"
        for key, value in expected_verification.items()
        if verification.get(key) != value
    )
    if verification.get("errors"):
        problems.append(f"verification errors={verification['errors']}")
    inventory = verification.get("inventory", {})
    if inventory.get("complete") is not True:
        problems.append(f"verification inventory is incomplete: {inventory}")
    if problems:
        raise SummaryError("continuum gates are not acceptable: " + "; ".join(problems))


def _validate_record(record: dict[str, Any], expected: dict[str, Any], context: str) -> None:
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise SummaryError(f"{context} record mismatch for: {', '.join(mismatches)}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as saved:
            return {name: np.asarray(saved[name]).copy() for name in saved.files}
    except (OSError, ValueError) as error:
        raise SummaryError(f"cannot read verified array file {path}: {error}") from error


def load_verified_results(results_dir: Path) -> dict[str, Any]:
    """Load and identity-check every record used by figures and summary."""
    if not results_dir.is_dir():
        raise SummaryError(f"results directory is missing: {results_dir}")
    try:
        lock = runner._load_lock(results_dir)
        continuum, study = runner.load_designs()
        datasets, paths, counts = runner.expand_continuum_study(continuum, study)
        mesh_report = runner._accepted_mesh(results_dir, lock)
        calibrated_k0 = runner._accepted_calibration(results_dir, lock, continuum)
    except ProtocolError as error:
        raise SummaryError(f"source-locked continuum gates failed: {error}") from error
    if counts != EXPECTED_COUNTS:
        raise SummaryError(f"expanded continuum counts do not match D2: {counts}")

    status_path = results_dir / "run_status.json"
    verification_path = results_dir / "verification_report.json"
    _require_file(status_path, "completed run status")
    _require_file(verification_path, "independent verification report")
    status = read_json(status_path)
    verification = read_json(verification_path)
    _validate_gates(status, verification)

    datasets_by_key = {item["key"]: item for item in datasets}
    expected_dataset_dirs = set(datasets_by_key)
    actual_dataset_dirs = _actual_directories(results_dir / "datasets")
    if actual_dataset_dirs != expected_dataset_dirs:
        raise SummaryError(
            "dataset directories differ from the frozen design: "
            f"missing={sorted(expected_dataset_dirs - actual_dataset_dirs)}, "
            f"extra={sorted(actual_dataset_dirs - expected_dataset_dirs)}"
        )
    expected_path_dirs = {item["key"] for item in paths}
    actual_path_dirs = _actual_directories(results_dir / "paths")
    if actual_path_dirs != expected_path_dirs:
        raise SummaryError(
            "path directories differ from the frozen design: "
            f"missing={sorted(expected_path_dirs - actual_path_dirs)}, "
            f"extra={sorted(actual_path_dirs - expected_path_dirs)}"
        )
    expected_reference_dirs = {
        f"shape-{shape}_{runner._soil_key(soil_e)}" for shape in continuum["inversion"]["shapes"] for soil_e in SOIL_E_PA
    }
    actual_reference_dirs = _actual_directories(results_dir / "references")
    if actual_reference_dirs != expected_reference_dirs:
        raise SummaryError(
            "reference directories differ from the frozen design: "
            f"missing={sorted(expected_reference_dirs - actual_reference_dirs)}, "
            f"extra={sorted(actual_reference_dirs - expected_reference_dirs)}"
        )

    source_hash = runner.hash_json(lock["source_files"])
    dataset_records: dict[str, dict[str, Any]] = {}
    dataset_hashes: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        record_path = results_dir / "datasets" / dataset["key"] / "dataset_record.json"
        _require_file(record_path, f"dataset record {dataset['key']}")
        record = read_json(record_path)
        expected = {
            "schema_version": "continuum-dataset-v3.0",
            "dataset_key": dataset["key"],
            "config": dataset,
            "design_hash": lock["design_hash"],
            "source_hash": source_hash,
            "protocol_hash": lock["protocol_hash"],
        }
        _validate_record(record, expected, f"dataset {dataset['key']}")
        if float(record["physical"]["soil_E_Pa"]) != float(dataset["soil_E_Pa"]):
            raise SummaryError(f"dataset soil modulus mismatch: {dataset['key']}")
        if float(record["physical"]["calibrated_inverse_k0_N_m2"]) != float(
            calibrated_k0[float(dataset["soil_E_Pa"])]
        ):
            raise SummaryError(f"dataset calibrated k0 mismatch: {dataset['key']}")
        dataset_records[dataset["key"]] = record
        dataset_hashes[dataset["key"]] = {
            "dataset_record.json": style.hash_file(record_path),
            "data.npz_sha256_recorded": record.get("npz_sha256"),
        }

    path_records: dict[str, dict[str, Any]] = {}
    candidate_records: dict[str, list[dict[str, Any]]] = {}
    selected_arrays: dict[str, dict[str, np.ndarray]] = {}
    path_hashes: dict[str, dict[str, Any]] = {}
    for path_config in paths:
        key = path_config["key"]
        dataset = datasets_by_key[path_config["dataset_key"]]
        dataset_record = dataset_records[dataset["key"]]
        path_dir = results_dir / "paths" / key
        path_file = path_dir / "path.json"
        _require_file(path_file, f"path record {key}")
        record = read_json(path_file)
        expected = {
            "schema_version": "continuum-path-v3.0",
            "path_key": key,
            "dataset_key": dataset["key"],
            "method": path_config["method"],
            "dataset_hash": dataset_record["npz_sha256"],
            "soil_E_Pa": dataset["soil_E_Pa"],
            "calibrated_k0_N_m2": dataset_record["physical"]["calibrated_inverse_k0_N_m2"],
            "design_hash": lock["design_hash"],
            "source_hash": source_hash,
            "protocol_hash": lock["protocol_hash"],
        }
        _validate_record(record, expected, f"path {key}")
        candidates: list[dict[str, Any]] = []
        candidate_hashes: dict[str, str] = {}
        for penalty_lambda in study["regularization"]["lambda_path"]:
            filename = runner._candidate_filename(float(penalty_lambda))
            candidate_path = path_dir / filename
            _require_file(candidate_path, f"candidate record {key}/{filename}")
            candidate = read_json(candidate_path)
            candidate_expected = {
                "schema_version": "continuum-candidate-v3.0",
                "path_key": key,
                "dataset_key": dataset["key"],
                "method": path_config["method"],
                "penalty_lambda": float(penalty_lambda),
                "candidate_file": filename,
                "dataset_hash": dataset_record["npz_sha256"],
                "design_hash": lock["design_hash"],
                "source_hash": source_hash,
                "protocol_hash": lock["protocol_hash"],
            }
            _validate_record(candidate, candidate_expected, f"candidate {key}/{filename}")
            if len(candidate.get("starts", [])) != len(study["regularization"]["starts"]):
                raise SummaryError(f"candidate start accounting is incomplete: {key}/{filename}")
            candidates.append(candidate)
            candidate_hashes[filename] = style.hash_file(candidate_path)
        if len(candidates) != len(study["regularization"]["lambda_path"]):
            raise SummaryError(f"candidate path is incomplete: {key}")

        arrays_hash = None
        if record.get("selected"):
            arrays_path = results_dir / record["arrays_file"]
            _require_file(arrays_path, f"selected arrays {key}")
            arrays_hash = style.hash_file(arrays_path)
            if arrays_hash != record.get("arrays_sha256"):
                raise SummaryError(f"selected arrays hash mismatch: {key}")
            arrays = _load_npz(arrays_path)
            required = {
                "score_grid",
                "reference_interface_d",
                "estimated_effective_winkler_d",
                "unused_x",
                "reference_strain",
                "predicted_strain",
                "sensor_x",
                "observed_strain",
                "predicted_train_strain",
                "gamma",
            }
            if set(arrays) != required:
                raise SummaryError(f"selected arrays schema mismatch: {key}")
            if any(not np.all(np.isfinite(value)) for value in arrays.values()):
                raise SummaryError(f"selected arrays contain nonfinite values: {key}")
            selected_arrays[key] = arrays
        elif "metrics" in record or "arrays_file" in record:
            raise SummaryError(f"failed path reports selected output: {key}")

        path_records[key] = record
        candidate_records[key] = candidates
        path_hashes[key] = {
            "path.json": style.hash_file(path_file),
            "candidates": candidate_hashes,
            "selected_arrays.npz": arrays_hash,
        }

    return {
        "results_dir": results_dir,
        "lock": lock,
        "continuum": continuum,
        "datasets": datasets,
        "paths": paths,
        "mesh_report": mesh_report,
        "calibrated_k0": calibrated_k0,
        "verification": verification,
        "dataset_records": dataset_records,
        "path_records": path_records,
        "candidate_records": candidate_records,
        "selected_arrays": selected_arrays,
        "input_hashes": {
            "continuum_source_lock.json": style.hash_file(results_dir / runner.LOCK_FILE),
            "mesh_report.json": style.hash_file(results_dir / runner.MESH_REPORT),
            "calibration_report.json": style.hash_file(results_dir / runner.CALIBRATION_REPORT),
            "verification_report.json": style.hash_file(verification_path),
            "datasets": dataset_hashes,
            "paths": path_hashes,
        },
    }


def _path_row(
    dataset: dict[str, Any],
    path_config: dict[str, Any],
    path: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    selection = path.get("selection", {})
    selected = bool(path.get("selected"))
    metrics = {name: path.get("metrics", {}).get(name) for name in NUMERIC_METRICS}
    if FORBIDDEN_RELATIVE_METRICS.intersection(path.get("metrics", {})):
        raise SummaryError(f"forbidden relative interface/Winkler metric in {path_config['key']}")
    if selected:
        gamma = np.asarray(path.get("selected_gamma", []), dtype=float)
        if gamma.shape != (3,) or not np.all(np.isfinite(gamma)):
            raise SummaryError(f"selected load amplitudes are missing: {path_config['key']}")
        metrics["selected_lambda"] = selection.get("selected_lambda")
        metrics["selected_load_amplitude_mean"] = float(np.mean(gamma))
        if metrics["unused_load_strain_relative_L2"] is None:
            raise SummaryError(f"selected path lacks prediction metric: {path_config['key']}")
    else:
        metrics = {name: None for name in NUMERIC_METRICS}

    starts = [start for candidate in candidates for start in candidate.get("starts", [])]
    successful_candidates = sum(bool(item.get("successful_start")) for item in candidates)
    successful_starts = sum(bool(item.get("success")) for item in starts)
    return {
        "dataset_key": dataset["key"],
        "path_key": path_config["key"],
        "method": path_config["method"],
        "configuration": {
            "shape": dataset["shape"],
            "soil_E_Pa": float(dataset["soil_E_Pa"]),
            "soil_E_MPa": float(dataset["soil_E_Pa"]) / 1.0e6,
            "tau": float(dataset["tau"]),
            "tau_percent": 100.0 * float(dataset["tau"]),
            "seed": int(dataset["seed"]),
            "sigma": float(dataset["sigma"]),
            "sensors": int(dataset["sensors"]),
        },
        "status": path.get("status"),
        "selected": selected,
        "compatibility": {
            "selection_compatible": path.get("metrics", {}).get("selection_compatible"),
            "optimizer_failed": bool(selection.get("optimizer_failed")),
            "no_feasible_candidate": bool(selection.get("no_feasible_candidate")),
            "path_bracketed": bool(selection.get("path_bracketed")),
            "unbracketed_high": bool(selection.get("unbracketed_high")),
            "selection_reason": selection.get("selection_reason"),
        },
        "selected_gamma": path.get("selected_gamma"),
        "metrics": metrics,
        "candidate_counts": {
            "candidates": len(candidates),
            "successful_candidates": successful_candidates,
            "optimizer_failed_candidates": len(candidates) - successful_candidates,
            "starts": len(starts),
            "successful_starts": successful_starts,
            "failed_starts": len(starts) - successful_starts,
        },
        "record_hashes": {
            "path.json": None,
            "selected_arrays.npz": None,
        },
    }


def path_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    datasets_by_key = {item["key"]: item for item in bundle["datasets"]}
    rows = []
    for path_config in bundle["paths"]:
        dataset = datasets_by_key[path_config["dataset_key"]]
        row = _path_row(
            dataset,
            path_config,
            bundle["path_records"][path_config["key"]],
            bundle["candidate_records"][path_config["key"]],
        )
        hashes = bundle["input_hashes"]["paths"][path_config["key"]]
        row["record_hashes"] = {
            "path.json": hashes["path.json"],
            "selected_arrays.npz": hashes["selected_arrays.npz"],
        }
        rows.append(row)
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    dimensions = ("soil_E_Pa", "shape", "tau", "method")
    for row in rows:
        key = tuple(row["configuration"][name] if name in row["configuration"] else row[name] for name in dimensions)
        grouped[key].append(row)
    return list(sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])))


def build_statistical_summary(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = []
    dimensions = ("soil_E_Pa", "shape", "tau", "method")
    for key, materialized in _group_rows(rows):
        group = dict(zip(dimensions, key))
        selected_rows = [row for row in materialized if row["selected"]]
        group.update(
            {
                "n_paths": len(materialized),
                "counts": {
                    "selected": len(selected_rows),
                    "optimizer_failed": len(materialized) - len(selected_rows),
                    "incompatible": sum(
                        row["compatibility"]["selection_compatible"] is False for row in selected_rows
                    ),
                    "unbracketed_high": sum(
                        row["compatibility"]["unbracketed_high"] for row in selected_rows
                    ),
                },
                "metrics": {
                    metric: style.aggregate_metric(materialized, metric)
                    for metric in NUMERIC_METRICS
                },
            }
        )
        groups.append(group)

    candidates = [item for values in bundle["candidate_records"].values() for item in values]
    starts = [start for candidate in candidates for start in candidate.get("starts", [])]
    reason_counts = Counter(row["compatibility"]["selection_reason"] for row in rows)
    return {
        "schema_version": "continuum-summary-v3.0",
        "source_results": str(bundle["results_dir"]),
        "verification": {
            "passed": True,
            "checked_candidates": bundle["verification"]["checked_candidates"],
            "checked_paths": bundle["verification"]["checked_paths"],
            "expected_counts": EXPECTED_COUNTS,
        },
        "counts": {
            **EXPECTED_COUNTS,
            "selected_paths": sum(row["selected"] for row in rows),
            "optimizer_failed_paths": sum(not row["selected"] for row in rows),
            "incompatible_selected_paths": sum(
                row["compatibility"]["selection_compatible"] is False
                for row in rows
                if row["selected"]
            ),
            "unbracketed_high_selected_paths": sum(
                row["compatibility"]["unbracketed_high"]
                for row in rows
                if row["selected"]
            ),
            "successful_candidates": sum(bool(item.get("successful_start")) for item in candidates),
            "optimizer_failed_candidates": sum(not bool(item.get("successful_start")) for item in candidates),
            "start_records": len(starts),
            "successful_starts": sum(bool(item.get("success")) for item in starts),
            "failed_starts": sum(not bool(item.get("success")) for item in starts),
            "selected_reasons": dict(sorted(reason_counts.items())),
        },
        "groups": groups,
        "paths": rows,
        "input_hashes": bundle["input_hashes"],
        "calibrated_k0_N_m2": {
            runner._soil_key(soil_e): k0
            for soil_e, k0 in bundle["calibrated_k0"].items()
        },
        "interpretation": bundle["continuum"]["interpretation"],
        "experimental_validation_claimed": False,
    }


def write_group_csv(summary: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for group in summary["groups"]:
            common = {
                "record_type": "group_metric",
                "soil_E_Pa": group["soil_E_Pa"],
                "soil_E_MPa": group["soil_E_Pa"] / 1.0e6,
                "shape": group["shape"],
                "tau": group["tau"],
                "tau_percent": 100.0 * group["tau"],
                "method": group["method"],
                "n_paths": group["n_paths"],
                "n_selected": group["counts"]["selected"],
                "n_optimizer_failed": group["counts"]["optimizer_failed"],
                "n_incompatible": group["counts"]["incompatible"],
                "n_unbracketed_high": group["counts"]["unbracketed_high"],
            }
            for metric, values in group["metrics"].items():
                writer.writerow(
                    {
                        **common,
                        "metric_name": metric,
                        "n": values["n"],
                        "n_valid": values["n_valid"],
                        "null_count": values["null_count"],
                        "failed_path_count": values["failed_path_count"],
                        "median": values["median"],
                        "q25": values["q25"],
                        "q75": values["q75"],
                    }
                )


def _load_reference_fields(
    results_dir: Path, lock: dict[str, Any], shape: str, soil_e: float
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    key = f"shape-{shape}_{runner._soil_key(soil_e)}"
    record_path = results_dir / "references" / key / "reference_record.json"
    arrays_path = results_dir / "references" / key / "reference_fields.npz"
    _require_file(record_path, f"reference record {key}")
    _require_file(arrays_path, f"reference arrays {key}")
    record = read_json(record_path)
    expected = {
        "schema_version": "continuum-reference-v3.0",
        "reference_key": key,
        "shape": shape,
        "soil_E_Pa": float(soil_e),
        "design_hash": lock["design_hash"],
        "source_hash": runner.hash_json(lock["source_files"]),
        "protocol_hash": lock["protocol_hash"],
        "arrays_sha256": style.hash_file(arrays_path),
    }
    _validate_record(record, expected, f"reference {key}")
    arrays = _load_npz(arrays_path)
    required = {
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
        "sensor_indices",
        "unused_indices",
    }
    if set(arrays) != required:
        raise SummaryError(f"reference arrays schema mismatch: {key}")
    load_centres = np.asarray(record["load_centres_m"], dtype=float)
    if load_centres.tolist() != runner.ALL_CENTERS.tolist():
        raise SummaryError(f"reference load centres mismatch: {key}")
    if arrays["soil_uz"].shape != (arrays["soil_nodes"].shape[0], load_centres.size):
        raise SummaryError(f"reference vertical displacement shape mismatch: {key}")
    hashes = {
        "reference_record.json": style.hash_file(record_path),
        "reference_fields.npz": style.hash_file(arrays_path),
    }
    return arrays, hashes


def _quad_triangles(quads: np.ndarray, nodes: np.ndarray, window: tuple[float, float, float, float]) -> Triangulation:
    x0, x1, z0, z1 = window
    triangles = np.vstack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    centres = nodes[triangles].mean(axis=1)
    mask = (
        (centres[:, 0] < x0)
        | (centres[:, 0] > x1)
        | (centres[:, 1] < z0)
        | (centres[:, 1] > z1)
    )
    triangulation = Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    triangulation.set_mask(mask)
    return triangulation


def _spring(axis: plt.Axes, x: float, top: float, bottom: float) -> None:
    ys = np.linspace(top, bottom, 9)
    xs = np.array([x, x, x - 0.30, x + 0.30, x - 0.30, x + 0.30, x - 0.30, x, x])
    axis.plot(xs, ys, color="#444444", linewidth=0.9, solid_capstyle="round")


def _beam_clamp(axis: plt.Axes, edge: float) -> None:
    """Clamped beam end (w = w' = 0): wall bar with outward hatch ticks."""
    sign = 1.0 if edge > 0.0 else -1.0
    axis.plot(
        [edge, edge], [0.55, 1.45], color="black", linewidth=2.2,
        solid_capstyle="butt", zorder=4,
    )
    for yy in (0.62, 0.92, 1.22):
        axis.plot(
            [edge, edge + 1.1 * sign], [yy, yy + 0.34],
            color="black", linewidth=0.8, zorder=4,
        )


def _draw_mechanical_model(axis: plt.Axes) -> None:
    beam_bottom, beam_top = 0.90, 1.30
    axis.add_patch(Rectangle((-50.0, -10.0), 100.0, 10.0, facecolor="#f4f4f2", edgecolor="none", zorder=0))
    axis.add_patch(Rectangle((-50.0, -10.55), 100.0, 0.55, facecolor="none", edgecolor="black", hatch="////", linewidth=0.0, zorder=1))
    axis.plot([-50.0, 50.0], [0.0, 0.0], color="black", linewidth=1.2, zorder=2)
    axis.plot([-50.0, 50.0], [-10.0, -10.0], color="black", linewidth=1.5, zorder=2)
    for edge in (-50.0, 50.0):
        axis.plot([edge, edge], [0.0, -10.0], color="black", linewidth=1.6)
    axis.add_patch(
        Rectangle(
            (-50.0, beam_bottom), 100.0, beam_top - beam_bottom,
            facecolor="#666666", edgecolor="black", linewidth=0.8, zorder=3,
        )
    )
    for x in (-41.0, -27.0, -13.0, 1.2, 15.0, 29.0, 43.0):
        _spring(axis, x, beam_bottom, 0.0)
    for edge in (-50.0, 50.0):
        _beam_clamp(axis, edge)
    # Soil side boundaries: horizontal rollers, distinct from the beam clamps.
    for edge in (-50.0, 50.0):
        sign = 1.0 if edge > 0.0 else -1.0
        for yy in (-2.0, -4.0, -6.0, -8.0):
            axis.add_patch(
                Circle(
                    (edge + 0.45 * sign, yy), 0.22,
                    facecolor="white", edgecolor="black", linewidth=0.8, zorder=3,
                )
            )
    axis.plot([0.0, 0.0], [1.90, 1.44], color="black", linewidth=1.3, zorder=5)
    axis.plot([0.0], [1.37], marker="v", markersize=7, color="black", zorder=5)
    axis.plot([43.0, 43.0], [beam_top + 0.03, 3.20], color="black", linewidth=0.7, zorder=4)
    axis.plot([-30.0, -18.0], [-0.75, -0.75], color="black", linewidth=0.9, zorder=4)
    axis.plot([-30.0], [-0.75], marker="<", markersize=5, color="black", zorder=4)
    axis.plot([-18.0], [-0.75], marker=">", markersize=5, color="black", zorder=4)
    axis.plot([-47.4, -47.4], [-9.7, -0.3], color="black", linewidth=0.9, zorder=4)
    axis.plot([-47.4], [-0.25], marker="v", markersize=5, color="black", zorder=4)
    axis.plot([-47.4], [-9.75], marker="^", markersize=5, color="black", zorder=4)
    axis.text(-57.0, 4.00, "surface beam, $EI$; $B=0.6$ m out of plane", ha="left", va="center")
    axis.text(57.0, 4.00, "bilateral vertical springs $k_s(1-d_I)$", ha="right", va="center")
    axis.text(-57.0, 2.55, "clamped beam ends: $w=w'=0$", ha="left", va="center")
    axis.text(-14.0, 2.55, "resultant of footprint load, $b=4.3$ m", ha="left", va="center")
    axis.text(-44.5, -5.0, "$H=10$ m", ha="left", va="center")
    axis.text(-24.0, -1.70, "surface tangential slip", ha="center", va="center")
    axis.text(-44.7, -8.7, "$u_x=0$", ha="left", va="center")
    axis.text(44.7, -8.7, "$u_x=0$", ha="right", va="center")
    axis.text(0.0, -10.95, "base $u_x=u_z=0$; remote ends at $x=\\pm 50$ m", ha="center", va="top")
    axis.set_xlim(-58.0, 58.0)
    axis.set_ylim(-12.6, 4.7)
    axis.set_xticks((-50.0, 0.0, 50.0))
    axis.set_yticks((0.0, -5.0, -10.0))
    axis.set_yticklabels(("0", "5", "10"))
    axis.set_xlabel("Schematic position (m)")
    axis.set_ylabel("Depth (m)")
    axis.set_title("(a) Mechanical model (not to scale)", pad=8)
    style._style_axis(axis)
    axis.grid(False)


def render_reference_figure(fields_by_e: dict[float, dict[str, np.ndarray]], out_dir: Path) -> dict[str, Any]:
    if set(fields_by_e) != set(SOIL_E_PA):
        raise SummaryError("reference figure requires exactly E=50 and 200 MPa verified fields")
    fig = plt.figure(figsize=(style.FIGURE_WIDTH_INCHES, 6.0))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.55, 1.00, 0.07),
        left=0.105,
        right=0.985,
        top=0.95,
        bottom=0.08,
        wspace=0.26,
        hspace=0.45,
    )
    schematic = fig.add_subplot(grid[0, :])
    _draw_mechanical_model(schematic)
    contour_axes = [fig.add_subplot(grid[1, index]) for index in range(2)]
    color_axis = fig.add_subplot(grid[2, :])
    window = (-12.0, 12.0, 0.0, 10.0)
    common_min = math.inf
    common_max = -math.inf
    prepared = {}
    central_index = 1
    for soil_e, arrays in fields_by_e.items():
        nodes = np.asarray(arrays["soil_nodes"], dtype=float)
        quads = np.asarray(arrays["soil_quads"], dtype=np.int64)
        values = np.asarray(arrays["soil_uz"][:, central_index], dtype=float) * 1000.0
        if nodes.shape != (nodes.shape[0], 2) or quads.ndim != 2 or quads.shape[1] != 4:
            raise SummaryError(f"reference mesh arrays are not nodal quadrilateral fields: E={soil_e}")
        if values.shape != (nodes.shape[0],) or not np.all(np.isfinite(values)):
            raise SummaryError(f"reference central-load displacement is invalid: E={soil_e}")
        prepared[soil_e] = (nodes, quads, values, _quad_triangles(quads, nodes, window))
        common_min = min(common_min, float(np.min(values)))
        common_max = max(common_max, float(np.max(values)))
    if not (math.isfinite(common_min) and math.isfinite(common_max)) or common_max <= common_min:
        raise SummaryError("reference displacement fields have no usable common color range")
    common_min = min(0.0, common_min)
    common_max = max(0.0, common_max)
    for axis, soil_e, label in zip(contour_axes, SOIL_E_PA, ("(b)", "(c)"), strict=True):
        _, _, values, triangulation = prepared[soil_e]
        mesh = axis.tripcolor(
            triangulation,
            values,
            cmap="viridis",
            shading="gouraud",
            vmin=common_min,
            vmax=common_max,
            rasterized=True,
        )
        axis.set_xlim(window[0], window[1])
        axis.set_ylim(window[2], window[3])
        axis.invert_yaxis()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks((-10, -5, 0, 5, 10))
        axis.set_yticks((0, 5, 10))
        axis.set_xlabel("Position $x$ (m)")
        axis.set_ylabel("Depth $z$ (m)")
        axis.set_title(f"{label} $E={soil_e / 1.0e6:.0f}$ MPa, asymmetric", pad=7)
        style._style_axis(axis)
    colorbar = fig.colorbar(mesh, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Vertical soil displacement $u_z$ (mm); positive down")
    colorbar.ax.tick_params(labelsize=style.FONT_SIZE_PT)
    return style.save_and_audit_figure(fig, out_dir, "continuum_reference")


def make_reference_figure(bundle: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    fields: dict[float, dict[str, np.ndarray]] = {}
    reference_hashes: dict[str, dict[str, str]] = {}
    for soil_e in SOIL_E_PA:
        arrays, hashes = _load_reference_fields(
            bundle["results_dir"], bundle["lock"], "asymmetric", soil_e
        )
        fields[soil_e] = arrays
        reference_hashes[runner._soil_key(soil_e)] = hashes
    report = render_reference_figure(fields, out_dir)
    report["reference_hashes"] = reference_hashes
    return report


def _exact_rows(
    rows: list[dict[str, Any]],
    *,
    shape: str,
    soil_e: float,
    tau: float,
    method: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["configuration"]["shape"] == shape
        and math.isclose(row["configuration"]["soil_E_Pa"], soil_e)
        and math.isclose(row["configuration"]["tau"], tau)
        and row["method"] == method
        and (seed is None or row["configuration"]["seed"] == seed)
    ]
    expected_count = 3 if seed is None else 1
    if len(selected) != expected_count:
        raise SummaryError(
            f"predesignated paths are missing or duplicated: {shape}/{soil_e}/{tau}/{method}/seed={seed}"
        )
    if not all(row["selected"] for row in selected):
        raise SummaryError(
            f"predesignated path failed; numerical figure cannot be generated: {shape}/{soil_e}/{tau}/{method}"
        )
    return selected


def make_prediction_figure(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(style.FIGURE_WIDTH_INCHES, 6.15), sharey=True)
    offsets = {method: (index - 1.5) * 0.17 for index, method in enumerate(METHODS)}
    for row_index, shape in enumerate(("asymmetric", "broad_shallow")):
        for column_index, soil_e in enumerate(SOIL_E_PA):
            axis = axes[row_index, column_index]
            for method in METHODS:
                values_by_tau = []
                for tau in TAU_LEVELS:
                    selected = _exact_rows(rows, shape=shape, soil_e=soil_e, tau=tau, method=method)
                    values = [row["metrics"]["unused_load_strain_relative_L2"] for row in selected]
                    if any(value is None for value in values):
                        raise SummaryError(f"prediction-quality value is missing: {shape}/{soil_e}/{tau}/{method}")
                    values_by_tau.append(100.0 * np.asarray(values, dtype=float))
                for index, values in enumerate(values_by_tau):
                    q25, median, q75 = np.percentile(values, (25.0, 50.0, 75.0))
                    x_position = index + offsets[method]
                    color = style.METHOD_COLORS[method]
                    axis.errorbar(
                        x_position,
                        median,
                        yerr=((median - q25,), (q75 - median,)),
                        fmt="none",
                        color=color,
                        elinewidth=1.0,
                        capsize=2.4,
                    )
                    axis.plot(
                        x_position,
                        median,
                        marker=style.METHOD_MARKERS[method],
                        linestyle="none",
                        markersize=5.0,
                        color=color,
                        markerfacecolor="white" if method in {"fixed", "oracle"} else color,
                        markeredgewidth=1.0,
                    )
            axis.set_xticks((0.0, 1.0), ("1", "10"))
            axis.set_xlim(-0.45, 1.45)
            axis.set_xlabel(r"Load measurement uncertainty $\tau$ (%)")
            if column_index == 0:
                axis.set_ylabel("Unused-load strain error (%)")
            panel = chr(ord("a") + row_index * 2 + column_index)
            axis.set_title(
                f"({panel}) {'Asymmetric' if shape == 'asymmetric' else 'Broad shallow'}, "
                f"$E={soil_e / 1.0e6:.0f}$ MPa",
                pad=7,
            )
            style._style_axis(axis)
    fig.legend(
        handles=style._method_handles(METHODS),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.115, right=0.985, top=0.92, bottom=0.185, wspace=0.25, hspace=0.50)
    return style.save_and_audit_figure(fig, out_dir, "continuum_prediction")


def make_response_figure(
    bundle: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path
) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 1, figsize=(style.FIGURE_WIDTH_INCHES, 6.15), sharex=True)
    consumed_hashes = {}
    for axis, soil_e, label in zip(axes, SOIL_E_PA, ("(a)", "(b)"), strict=True):
        fixed_row = _exact_rows(
            rows,
            shape=RESPONSE_SHAPE,
            soil_e=soil_e,
            tau=RESPONSE_TAU,
            method="fixed",
            seed=RESPONSE_SEED,
        )[0]
        fixed_arrays = bundle["selected_arrays"][fixed_row["path_key"]]
        axis.plot(
            fixed_arrays["unused_x"],
            fixed_arrays["reference_strain"] * 1.0e6,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="Continuum reference",
        )
        for method in METHODS:
            selected = _exact_rows(
                rows,
                shape=RESPONSE_SHAPE,
                soil_e=soil_e,
                tau=RESPONSE_TAU,
                method=method,
                seed=RESPONSE_SEED,
            )
            row = selected[0]
            arrays = bundle["selected_arrays"][row["path_key"]]
            if not (
                np.array_equal(fixed_arrays["unused_x"], arrays["unused_x"])
                and np.array_equal(fixed_arrays["reference_strain"], arrays["reference_strain"])
            ):
                raise SummaryError(f"response references differ across methods: E={soil_e}")
            axis.plot(
                arrays["unused_x"],
                arrays["predicted_strain"] * 1.0e6,
                color=style.METHOD_COLORS[method],
                linestyle=style.METHOD_LINESTYLES[method],
                linewidth=1.10,
                marker=style.METHOD_MARKERS[method],
                markersize=3.1,
                markevery=15,
                markerfacecolor="none",
                markeredgewidth=0.75,
                label=style.METHOD_LABELS[method],
            )
            consumed_hashes[row["path_key"]] = row["record_hashes"]["selected_arrays.npz"]
        axis.set_xlim(-12.0, 12.0)
        axis.set_xticks((-10, -5, 0, 5, 10))
        axis.set_ylabel("Finite-gauge strain ($\\mu\\varepsilon$)")
        axis.set_title(f"{label} Asymmetric, $E={soil_e / 1.0e6:.0f}$ MPa", pad=7)
        style._style_axis(axis)
    axes[1].set_xlabel("Position $x$ (m)")
    handles = [
        plt.Line2D((0,), (0,), color="black", linestyle="--", linewidth=1.3, label="Continuum reference")
    ] + style._method_handles(METHODS)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=5, frameon=False)
    fig.subplots_adjust(left=0.160, right=0.985, top=0.92, bottom=0.175, hspace=0.42)
    report = style.save_and_audit_figure(fig, out_dir, "continuum_response")
    report["selected_array_hashes"] = consumed_hashes
    return report


def _caption_conditions() -> dict[str, dict[str, Any]]:
    return {
        "continuum_reference": {
            "caption": (
                "(a) Surface beam (out-of-plane width $B=0.6$ m) coupled to a plane-strain "
                "elastic layer through bilateral vertical springs $k_s(1-d_I)$ across the "
                "schematic beam-soil gap; $H=10$ m, beam ends clamped ($w=w'=0$) at "
                "$x=\\pm 50$ m, lateral rollers $u_x=0$, base $u_x=u_z=0$, surface tangential "
                "slip, and the downward resultant of the $b=4.3$ m loading footprint. "
                "(b,c) Nodal vertical soil displacement from the verified fine-grid "
                "asymmetric-interface central-load reference over $x\\in[-12,12]$ m and "
                "$z\\in[0,10]$ m (positive down) for $E=50$ and 200 MPa, with shared color "
                "limits spanning the actual minimum and maximum of both fields. The "
                "mechanical drawing is not to scale."
            ),
            "conditions": {
                "shape": "asymmetric",
                "soil_E_MPa": [50, 200],
                "load_centre_m": 0.0,
                "field": "nodal soil_uz",
                "units": "mm, positive downward",
                "mesh": "verified fine-grid reference fields",
            },
        },
        "continuum_prediction": {
            "caption": (
                "Unused-load relative L2 strain error for fixed, free, calibrated, and oracle "
                "load information. Markers are medians over three seeds and bars are the 25th "
                "and 75th percentiles. The two $\\tau$ values are categorical experiment levels, "
                "not a continuous fitted relationship."
            ),
            "conditions": {
                "shapes": ["asymmetric", "broad_shallow"],
                "soil_E_MPa": [50, 200],
                "tau": [0.01, 0.10],
                "seeds": [1201, 1202, 1203],
                "metric": "unused-load strain relative L2",
                "bars": "q25-median-q75 over seeds",
            },
        },
        "continuum_response": {
            "caption": (
                "Actual unused-load finite-gauge strain profiles for the asymmetric case with "
                "$\\tau=10\\%$ and seed 1201. Black dashed curves are the continuum reference; "
                "colored curves are the verified selected predictions."
            ),
            "conditions": {
                "shape": "asymmetric",
                "soil_E_MPa": [50, 200],
                "tau": 0.10,
                "seed": 1201,
                "x_window_m": [-12, 12],
            },
        },
    }


def _resolve_relative(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    try:
        resolved = path.resolve()
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise SummaryError(f"path escapes the project root: {value}") from error
    return resolved


def _output_targets(out_dir: Path, summary_dir: Path) -> list[Path]:
    return [
        item
        for stem in ("continuum_reference", "continuum_prediction", "continuum_response")
        for item in (out_dir / f"{stem}.pdf", out_dir / f"{stem}.png")
    ] + [
        out_dir / "continuum_layout_audit.json",
        summary_dir / "continuum_summary.json",
        summary_dir / "continuum_summary.csv",
    ]


def generate_all(results_arg: Path, out_dir_arg: Path, summary_dir_arg: Path) -> dict[str, Any]:
    results_dir = _resolve_relative(results_arg)
    out_dir = _resolve_relative(out_dir_arg)
    summary_dir = _resolve_relative(summary_dir_arg)
    existing = [str(path) for path in _output_targets(out_dir, summary_dir) if path.exists()]
    if existing:
        raise SummaryError("refusing to overwrite prior continuum output: " + "; ".join(existing))

    bundle = load_verified_results(results_dir)
    rows = path_rows(bundle)
    summary = build_statistical_summary(bundle, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "continuum_reference": make_reference_figure(bundle, out_dir),
        "continuum_prediction": make_prediction_figure(rows, out_dir),
        "continuum_response": make_response_figure(bundle, rows, out_dir),
    }
    layout_report = {
        "schema_version": "continuum-layout-audit-v3.0",
        "created_utc": style.utc_now(),
        "figures": {
            name: {
                "warnings": report["layout_warnings"],
                "inspection": report["inspection"],
            }
            for name, report in figures.items()
        },
        "policy": {
            "width_inches": style.FIGURE_WIDTH_INCHES,
            "font_size_pt": style.FONT_SIZE_PT,
            "tight_bbox_used": False,
            "note": "Automated warnings are screening only; root must inspect the actual PNG/PDF.",
        },
    }
    summary["figures"] = figures
    summary["captions_and_conditions"] = _caption_conditions()
    summary["provenance"] = {
        "results_root": str(results_dir),
        "source_lock_schema": bundle["lock"]["schema_version"],
        "mesh_passed": bundle["mesh_report"]["passed"],
        "calibration_passed": True,
        "verification_passed": bundle["verification"]["passed"],
        "calibrated_k0_source": "accepted intact-interface calibration records",
        "recomputation": "none; verified saved path/reference arrays were consumed",
    }
    style.write_new_json(out_dir / "continuum_layout_audit.json", layout_report)
    style.write_new_json(summary_dir / "continuum_summary.json", summary)
    write_group_csv(summary, summary_dir / "continuum_summary.csv")
    return {
        "status": "completed_with_layout_warnings"
        if any(report["layout_warnings"] for report in figures.values())
        else "completed",
        "figures": {
            name: {"pdf": report["pdf"], "png": report["png"], "warnings": report["layout_warnings"]}
            for name, report in figures.items()
        },
        "summary_json": str(summary_dir / "continuum_summary.json"),
        "summary_csv": str(summary_dir / "continuum_summary.csv"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--summary-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = generate_all(args.results, args.out_dir, args.summary_dir)
    except (SummaryError, ProtocolError, OSError, ValueError, KeyError) as error:
        print(
            json.dumps(
                {"status": "error", "error": f"{type(error).__name__}: {error}"},
                allow_nan=False,
            )
        )
        return 2
    print(json.dumps(runner.safe_json(result), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
