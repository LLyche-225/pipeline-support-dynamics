"""Publication figures and manuscript summary for the v3 load-information study.

The command consumes only a completed, independently verified formal result
tree.  It does not synthesize replacement numerical data and does not rerun the
campaign.
"""

from __future__ import annotations

import os

# The user's default Matplotlib configuration directory is not writable in this
# sandbox.  A private temporary cache avoids harmless but noisy fallback writes.
os.environ["MPLCONFIGDIR"] = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "matplotlib-load-information-v3"
)

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "vendor", ROOT / "code"):
    _absolute = str(_path.resolve())
    if _absolute not in sys.path:
        sys.path.insert(0, _absolute)

import run_load_information as runner  # noqa: E402
from beam_inverse import HermiteBeamFE  # noqa: E402
from campaign import PhysicalModel  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

def _import_figure_layout_audit():
    """Import the layout-audit helper without an absolute per-user path.

    The project-local copy ``vendor/figure_layout_audit.py`` (see
    ``vendor/FIGURE_AUDIT_PROVENANCE.md``) is preferred so the research
    package reproduces on any machine.  The original skill installation
    under the current user's Codex home is only an optional fallback.
    """
    vendor_dir = ROOT / "vendor"
    if (vendor_dir / "figure_layout_audit.py").is_file():
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        import figure_layout_audit

        return figure_layout_audit
    fallback_dir = (
        Path.home() / ".codex" / "skills" / "paper-figure-consistency" / "scripts"
    )
    if (fallback_dir / "figure_layout_audit.py").is_file():
        if str(fallback_dir) not in sys.path:
            sys.path.append(str(fallback_dir))
        import figure_layout_audit

        return figure_layout_audit
    raise ImportError(
        "figure_layout_audit.py is missing from vendor/ and from the "
        "optional skill fallback; see vendor/FIGURE_AUDIT_PROVENANCE.md"
    )


figure_layout_audit = _import_figure_layout_audit()  # noqa: E402
audit_figure = figure_layout_audit.audit_figure  # noqa: E402


FIGURE_WIDTH_INCHES = 6.5
FIGURE_WIDTH_PT = 468.0
FONT_SIZE_PT = 10
PNG_DPI = 300
EXPECTED_COUNTS = {
    "datasets": 200,
    "paths": 848,
    "candidates": 5088,
    "starts": 10176,
    "overlap_datasets": 16,
}
MAIN_METHODS = ("fixed", "free", "calibrated", "oracle")
SENSITIVITY_METHODS = ("overconfident", "calibrated", "conservative")
SENSITIVITY_FACTORS = {"overconfident": 0.2, "calibrated": 1.0, "conservative": 3.0}
METHOD_LABELS = {
    "fixed": "Fixed",
    "free": "Free",
    "calibrated": "Calibrated",
    "oracle": "Oracle",
    "overconfident": "Overconfident",
    "conservative": "Conservative",
}
METHOD_COLORS = {
    "fixed": "#000000",
    "free": "#0072B2",
    "calibrated": "#009E73",
    "oracle": "#D55E00",
    "overconfident": "#0072B2",
    "conservative": "#D55E00",
}
METHOD_MARKERS = {
    "fixed": "o",
    "free": "s",
    "calibrated": "^",
    "oracle": "D",
    "overconfident": "v",
    "conservative": "^",
}
METHOD_LINESTYLES = {
    "fixed": "-",
    "free": "--",
    "calibrated": "-.",
    "oracle": ":",
    "overconfident": "--",
    "conservative": ":",
}
METRICS = (
    "loss_area_m",
    "loss_area_absolute_error_m",
    "absolute_support_L2",
    "support_rel_L2_except_intact",
    "max_false_loss_for_intact",
    "gamma_RMSE",
    "unused_load_strain_relative_L2",
    "unused_load_peak_strain_error",
)
TAU_LEVELS = (0.01, 0.05, 0.10)
FIELDS_SEED = 1101
FIELDS_SIGMA = 0.000005
FIELDS_SENSORS = 20

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


class SummaryError(RuntimeError):
    """Formal results are absent, incomplete, or inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise SummaryError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(runner.safe_json(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise SummaryError(f"missing {description}: {path}")


def _validate_expected_expansion(design: dict) -> tuple[list[dict], list[dict]]:
    datasets, paths, counts = runner.expand_study(design)
    mismatches = [
        f"{name}={counts.get(name, 'missing')} (expected {expected})"
        for name, expected in EXPECTED_COUNTS.items()
        if counts.get(name) != expected
    ]
    if mismatches:
        raise SummaryError(
            "study expansion does not match the frozen D1 intake: "
            + "; ".join(mismatches)
        )
    if len(datasets) != EXPECTED_COUNTS["datasets"] or len(paths) != EXPECTED_COUNTS["paths"]:
        raise SummaryError("expanded dataset/path lists have unexpected length")
    return datasets, paths


def _validate_run_status(status: dict) -> None:
    expected = {
        "schema_version": "load-information-run-status-v3.0",
        "status": "completed",
        "completed_paths": EXPECTED_COUNTS["paths"],
        "selected_path_count": EXPECTED_COUNTS["paths"],
    }
    for key, value in expected.items():
        if status.get(key) != value:
            raise SummaryError(
                f"run status has {key}={status.get(key)!r}; expected {value!r}"
            )
    reported = status.get("expected_full_counts", {})
    for key, value in EXPECTED_COUNTS.items():
        if reported.get(key) != value:
            raise SummaryError(
                f"run status full count {key}={reported.get(key)!r}; expected {value}"
            )


def _validate_verification(report: dict) -> None:
    expected = {
        "schema_version": "load-information-verification-v3.0",
        "passed": True,
        "incomplete": False,
        "checked_candidates": EXPECTED_COUNTS["candidates"],
        "checked_completed_paths": EXPECTED_COUNTS["paths"],
        "expected_candidates": EXPECTED_COUNTS["candidates"],
        "expected_paths": EXPECTED_COUNTS["paths"],
    }
    problems = [
        f"{key}={report.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if report.get(key) != value
    ]
    if report.get("errors"):
        problems.append(f"verification errors={report['errors']}")
    if report.get("incomplete_paths"):
        problems.append(f"incomplete paths={report['incomplete_paths']}")
    if problems:
        raise SummaryError("independent verification is not acceptable: " + "; ".join(problems))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]).copy() for name in saved.files}


def _actual_directories(parent: Path) -> set[str]:
    if not parent.is_dir():
        return set()
    return {item.name for item in parent.iterdir() if item.is_dir()}


def _check_record_identity(
    record: dict,
    expected: dict[str, Any],
    context: str,
) -> None:
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise SummaryError(f"{context} record mismatch for: {', '.join(mismatches)}")


def load_verified_results(results_dir: Path) -> dict[str, Any]:
    """Read and identity-check the complete formal result tree."""
    if not results_dir.is_dir():
        raise SummaryError(f"results directory is missing: {results_dir}")
    design = runner.load_design()
    lock = runner.load_and_check_lock(results_dir, design)
    datasets, paths = _validate_expected_expansion(design)

    status_path = results_dir / "run_status.json"
    verification_path = results_dir / "verification_report.json"
    _require_file(status_path, "completed run status")
    _require_file(verification_path, "independent verification report")
    status = runner.read_json(status_path)
    verification = runner.read_json(verification_path)
    _validate_run_status(status)
    _validate_verification(verification)

    expected_dataset_dirs = {item["key"] for item in datasets}
    actual_dataset_dirs = _actual_directories(results_dir / "datasets")
    extra_datasets = sorted(actual_dataset_dirs - expected_dataset_dirs)
    if extra_datasets:
        raise SummaryError(f"unexpected dataset directories: {extra_datasets}")
    expected_path_dirs = {item["key"] for item in paths}
    actual_path_dirs = _actual_directories(results_dir / "paths")
    extra_paths = sorted(actual_path_dirs - expected_path_dirs)
    if extra_paths:
        raise SummaryError(f"unexpected path directories: {extra_paths}")

    input_hashes: dict[str, Any] = {
        "study_design_v3.json": hash_file(ROOT / "study_design_v3.json"),
        "load_information_lock.json": hash_file(results_dir / runner.LOCK_FILE),
        "run_status.json": hash_file(status_path),
        "verification_report.json": hash_file(verification_path),
        "runner_source_files": dict(lock["source_files"]),
        "runner_test_files": dict(lock["test_files"]),
        "runner_vendor_files": dict(lock["vendor"]["files"]),
        "code/run_load_information.py": hash_file(ROOT / "code/run_load_information.py"),
        "code/make_load_information_figures.py": hash_file(Path(__file__)),
        "figure_layout_audit.py": hash_file(Path(figure_layout_audit.__file__)),
        "datasets": {},
        "paths": {},
    }
    source_hash = runner.hash_json(lock["source_files"])

    dataset_records: dict[str, dict] = {}
    for dataset in datasets:
        dataset_dir = results_dir / "datasets" / dataset["key"]
        record_path = dataset_dir / "dataset_record.json"
        _require_file(record_path, f"dataset record for {dataset['key']}")
        record = runner.read_json(record_path)
        _check_record_identity(
            record,
            {
                "schema_version": "load-information-dataset-v3.0",
                "dataset_key": dataset["key"],
                "config": dataset,
                "config_hash": dataset["config_hash"],
                "memberships": dataset["memberships"],
                "design_hash": lock["design_hash"],
                "protocol_hash": lock["protocol_hash"],
                "source_hash": source_hash,
            },
            dataset["key"],
        )
        npz_path = results_dir / record["npz_file"]
        _require_file(npz_path, f"dataset arrays for {dataset['key']}")
        if hash_file(npz_path) != record["npz_sha256"]:
            raise SummaryError(f"dataset NPZ hash mismatch: {dataset['key']}")
        dataset_records[dataset["key"]] = record
        input_hashes["datasets"][dataset["key"]] = {
            "dataset_record.json": hash_file(record_path),
            "npz": record["npz_sha256"],
            "npz_file_current_hash": hash_file(npz_path),
        }

    datasets_by_key = {item["key"]: item for item in datasets}
    loaded_paths: dict[str, dict] = {}
    candidate_records: dict[str, list[dict]] = {}
    field_arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    lambdas = design["regularization"]["lambda_path"]

    for path_config in paths:
        key = path_config["key"]
        path_dir = results_dir / "paths" / key
        path_file = path_dir / "path.json"
        _require_file(path_file, f"completed path record for {key}")
        path_record = runner.read_json(path_file)
        dataset = datasets_by_key[path_config["dataset_key"]]
        dataset_record = dataset_records[path_config["dataset_key"]]
        _check_record_identity(
            path_record,
            {
                "schema_version": "load-information-path-v3.0",
                "path_key": key,
                "dataset_key": dataset["key"],
                "method": path_config["method"],
                "method_family": path_config["method_family"],
                "lambda_path": lambdas,
                "starts": design["regularization"]["starts"],
                "design_hash": lock["design_hash"],
                "protocol_hash": lock["protocol_hash"],
                "source_hash": source_hash,
                "dataset_hash": dataset_record["npz_sha256"],
                "memberships": dataset["memberships"],
            },
            key,
        )
        candidates = []
        path_hashes: dict[str, Any] = {"path.json": hash_file(path_file), "candidates": {}}
        for penalty_lambda in lambdas:
            candidate_file = runner._candidate_filename(penalty_lambda)
            candidate_path = path_dir / candidate_file
            _require_file(candidate_path, f"candidate {penalty_lambda:.17g} for {key}")
            candidate = runner.read_json(candidate_path)
            _check_record_identity(
                candidate,
                {
                    "schema_version": "load-information-candidate-v3.0",
                    "path_key": key,
                    "dataset_key": dataset["key"],
                    "method": path_config["method"],
                    "penalty": "H1",
                    "penalty_lambda": penalty_lambda,
                    "design_hash": lock["design_hash"],
                    "protocol_hash": lock["protocol_hash"],
                    "source_hash": source_hash,
                    "dataset_hash": dataset_record["npz_sha256"],
                },
                f"{key} lambda={penalty_lambda:.17g}",
            )
            if len(candidate.get("starts", [])) != len(design["regularization"]["starts"]):
                raise SummaryError(f"candidate start records are incomplete for {key}")
            candidates.append(candidate)
            path_hashes["candidates"][candidate_file] = hash_file(candidate_path)
        candidate_records[key] = candidates

        if path_record.get("selected"):
            selected_lambda = path_record["selection"]["selected_lambda"]
            selected = [
                item for item in candidates if item["penalty_lambda"] == selected_lambda
            ]
            if len(selected) != 1:
                raise SummaryError(f"selected candidate is ambiguous or absent for {key}")
            arrays_file = path_dir / Path(path_record["arrays_file"]).name
            _require_file(arrays_file, f"selected arrays for {key}")
            if hash_file(arrays_file) != path_record["arrays_sha256"]:
                raise SummaryError(f"selected-array hash mismatch for {key}")
            arrays = _load_npz(arrays_file)
            required_arrays = {
                "score_x",
                "reference_d",
                "estimated_d",
                "unused_x",
                "reference_strain",
                "predicted_strain",
            }
            # The runner also stores coefficients/gamma/train_prediction in the
            # same NPZ; the figures consume exactly these six arrays.
            if not required_arrays.issubset(arrays):
                raise SummaryError(f"selected-array schema mismatch for {key}")
            for name in required_arrays:
                if not np.all(np.isfinite(arrays[name])):
                    raise SummaryError(f"nonfinite selected array {name} for {key}")
            if arrays["score_x"].shape != arrays["reference_d"].shape:
                raise SummaryError(f"support-array shape mismatch for {key}")
            path_hashes["selected_arrays.npz"] = hash_file(arrays_file)
            field_arrays[(dataset["key"], path_config["method"])] = arrays

        loaded_paths[key] = path_record
        input_hashes["paths"][key] = path_hashes

    if len(loaded_paths) != EXPECTED_COUNTS["paths"]:
        raise SummaryError("not every expected completed path was loaded")
    if sum(len(items) for items in candidate_records.values()) != EXPECTED_COUNTS["candidates"]:
        raise SummaryError("not every expected candidate was loaded")

    return {
        "results_dir": results_dir,
        "design": design,
        "lock": lock,
        "run_status": status,
        "verification": verification,
        "datasets": datasets,
        "paths": paths,
        "dataset_records": dataset_records,
        "path_records": loaded_paths,
        "candidate_records": candidate_records,
        "field_arrays": field_arrays,
        "input_hashes": input_hashes,
    }


def _candidate_status(candidate: dict) -> str:
    return "successful" if candidate.get("successful_start") else "optimizer_failed"


def _path_summary(
    dataset: dict,
    dataset_record: dict,
    path_config: dict,
    path: dict,
    candidates: list[dict],
) -> dict:
    selection = path.get("selection", {})
    metrics = {name: path.get("metrics", {}).get(name) for name in METRICS}
    starts = [start for candidate in candidates for start in candidate.get("starts", [])]
    successful_candidates = sum(bool(item.get("successful_start")) for item in candidates)
    successful_starts = sum(bool(item.get("success")) for item in starts)
    selected_gamma = path.get("selected_gamma")
    selected_lambda = selection.get("selected_lambda")
    selected_file = None
    if path.get("selected"):
        selected_file = runner._candidate_filename(float(selected_lambda))
    return {
        "dataset_key": dataset["key"],
        "path_key": path_config["key"],
        "configuration": {
            "shape": dataset["shape"],
            "tau": dataset["tau"],
            "tau_percent": 100.0 * dataset["tau"],
            "sigma": dataset["sigma"],
            "sigma_microstrain": dataset["sigma"] * 1.0e6,
            "sensors": dataset["sensors"],
            "seed": dataset["seed"],
        },
        "memberships": dataset["memberships"],
        "method": path_config["method"],
        "method_family": path_config["method_family"],
        "status": path.get("status"),
        "selected": bool(path.get("selected")),
        "compatibility": {
            "optimizer_failed": bool(selection.get("optimizer_failed")),
            "no_feasible_candidate": bool(selection.get("no_feasible_candidate")),
            "path_bracketed": bool(selection.get("path_bracketed")),
            "unbracketed_high": bool(selection.get("unbracketed_high")),
            "selection_reason": selection.get("selection_reason"),
        },
        "selected_lambda": selected_lambda,
        "selected_gamma": selected_gamma,
        "calibrated_mean": dataset_record["measured_load_mean"],
        "true_load_standard_uncertainty": dataset_record["true_load_standard_uncertainty"],
        "metrics": metrics,
        "selection_counts": {
            "N_strain_observations": selection.get("N_strain_observations"),
            "M_calibration_observations": selection.get("M_calibration_observations"),
            "threshold": selection.get("threshold"),
            "selection_norm": selection.get("selection_norm"),
            "data_norm": selection.get("data_norm"),
            "calibration_norm": selection.get("calibration_norm"),
        },
        "counts": {
            "candidates": len(candidates),
            "successful_candidates": successful_candidates,
            "optimizer_failed_candidates": len(candidates) - successful_candidates,
            "starts": len(starts),
            "successful_starts": successful_starts,
            "failed_starts": len(starts) - successful_starts,
        },
        "candidate_statuses": {
            _float_key(item["penalty_lambda"]): _candidate_status(item) for item in candidates
        },
        "candidate_details": [
            {
                "penalty_lambda": item["penalty_lambda"],
                "candidate_file": runner._candidate_filename(item["penalty_lambda"]),
                "status": _candidate_status(item),
                "successful_start": bool(item["successful_start"]),
                "chosen_start": item["chosen_start"],
                "successful_starts": sum(bool(start.get("success")) for start in item["starts"]),
                "failed_starts": sum(not start.get("success") for start in item["starts"]),
                "start_records": [
                    {
                        "success": bool(start.get("success")),
                        "status": start.get("status"),
                        "cost": start.get("cost"),
                        "nfev": start.get("nfev"),
                        "optimality": start.get("optimality"),
                        "projected_first_order": start.get("projected_first_order"),
                        "wall_time_s": start.get("wall_time_s"),
                    }
                    for start in item["starts"]
                ],
                "elapsed_s": float(item.get("elapsed_s", 0.0)),
            }
            for item in candidates
        ],
        "selected_candidate_file": selected_file,
        "computational_time_descriptive_s": sum(
            float(item.get("elapsed_s", 0.0)) for item in candidates
        ),
    }


def _float_key(value: float) -> str:
    return runner._format_number(float(value))


def path_summaries(bundle: dict[str, Any]) -> list[dict]:
    datasets_by_key = {item["key"]: item for item in bundle["datasets"]}
    summaries = []
    for path_config in bundle["paths"]:
        dataset = datasets_by_key[path_config["dataset_key"]]
        summaries.append(
            _path_summary(
                dataset,
                bundle["dataset_records"][dataset["key"]],
                path_config,
                bundle["path_records"][path_config["key"]],
                bundle["candidate_records"][path_config["key"]],
            )
        )
    return summaries


def aggregate_metric(rows: Iterable[dict], metric: str) -> dict[str, Any]:
    """Summarize a metric while explicitly counting null and failed records."""
    materialized = list(rows)
    replicates = []
    values = []
    for row in materialized:
        value = row.get("metrics", {}).get(metric)
        if value is None:
            continue
        values.append(float(value))
        replicates.append(
            {
                "dataset_key": row["dataset_key"],
                "seed": row["configuration"]["seed"],
                "value": float(value),
            }
        )
    if values:
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        statistics = {
            "median": float(median),
            "q25": float(q25),
            "q75": float(q75),
        }
    else:
        statistics = {"median": None, "q25": None, "q75": None}
    return {
        "n": len(materialized),
        "n_valid": len(values),
        "null_count": len(materialized) - len(values),
        "failed_path_count": sum(not row.get("selected", False) for row in materialized),
        **statistics,
        "replicates": replicates,
    }


def _group_rows(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[Any, ...], list[dict]] = {}
    for row in rows:
        key = tuple(_dimension_value(row, name) for name in dimensions)
        grouped.setdefault(key, []).append(row)
    output = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        materialized = grouped[key]
        result = {"n_paths": len(materialized), "n_selected": sum(r.get("selected", False) for r in materialized)}
        result.update({name: value for name, value in zip(dimensions, key)})
        output.append((result, materialized))
    # Preserve a plain list after pairing; callers consume tuples immediately.
    return output


def _dimension_value(row: dict, name: str) -> Any:
    if name in row["configuration"]:
        return row["configuration"][name]
    return row.get(name)


def paired_metric_comparison(
    rows: list[dict], method_a: str, method_b: str, metric: str
) -> dict[str, Any]:
    """Pair values only at the exact same dataset configuration and seed."""
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row["method"] not in {method_a, method_b}:
            continue
        grouped.setdefault(row["dataset_key"], {})[row["method"]] = row
    pairs = []
    missing_method_datasets = []
    null_datasets = []
    for dataset_key in sorted(grouped):
        pair = grouped[dataset_key]
        if method_a not in pair or method_b not in pair:
            missing_method_datasets.append(dataset_key)
            continue
        value_a = pair[method_a].get("metrics", {}).get(metric)
        value_b = pair[method_b].get("metrics", {}).get(metric)
        if value_a is None or value_b is None:
            null_datasets.append(dataset_key)
            continue
        pairs.append(
            {
                "dataset_key": dataset_key,
                "seed": pair[method_a]["configuration"]["seed"],
                method_a: float(value_a),
                method_b: float(value_b),
                "delta_a_minus_b": float(value_a) - float(value_b),
            }
        )
    deltas = [item["delta_a_minus_b"] for item in pairs]
    if deltas:
        q25, median, q75 = np.percentile(deltas, [25, 50, 75])
        delta_stats = {"median": float(median), "q25": float(q25), "q75": float(q75)}
    else:
        delta_stats = {"median": None, "q25": None, "q75": None}
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_dataset_keys": len(grouped),
        "n_valid_pairs": len(pairs),
        "missing_method_dataset_count": len(missing_method_datasets),
        "missing_method_datasets": missing_method_datasets,
        "null_pair_dataset_count": len(null_datasets),
        "null_pair_datasets": null_datasets,
        "delta_a_minus_b": delta_stats,
        "pairs": pairs,
    }


def _build_group_definitions(rows: list[dict]) -> list[tuple[str, list[dict], tuple[str, ...], list[str]]]:
    primary_main = [
        row
        for row in rows
        if "primary" in row["memberships"]
        and row["method_family"] == "primary_or_resolution"
        and row["method"] in MAIN_METHODS
        and row["configuration"]["sigma"] == 0.000005
        and row["configuration"]["sensors"] == 20
    ]
    resolution = [
        row
        for row in rows
        if "resolution" in row["memberships"]
        and row["configuration"]["shape"] == "asymmetric"
        and row["method"] in {"calibrated", "free"}
    ]
    sensitivity_tau = 0.05
    confidence = [
        row
        for row in rows
        if "primary" in row["memberships"]
        and row["configuration"]["tau"] == sensitivity_tau
        and row["configuration"]["sigma"] == 0.000005
        and row["configuration"]["sensors"] == 20
        and row["method"] in set(SENSITIVITY_METHODS) | {"fixed", "free"}
    ]
    return [
        ("primary_quality", primary_main, ("shape", "tau", "method"), list(MAIN_METHODS)),
        ("resolution_maps", resolution, ("tau", "sensors", "sigma", "method"), ["calibrated", "free"]),
        ("confidence_sensitivity", confidence, ("shape", "method"), list(SENSITIVITY_METHODS + ("fixed", "free"))),
    ]


def build_statistical_summary(bundle: dict[str, Any], rows: list[dict]) -> dict[str, Any]:
    groups = []
    paired = []
    for group_name, selected, dimensions, methods in _build_group_definitions(rows):
        for summary, materialized in _group_rows(selected, dimensions):
            summary = dict(summary)
            summary["group"] = group_name
            summary["metrics"] = {metric: aggregate_metric(materialized, metric) for metric in METRICS}
            summary["computational_time_descriptive_s"] = aggregate_timing(materialized)
            groups.append(summary)

        pairing_dimensions = tuple(name for name in dimensions if name != "method")
        for stratum, materialized in _group_rows(selected, pairing_dimensions):
            available = [method for method in methods if any(r["method"] == method for r in materialized)]
            for method_a, method_b in combinations(available, 2):
                for metric in METRICS:
                    comparison = paired_metric_comparison(materialized, method_a, method_b, metric)
                    comparison.update(
                        {
                            "group": group_name,
                            "shape": stratum.get("shape"),
                            "tau": stratum.get("tau"),
                            "sensors": stratum.get("sensors"),
                            "sigma": stratum.get("sigma"),
                        }
                    )
                    paired.append(comparison)

    candidate_rows = [item for values in bundle["candidate_records"].values() for item in values]
    start_rows = [item for candidate in candidate_rows for item in candidate["starts"]]
    method_counts = Counter(row["method"] for row in rows)
    membership_counts = {
        membership: sum(membership in row["memberships"] for row in rows) for membership in ("primary", "resolution")
    }
    return {
        "schema_version": "load-information-summary-v3.0",
        "created_utc": utc_now(),
        "source_results": str(bundle["results_dir"]),
        "verification": {
            "passed": bundle["verification"]["passed"],
            "checked_candidates": bundle["verification"]["checked_candidates"],
            "checked_completed_paths": bundle["verification"]["checked_completed_paths"],
        },
        "counts": {
            "datasets": len(bundle["datasets"]),
            "paths": len(rows),
            "path_records_loaded": len(rows),
            "candidate_records_loaded": len(candidate_rows),
            "successful_candidates": sum(bool(item["successful_start"]) for item in candidate_rows),
            "optimizer_failed_candidates": sum(not item["successful_start"] for item in candidate_rows),
            "start_records_loaded": len(start_rows),
            "successful_starts": sum(bool(item["success"]) for item in start_rows),
            "failed_starts": sum(not item["success"] for item in start_rows),
            "selected_paths": sum(bool(row["selected"]) for row in rows),
            "optimizer_failed_paths": sum(row["status"] == "optimizer_failed" for row in rows),
            "method_paths": dict(sorted(method_counts.items())),
            "membership_paths": membership_counts,
            "overlap_datasets": sum(
                {"primary", "resolution"}.issubset(set(item["memberships"]))
                for item in bundle["datasets"]
            ),
        },
        "dataset_configurations": bundle["datasets"],
        "paths": rows,
        "groups": groups,
        "paired_comparisons": paired,
        "computational_time_note": "Times are descriptive accounting only; no speed or speedup claim is made.",
        "input_hashes": bundle["input_hashes"],
    }


def aggregate_timing(rows: list[dict]) -> dict[str, Any]:
    values = [float(row.get("computational_time_descriptive_s", 0.0)) for row in rows]
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        "n": len(values),
        "median_s": float(median),
        "q25_s": float(q25),
        "q75_s": float(q75),
    }


CSV_COLUMNS = (
    ["record_type", "group", "dataset_key", "path_key", "shape", "tau", "tau_percent",
     "sigma_microstrain", "sensors", "seed", "memberships", "method", "method_family",
     "status", "selected", "optimizer_failed", "no_feasible_candidate", "selection_reason",
     "selected_lambda", "selected_gamma_0", "selected_gamma_1", "selected_gamma_2",
     "calibrated_mean_0", "calibrated_mean_1", "calibrated_mean_2"]
    + [f"metric_{name}" for name in METRICS]
    + ["candidate_lambda", "candidate_status", "candidate_successful",
       "candidate_successful_starts", "candidate_failed_starts", "candidate_elapsed_s",
       "path_candidate_count", "path_successful_candidates", "path_optimizer_failed_candidates",
       "path_start_count", "path_successful_starts", "path_failed_starts",
       "metric_name", "n", "n_valid", "null_count", "median", "q25", "q75",
       "method_a", "method_b", "n_dataset_keys", "n_valid_pairs",
       "missing_method_dataset_count", "null_pair_dataset_count", "delta_median",
       "value_a", "value_b", "delta_a_minus_b",
       "dataset_hash", "path_record_sha256", "selected_arrays_sha256", "candidate_file_sha256"]
)


def _memberships(value: list[str]) -> str:
    return "|".join(value)


def _common_path_fields(row: dict) -> dict[str, Any]:
    config = row["configuration"]
    gamma = row.get("selected_gamma") or [None, None, None]
    mean = row.get("calibrated_mean") or [None, None, None]
    compatibility = row["compatibility"]
    result = {
        "dataset_key": row["dataset_key"],
        "path_key": row["path_key"],
        "shape": config["shape"],
        "tau": config["tau"],
        "tau_percent": config["tau_percent"],
        "sigma_microstrain": config["sigma_microstrain"],
        "sensors": config["sensors"],
        "seed": config["seed"],
        "memberships": _memberships(row["memberships"]),
        "method": row["method"],
        "method_family": row["method_family"],
        "status": row["status"],
        "selected": row["selected"],
        "optimizer_failed": compatibility["optimizer_failed"],
        "no_feasible_candidate": compatibility["no_feasible_candidate"],
        "selection_reason": compatibility["selection_reason"],
        "selected_lambda": row["selected_lambda"],
        "selected_gamma_0": gamma[0],
        "selected_gamma_1": gamma[1],
        "selected_gamma_2": gamma[2],
        "calibrated_mean_0": mean[0],
        "calibrated_mean_1": mean[1],
        "calibrated_mean_2": mean[2],
    }
    result.update({f"metric_{name}": row["metrics"][name] for name in METRICS})
    result.update(
        {
            "path_candidate_count": row["counts"]["candidates"],
            "path_successful_candidates": row["counts"]["successful_candidates"],
            "path_optimizer_failed_candidates": row["counts"]["optimizer_failed_candidates"],
            "path_start_count": row["counts"]["starts"],
            "path_successful_starts": row["counts"]["successful_starts"],
            "path_failed_starts": row["counts"]["failed_starts"],
        }
    )
    return result


def write_flat_csv(summary: dict[str, Any], path: Path) -> None:
    bundle_hashes = summary["input_hashes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in summary["paths"]:
            common = _common_path_fields(row)
            common.update(
                {
                    "record_type": "path",
                    "dataset_hash": bundle_hashes["datasets"][row["dataset_key"]]["npz"],
                    "path_record_sha256": bundle_hashes["paths"][row["path_key"]]["path.json"],
                    "selected_arrays_sha256": bundle_hashes["paths"][row["path_key"]].get(
                        "selected_arrays.npz"
                    ),
                }
            )
            writer.writerow(common)
            for candidate in row["candidate_details"]:
                penalty = float(candidate["penalty_lambda"])
                candidate_file = candidate["candidate_file"]
                candidate_hash = bundle_hashes["paths"][row["path_key"]]["candidates"][candidate_file]
                candidate_row = {
                    **common,
                    "record_type": "candidate",
                    "candidate_lambda": penalty,
                    "candidate_status": candidate["status"],
                    "candidate_successful": candidate["successful_start"],
                    "candidate_successful_starts": candidate["successful_starts"],
                    "candidate_failed_starts": candidate["failed_starts"],
                    "candidate_elapsed_s": candidate["elapsed_s"],
                    "candidate_file_sha256": candidate_hash,
                }
                candidate_row.update(
                    {
                        f"metric_{name}": None
                        for name in METRICS
                    }
                )
                writer.writerow(candidate_row)
        for group in summary["groups"]:
            for metric_name, stats in group["metrics"].items():
                writer.writerow(
                    {
                        "record_type": "group_summary",
                        "group": group["group"],
                        "shape": group.get("shape"),
                        "tau": group.get("tau"),
                        "sigma_microstrain": (
                            None if group.get("sigma") is None else group["sigma"] * 1.0e6
                        ),
                        "sensors": group.get("sensors"),
                        "method": group.get("method"),
                        "metric_name": metric_name,
                        "n": stats["n"],
                        "n_valid": stats["n_valid"],
                        "null_count": stats["null_count"],
                        "median": stats["median"],
                        "q25": stats["q25"],
                        "q75": stats["q75"],
                    }
                )
        for comparison in summary["paired_comparisons"]:
            common_comparison = {
                "record_type": "paired_comparison",
                "group": comparison["group"],
                "shape": comparison.get("shape"),
                "tau": comparison.get("tau"),
                "sensors": comparison.get("sensors"),
                "sigma_microstrain": comparison.get("sigma"),
                "metric_name": comparison["metric"],
                "method_a": comparison["method_a"],
                "method_b": comparison["method_b"],
                "n_dataset_keys": comparison["n_dataset_keys"],
                "n_valid_pairs": comparison["n_valid_pairs"],
                "missing_method_dataset_count": comparison["missing_method_dataset_count"],
                "null_pair_dataset_count": comparison["null_pair_dataset_count"],
                "delta_median": comparison["delta_a_minus_b"]["median"],
            }
            writer.writerow(
                common_comparison
            )
            for pair in comparison["pairs"]:
                writer.writerow(
                    {
                        **common_comparison,
                        "record_type": "paired_value",
                        "dataset_key": pair["dataset_key"],
                        "seed": pair["seed"],
                        "value_a": pair[comparison["method_a"]],
                        "value_b": pair[comparison["method_b"]],
                        "delta_a_minus_b": pair["delta_a_minus_b"],
                    }
                )


def _figure_text_sizes(fig: plt.Figure) -> list[float]:
    artists = list(fig.texts)
    for legend in fig.legends:
        artists.extend(legend.get_texts())
    for axis in fig.axes:
        artists.extend(axis.texts)
        artists.extend((axis.title, axis.xaxis.label, axis.yaxis.label))
        artists.extend(axis.get_xticklabels())
        artists.extend(axis.get_yticklabels())
        legend = axis.get_legend()
        if legend is not None:
            artists.extend(legend.get_texts())
    sizes = []
    for artist in artists:
        if getattr(artist, "get_text", lambda: "")():
            try:
                sizes.append(float(artist.get_fontsize()))
            except AttributeError:
                pass
    return sorted(set(sizes))


def inspect_figure(fig: plt.Figure) -> dict[str, Any]:
    return {
        "width_inches": float(fig.get_size_inches()[0]),
        "height_inches": float(fig.get_size_inches()[1]),
        "axes_count": len(fig.axes),
        "font_size_setting_pt": FONT_SIZE_PT,
        "text_sizes_pt": _figure_text_sizes(fig),
        "pdf_fonttype": int(matplotlib.rcParams["pdf.fonttype"]),
        "mathtext_fontset": matplotlib.rcParams["mathtext.fontset"],
        "serif_family": matplotlib.rcParams["font.serif"],
    }


def _pdf_metadata(path: Path) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    page = reader.pages[0]
    fonts = []
    try:
        font_objects = page["/Resources"].get("/Font", {})
        for font in font_objects.values():
            obj = font.get_object()
            fonts.append(str(obj.get("/BaseFont", "")))
    except Exception:
        fonts = []
    return {
        "media_box_width_pt": float(page.mediabox.width),
        "media_box_height_pt": float(page.mediabox.height),
        "fonts": fonts,
    }


def save_and_audit_figure(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, Any]:
    for path in (out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"):
        if path.exists():
            raise SummaryError(f"refusing to overwrite existing output: {path}")
    fig.canvas.draw()
    warnings = audit_figure(fig, stem, clearance_px=3.0)
    inspection = inspect_figure(fig)
    if inspection["text_sizes_pt"] and set(inspection["text_sizes_pt"]) != {FONT_SIZE_PT}:
        warnings.append(
            f"{stem}: non-10pt text found {inspection['text_sizes_pt']}"
        )
    if not math.isclose(inspection["width_inches"], FIGURE_WIDTH_INCHES, abs_tol=1e-12):
        warnings.append(
            f"{stem}: width is {inspection['width_inches']:.9g} in, expected 6.5 in"
        )
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(png_path, format="png", dpi=PNG_DPI, bbox_inches=None, pad_inches=0)
    pdf_data = _pdf_metadata(pdf_path)
    if not math.isclose(pdf_data["media_box_width_pt"], FIGURE_WIDTH_PT, abs_tol=0.1):
        warnings.append(
            f"{stem}: PDF width is {pdf_data['media_box_width_pt']:.3f} pt, expected 468 pt"
        )
    plt.close(fig)
    return {
        "pdf": str(pdf_path),
        "png": str(png_path),
        "pdf_sha256": hash_file(pdf_path),
        "png_sha256": hash_file(png_path),
        "layout_warnings": warnings,
        "inspection": {**inspection, **pdf_data},
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.tick_params(labelsize=FONT_SIZE_PT)
    axis.grid(True, alpha=0.22, linewidth=0.5)
    axis.set_axisbelow(True)


def _tau_axis(axis: plt.Axes) -> None:
    axis.set_xticks([0, 1, 2])
    axis.set_xticklabels([str(int(100 * value)) for value in TAU_LEVELS])
    axis.set_xlim(-0.35, 2.35)
    axis.set_xlabel(r"Load measurement uncertainty $\tau$ (%)")


def _method_handles(methods: Iterable[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=5,
            linewidth=1.2,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]


def _rows(
    rows: list[dict],
    *,
    shape: str,
    method: str,
    tau: float,
    membership: str = "primary",
    method_family: str = "primary_or_resolution",
) -> list[dict]:
    return [
        row
        for row in rows
        if membership in row["memberships"]
        and row["method_family"] == method_family
        and row["configuration"]["shape"] == shape
        and row["method"] == method
        and math.isclose(row["configuration"]["tau"], tau, abs_tol=1e-15)
        and row["configuration"]["sigma"] == FIELDS_SIGMA
        and row["configuration"]["sensors"] == FIELDS_SENSORS
    ]


def _plot_median_iqr(
    axis: plt.Axes,
    rows: list[dict],
    method: str,
    metric: str,
    scale: float = 1.0,
) -> None:
    x = np.arange(len(TAU_LEVELS))
    medians = []
    lowers = []
    uppers = []
    for tau in TAU_LEVELS:
        stats = aggregate_metric(_rows(rows, shape=rows[0]["configuration"]["shape"], method=method, tau=tau), metric)
        medians.append(None if stats["median"] is None else stats["median"] * scale)
        lowers.append(None if stats["q25"] is None else stats["q25"] * scale)
        uppers.append(None if stats["q75"] is None else stats["q75"] * scale)
    finite = [index for index, value in enumerate(medians) if value is not None]
    if finite:
        x_valid = np.asarray([x[index] for index in finite], dtype=float)
        lower = np.asarray([lowers[index] for index in finite], dtype=float)
        upper = np.asarray([uppers[index] for index in finite], dtype=float)
        median = np.asarray([medians[index] for index in finite], dtype=float)
        axis.fill_between(
            x_valid,
            lower,
            upper,
            color=METHOD_COLORS[method],
            alpha=0.17,
            linewidth=0,
        )
        axis.plot(
            x_valid,
            median,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=4.5,
            linewidth=1.2,
        )


def make_load_quality_figure(rows: list[dict], out_dir: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_INCHES, 5.5))
    panels = (
        (axes[0, 0], "intact", "loss_area_m", "(a) Intact"),
        (axes[0, 1], "asymmetric", "support_rel_L2_except_intact", "(b) Asymmetric"),
        (axes[1, 0], "double", "support_rel_L2_except_intact", "(c) Double"),
        (axes[1, 1], "broad_shallow", "support_rel_L2_except_intact", "(d) Broad shallow"),
    )
    primary = [
        row
        for row in rows
        if "primary" in row["memberships"]
        and row["method_family"] == "primary_or_resolution"
        and row["method"] in MAIN_METHODS
        and row["configuration"]["sigma"] == FIELDS_SIGMA
        and row["configuration"]["sensors"] == FIELDS_SENSORS
    ]
    for axis, shape, metric, title in panels:
        shape_rows = [row for row in primary if row["configuration"]["shape"] == shape]
        for method in MAIN_METHODS:
            method_rows = [row for row in shape_rows if row["method"] == method]
            if not method_rows:
                raise SummaryError(f"load-quality series is absent: {shape}/{method}")
            _plot_median_iqr(axis, method_rows, method, metric, scale=100.0 if shape != "intact" else 1.0)
        # Loss area and relative errors are non-negative; pin the lower
        # limit to zero and keep only ticks inside the view interval so no
        # clipped stray tick labels crowd neighbouring panels.
        _, ymax = axis.get_ylim()
        axis.set_ylim(0.0, ymax)
        yticks = MaxNLocator(nbins=6, min_n_ticks=3).tick_values(0.0, ymax)
        axis.set_yticks(yticks[(yticks >= 0.0) & (yticks <= ymax)])
        _tau_axis(axis)
        axis.set_ylabel("Spurious loss area (m)" if shape == "intact" else "Relative support error (%)")
        axis.set_title(title, pad=7)
        _style_axis(axis)
    handles = _method_handles(MAIN_METHODS)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.115, right=0.98, top=0.91, bottom=0.145, wspace=0.30, hspace=0.44)
    return save_and_audit_figure(fig, out_dir, "load_quality")


def make_prediction_quality_figure(rows: list[dict], out_dir: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_INCHES, 3.55))
    for axis, shape, title in (
        (axes[0], "asymmetric", "(a) Asymmetric"),
        (axes[1], "broad_shallow", "(b) Broad shallow"),
    ):
        shape_rows = [
            row
            for row in rows
            if "primary" in row["memberships"]
            and row["method_family"] == "primary_or_resolution"
            and row["configuration"]["shape"] == shape
            and row["configuration"]["sigma"] == FIELDS_SIGMA
            and row["configuration"]["sensors"] == FIELDS_SENSORS
        ]
        for method in MAIN_METHODS:
            method_rows = [row for row in shape_rows if row["method"] == method]
            if not method_rows:
                raise SummaryError(f"prediction-quality series is absent: {shape}/{method}")
            _plot_median_iqr(
                axis,
                method_rows,
                method,
                "unused_load_strain_relative_L2",
                scale=100.0,
            )
        _tau_axis(axis)
        axis.set_ylabel("Unused-load strain error (%)")
        axis.set_title(title, pad=7)
        _style_axis(axis)
    fig.legend(
        handles=_method_handles(MAIN_METHODS),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.27, wspace=0.30)
    return save_and_audit_figure(fig, out_dir, "prediction_quality")


def make_fields_prediction_figure(bundle: dict[str, Any], rows: list[dict], out_dir: Path) -> dict[str, Any]:
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 6.25))
    axes = {
        ("asymmetric", "support"): fig.add_axes([0.11, 0.62, 0.33, 0.24]),
        ("asymmetric", "response"): fig.add_axes([0.63, 0.62, 0.33, 0.24]),
        ("double", "support"): fig.add_axes([0.11, 0.16, 0.33, 0.24]),
        ("double", "response"): fig.add_axes([0.63, 0.16, 0.33, 0.24]),
    }
    for shape in ("asymmetric", "double"):
        reference_plotted = False
        for method_index, method in enumerate(MAIN_METHODS):
            selected = [
                row
                for row in _rows(rows, shape=shape, method=method, tau=0.10)
                if row["configuration"]["seed"] == FIELDS_SEED
            ]
            if len(selected) != 1 or not selected[0]["selected"]:
                raise SummaryError(
                    f"predesignated field path is missing or failed: {shape}/{method}/seed1101"
                )
            arrays = bundle["field_arrays"][(selected[0]["dataset_key"], method)]
            support_axis = axes[(shape, "support")]
            response_axis = axes[(shape, "response")]
            if not reference_plotted:
                # All methods share the same reference field on this dataset;
                # draw it once per panel instead of once per method so the
                # dashed black curve is not buried under repeated overplotting.
                support_axis.plot(
                    arrays["score_x"],
                    arrays["reference_d"],
                    color="black",
                    linestyle="--",
                    linewidth=1.3,
                )
                response_axis.plot(
                    arrays["unused_x"],
                    arrays["reference_strain"] * 1e6,
                    color="black",
                    linestyle="--",
                    linewidth=1.3,
                )
                reference_plotted = True
            # Lines keep every data point; only markers are thinned to about
            # ten hollow markers per curve, offset per method so that close
            # curves stay distinguishable and the reference remains visible.
            support_step = max(1, len(arrays["score_x"]) // 10)
            response_step = max(1, len(arrays["unused_x"]) // 10)
            support_offset = method_index * (support_step // len(MAIN_METHODS))
            response_offset = method_index * (response_step // len(MAIN_METHODS))
            support_axis.plot(
                arrays["score_x"],
                arrays["estimated_d"],
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=1.1,
                marker=METHOD_MARKERS[method],
                markersize=3.1,
                markevery=(support_offset, support_step),
                markerfacecolor="none",
                markeredgewidth=0.7,
            )
            response_axis.plot(
                arrays["unused_x"],
                arrays["predicted_strain"] * 1e6,
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=1.1,
                marker=METHOD_MARKERS[method],
                markersize=3.1,
                markevery=(response_offset, response_step),
                markerfacecolor="none",
                markeredgewidth=0.7,
            )
    titles = {
        ("asymmetric", "support"): "(a) Asymmetric support",
        ("asymmetric", "response"): "(b) Asymmetric unused response",
        ("double", "support"): "(c) Double support",
        ("double", "response"): "(d) Double unused response",
    }
    for key, axis in axes.items():
        axis.set_xticks((-10, 0, 10))
        axis.set_xlabel("Position (m)")
        axis.set_ylabel("Support loss $d$" if key[1] == "support" else r"Strain ($\mu\varepsilon$)")
        axis.set_title(titles[key], pad=7)
        _style_axis(axis)
    handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.3, label="Reference")
    ] + _method_handles(MAIN_METHODS)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        frameon=False,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    return save_and_audit_figure(fig, out_dir, "fields_prediction")


def make_measurement_map_figure(rows: list[dict], out_dir: Path) -> dict[str, Any]:
    sensor_levels = (10, 20, 40)
    sigma_levels = (1.0, 5.0, 20.0)
    selected_rows = [
        row
        for row in rows
        if "resolution" in row["memberships"]
        and row["configuration"]["shape"] == "asymmetric"
        and row["method"] in {"calibrated", "free"}
    ]
    maps = {}
    for tau in (0.01, 0.10):
        for method in ("calibrated", "free"):
            matrix = np.full((3, 3), np.nan)
            valid = np.zeros((3, 3), dtype=int)
            total = np.zeros((3, 3), dtype=int)
            for row in selected_rows:
                if row["method"] != method or not math.isclose(row["configuration"]["tau"], tau, abs_tol=1e-15):
                    continue
                iy = sigma_levels.index(row["configuration"]["sigma_microstrain"])
                ix = sensor_levels.index(row["configuration"]["sensors"])
                value = row["metrics"]["support_rel_L2_except_intact"]
                total[iy, ix] += 1
                if value is not None:
                    valid[iy, ix] += 1
                    matrix[iy, ix] = float(value)
            # There is one value per seed here; median handles future additions.
            for iy, ix in np.ndindex(3, 3):
                if valid[iy, ix]:
                    matrix[iy, ix] = np.median(
                        [
                            row["metrics"]["support_rel_L2_except_intact"]
                            for row in selected_rows
                            if row["method"] == method
                            and math.isclose(row["configuration"]["tau"], tau, abs_tol=1e-15)
                            and row["configuration"]["sigma_microstrain"] == sigma_levels[iy]
                            and row["configuration"]["sensors"] == sensor_levels[ix]
                            and row["metrics"]["support_rel_L2_except_intact"] is not None
                        ]
                    )
            maps[(tau, method)] = (matrix, valid, total)
    finite_blocks = [values[np.isfinite(values)] for values, _, _ in maps.values()]
    if not any(block.size for block in finite_blocks):
        raise SummaryError("resolution maps contain no valid support errors")
    finite_values = np.concatenate(finite_blocks)
    # Cells, color scale, and colorbar ticks all use percent units so the
    # colorbar agrees with the numeric annotations (summary values are kept
    # as fractions and are not affected).
    vmin, vmax = 100.0 * float(np.min(finite_values)), 100.0 * float(np.max(finite_values))

    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 5.8))
    axes = {}
    positions = {
        (0.01, "calibrated"): [0.10, 0.67, 0.34, 0.20],
        (0.01, "free"): [0.60, 0.67, 0.34, 0.20],
        (0.10, "calibrated"): [0.10, 0.33, 0.34, 0.20],
        (0.10, "free"): [0.60, 0.33, 0.34, 0.20],
    }
    image = None
    for (tau, method), position in positions.items():
        axis = fig.add_axes(position)
        values, valid, total = maps[(tau, method)]
        image = axis.imshow(
            100.0 * values,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="Greys",
        )
        axis.set_xticks(range(3), labels=[str(value) for value in sensor_levels])
        axis.set_yticks(range(3), labels=[str(value) for value in sigma_levels])
        if tau == 0.10:
            axis.set_xlabel("Sensors per load")
        if method == "calibrated":
            axis.set_ylabel(r"$\sigma$ ($\mu\varepsilon$)")
        panel = {
            (0.01, "calibrated"): "a",
            (0.01, "free"): "b",
            (0.10, "calibrated"): "c",
            (0.10, "free"): "d",
        }[(tau, method)]
        axis.set_title(
            f"({panel}) $\\tau$={100 * tau:g}% {METHOD_LABELS[method].lower()}",
            pad=7,
        )
        axis.tick_params(labelsize=FONT_SIZE_PT)
        for iy, ix in np.ndindex(3, 3):
            value = values[iy, ix]
            displayed = 100.0 * value
            normalized = (
                0.5
                if not np.isfinite(value)
                else (displayed - vmin) / (vmax - vmin if vmax > vmin else 1)
            )
            text = "--" if not np.isfinite(value) else f"{100 * value:.1f}"
            axis.text(
                ix,
                iy,
                text,
                ha="center",
                va="center",
                fontsize=FONT_SIZE_PT,
                color="white" if normalized > 0.55 else "black",
            )
        axes[(tau, method)] = axis
    color_axis = fig.add_axes([0.10, 0.14, 0.84, 0.035])
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Median relative support error (%)")
    # Round percent ticks strictly inside the displayed range.
    ticks = MaxNLocator(nbins=6, min_n_ticks=3).tick_values(vmin, vmax)
    colorbar.set_ticks(ticks[(ticks >= vmin) & (ticks <= vmax)])
    colorbar.ax.tick_params(labelsize=FONT_SIZE_PT)
    colorbar.ax.xaxis.label.set_size(FONT_SIZE_PT)
    return save_and_audit_figure(fig, out_dir, "measurement_map")


def make_confidence_figure(rows: list[dict], out_dir: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_INCHES, 3.75))
    factors = (0.2, 1.0, 3.0)
    for axis, shape, title in (
        (axes[0], "asymmetric", "(a) Asymmetric"),
        (axes[1], "broad_shallow", "(b) Broad shallow"),
    ):
        shape_rows = [
            row
            for row in rows
            if "primary" in row["memberships"]
            and row["configuration"]["shape"] == shape
            and math.isclose(row["configuration"]["tau"], 0.05, abs_tol=1e-15)
            and row["configuration"]["sigma"] == FIELDS_SIGMA
            and row["configuration"]["sensors"] == FIELDS_SENSORS
        ]
        for method in SENSITIVITY_METHODS:
            if method == "calibrated":
                # The factor-1 calibrated treatment is the reused primary
                # path; there is no separate confidence-sensitivity run.
                method_rows = [
                    row
                    for row in shape_rows
                    if row["method"] == "calibrated"
                    and row["method_family"] == "primary_or_resolution"
                ]
            else:
                method_rows = [
                    row
                    for row in shape_rows
                    if row["method"] == method
                    and row["method_family"] == "confidence_sensitivity"
                ]
            if not method_rows:
                raise SummaryError(f"confidence series is absent: {shape}/{method}")
            stats = aggregate_metric(method_rows, "support_rel_L2_except_intact")
            if stats["median"] is not None:
                # Each treatment has exactly one assigned uncertainty factor;
                # draw it as a discrete median with its IQR at that factor
                # only, never as a horizontal curve across all factors.
                factor = SENSITIVITY_FACTORS[method]
                median = 100.0 * stats["median"]
                lower = 100.0 * stats["q25"]
                upper = 100.0 * stats["q75"]
                axis.errorbar(
                    [factor],
                    [median],
                    yerr=[[median - lower], [upper - median]],
                    color=METHOD_COLORS[method],
                    linestyle="none",
                    marker=METHOD_MARKERS[method],
                    markersize=4.8,
                    linewidth=1.2,
                    capsize=3.0,
                )
        for method, linestyle in (("fixed", "--"), ("free", ":")):
            method_rows = [
                row
                for row in shape_rows
                if row["method"] == method and row["method_family"] == "primary_or_resolution"
            ]
            stats = aggregate_metric(method_rows, "support_rel_L2_except_intact")
            if stats["median"] is not None:
                axis.axhline(100.0 * stats["median"], color="black", linestyle=linestyle, linewidth=1.2)
                axis.axhspan(100.0 * stats["q25"], 100.0 * stats["q75"], color="black", alpha=0.08, linewidth=0)
        axis.set_xticks(factors, labels=["0.2", "1", "3"])
        axis.set_xlim(0.0, 3.25)
        axis.set_xlabel(r"Assigned uncertainty factor at $\tau$=5%")
        axis.set_ylabel("Relative field error (%)")
        axis.set_title(title, pad=7)
        _style_axis(axis)
    handles = _method_handles(SENSITIVITY_METHODS) + [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Fixed"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.2, label="Free"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=5, frameon=False)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.85, bottom=0.28, wspace=0.30)
    return save_and_audit_figure(fig, out_dir, "confidence")


def _fields_caption_data(rows: list[dict]) -> list[dict[str, Any]]:
    output = []
    for shape in ("asymmetric", "double"):
        for method in MAIN_METHODS:
            selected = [
                row
                for row in _rows(rows, shape=shape, method=method, tau=0.10)
                if row["configuration"]["seed"] == FIELDS_SEED
            ]
            if len(selected) != 1:
                raise SummaryError(
                    f"predesignated field path is absent: {shape}/{method}/seed{FIELDS_SEED}"
                )
            row = selected[0]
            output.append(
                {
                    "shape": shape,
                    "method": method,
                    "dataset_key": row["dataset_key"],
                    "seed": FIELDS_SEED,
                    "tau": 0.10,
                    "sigma_microstrain": FIELDS_SIGMA * 1.0e6,
                    "sensors": FIELDS_SENSORS,
                    "selected_lambda": row["selected_lambda"],
                    "selected_gamma": row["selected_gamma"],
                    "calibrated_mean": row["calibrated_mean"],
                    "selected": row["selected"],
                }
            )
    return output


def _information_curves() -> dict[str, list[dict]]:
    physical = PhysicalModel()
    sensors = np.linspace(-12.0, 12.0, 20)
    forward = HermiteBeamFE(
        [-4.0, 0.0, 4.0],
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
    coefficients = np.zeros(forward.basis.n_variables)
    strains = forward.strain_and_jacobian(coefficients, [sensors] * 3)[0].reshape(3, -1)
    tau = np.linspace(0.0, 0.12, 241)
    curves = []
    for sigma_microstrain in (1.0, 5.0, 20.0):
        sigma = sigma_microstrain * 1.0e-6
        for load_index, center in enumerate((-4.0, 0.0, 4.0)):
            response_norm2 = float(np.sum((strains[load_index] / sigma) ** 2))
            chi = 1.0 / (1.0 + tau**2 * response_norm2)
            curves.append(
                {
                    "sigma_microstrain": sigma_microstrain,
                    "load_center_m": center,
                    "tau": tau.tolist(),
                    "chi": chi.tolist(),
                    "response_norm2_whitened": response_norm2,
                }
            )
    return {
        "definition": "chi_j=1/(1+tau_j^2 ||f_j/sigma||_2^2)",
        "actual_tau_levels": list(TAU_LEVELS),
        "curves": curves,
    }


def make_information_figure(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    information = _information_curves()
    fig, axis = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, 3.55))
    colors = {1.0: "#000000", 5.0: "#0072B2", 20.0: "#009E73"}
    markers = {-4.0: "o", 0.0: "s", 4.0: "^"}
    handles = []
    for curve in information["curves"]:
        sigma = curve["sigma_microstrain"]
        center = curve["load_center_m"]
        tau = np.asarray(curve["tau"])
        chi = np.asarray(curve["chi"])
        color = colors[sigma]
        marker = markers[center]
        axis.plot(
            100.0 * tau,
            chi,
            color=color,
            linewidth=1.1,
            linestyle="--" if center == 0.0 else "-",
            marker=marker,
            markevery=30,
            markersize=3.5,
            markerfacecolor="none",
            markeredgewidth=0.7,
            label=f"$\\sigma$={sigma:g} $\\mu\\varepsilon$, x={center:+g} m",
        )
        handles.append(
            Line2D(
                [0], [0], color=color, linestyle="--" if center == 0.0 else "-",
                marker=marker, markersize=4, markerfacecolor="none", linewidth=1.2,
                label=f"$\\sigma$={sigma:g}, x={center:+g}",
            )
        )
    for tau in TAU_LEVELS:
        axis.axvline(100.0 * tau, color="#777777", linestyle=":", linewidth=0.8)
    axis.set_xlim(0.0, 12.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel(r"Load standard uncertainty $\tau$ (%)")
    axis.set_ylabel(r"Retained fraction $\chi_j$")
    axis.set_title("Analytic response-parallel information at intact support", pad=7)
    _style_axis(axis)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.85, bottom=0.35)
    report = save_and_audit_figure(fig, out_dir, "information")
    return report, information


def make_model_figure(out_dir: Path) -> dict[str, Any]:
    """Schematic model diagram: mechanical setup, load states, and process.

    All geometry is schematic and not to scale; the canvas uses arbitrary
    units while the annotations state the physical values of the campaign
    (inspected span [-12, 12] m, gauge length 0.20 m, footprint 4.3 m,
    nominal load 750 kN, calibration load centres -4/0/+4 m, unused
    prediction centre +6 m).  No measured support profile is shown: the
    reduced-stiffness region is only marked as bounded.
    """
    blue = "#0072B2"
    green = "#009E73"
    orange = "#D55E00"
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 5.9))
    mechanical = fig.add_axes([0.025, 0.560, 0.95, 0.395])
    states = fig.add_axes([0.025, 0.280, 0.95, 0.220])
    process = fig.add_axes([0.025, 0.020, 0.95, 0.195])
    for axis in (mechanical, states, process):
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.0, 100.0)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    # (a) Mechanical configuration: beam, clamps, spring bed, gauge inset.
    mechanical.set_title("(a) Mechanical configuration (schematic, not to scale)", pad=6)
    mechanical.add_patch(
        FancyBboxPatch(
            (2.0, 60.0), 34.0, 38.0,
            boxstyle="round,pad=0,rounding_size=1.5",
            facecolor="#FAFAFA", edgecolor="#888888", linewidth=0.8,
        )
    )
    mechanical.text(19.0, 93.5, "finite-gauge sensor", ha="center", va="center")
    mechanical.plot([5.0, 33.0], [82.0, 82.0], color="black", linewidth=1.6)
    mechanical.add_patch(
        Rectangle(
            (15.75, 79.6), 6.5, 4.8,
            facecolor=blue, alpha=0.25, edgecolor=blue, linewidth=1.1,
        )
    )
    mechanical.plot([15.75, 22.25], [74.5, 74.5], color="black", linewidth=0.8)
    for x_tick in (15.75, 22.25):
        mechanical.plot([x_tick, x_tick], [73.6, 75.4], color="black", linewidth=0.8)
    mechanical.text(19.0, 69.2, "gauge length $g=0.20$ m", ha="center", va="center")
    for x0, x1 in ((4.0, 10.0), (18.0, 82.0), (90.0, 96.0)):
        mechanical.plot([x0, x1], [40.0, 40.0], color="black", linewidth=2.0)
    mechanical.plot(
        [12.0, 14.0, 16.0, 84.0, 86.0, 88.0], [40.0] * 6,
        linestyle="none", marker="o", markersize=1.8, color="black",
    )
    for x_wall, direction in ((4.0, -1.0), (96.0, 1.0)):
        mechanical.plot([x_wall, x_wall], [33.0, 47.0], color="black", linewidth=2.5)
        for y_hatch in (35.5, 39.0, 42.5, 46.0):
            mechanical.plot(
                [x_wall, x_wall + direction * 2.4],
                [y_hatch, y_hatch - 2.4],
                color="black", linewidth=0.9,
            )
    mechanical.add_patch(
        Rectangle(
            (44.0, 30.0), 12.0, 10.0,
            facecolor=green, alpha=0.15, edgecolor=green, linewidth=0.8,
            linestyle=(0, (4, 2)), zorder=0.5,
        )
    )
    mechanical.plot([18.0, 82.0], [30.0, 30.0], color="black", linewidth=1.2)
    for x_spring in np.arange(20.0, 80.5, 4.0):
        weakened = 44.0 <= x_spring <= 56.0
        mechanical.plot(
            [x_spring, x_spring - 0.8, x_spring + 0.8, x_spring - 0.8, x_spring + 0.8, x_spring],
            [40.0, 38.5, 36.5, 34.5, 32.5, 30.0],
            color=green if weakened else "black",
            linewidth=1.0,
        )
    # The inspected-span dimension covers the entire displayed beam between
    # the omitted-domain ellipses (x = 18 to x = 82); the bounded
    # reduced-stiffness region stays a separate dashed-band annotation.
    mechanical.plot([18.0, 82.0], [23.5, 23.5], color="black", linewidth=0.9)
    for x_tick in (18.0, 82.0):
        mechanical.plot([x_tick, x_tick], [22.4, 24.6], color="black", linewidth=0.9)
    mechanical.text(
        50.0, 17.2, "inspected span $[-12\\,\\mathrm{m},\\,12\\,\\mathrm{m}]$",
        ha="center", va="center",
    )
    mechanical.text(
        50.0, 7.2, "bounded region of reduced stiffness (schematic)",
        ha="center", va="center",
    )
    mechanical.text(40.0, 47.2, "beam on elastic spring bed", ha="center", va="center")
    mechanical.text(10.5, 52.8, "clamped remote end", ha="center", va="center")
    mechanical.text(89.5, 52.8, "clamped remote end", ha="center", va="center")

    # (b) Separate load states: one footprint load per state, drawn in its own
    # column so the three calibration loads cannot read as simultaneous.
    states.set_title("(b) Separate load states (schematic, not to scale)", pad=6)
    state_columns = (
        (13.5, -3.4, "state 1:\ncentre $-4$ m", blue, "-", 0.30),
        (37.5, 0.0, "state 2:\ncentre $0$ m", blue, "-", 0.30),
        (61.5, 3.4, "state 3:\ncentre $+4$ m", blue, "-", 0.30),
        (85.5, 5.1, "prediction (unused):\ncentre $+6$ m", orange, (0, (4, 2)), 0.22),
    )
    for x_column, offset, label, color, dash, alpha in state_columns:
        x_load = x_column + offset
        states.text(x_column, 85.0, label, ha="center", va="center")
        states.text(x_load, 62.0, "750 kN", ha="center", va="center")
        states.plot(
            [x_load, x_load], [54.0, 47.5],
            color=color, linewidth=1.3, linestyle=dash,
        )
        states.plot(
            [x_load], [46.0], linestyle="none",
            marker="v", markersize=8, color=color,
        )
        states.add_patch(
            Rectangle(
                (x_load - 2.2, 40.0), 4.4, 6.0,
                facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.1,
                linestyle=dash,
            )
        )
        states.plot(
            [x_column - 10.0, x_column + 10.0], [40.0, 40.0],
            color="black", linewidth=1.6,
        )
        states.text(x_column, 26.0, "$b=4.3$ m footprint", ha="center", va="center")

    # (c) Load-information process; arrows terminate on box boundaries.
    process.set_title("(c) Load-information process", pad=6)
    steps = (
        "whitened strains $\\tilde y_j$\n+ independent load\nobservations",
        "equilibrium and\nbounded amplitude\nupdate",
        "bounded\nsupport-field\niteration",
        "unused-load\nprediction",
    )
    for index, text in enumerate(steps):
        x_box = 1.0 + 25.0 * index
        process.add_patch(
            FancyBboxPatch(
                (x_box, 24.0), 22.5, 52.0,
                boxstyle="round,pad=0,rounding_size=2",
                facecolor="white", edgecolor="black", linewidth=1.0,
            )
        )
        process.text(x_box + 11.25, 50.0, text, ha="center", va="center")
        if index < 3:
            process.plot(
                [x_box + 22.5, x_box + 24.2], [50.0, 50.0],
                color="black", linewidth=1.3,
            )
            process.plot(
                [x_box + 25.0], [50.0], linestyle="none",
                marker=">", markersize=8, color="black",
            )
    return save_and_audit_figure(fig, out_dir, "model")


def _resolve_relative(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise SummaryError(f"path escapes the project root: {value}") from error
    return path.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--summary-dir", required=True, type=Path)
    return parser


def generate_all(results_arg: Path, out_dir_arg: Path, summary_dir_arg: Path) -> dict[str, Any]:
    results_dir = _resolve_relative(ROOT, results_arg)
    out_dir = _resolve_relative(ROOT, out_dir_arg)
    summary_dir = _resolve_relative(ROOT, summary_dir_arg)
    if out_dir == ROOT / "manuscript" / "figures":
        raise SummaryError("refusing to write into the existing v2 figure directory")
    targets = [
        item
        for stem in (
            "model",
            "load_quality",
            "prediction_quality",
            "fields_prediction",
            "measurement_map",
            "confidence",
            "information",
        )
        for item in (out_dir / f"{stem}.pdf", out_dir / f"{stem}.png")
    ] + [summary_dir / "load_information_summary.json", summary_dir / "load_information_summary.csv"]
    targets.append(out_dir / "layout_audit.json")
    existing = [str(item) for item in targets if item.exists()]
    if existing:
        raise SummaryError("refusing to overwrite prior figure/summary output: " + "; ".join(existing))

    bundle = load_verified_results(results_dir)
    rows = path_summaries(bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "model": make_model_figure(out_dir),
        "load_quality": make_load_quality_figure(rows, out_dir),
        "prediction_quality": make_prediction_quality_figure(rows, out_dir),
        "fields_prediction": make_fields_prediction_figure(bundle, rows, out_dir),
        "measurement_map": make_measurement_map_figure(rows, out_dir),
        "confidence": make_confidence_figure(rows, out_dir),
    }
    information_report, information = make_information_figure(out_dir)
    figures["information"] = information_report

    summary = build_statistical_summary(bundle, rows)
    summary["figures"] = figures
    summary["fields_prediction_caption_data"] = _fields_caption_data(rows)
    summary["information_analysis"] = information
    layout_report = {
        "schema_version": "load-information-layout-audit-v3.0",
        "created_utc": utc_now(),
        "figures": {name: {"warnings": report["layout_warnings"], "inspection": report["inspection"]} for name, report in figures.items()},
        "policy": {
            "width_inches": FIGURE_WIDTH_INCHES,
            "font_size_pt": FONT_SIZE_PT,
            "tight_bbox_used": False,
            "note": "Warnings are retained for root visual review; qa-ignore is not applied.",
        },
    }
    write_new_json(out_dir / "layout_audit.json", layout_report)
    write_new_json(summary_dir / "load_information_summary.json", summary)
    csv_path = summary_dir / "load_information_summary.csv"
    if csv_path.exists():
        raise SummaryError(f"refusing to overwrite existing output: {csv_path}")
    write_flat_csv(summary, csv_path)
    return {
        "status": "completed_with_layout_warnings" if any(report["layout_warnings"] for report in figures.values()) else "completed",
        "figures": {name: {"pdf": report["pdf"], "png": report["png"], "warnings": report["layout_warnings"]} for name, report in figures.items()},
        "summary_json": str(summary_dir / "load_information_summary.json"),
        "summary_csv": str(csv_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = generate_all(args.results, args.out_dir, args.summary_dir)
    except (SummaryError, runner.ProtocolError, OSError, ValueError) as error:
        print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}, allow_nan=False))
        return 2
    print(json.dumps(runner.safe_json(result), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
