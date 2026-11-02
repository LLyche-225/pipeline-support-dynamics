#!/usr/bin/env python3
"""Export frozen v3 inputs needed by the dynamic-transfer calculations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "revision_v3_load_information_20261102"
SHAPES = ("intact", "asymmetric", "double", "broad_shallow")
TAUS = (0.01, 0.10)
SEEDS = tuple(range(1101, 1107))
METHODS = ("fixed", "free", "calibrated", "oracle")
N_COEFFICIENTS = 29
INHERITED_LAMBDAS = (0.03, 0.3, 3.0, 30.0, 300.0, 3000.0)
EXTENSION_LAMBDAS = (3.0e4, 3.0e5, 3.0e6, 3.0e7)


class ExtractionError(RuntimeError):
    """A source record or selection is inconsistent with the frozen protocol."""


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file(source: Path, relative: str, role: str) -> dict[str, str]:
    path = source / relative
    if not path.is_file():
        raise ExtractionError(f"missing source file: {relative}")
    return {"path": relative, "role": role, "sha256": sha256_file(path)}


def require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ExtractionError(f"{name} is not boolean")
    return value


def require_coefficients(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != N_COEFFICIENTS:
        raise ExtractionError(f"{name} does not contain {N_COEFFICIENTS} coefficients")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ExtractionError(f"{name} contains a non-numeric coefficient")
    coefficients = [float(item) for item in value]
    if not all(math.isfinite(item) for item in coefficients):
        raise ExtractionError(f"{name} contains a non-finite coefficient")
    return coefficients


def candidate_source(
    source: Path,
    source_directory: str,
    relative_files: list[str],
    lambda_path: list[float],
    selected_lambda: float,
) -> tuple[str, dict[str, Any]]:
    if len(relative_files) != len(lambda_path):
        raise ExtractionError("candidate file and lambda paths have different lengths")
    for relative, penalty_lambda in zip(relative_files, lambda_path):
        if float(penalty_lambda) == float(selected_lambda):
            source_relative = f"{source_directory}/{relative}"
            return source_relative, read_json(source / source_relative)
    raise ExtractionError(f"selected lambda {selected_lambda} has no candidate source")


def export_static_fields(source: Path) -> list[dict[str, Any]]:
    endpoint_verification = read_json(
        source / "results_regularization_endpoint/verification_report.json"
    )
    if (
        endpoint_verification.get("passed") is not True
        or endpoint_verification.get("incomplete") is not False
        or endpoint_verification.get("checked_null_checks") != 848
        or endpoint_verification.get("checked_endpoint_treatments") != 361
        or endpoint_verification.get("selected_path_count") != 848
    ):
        raise ExtractionError("v3 endpoint verification is incomplete or unexpected")

    summary = read_json(
        source / "summaries_load_information/load_information_summary.json"
    )
    configurations = {
        (item["shape"], float(item["tau"]), int(item["seed"])): item
        for item in summary["dataset_configurations"]
        if item["shape"] in SHAPES
        and float(item["tau"]) in TAUS
        and int(item["seed"]) in SEEDS
        and int(item["sensors"]) == 20
        and float(item["sigma"]) == 5.0e-6
        and "primary" in item["memberships"]
    }
    if len(configurations) != 48:
        raise ExtractionError(f"expected 48 matched datasets, found {len(configurations)}")

    records: list[dict[str, Any]] = []
    for shape in SHAPES:
        for tau in TAUS:
            for seed in SEEDS:
                configuration = configurations[(shape, tau, seed)]
                dataset_key = configuration["key"]
                for method in METHODS:
                    path_key = f"{dataset_key}__method-{method}"
                    path_relative = f"results_load_information/paths/{path_key}/path.json"
                    null_relative = (
                        f"results_regularization_endpoint/null_checks/{path_key}/null_check.json"
                    )
                    endpoint_relative = (
                        f"results_regularization_endpoint/endpoints/{path_key}/endpoint.json"
                    )
                    path_record = read_json(source / path_relative)
                    null_record = read_json(source / null_relative)
                    inherited_selection = dict(path_record["selection"])

                    if (
                        path_record.get("path_key") != path_key
                        or path_record.get("dataset_key") != dataset_key
                        or path_record.get("method") != method
                        or null_record.get("path_key") != path_key
                        or null_record.get("dataset_key") != dataset_key
                        or null_record.get("method") != method
                    ):
                        raise ExtractionError(f"path identity mismatch: {path_key}")
                    if null_record["source_path_record_sha256"] != sha256_file(source / path_relative):
                        raise ExtractionError(f"path hash mismatch: {path_key}")

                    upper = require_bool(
                        inherited_selection["unbracketed_high"],
                        f"{path_key}: unbracketed_high",
                    )
                    null_compatible = require_bool(
                        null_record["null_compatible"], f"{path_key}: null_compatible"
                    )
                    source_files = [
                        source_file(source, path_relative, "inherited_path_record"),
                        source_file(source, null_relative, "null_compatibility_check"),
                    ]

                    if upper:
                        endpoint_record = read_json(source / endpoint_relative)
                        if endpoint_record.get("path_key") != path_key:
                            raise ExtractionError(f"endpoint identity mismatch: {path_key}")
                        decision = endpoint_record["decision"]
                        expected_action = (
                            "lambda_infinity" if null_compatible else "finite_extensions"
                        )
                        if (
                            endpoint_record["action"] != expected_action
                            or decision["action"] != expected_action
                            or decision["null_compatible"] != null_compatible
                            or decision["originally_unbracketed_high"] is not True
                        ):
                            raise ExtractionError(f"endpoint action mismatch: {path_key}")
                        selection = endpoint_record["selection"]
                        selected_origin = selection["selected_origin"]
                        selected_lambda = selection["selected_lambda"]

                        if null_compatible:
                            if selected_origin != "null_zero_field" or selected_lambda != "infinity":
                                raise ExtractionError(f"admissible null was not selected: {path_key}")
                            coefficients = require_coefficients(
                                endpoint_record["selected_coefficients"], path_key
                            )
                            if coefficients != require_coefficients(null_record["coefficients"], path_key):
                                raise ExtractionError(f"null coefficient mismatch: {path_key}")
                            coefficient_relative = null_relative
                        else:
                            if selected_origin == "extension":
                                candidate_files = endpoint_record["extension_candidate_files"]
                                lambda_path = list(EXTENSION_LAMBDAS)
                            elif selected_origin == "inherited":
                                candidate_files = endpoint_record["inherited_candidate_files"]
                                lambda_path = list(INHERITED_LAMBDAS)
                            else:
                                raise ExtractionError(f"unknown endpoint origin: {path_key}")
                            coefficient_relative, candidate = candidate_source(
                                source,
                                "results_regularization_endpoint"
                                if selected_origin == "extension"
                                else "results_load_information",
                                candidate_files,
                                lambda_path,
                                selected_lambda,
                            )
                            if candidate.get("path_key") != path_key or candidate.get("method") != method:
                                raise ExtractionError(f"candidate identity mismatch: {path_key}")
                            coefficients = require_coefficients(
                                endpoint_record["selected_coefficients"], path_key
                            )
                            if coefficients != require_coefficients(candidate["coefficients"], path_key):
                                raise ExtractionError(f"endpoint coefficient mismatch: {path_key}")

                        source_files.append(
                            source_file(source, endpoint_relative, "endpoint_selection")
                        )
                        selection_source = endpoint_relative
                        selection_action = expected_action
                    else:
                        if (source / endpoint_relative).exists():
                            raise ExtractionError(f"unexpected endpoint treatment: {path_key}")
                        selected_origin = "inherited"
                        selected_lambda = inherited_selection["selected_lambda"]
                        if isinstance(selected_lambda, bool) or not isinstance(selected_lambda, (int, float)):
                            raise ExtractionError(f"invalid inherited lambda: {path_key}")
                        coefficient_relative, candidate = candidate_source(
                            source,
                            "results_load_information",
                            path_record["candidate_files"],
                            path_record["lambda_path"],
                            selected_lambda,
                        )
                        coefficients = require_coefficients(
                            path_record["selected_coefficients"], path_key
                        )
                        if coefficients != require_coefficients(candidate["coefficients"], path_key):
                            raise ExtractionError(f"inherited coefficient mismatch: {path_key}")
                        selection_source = path_relative
                        selection_action = "null_check_only"

                    coefficient_entry = source_file(
                        source, coefficient_relative, "selected_coefficients"
                    )
                    if all(entry["path"] != coefficient_relative for entry in source_files):
                        source_files.append(coefficient_entry)
                    records.append(
                        {
                            "shape": shape,
                            "seed": seed,
                            "tau": tau,
                            "method": method,
                            "coefficients": coefficients,
                            "dataset_key": dataset_key,
                            "path_key": path_key,
                            "status": path_record["status"],
                            "inherited_compatible": not require_bool(
                                inherited_selection["no_feasible_candidate"],
                                f"{path_key}: no_feasible_candidate",
                            )
                            and not require_bool(
                                inherited_selection["optimizer_failed"],
                                f"{path_key}: optimizer_failed",
                            ),
                            "inherited_selection": inherited_selection,
                            "null_compatible": null_compatible,
                            "selection_action": selection_action,
                            "selected_origin": selected_origin,
                            "selected_lambda": selected_lambda,
                            "selection_source": {
                                "path": selection_source,
                                "sha256": sha256_file(source / selection_source),
                            },
                            "source_files": source_files,
                        }
                    )
    return records


def export_layer_calibration(source: Path) -> dict[str, Any]:
    report = read_json(
        source / "results_continuum/calibration/calibration_report.json"
    )
    reported = {
        float(record["soil_E_Pa"]): float(record["k0_N_m2"])
        for record in report["records"]
    }
    calibrations = []
    for label, soil_E_Pa in (("E50MPa", 5.0e7), ("E200MPa", 2.0e8)):
        relative = f"results_continuum/calibration/{label}.json"
        record = read_json(source / relative)
        k0 = float(record["k0_N_m2"])
        if (
            float(record["soil_E_Pa"]) != soil_E_Pa
            or reported[soil_E_Pa] != k0
            or not math.isfinite(k0)
            or k0 <= 0.0
            or record.get("passed") is not True
            or record.get("bound_active") is not False
            or record.get("defect_response_used") is not False
            or record.get("unused_response_used") is not False
        ):
            raise ExtractionError(f"invalid intact calibration: {label}")
        calibrations.append(
            {
                "label": label,
                "soil_E_Pa": soil_E_Pa,
                "k0_N_m2": k0,
                "source_file": relative,
                "source_sha256": sha256_file(source / relative),
                "design_hash": record["design_hash"],
                "protocol_hash": record["protocol_hash"],
                "source_hash": record["source_hash"],
            }
        )
    return {
        "schema_version": "dynamic-layer-calibration-input-v1",
        "calibrations": calibrations,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_and_print(records: list[dict[str, Any]], layer: dict[str, Any]) -> None:
    if len(records) != 192:
        raise ExtractionError(f"expected 192 fields, found {len(records)}")
    identities = [(record["shape"], record["seed"], record["tau"], record["method"]) for record in records]
    if len(set(identities)) != len(identities):
        raise ExtractionError("duplicate shape/seed/tau/method identity")
    expected_keys = {"shape", "seed", "tau", "method", "coefficients"}
    if any(not expected_keys <= set(record) for record in records):
        raise ExtractionError("required static-field key is missing")

    def counts(name: str) -> Counter[Any]:
        return Counter(record[name] for record in records)

    if counts("shape") != Counter({shape: 48 for shape in SHAPES}):
        raise ExtractionError("shape counts do not match the protocol")
    if counts("tau") != Counter({0.01: 96, 0.10: 96}):
        raise ExtractionError("tau counts do not match the protocol")
    if counts("seed") != Counter({seed: 32 for seed in SEEDS}):
        raise ExtractionError("seed counts do not match the protocol")
    if counts("method") != Counter({method: 48 for method in METHODS}):
        raise ExtractionError("method counts do not match the protocol")
    endpoint_counts = Counter(
        (record["selection_action"], record["selected_origin"]) for record in records
    )
    if any(
        action not in {"null_check_only", "lambda_infinity", "finite_extensions"}
        or origin not in {"inherited", "null_zero_field", "extension"}
        for action, origin in endpoint_counts
    ):
        raise ExtractionError("invalid endpoint action or selection origin")
    if len(layer["calibrations"]) != 2:
        raise ExtractionError("expected two soil-modulus calibrations")

    print("static fields: 192")
    print("methods: " + ", ".join(f"{method}=48" for method in METHODS))
    print("selection origins: " + ", ".join(f"{key}={value}" for key, value in sorted(counts("selected_origin").items())))
    print("endpoint actions: " + ", ".join(f"{key}={value}" for key, value in sorted(counts("selection_action").items())))
    print(f"null compatible: {counts('null_compatible')[True]}")
    print(f"inherited incompatible retained: {sum(not record['inherited_compatible'] for record in records)}")
    print(
        "layer k0 N/m2: "
        + ", ".join(
            f"{item['label']}={item['k0_N_m2']:.17g}" for item in layer["calibrations"]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.resolve()

    records = export_static_fields(source)
    layer = export_layer_calibration(source)
    validate_and_print(records, layer)
    static_path = ROOT / "inputs/static_fields.json"
    layer_path = ROOT / "inputs/layer_calibration.json"
    write_json(static_path, records)
    write_json(layer_path, layer)
    print(f"wrote {static_path.relative_to(ROOT)} ({sha256_file(static_path)})")
    print(f"wrote {layer_path.relative_to(ROOT)} ({sha256_file(layer_path)})")


if __name__ == "__main__":
    main()
