"""Formal experiment runner for support inversion comparison.

Set BLAS/OMP threads to 1 before NumPy import.
Record versions, platform, timestamps, source SHA256.
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# Import after thread settings
from campaign import PhysicalModel, DataConfig, generate_dataset, run_inversion
from run_campaign import (
    convert_to_json_serializable, hash_protocol, get_environment_info,
    get_source_hashes, select_candidate, score_selected, hash_file
)


def expand_primary(plan: dict) -> list:
    """Expand primary study: 5 shapes × 2 regimes × 5 seeds, with method routing.

    Returns list of (shape, regime, seed, method) tuples.
    Total: 50 datasets, 225 paths (200 main + 25 oracle).
    """
    shapes = list(plan["shapes"].keys())
    regimes = ["calibrated", "amplitude_bias"]
    seeds = plan["data"]["seeds"]
    main_methods = plan["primary_methods"]

    paths = []

    for shape in shapes:
        for regime in regimes:
            for seed in seeds:
                # All datasets get 4 main methods
                for method in main_methods:
                    paths.append((shape, regime, seed, method))

                # Only amplitude_bias cases get H1_oracle
                if regime == "amplitude_bias":
                    paths.append((shape, regime, seed, "H1_oracle"))

    return paths


def expand_robustness(plan: dict) -> list:
    """Expand robustness study: 3 shapes × 3 regimes × 3 seeds, H1_fixed/H1_profile only.

    Returns list of (shape, regime, seed, method) tuples.
    Total: 27 datasets, 54 paths.
    """
    robustness_config = plan["robustness"]
    shapes = robustness_config["shapes"]
    regimes = robustness_config["regimes"]
    seeds = robustness_config["seeds"]
    methods = robustness_config["primary_comparison"]

    paths = []

    for shape in shapes:
        for regime in regimes:
            for seed in seeds:
                for method in methods:
                    paths.append((shape, regime, seed, method))

    return paths


def expand_joint(plan: dict, primary_dir: Path) -> list:
    """Expand joint study: 5 biased shapes at first seed, read H1_profile lambda.

    Returns list of (shape, seed, lambda, repeat_idx, method) tuples.
    Total: 5 datasets, 30 fits (5 × 3 repeats × 2 methods).
    """
    joint_config = plan["same_objective_check"]
    shapes = joint_config["shapes"]
    seed = joint_config["seed"]

    paths = []

    for shape in shapes:
        # Read H1_profile result to get selected lambda
        result_path = primary_dir / "results" / f"{shape}_amplitude_bias_seed{seed}_H1_profile.json"

        if not result_path.exists():
            raise FileNotFoundError(
                f"Joint study requires primary H1_profile result: {result_path}"
            )

        with open(result_path) as f:
            result = json.load(f)

        selected_lambda = result["selection"]["selected_lambda"]

        # 3 repeats, alternating order by repeat
        for repeat_idx in range(3):
            if repeat_idx % 2 == 0:
                # Even repeats: profile then joint
                paths.append((shape, seed, selected_lambda, repeat_idx, "H1_profile"))
                paths.append((shape, seed, selected_lambda, repeat_idx, "H1_joint"))
            else:
                # Odd repeats: joint then profile
                paths.append((shape, seed, selected_lambda, repeat_idx, "H1_joint"))
                paths.append((shape, seed, selected_lambda, repeat_idx, "H1_profile"))

    return paths


def run_path(
    shape: str,
    regime: str,
    seed: int,
    method: str,
    physical: PhysicalModel,
    data_config: DataConfig,
    plan: dict,
    output_dir: Path,
    data_dir: Path,
) -> dict:
    """Run one complete regularization path for a dataset/method combination.

    Returns candidate selection result with all lambda fits and selected scoring.
    """
    dataset_name = f"{shape}_{regime}_seed{seed}"
    dataset_npz = data_dir / f"{dataset_name}.npz"

    # Generate dataset if it doesn't exist
    if not dataset_npz.exists():
        print(f"  Generating dataset: {dataset_name}")
        dataset_meta = generate_dataset(shape, regime, seed, physical, data_config, data_dir)
        dataset_npz = Path(dataset_meta["npz_path"])
    else:
        print(f"  Using existing dataset: {dataset_name}")

    # Load data to get N for selection
    data = np.load(dataset_npz, allow_pickle=False)
    N_total = sum(len(s) for s in data["sensor_locations"])

    lambdas = plan["optimizers"]["lambda_path_proposed"]
    starts = tuple(plan["optimizers"]["starts"])
    max_nfev = plan["optimizers"]["max_nfev"]

    # For oracle, need actual multipliers
    oracle_mults = data["multipliers"].tolist() if method == "H1_oracle" else None

    # Time complete path
    path_start = time.perf_counter()

    candidates = []
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    for lam in lambdas:
        print(f"    Lambda = {lam:.1f}", end=" ", flush=True)
        result = run_inversion(
            dataset_npz, method, lam, physical,
            starts=starts, max_nfev=max_nfev,
            oracle_multipliers=oracle_mults
        )

        # Build candidate record
        candidate = {
            "method": method,
            "penalty_lambda": lam,
            "coefficients": result["coefficients"].tolist(),
            "gamma": result["gamma"].tolist(),
            "train_prediction": result["train_prediction"].tolist(),
            "train_norm": result["train_norm"],
            "cost": result["starts"][result["chosen_start"]]["cost"],
            "successful_start": result["successful_start"],
            "all_starts": result["starts"],
            "chosen_start": result["chosen_start"],
            "assembly_time_s": result["assembly_time_s"],
            "fit_time_s": result["fit_time_s"],
        }
        candidates.append(candidate)

        # Persist candidate immediately
        candidate_serializable = convert_to_json_serializable(candidate)
        candidate_path = candidates_dir / f"{dataset_name}_{method}_lambda{lam:g}.json"
        with open(candidate_path, 'w') as f:
            json.dump(candidate_serializable, f, indent=2, allow_nan=False)

        status = "SUCCESS" if result["successful_start"] else "FAILED"
        print(f"norm={result['train_norm']:.3f} {status}")

    path_time = time.perf_counter() - path_start

    # Selection
    selection = select_candidate(candidates, N_total)
    print(f"    Selection: lambda={selection['selected_lambda']}, reason={selection['selection_reason']}")

    # Score selected
    selected_idx = next(i for i, c in enumerate(candidates)
                       if c["penalty_lambda"] == selection["selected_lambda"])
    selected_candidate = candidates[selected_idx]

    # Reconstruct full result for scoring
    full_result = {
        "coefficients": np.array(selected_candidate["coefficients"]),
        "gamma": np.array(selected_candidate["gamma"]),
    }

    metrics, arrays = score_selected(dataset_npz, full_result, physical)

    # Save arrays first, then compute SHA256
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    arrays_path = results_dir / f"{dataset_name}_{method}_arrays.npz"
    np.savez(arrays_path, **arrays)

    # Compute array SHA256
    array_sha256 = hash_file(arrays_path)

    # Compute input dataset SHA256
    input_sha256 = hash_file(dataset_npz)

    # Build final result with complete payload
    result_data = {
        "dataset": {
            "shape": shape,
            "regime": regime,
            "seed": seed,
            "sha256": input_sha256,
            "npz_relative_path": f"data/{dataset_npz.name}",
        },
        "method": method,
        "candidates": candidates,
        "selection": selection,
        "metrics": metrics,
        "selected_array_filename": arrays_path.name,
        "array_sha256": array_sha256,
        "complete_path_seconds": path_time,
        "nfev": sum(s["nfev"] for s in selected_candidate["all_starts"]),
        "success": selected_candidate["successful_start"],
        "projected_optimality": selected_candidate["all_starts"][selected_candidate["chosen_start"]].get("projected_first_order", None),
    }

    # Save result JSON once with complete payload
    result_serializable = convert_to_json_serializable(result_data)
    result_path = results_dir / f"{dataset_name}_{method}.json"
    with open(result_path, 'w') as f:
        json.dump(result_serializable, f, indent=2, allow_nan=False)

    return result_data


def run_primary_formal(args):
    """Run primary formal study."""
    # Reject --resume immediately (not supported)
    if args.resume:
        print("ERROR: --resume is not supported. Use a fresh output directory.")
        return 1

    plan_path = Path(args.plan)
    with open(plan_path) as f:
        plan = json.load(f)

    output_dir = Path(args.out)

    # Reject any nonempty output directory
    if output_dir.exists():
        if any(output_dir.iterdir()):
            print(f"ERROR: Output directory is not empty: {output_dir}")
            print("Choose a different output directory for each formal run.")
            return 1

    # Record environment and hashes
    code_dir = Path(__file__).parent
    env_info = get_environment_info()
    source_hashes = get_source_hashes(code_dir)

    # Add run_formal.py hash
    run_formal_path = Path(__file__)
    source_hashes["run_formal.py"] = hash_file(run_formal_path)

    protocol_hash = hash_protocol(plan)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build physical model and data config
    physical = PhysicalModel(**plan["physical_model"])
    data_fields = {k: v for k, v in plan["data"].items()
                   if k in ["train_load_centers_m", "unused_load_center_m",
                           "unused_load_multiplier", "sensors_per_load",
                           "sensor_interval_m", "sigma_strain"]}
    data_config = DataConfig(**data_fields)

    # Expand paths
    paths = expand_primary(plan)
    print(f"Primary study: {len(paths)} paths")

    # Record start
    start_time = datetime.utcnow()

    # Create manifest with complete plan copy
    manifest = {
        "mode": "primary",
        "status": "running",
        "protocol_hash": protocol_hash,
        "source_hashes": source_hashes,
        "environment": env_info,
        "plan_path": str(plan_path.name),  # Relative filename only
        "effective_plan": plan,  # Complete plan copy
        "start_time_utc": start_time.isoformat() + "Z",
        "total_paths": len(paths),
        "completed_paths": 0,
    }

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    # Run all paths
    completed = 0
    for shape, regime, seed, method in paths:
        print(f"\n[{completed+1}/{len(paths)}] {shape}/{regime}/seed{seed}/{method}")

        try:
            run_path(shape, regime, seed, method, physical, data_config, plan, output_dir, data_dir)
            completed += 1

            # Update manifest
            manifest["completed_paths"] = completed
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, allow_nan=False)

        except Exception as e:
            print(f"ERROR: Path failed: {e}")
            manifest["status"] = "failed"
            manifest["error"] = str(e)
            manifest["failed_at_path"] = f"{shape}/{regime}/seed{seed}/{method}"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, allow_nan=False)
            return 1

    # Record completion
    end_time = datetime.utcnow()
    manifest["status"] = "complete"
    manifest["end_time_utc"] = end_time.isoformat() + "Z"
    manifest["completed_paths"] = completed

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    print(f"\nPrimary study complete: {completed} paths")
    return 0


def run_robustness_formal(args):
    """Run robustness formal study."""
    # Reject --resume immediately (not supported)
    if args.resume:
        print("ERROR: --resume is not supported. Use a fresh output directory.")
        return 1

    plan_path = Path(args.plan)
    with open(plan_path) as f:
        plan = json.load(f)

    output_dir = Path(args.out)

    # Reject any nonempty output directory
    if output_dir.exists():
        if any(output_dir.iterdir()):
            print(f"ERROR: Output directory is not empty: {output_dir}")
            print("Choose a different output directory for each formal run.")
            return 1

    # Record environment and hashes
    code_dir = Path(__file__).parent
    env_info = get_environment_info()
    source_hashes = get_source_hashes(code_dir)

    # Add run_formal.py hash
    run_formal_path = Path(__file__)
    source_hashes["run_formal.py"] = hash_file(run_formal_path)

    protocol_hash = hash_protocol(plan)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build physical model and data config
    physical = PhysicalModel(**plan["physical_model"])
    data_fields = {k: v for k, v in plan["data"].items()
                   if k in ["train_load_centers_m", "unused_load_center_m",
                           "unused_load_multiplier", "sensors_per_load",
                           "sensor_interval_m", "sigma_strain"]}
    data_config = DataConfig(**data_fields)

    # Expand paths
    paths = expand_robustness(plan)
    print(f"Robustness study: {len(paths)} paths")

    # Record start
    start_time = datetime.utcnow()

    # Create manifest with complete plan copy
    manifest = {
        "mode": "robustness",
        "status": "running",
        "protocol_hash": protocol_hash,
        "source_hashes": source_hashes,
        "environment": env_info,
        "plan_path": str(plan_path.name),  # Relative filename only
        "effective_plan": plan,  # Complete plan copy
        "start_time_utc": start_time.isoformat() + "Z",
        "total_paths": len(paths),
        "completed_paths": 0,
    }

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    # Run all paths
    completed = 0
    for shape, regime, seed, method in paths:
        print(f"\n[{completed+1}/{len(paths)}] {shape}/{regime}/seed{seed}/{method}")

        try:
            run_path(shape, regime, seed, method, physical, data_config, plan, output_dir, data_dir)
            completed += 1

            # Update manifest
            manifest["completed_paths"] = completed
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, allow_nan=False)

        except Exception as e:
            print(f"ERROR: Path failed: {e}")
            manifest["status"] = "failed"
            manifest["error"] = str(e)
            manifest["failed_at_path"] = f"{shape}/{regime}/seed{seed}/{method}"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, allow_nan=False)
            return 1

    # Record completion
    end_time = datetime.utcnow()
    manifest["status"] = "complete"
    manifest["end_time_utc"] = end_time.isoformat() + "Z"
    manifest["completed_paths"] = completed

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    print(f"\nRobustness study complete: {completed} paths")
    return 0


def run_joint_formal(args):
    """Run joint formal study."""
    # Reject --resume immediately (not supported)
    if args.resume:
        print("ERROR: --resume is not supported. Use a fresh output directory.")
        return 1

    if not args.primary:
        print("ERROR: --primary directory required for joint study")
        return 1

    plan_path = Path(args.plan)
    with open(plan_path) as f:
        plan = json.load(f)

    output_dir = Path(args.out)
    primary_dir = Path(args.primary)

    # Reject any nonempty output directory
    if output_dir.exists():
        if any(output_dir.iterdir()):
            print(f"ERROR: Output directory is not empty: {output_dir}")
            print("Choose a different output directory for each formal run.")
            return 1

    # Validate primary manifest before creating output
    primary_manifest_path = primary_dir / "run_manifest.json"
    if not primary_manifest_path.exists():
        print(f"ERROR: Primary manifest not found: {primary_manifest_path}")
        return 1

    with open(primary_manifest_path) as f:
        primary_manifest = json.load(f)

    # Require primary complete
    if primary_manifest.get("status") != "complete":
        print(f"ERROR: Primary run is not complete (status: {primary_manifest.get('status')})")
        return 1

    # Record environment and hashes
    code_dir = Path(__file__).parent
    env_info = get_environment_info()
    source_hashes = get_source_hashes(code_dir)

    # Add run_formal.py hash
    run_formal_path = Path(__file__)
    source_hashes["run_formal.py"] = hash_file(run_formal_path)

    protocol_hash = hash_protocol(plan)

    # Verify protocol and all source hashes match primary
    if primary_manifest["protocol_hash"] != protocol_hash:
        print("ERROR: Protocol hash mismatch between primary and current plan.")
        return 1

    for key in source_hashes.keys():
        if primary_manifest["source_hashes"].get(key) != source_hashes.get(key):
            print(f"ERROR: Source hash mismatch for {key} between primary and current.")
            return 1

    # Build physical model
    physical = PhysicalModel(**plan["physical_model"])

    starts = tuple(plan["optimizers"]["starts"])
    max_nfev = plan["optimizers"]["max_nfev"]

    # Expand joint paths
    try:
        paths = expand_joint(plan, primary_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Verify dataset hashes match primary for all shapes
    joint_config = plan["same_objective_check"]
    shapes = joint_config["shapes"]
    seed = joint_config["seed"]

    for shape in shapes:
        dataset_name = f"{shape}_amplitude_bias_seed{seed}"
        dataset_npz = primary_dir / "data" / f"{dataset_name}.npz"

        if not dataset_npz.exists():
            print(f"ERROR: Dataset not found in primary: {dataset_npz}")
            return 1

        # Read primary result to get expected dataset hash
        result_path = primary_dir / "results" / f"{dataset_name}_H1_profile.json"
        with open(result_path) as f:
            result = json.load(f)

        expected_sha = result["dataset"]["sha256"]
        actual_sha = hash_file(dataset_npz)

        if expected_sha != actual_sha:
            print(f"ERROR: Dataset SHA mismatch for {dataset_name}")
            print(f"  Expected: {expected_sha}")
            print(f"  Actual: {actual_sha}")
            return 1

    print(f"Joint study: {len(paths)} fits")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Record start
    start_time = datetime.utcnow()

    # Create manifest with complete plan copy
    manifest = {
        "mode": "joint",
        "status": "running",
        "protocol_hash": protocol_hash,
        "source_hashes": source_hashes,
        "environment": env_info,
        "plan_path": str(plan_path.name),  # Relative filename only
        "effective_plan": plan,  # Complete plan copy
        "primary_dir": os.path.relpath(primary_dir, output_dir),
        "start_time_utc": start_time.isoformat() + "Z",
        "total_fits": len(paths),
        "completed_fits": 0,
    }

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    # Group paths by dataset
    dataset_groups = {}
    for shape, seed, lam, repeat_idx, method in paths:
        key = (shape, seed)
        if key not in dataset_groups:
            dataset_groups[key] = []
        dataset_groups[key].append((lam, repeat_idx, method))

    # Run all fits
    completed = 0
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for (shape, seed), fits in dataset_groups.items():
        print(f"\n=== {shape}/seed{seed} ===")

        # Read dataset from primary
        dataset_name = f"{shape}_amplitude_bias_seed{seed}"
        dataset_npz = primary_dir / "data" / f"{dataset_name}.npz"

        if not dataset_npz.exists():
            print(f"ERROR: Dataset not found in primary: {dataset_npz}")
            manifest["status"] = "failed"
            manifest["error"] = f"Missing dataset: {dataset_npz}"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, allow_nan=False)
            return 1

        # Run all fits for this dataset in order
        fit_records = []
        for lam, repeat_idx, method in fits:
            print(f"  [{completed+1}/{len(paths)}] repeat{repeat_idx} {method} lambda={lam:.1f}")

            try:
                result = run_inversion(
                    dataset_npz, method, lam, physical,
                    starts=starts, max_nfev=max_nfev,
                    oracle_multipliers=None
                )

                fit_record = {
                    "repeat_idx": repeat_idx,
                    "method": method,
                    "penalty_lambda": lam,
                    "execution_order": completed,
                    "coefficients": result["coefficients"].tolist(),
                    "gamma": result["gamma"].tolist(),
                    "train_prediction": result["train_prediction"].tolist(),
                    "train_norm": result["train_norm"],
                    "cost": result["starts"][result["chosen_start"]]["cost"],
                    "successful_start": result["successful_start"],
                    "all_starts": result["starts"],
                    "chosen_start": result["chosen_start"],
                    "assembly_time_s": result["assembly_time_s"],
                    "fit_time_s": result["fit_time_s"],
                    "nfev": sum(s["nfev"] for s in result["starts"]),
                    "status": result["starts"][result["chosen_start"]]["status"],
                    "projected_first_order": result["starts"][result["chosen_start"]].get("projected_first_order", None),
                }

                fit_records.append(fit_record)
                completed += 1

                print(f"    norm={result['train_norm']:.3f} time={result.get('fit_time_s', 0.0):.2f}s")

                # Update manifest
                manifest["completed_fits"] = completed
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2, allow_nan=False)

            except Exception as e:
                print(f"ERROR: Fit failed: {e}")
                manifest["status"] = "failed"
                manifest["error"] = str(e)
                manifest["failed_at_fit"] = f"{shape}/seed{seed}/repeat{repeat_idx}/{method}"
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2, allow_nan=False)
                return 1

        # Save all fit records for this dataset
        joint_result = {
            "dataset": {
                "shape": shape,
                "seed": seed,
                "sha256": hash_file(dataset_npz),
                "npz_relative_path": f"data/{dataset_npz.name}",
                "path_base": "primary_dir_in_joint_manifest",
            },
            "fits": fit_records,
        }

        joint_result_serializable = convert_to_json_serializable(joint_result)
        result_path = results_dir / f"{dataset_name}_joint_comparison.json"

        with open(result_path, 'w') as f:
            json.dump(joint_result_serializable, f, indent=2, allow_nan=False)

        print(f"  Saved: {result_path.name}")

    # Record completion
    end_time = datetime.utcnow()
    manifest["status"] = "complete"
    manifest["end_time_utc"] = end_time.isoformat() + "Z"
    manifest["completed_fits"] = completed

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    print(f"\nJoint study complete: {completed} fits")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Formal experiment runner for support inversion comparison"
    )
    parser.add_argument(
        "mode",
        choices=["primary", "robustness", "joint"],
        help="Experiment mode"
    )
    parser.add_argument(
        "--plan",
        required=True,
        help="Path to campaign plan JSON"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--primary",
        help="Primary results directory (required for joint mode)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Unsupported; rejected without writing. Use a fresh output directory."
    )

    args = parser.parse_args()

    if args.mode == "primary":
        return run_primary_formal(args)
    elif args.mode == "robustness":
        return run_robustness_formal(args)
    elif args.mode == "joint":
        return run_joint_formal(args)


if __name__ == "__main__":
    sys.exit(main())



