"""Campaign runner CLI for support inversion comparison.

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
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
import platform

import numpy as np

# Import after thread settings
from campaign import PhysicalModel, DataConfig, generate_dataset, run_inversion
from verification import run_verification_suite


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # Return as-is, will fail if not JSON serializable
        return obj


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_source_hashes(code_dir: Path) -> dict:
    """Hash benchmark sources only."""
    benchmark_sources = [
        "beam_inverse.py",
        "campaign.py",
        "shape_factory.py",
        "skfem_reference.py",
        "run_campaign.py",
        "verification.py",
        "green_verification.py",
    ]

    hashes = dict()
    for name in benchmark_sources:
        path = code_dir / name
        if path.exists():
            hashes[name] = hash_file(path)
    return hashes


def get_environment_info() -> dict:
    """Collect environment information."""
    import numpy
    import scipy

    # Get skfem version from vendor
    try:
        vendor_path = Path(__file__).resolve().parents[1] / "vendor"
        sys.path.insert(0, str(vendor_path))
        import skfem
        skfem_version = skfem.__version__
    except:
        skfem_version = "unknown"

    thread_env = dict()
    for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        thread_env[key] = os.environ.get(key, 'unset')

    return {
        "python_version": sys.version,
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "skfem_version": skfem_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "thread_settings": thread_env,
    }


def hash_protocol(plan: dict) -> str:
    """Hash the protocol specification."""
    # Use canonical JSON for consistent hashing
    canonical = json.dumps(plan, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def select_candidate(candidates: list, N: int) -> dict:
    """Select candidate according to specification.

    Among candidates with successful_start=True AND train_norm<=1.2*sqrt(N),
    select LARGEST lambda. If none, choose minimum train_norm among successful
    candidates and flag no_feasible_candidate. If no candidate successful,
    return explicit optimizer_failed.
    """
    threshold = 1.2 * np.sqrt(N)

    # Find successful candidates
    successful = [c for c in candidates if c.get("successful_start", False)]

    if not successful:
        # No successful candidates - optimizer failed
        # Return best failed result as diagnostic
        best_failed = min(candidates, key=lambda c: c["train_norm"])
        return {
            "selected_lambda": best_failed["penalty_lambda"],
            "train_norm": best_failed["train_norm"],
            "optimizer_failed": True,
            "no_feasible_candidate": False,
            "path_bracketed": False,
            "unbracketed_high": False,
            "selection_reason": "optimizer_failed",
        }

    # Find feasible successful candidates (train_norm <= threshold)
    feasible = [c for c in successful if c["train_norm"] <= threshold]

    if not feasible:
        # No feasible candidate - choose minimum train_norm among successful
        best = min(successful, key=lambda c: c["train_norm"])
        return {
            "selected_lambda": best["penalty_lambda"],
            "train_norm": best["train_norm"],
            "optimizer_failed": False,
            "no_feasible_candidate": True,
            "path_bracketed": False,
            "unbracketed_high": False,
            "selection_reason": "no_feasible_min_successful_norm",
        }

    # Select largest lambda among feasible
    selected = max(feasible, key=lambda c: c["penalty_lambda"])

    # Check if unbracketed_high: selected is at largest tested lambda
    lambdas_tested = sorted([c["penalty_lambda"] for c in candidates])
    unbracketed_high = (selected["penalty_lambda"] == lambdas_tested[-1])

    # Check if path_bracketed: exists higher SUCCESSFUL candidate violating threshold
    higher_successful_lambdas = [
        c["penalty_lambda"] for c in successful
        if c["penalty_lambda"] > selected["penalty_lambda"]
    ]

    # Among higher successful lambdas, check if any violate threshold
    path_bracketed = False
    if higher_successful_lambdas:
        for c in successful:
            if c["penalty_lambda"] > selected["penalty_lambda"] and c["train_norm"] > threshold:
                path_bracketed = True
                break

    return {
        "selected_lambda": selected["penalty_lambda"],
        "train_norm": selected["train_norm"],
        "optimizer_failed": False,
        "no_feasible_candidate": False,
        "path_bracketed": path_bracketed,
        "unbracketed_high": unbracketed_high,
        "selection_reason": "largest_feasible_lambda",
    }


def score_selected(dataset_npz: Path, result: dict, physical: PhysicalModel) -> tuple:
    """Score selected result on fixed midpoint grid."""
    from beam_inverse import HermiteBeamFE, SupportBasis
    from shape_factory import get_shape_function

    data = np.load(dataset_npz, allow_pickle=False)

    # Fixed midpoint grid: x = -12 + (arange(2400) + 0.5) * 0.01
    score_x = -12.0 + (np.arange(2400) + 0.5) * 0.01
    dx = 0.01

    # Evaluate estimated support at midpoint grid
    basis = SupportBasis(n_coefficients=physical.spline_count)
    full_coefs = basis.full_coefficients(result["coefficients"])
    estimated_d = basis.values(score_x) @ full_coefs

    # Evaluate reference shape directly at midpoint grid
    shape_name = str(data["shape_name"])
    shape_func = get_shape_function(shape_name)
    reference_d = shape_func(score_x)

    # Compute loss area using midpoint quadrature
    loss_area = dx * np.sum(estimated_d)
    loss_area_reference = dx * np.sum(reference_d)
    loss_area_error = abs(loss_area - loss_area_reference)

    # Absolute L2
    absolute_L2 = np.sqrt(dx * np.sum((estimated_d - reference_d)**2))

    # Relative L2 (null for intact)
    if shape_name == "intact":
        relative_L2 = None
    else:
        relative_L2 = absolute_L2 / np.sqrt(dx * np.sum(reference_d**2))

    # Max false loss for intact
    if shape_name == "intact":
        max_false_loss = float(np.max(estimated_d))
    else:
        max_false_loss = None

    # Gamma RMSE (uses actual multipliers from data)
    actual_multipliers = data["multipliers"]
    gamma_RMSE = float(np.sqrt(np.mean((result["gamma"] - actual_multipliers)**2)))

    # Unused load prediction
    unused_center = float(data["unused_center"])
    unused_eval_grid = data["unused_eval_grid"]
    unused_truth_strain = data["unused_truth_strain"]

    # Build nominal FE at unused center with multiplier 1
    fe_unused = HermiteBeamFE(
        [unused_center],
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

    predicted_unused_strain = fe_unused.strain_and_jacobian(result["coefficients"], unused_eval_grid)[0]

    # Unused load metrics
    unused_rel_L2 = float(
        np.linalg.norm(predicted_unused_strain - unused_truth_strain) /
        np.linalg.norm(unused_truth_strain)
    )
    # Peak strain error: abs(max(abs(predicted)) - max(abs(reference))) / max(abs(reference))
    predicted_peak = float(np.max(np.abs(predicted_unused_strain)))
    reference_peak = float(np.max(np.abs(unused_truth_strain)))
    unused_max_error = float(
        np.abs(predicted_peak - reference_peak) / reference_peak
    )

    metrics = {
        "loss_area_m": float(loss_area),
        "loss_area_absolute_error_m": float(loss_area_error),
        "absolute_support_L2": float(absolute_L2),
        "support_rel_L2_except_intact": relative_L2,
        "max_false_loss_for_intact": max_false_loss,
        "gamma_RMSE": gamma_RMSE,
        "unused_load_strain_relative_L2": unused_rel_L2,
        "unused_load_peak_strain_error": unused_max_error,
    }

    # Save arrays
    arrays = {
        "score_x": score_x,
        "reference_d": reference_d,
        "estimated_d": estimated_d,
        "unused_x": unused_eval_grid,
        "reference_strain": unused_truth_strain,
        "predicted_strain": predicted_unused_strain,
    }

    return metrics, arrays


def run_verify(args):
    """Run verification mode."""
    output_path = Path(args.out) / "verification_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running verification suite...")
    results = run_verification_suite(output_path)

    if not results["summary"]["all_tests_pass"]:
        print("VERIFICATION FAILED")
        for test_name, test_result in results.items():
            if test_name != "summary" and not test_result.get("passes", test_result.get("all_pass", False)):
                print(f"  FAILED: {test_name}")
        sys.exit(1)
    else:
        print(f"All {results['summary']['total_tests']} tests passed")

    return 0


def run_pilot(args):
    """Run pilot mode."""
    plan_path = Path(args.plan)
    with open(plan_path) as f:
        plan = json.load(f)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record environment and hashes
    code_dir = Path(__file__).parent
    env_info = get_environment_info()
    source_hashes = get_source_hashes(code_dir)
    protocol_hash = hash_protocol(plan)

    metadata = {
        "mode": "pilot",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": env_info,
        "source_hashes": source_hashes,
        "protocol_hash": protocol_hash,
        "protocol_path": str(plan_path),
    }

    # Pilot uses ONLY pilot_gaussian + regime pilot + seed 2026110199
    shape = "pilot_gaussian"
    regime = "pilot"
    seed = 2026110199

    physical = PhysicalModel(**plan["physical_model"])
    # DataConfig only gets its declared fields, not seeds/regimes
    data_fields = {k: v for k, v in plan["data"].items()
                   if k in ["train_load_centers_m", "unused_load_center_m",
                           "unused_load_multiplier", "sensors_per_load",
                           "sensor_interval_m", "sigma_strain"]}
    data_config = DataConfig(**data_fields)

    print(f"Generating pilot dataset: {shape}/{regime}/seed{seed}")
    dataset_meta = generate_dataset(shape, regime, seed, physical, data_config, output_dir)
    dataset_path = Path(dataset_meta["npz_path"])

    # Use relative path for dataset in metadata
    try:
        dataset_meta["npz_path"] = str(dataset_path.relative_to(output_dir))
    except ValueError:
        # Keep absolute if relative fails
        dataset_meta["npz_path"] = str(dataset_path)

    # Load data to get N
    data = np.load(dataset_path, allow_pickle=False)
    N_total = sum(len(s) for s in data["sensor_locations"])

    lambdas = plan["optimizers"]["lambda_path_proposed"]
    starts = tuple(plan["optimizers"]["starts"])
    max_nfev = plan["optimizers"]["max_nfev"]

    # Run 4 main methods + oracle
    methods = plan["primary_methods"] + ["H1_oracle"]

    for method in methods:
        print(f"\n=== Running method: {method} ===")

        # For oracle, need actual multipliers
        oracle_mults = data["multipliers"].tolist() if method == "H1_oracle" else None

        candidates = []
        complete_path_start = datetime.utcnow()

        for lam in lambdas:
            print(f"  Lambda = {lam:.1f}", end=" ", flush=True)
            result = run_inversion(
                dataset_path, method, lam, physical,
                starts=starts, max_nfev=max_nfev,
                oracle_multipliers=oracle_mults
            )

            # Extract candidate info with train_prediction and cost
            chosen_start_result = result["starts"][result["chosen_start"]]
            candidate = {
                "penalty_lambda": lam,
                "train_norm": result["train_norm"],
                "train_prediction": result["train_prediction"].tolist(),
                "cost": chosen_start_result["cost"],
                "successful_start": result["successful_start"],
                "nfev": sum(s["nfev"] for s in result["starts"]),
                "fit_time_s": result["fit_time_s"],
                "assembly_time_s": result["assembly_time_s"],
                "coefficients": result["coefficients"].tolist(),
                "gamma": result["gamma"].tolist(),
                "chosen_start": result["chosen_start"],
                "starts": result["starts"],
            }
            candidates.append(candidate)

            status = "SUCCESS" if result["successful_start"] else "FAILED"
            print(f"norm={result['train_norm']:.3f} {status} nfev={candidate['nfev']} time={result['fit_time_s']:.2f}s")

            # Save each candidate immediately to separate file
            candidate_path = output_dir / f"pilot_{method}_candidate_lambda{lam:.1f}.json"
            candidate_data = convert_to_json_serializable({
                "dataset": dataset_meta,
                "method": method,
                "candidate": candidate,
                "metadata": metadata,
            })
            with open(candidate_path, 'w') as f:
                json.dump(candidate_data, f, indent=2, allow_nan=False)

        complete_path_elapsed = (datetime.utcnow() - complete_path_start).total_seconds()

        # Selection
        selection = select_candidate(candidates, N_total)
        print(f"\nSelection: lambda={selection['selected_lambda']}, reason={selection['selection_reason']}")

        # Score selected
        selected_idx = next(i for i, c in enumerate(candidates) if c["penalty_lambda"] == selection["selected_lambda"])
        selected_result = candidates[selected_idx]

        # Reconstruct full result for scoring
        full_result = {
            "coefficients": np.array(selected_result["coefficients"]),
            "gamma": np.array(selected_result["gamma"]),
        }

        metrics, arrays = score_selected(dataset_path, full_result, physical)

        # Save result with complete_path_s
        result_data = {
            "dataset": dataset_meta,
            "method": method,
            "candidates": candidates,
            "selection": selection,
            "metrics": metrics,
            "complete_path_s": complete_path_elapsed,
            "metadata": metadata,
        }

        # Convert to JSON-serializable format
        result_data_serializable = convert_to_json_serializable(result_data)

        result_path = output_dir / f"pilot_{method}.json"
        with open(result_path, 'w') as f:
            json.dump(result_data_serializable, f, indent=2, allow_nan=False)

        # Save arrays
        arrays_path = output_dir / f"pilot_{method}_arrays.npz"
        np.savez(arrays_path, **arrays)

        print(f"Results saved to {result_path}")

        # Store H1_profile selection for joint comparison
        if method == "H1_profile":
            h1_profile_lambda = selection["selected_lambda"]

    # Run profile/joint comparison at selected H1_profile lambda
    print(f"\n=== Profile/Joint comparison at lambda={h1_profile_lambda} ===")

    comparison_results = []
    for method_name in ["H1_profile", "H1_joint"]:
        print(f"Running {method_name}...")
        result = run_inversion(
            dataset_path, method_name, h1_profile_lambda, physical,
            starts=starts, max_nfev=max_nfev,
            oracle_multipliers=None
        )

        # Include all start records in joint record
        chosen_start_result = result["starts"][result["chosen_start"]]
        comparison_results.append({
            "method": method_name,
            "penalty_lambda": h1_profile_lambda,
            "train_norm": result["train_norm"],
            "train_prediction": result["train_prediction"].tolist(),
            "cost": chosen_start_result["cost"],
            "successful_start": result["successful_start"],
            "coefficients": result["coefficients"].tolist(),
            "gamma": result["gamma"].tolist(),
            "fit_time_s": result["fit_time_s"],
            "assembly_time_s": result["assembly_time_s"],
            "nfev": sum(s["nfev"] for s in result["starts"]),
            "chosen_start": result["chosen_start"],
            "starts": result["starts"],
        })

        print(f"  {method_name}: norm={result['train_norm']:.3f} time={result['fit_time_s']:.2f}s")

    # Save comparison
    comparison_data = {
        "lambda": h1_profile_lambda,
        "results": comparison_results,
        "metadata": metadata,
    }
    comparison_data_serializable = convert_to_json_serializable(comparison_data)

    comparison_path = output_dir / "pilot_profile_joint_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data_serializable, f, indent=2, allow_nan=False)

    print(f"Comparison saved to {comparison_path}")

    print("\nPilot complete!")
    return 0


def run_primary(args):
    """Run primary mode (placeholder)."""
    print("Primary mode not yet implemented")
    return 1


def run_robustness(args):
    """Run robustness mode (placeholder)."""
    print("Robustness mode not yet implemented")
    return 1


def run_joint(args):
    """Run joint mode (placeholder)."""
    print("Joint mode not yet implemented")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Campaign runner for support inversion")
    parser.add_argument("mode", choices=["verify", "pilot", "primary", "robustness", "joint"])
    parser.add_argument("--plan", required=False, help="Path to campaign plan JSON")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete run")

    args = parser.parse_args()

    if args.mode == "verify":
        return run_verify(args)

    # Other modes require plan
    if not args.plan:
        print("Error: --plan required for this mode")
        return 1

    if args.mode == "pilot":
        return run_pilot(args)
    elif args.mode == "primary":
        return run_primary(args)
    elif args.mode == "robustness":
        return run_robustness(args)
    elif args.mode == "joint":
        return run_joint(args)


if __name__ == "__main__":
    sys.exit(main())
