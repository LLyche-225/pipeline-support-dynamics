"""Replay selected finite-prior fits by direct joint optimization."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_load_information as runner
from beam_inverse import fit_support
from campaign import PhysicalModel

output = ROOT / "results_load_information"
design = runner.load_design()
runner.load_and_check_lock(output, design)
datasets, _, _ = runner.expand_study(design)
records = []
for config in datasets:
    if (config["shape"] not in ["intact", "asymmetric", "broad_shallow"] or config["tau"] not in [.01, .1]
            or config["seed"] != 1101 or config["sigma"] != 5e-6 or config["sensors"] != 20):
        continue
    path_dir = output / "paths" / runner.path_key(config, "calibrated")
    selected = runner.read_json(path_dir / "path.json")
    lam = selected["selection"]["selected_lambda"]
    candidate = runner.read_json(path_dir / runner._candidate_filename(lam))
    data_file = output / selected["dataset_file"]
    forward = runner.build_forward(data_file, PhysicalModel())
    points, observed, sigma, _ = runner._observation_inputs(data_file)
    joint = fit_support(forward, points, observed, sigma, gamma_mode="joint", penalty="h1",
                        penalty_lambda=lam, load_mean=np.asarray(candidate["load_mean"]), load_std=config["tau"],
                        starts=(.05, .30), max_nfev=200)
    cost = joint["starts"][joint["chosen_start"]]["cost"]
    gap = abs(cost - candidate["objective"]) / max(1., cost, candidate["objective"])
    records.append({"dataset": config["key"], "lambda": lam,
                    "profile_objective": candidate["objective"], "joint_objective": float(cost),
                    "relative_objective_gap": gap, "successful_joint": bool(joint["successful_start"]),
                    "coefficient_norm_gap": float(np.linalg.norm(joint["coefficients"] - np.asarray(candidate["coefficients"]))),
                    "gamma_norm_gap": float(np.linalg.norm(joint["gamma"] - np.asarray(candidate["gamma"]))),
                    "joint_starts": runner.safe_json(joint["starts"])})
print(json.dumps(runner.safe_json({"cases": records}), indent=2, allow_nan=False))
assert len(records) == 6
assert all(record["successful_joint"] and record["relative_objective_gap"] < 1e-6 for record in records)
