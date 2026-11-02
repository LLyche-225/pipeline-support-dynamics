"""Independently evaluate the intact model using only recorded observations."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from campaign import PhysicalModel
import run_load_information as runner

source = ROOT / "results_load_information"
design = runner.load_design()
datasets, paths, _ = runner.expand_study(design)
configs = {item["key"]: item for item in datasets}
cache = {}
rows = []
for path in paths:
    key = path["dataset_key"]
    config = configs[key]
    record = runner.read_json(source / "datasets" / key / "dataset_record.json")
    if key not in cache:
        file = source / record["npz_file"]
        forward = runner.build_forward(file, PhysicalModel())
        points, observations, sigmas, _ = runner._observation_inputs(file)
        strain, _ = forward.strain_and_jacobian(np.zeros(forward.basis.n_variables), points)
        cache[key] = (strain.reshape(3, -1), np.array(observations), np.array(sigmas))
    f, y, sigma = cache[key]
    inputs = runner.method_inputs(path["method"], record["measured_load_mean"], config["tau"])
    mean = np.asarray(inputs["load_mean"])
    tau = inputs["load_std"]
    fw, yw = f / sigma, y / sigma
    if tau == 0:
        gamma = mean
        cal = np.zeros(3)
    else:
        precision = 0 if np.isinf(tau) else 1 / tau**2
        gamma = np.clip((np.sum(fw * yw, axis=1) + precision * mean) /
                        (np.sum(fw**2, axis=1) + precision), .7, 1.3)
        cal = np.zeros(3) if precision == 0 else (gamma - mean) / tau
    residual = gamma[:, None] * fw - yw
    norm = float(np.sqrt(np.sum(residual**2) + np.sum(cal**2)))
    threshold = 1.2 * np.sqrt(y.size + (3 if inputs["finite_positive_prior"] else 0))
    original = runner.read_json(source / "paths" / path["key"] / "path.json")
    rows.append(dict(path_key=path["key"], shape=config["shape"], method=path["method"],
                     original_upper=original["selection"]["unbracketed_high"],
                     gamma=gamma.tolist(), data_norm=float(np.linalg.norm(residual)),
                     calibration_norm=float(np.linalg.norm(cal)), selection_norm=norm,
                     threshold=float(threshold), null_compatible=bool(norm <= threshold)))
report = dict(count=len(rows), upper=sum(r["original_upper"] for r in rows),
              null_compatible=sum(r["null_compatible"] for r in rows),
              upper_null_compatible=sum(r["original_upper"] and r["null_compatible"] for r in rows),
              upper_needing_extension=sum(r["original_upper"] and not r["null_compatible"] for r in rows),
              null_by_shape=dict(Counter(r["shape"] for r in rows if r["null_compatible"])), rows=rows)
with (ROOT / "audit/null_independent.json").open("w") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
