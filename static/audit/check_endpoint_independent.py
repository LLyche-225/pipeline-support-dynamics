"""Independent endpoint replay and before/after reductions; no fitting."""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_load_information as runner
from beam_inverse import HermiteBeamFE
from campaign import PhysicalModel

source = ROOT / "results_load_information"
endpoint = ROOT / "results_regularization_endpoint"
read = lambda path: json.loads(path.read_text())
design = runner.load_design()
datasets, paths, _ = runner.expand_study(design)
configs = {d["key"]: d for d in datasets}
null_rows = {r["path_key"]: r for r in read(ROOT / "audit/null_independent.json")["rows"]}
cache = {}
errors = defaultdict(float)
counts = Counter()
primary = defaultdict(list)
all_rows = []
unused_fe = HermiteBeamFE([6.0])
local, local_w = np.polynomial.legendre.leggauss(6)

def inputs(method, mean, tau):
    if method == "oracle":
        return np.ones(3), 0.0
    if method == "fixed":
        return mean, 0.0
    if method == "free":
        return mean, np.inf
    return mean, tau * {"calibrated": 1, "overconfident": .2, "conservative": 3}[method]

def evaluate(coeff, mean, tau, lam, forward, points, y, sigma):
    f = forward.strain_and_jacobian(np.asarray(coeff), points)[0].reshape(3, -1)
    fw, yw = f / sigma, y / sigma
    if tau == 0:
        gamma = mean
        cal_sq = 0.
    else:
        precision = 0. if np.isinf(tau) else tau ** -2
        gamma = np.clip((np.sum(fw * yw, axis=1) + precision * mean) /
                        (np.sum(fw ** 2, axis=1) + precision), .7, 1.3)
        cal_sq = 0. if precision == 0 else float(np.sum(((gamma - mean) / tau) ** 2))
    data_sq = float(np.sum((gamma[:, None] * fw - yw) ** 2))
    edges = np.unique(forward.basis.knots)
    xq = (edges[:-1, None] + np.diff(edges)[:, None] * (local + 1) / 2).ravel()
    wq = (np.diff(edges)[:, None] * local_w / 2).ravel()
    spline = BSpline(forward.basis.knots, np.r_[0., coeff, 0.], 3)
    reg = float(lam * np.sum(wq * spline.derivative()(xq) ** 2))
    return gamma, gamma[:, None] * f, data_sq + cal_sq + reg

for path in paths:
    key = path["key"]
    conf = configs[path["dataset_key"]]
    original = read(source / "paths" / key / "path.json")
    null = read(endpoint / "null_checks" / key / "null_check.json")
    ref_null = null_rows[key]
    assert null["null_compatible"] == ref_null["null_compatible"]
    gap = abs(null["augmented_norm"] - ref_null["selection_norm"])
    assert gap < 1e-7, (key, gap)
    errors["max_null_norm_absolute"] = max(errors["max_null_norm_absolute"], gap)
    np.testing.assert_allclose(null["gamma"], ref_null["gamma"], rtol=1e-11, atol=1e-12)
    assert np.linalg.norm(null["predicted_strain"]) > 0
    counts["null_checks"] += 1
    counts["null_compatible"] += null["null_compatible"]
    final = original
    if original["selection"]["unbracketed_high"]:
        counts["upper_paths"] += 1
        final = read(endpoint / "endpoints" / key / "endpoint.json")
        candidates = [read(p) for p in (source / "paths" / key).glob("candidate_lambda_*.json")
                      if not p.name.startswith("._")]
        extensions = [read(p) for p in (endpoint / "endpoints" / key).glob("candidate_lambda_*.json")
                      if not p.name.startswith("._")]
        if null["null_compatible"]:
            assert not extensions
            assert final["selection"]["selected_lambda"] == "infinity"
            assert np.all(np.asarray(final["selected_coefficients"]) == 0)
            counts["zero_field_endpoints"] += 1
        else:
            counts["extended_paths"] += 1
            assert sorted(c["penalty_lambda"] for c in extensions) == [3e4, 3e5, 3e6, 3e7]
            if conf["key"] not in cache:
                data_record = read(source / "datasets" / conf["key"] / "dataset_record.json")
                data_path = source / data_record["npz_file"]
                points, values, sigmas, _ = runner._observation_inputs(data_path)
                cache[conf["key"]] = (runner.build_forward(data_path, PhysicalModel()),
                                     points, np.asarray(values), np.asarray(sigmas),
                                     np.asarray(data_record["measured_load_mean"]))
            forward, points, y, sigma, measured = cache[conf["key"]]
            mean, tau = inputs(path["method"], measured, conf["tau"])
            for candidate in extensions:
                counts["extension_candidates"] += 1
                for start in candidate["starts"]:
                    counts["extension_starts"] += 1
                    counts["failed_starts"] += not start["success"]
                    _, _, obj = evaluate(start["x"], mean, tau, candidate["penalty_lambda"],
                                         forward, points, y, sigma)
                    # The fit record stores the full squared norm, not SciPy's half-cost.
                    rel = abs(obj - start["cost"]) / max(1., abs(obj))
                    assert rel < 1e-8, (key, rel)
                    errors["max_start_objective_relative"] = max(errors["max_start_objective_relative"], rel)
                gamma, pred, obj = evaluate(candidate["coefficients"], mean, tau,
                                           candidate["penalty_lambda"], forward, points, y, sigma)
                np.testing.assert_allclose(gamma, candidate["gamma"], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(pred, candidate["train_prediction"], rtol=1e-10, atol=1e-12)
                assert abs(obj - candidate["objective"]) < 1e-8 * max(1., abs(obj))
            successful = [c for c in candidates + extensions if c["successful_start"]]
            compatible = [c for c in successful if c["norms"]["selection_norm"] <= null["threshold"]]
            chosen = max(compatible, key=lambda c: c["penalty_lambda"]) if compatible else min(
                successful, key=lambda c: c["norms"]["selection_norm"])
            assert final["selection"]["selected_lambda"] == chosen["penalty_lambda"]
            counts["still_unbracketed"] += final["selection"]["still_unbracketed_high"]
            counts["extended_incompatible"] += not bool(compatible)
        with np.load(endpoint / final["arrays_file"], allow_pickle=False) as arrays:
            prediction = unused_fe.strain_and_jacobian(
                final["selected_coefficients"], arrays["unused_x"])[0].ravel()
            np.testing.assert_allclose(prediction, arrays["predicted_strain"].ravel(), rtol=1e-10, atol=1e-12)
            field = BSpline(unused_fe.basis.knots,
                            np.r_[0., final["selected_coefficients"], 0.], 3)(arrays["score_x"])
            np.testing.assert_allclose(field, arrays["estimated_d"], rtol=1e-10, atol=1e-12)
            area = float(np.sum(field) * .01)
            prediction_error = float(np.linalg.norm(prediction - arrays["reference_strain"].ravel()) /
                                     np.linalg.norm(arrays["reference_strain"]))
            assert abs(area - final["metrics"]["loss_area_m"]) < 1e-10
            assert abs(prediction_error - final["metrics"]["unused_load_strain_relative_L2"]) < 1e-10
            if conf["shape"] != "intact":
                err = float(np.linalg.norm(field - arrays["reference_d"]) / np.linalg.norm(arrays["reference_d"]))
                assert abs(err - final["metrics"]["support_rel_L2_except_intact"]) < 1e-10
            counts["endpoint_score_checks"] += 1
    record = {"path_key": key, "shape": conf["shape"], "tau": conf["tau"], "method": path["method"],
              "old": original["metrics"], "new": final["metrics"],
              "old_lambda": original["selection"]["selected_lambda"],
              "new_lambda": final["selection"]["selected_lambda"],
              "null_compatible": null["null_compatible"],
              "incompatible": final["selection"]["no_feasible_candidate"]}
    all_rows.append(record)
    if "primary" in conf["memberships"]:
        primary[(conf["shape"], conf["tau"], path["method"])].append(record)

groups = []
for (shape, tau, method), rows in sorted(primary.items()):
    item = {"shape": shape, "tau": tau, "method": method, "n": len(rows),
            "null_compatible": sum(r["null_compatible"] for r in rows),
            "incompatible": sum(r["incompatible"] for r in rows),
            "new_lambdas": [r["new_lambda"] for r in rows]}
    for metric in ["loss_area_m", "support_rel_L2_except_intact", "unused_load_strain_relative_L2"]:
        for stage in ["old", "new"]:
            values = [r[stage][metric] for r in rows if r[stage][metric] is not None]
            item[stage + "_" + metric] = float(np.median(values)) if values else None
    groups.append(item)
report = {"passed": True, "counts": dict(counts), "errors": dict(errors),
          "final_incompatible_paths": sum(r["incompatible"] for r in all_rows),
          "primary_groups": groups, "rows": all_rows}
(ROOT / "audit/endpoint_independent.json").write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({k:v for k,v in report.items() if k not in {"rows","primary_groups"}}, indent=2))

