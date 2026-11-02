"""Independent descriptive reduction of frozen path records for manuscript QA."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "results_load_information"
verification = json.loads((output / "verification_report.json").read_text())
assert verification["passed"] and not verification["incomplete"]
datasets = {p.parent.name: json.loads(p.read_text())["config"]
            for p in (output / "datasets").glob("*/dataset_record.json")}
paths = [json.loads(p.read_text()) for p in (output / "paths").glob("*/path.json")]
assert len(datasets) == 200 and len(paths) == 848
groups = defaultdict(list)
for path in paths:
    config = datasets[path["dataset_key"]]
    for membership in config["memberships"]:
        key = (membership, config["shape"], config["tau"], config["sigma"],
               config["sensors"], path["method"])
        groups[key].append(path)

def describe(values):
    values = [float(x) for x in values if x is not None]
    return None if not values else dict(zip(
        ["median", "q25", "q75", "min", "max"],
        map(float, [np.median(values), np.quantile(values, .25), np.quantile(values, .75),
                    np.min(values), np.max(values)])))

summaries = []
for key, rows in sorted(groups.items()):
    metric_names = sorted({m for r in rows for m in r.get("metrics", {})})
    summaries.append({
        **dict(zip(["study", "shape", "tau", "sigma", "sensors", "method"], key)),
        "n": len(rows),
        "incompatible": sum(p["selection"]["no_feasible_candidate"] for p in rows),
        "upper_lambda": sum(p["selection"]["unbracketed_high"] for p in rows),
        "metrics": {m: describe([p.get("metrics", {}).get(m) for p in rows]) for m in metric_names},
        "data_norm": describe([p["selection"]["data_norm"] for p in rows]),
        "calibration_norm": describe([p["selection"]["calibration_norm"] for p in rows]),
    })
candidates = [json.loads(p.read_text()) for p in (output / "paths").glob("*/candidate_lambda_*.json")]
assert len(candidates) == 5088
starts = [s for p in candidates for s in p["starts"]]
assert len(starts) == 10176
counts = {
    "datasets": len(datasets), "paths": len(paths), "candidates": len(candidates),
    "starts": len(starts), "selected": sum(bool(p["selected"]) for p in paths),
    "incompatible": sum(p["selection"]["no_feasible_candidate"] for p in paths),
    "upper_lambda": sum(p["selection"]["unbracketed_high"] for p in paths),
    "unsuccessful_candidates": sum(not p["successful_start"] for p in candidates),
    "unsuccessful_starts": sum(not s["success"] for s in starts),
    "start_status_counts": dict(Counter(s["status"] for s in starts)),
}
print(json.dumps({"counts": counts, "groups": summaries}, indent=2, allow_nan=False))
