"""Compare plotted summary statistics directly with immutable path records."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
summary_file = ROOT / sys.argv[1]
result_dir = ROOT / sys.argv[2]
summary = json.loads(summary_file.read_text())
records = {}
for file in (result_dir / "paths").glob("*/path.json"):
    record = json.loads(file.read_text())
    records[(record["dataset_key"], record["method"])] = record
checks = 0
replicate_checks = 0
for group in summary["groups"]:
    for metric, statistics in group["metrics"].items():
        values = []
        seen = set()
        for item in statistics["replicates"]:
            key = (item["dataset_key"], group["method"])
            assert key not in seen, (group, "duplicate replicate")
            seen.add(key)
            if metric == "selected_lambda":
                value = records[key]["selection"]["selected_lambda"]
            elif metric == "selected_load_amplitude_mean":
                value = float(np.mean(records[key]["selected_gamma"]))
            else:
                value = records[key]["metrics"][metric]
            assert value is not None and np.isfinite(value)
            assert np.isclose(value, item["value"], rtol=1e-13, atol=1e-14)
            values.append(value)
            replicate_checks += 1
        if values:
            for name, expected in zip(("q25", "median", "q75"), np.quantile(values, [.25, .5, .75])):
                assert np.isclose(expected, statistics[name], rtol=1e-13, atol=1e-14), (metric, name)
                checks += 1
        else:
            assert all(statistics[key] is None for key in ("median", "q25", "q75"))
report = dict(passed=True, groups=len(summary["groups"]), statistic_checks=checks,
              replicate_checks=replicate_checks, source_paths=len(records),
              summary=str(summary_file.relative_to(ROOT)), results=str(result_dir.relative_to(ROOT)))
print(json.dumps(report, indent=2))
