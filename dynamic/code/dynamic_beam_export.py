"""Load or provisionally rebuild the frozen static-field list for step 3.

Primary source: inputs/static_fields.json (frozen export, owned by another
worker). Fallback when absent: provisional export rebuilt from the read-only
v3 result trees under the documented selection order -- v3 admissible
zero-field first, then extended-endpoint selections, then inherited v3 path
selections. The provisional copy is written to results/beam/ (owned scope)
with full provenance in audit/beam_fields_provenance.json. No re-estimation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent / "revision_v3_load_information_20261102"
V3_PATHS = V3 / "results_load_information" / "paths"
V3_ENDPOINT = V3 / "results_regularization_endpoint"
FROZEN = ROOT / "inputs" / "static_fields.json"
PROVISIONAL = ROOT / "results" / "beam" / "static_fields.json"
PROVENANCE = ROOT / "audit" / "beam_fields_provenance.json"

SHAPES = ("intact", "asymmetric", "broad_shallow", "double")
TAUS = ("0.01", "0.10000000000000001")
SEEDS = tuple(range(1101, 1107))
METHODS = ("calibrated", "fixed", "free", "oracle")
SIGMA_KEY = "5.0000000000000004e-06"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _path_key(shape: str, tau: str, seed: int, method: str) -> str:
    return (f"shape-{shape}_tau-{tau}_sigma-{SIGMA_KEY}_sensors-20"
            f"_seed-{seed}__method-{method}")


def build_provisional() -> list[dict]:
    """Reconstruct the 192 selected fields from the read-only v3 trees."""
    records = []
    for shape in SHAPES:
        for tau in TAUS:
            for seed in SEEDS:
                for method in METHODS:
                    key = _path_key(shape, tau, seed, method)
                    path_file = V3_PATHS / key / "path.json"
                    null_file = V3_ENDPOINT / "null_checks" / key / "null_check.json"
                    ep_file = V3_ENDPOINT / "endpoints" / key / "endpoint.json"
                    path_rec = json.loads(path_file.read_text())
                    null_rec = json.loads(null_file.read_text())
                    base = {
                        "key": key,
                        "shape": shape,
                        "seed": seed,
                        "tau": float(tau),
                        "method": method,
                        "dataset_hash": path_rec["dataset_hash"],
                        "inherited_arrays_sha256": path_rec["arrays_sha256"],
                        "inherited_status": path_rec["status"],
                        "inherited_selected_lambda": path_rec["selection"]["selected_lambda"],
                        "inherited_unbracketed_high": path_rec["selection"]["unbracketed_high"],
                        "null_augmented_norm": null_rec["augmented_norm"],
                        "null_threshold": null_rec["threshold"],
                    }
                    if null_rec["null_compatible"]:
                        base.update(
                            selection_origin="null_zero_field",
                            status="null_admissible",
                            coefficients=[0.0] * 29,
                            gamma=[float(g) for g in null_rec["gamma"]],
                            source_record_sha256=_sha256(null_file),
                        )
                    elif ep_file.exists():
                        ep = json.loads(ep_file.read_text())
                        base.update(
                            selection_origin="endpoint_" + ep["selection"]["selected_origin"],
                            status=("unbracketed_high" if ep["selection"]["still_unbracketed_high"]
                                    else "endpoint_selected"),
                            coefficients=[float(c) for c in ep["selected_coefficients"]],
                            gamma=[float(g) for g in ep["selected_gamma"]],
                            selected_lambda=ep["selection"]["selected_lambda"],
                            source_arrays_sha256=ep["arrays_sha256"],
                            source_record_sha256=_sha256(ep_file),
                        )
                    else:
                        base.update(
                            selection_origin="inherited",
                            status=path_rec["status"],
                            coefficients=[float(c) for c in path_rec["selected_coefficients"]],
                            gamma=[float(g) for g in path_rec["selected_gamma"]],
                            selected_lambda=path_rec["selection"]["selected_lambda"],
                            source_arrays_sha256=path_rec["arrays_sha256"],
                            source_record_sha256=_sha256(path_file),
                        )
                    records.append(base)
    return records


def load_fields() -> tuple[list[dict], dict]:
    """Return (records, provenance) preferring the frozen input export."""
    if FROZEN.exists():
        records = json.loads(FROZEN.read_text())
        return records, {"source": "frozen inputs/static_fields.json",
                         "sha256": _sha256(FROZEN), "provisional": False}
    records = build_provisional()
    PROVISIONAL.parent.mkdir(parents=True, exist_ok=True)
    PROVISIONAL.write_text(json.dumps(records, indent=1))
    provenance = {
        "source": "provisional rebuild from read-only v3 trees",
        "provisional": True,
        "sha256": _sha256(PROVISIONAL),
        "v3_paths_dir": str(V3_PATHS),
        "v3_endpoint_dir": str(V3_ENDPOINT),
        "selection_order": ["null_zero_field (v3 admissible)",
                            "extended-endpoint selection", "inherited v3 selection"],
        "record_count": len(records),
        "origin_counts": {o: sum(r["selection_origin"].startswith(o.split("_")[0])
                                 if o.startswith("endpoint") else r["selection_origin"] == o
                                 for r in records)
                          for o in ("null_zero_field", "endpoint", "inherited")},
        "file": str(PROVISIONAL),
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=1))
    return records, provenance


if __name__ == "__main__":
    records, prov = load_fields()
    print(json.dumps(prov, indent=1))
    counts = {}
    for r in records:
        o = r.get("selected_origin", r.get("selection_origin", "?"))
        counts[o] = counts.get(o, 0) + 1
    print("origins:", counts)
