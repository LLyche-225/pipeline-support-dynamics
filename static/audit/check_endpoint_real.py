"""Root acceptance: three real endpoint branches and read-only verification."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import check_regularization_endpoint as endpoint

out = ROOT / sys.argv[1]
source = ROOT / "results_load_information"
design = endpoint.load_design()
rows = json.loads((ROOT / "audit/null_independent.json").read_text())["rows"]
chosen = [
    next(r for r in rows if r["original_upper"] and r["null_compatible"]),
    next(r for r in rows if r["original_upper"] and not r["null_compatible"]),
    next(r for r in rows if not r["original_upper"]),
]
keys = [r["path_key"] for r in chosen]
endpoint.freeze_protocol(out, source, design)
status = endpoint.execute_run(out, source, design, resume=False, path_keys=keys)
assert status["status"] == "completed"
report = endpoint.verify_protocol(out, source, design)
assert report["passed"] and report["incomplete"], report
assert report["checked_null_checks"] == 3
assert report["checked_extension_candidates"] == 4
assert report["checked_endpoint_treatments"] == 2
for row in chosen:
    record = json.loads((out / "null_checks" / row["path_key"] / "null_check.json").read_text())
    assert record["null_compatible"] == row["null_compatible"]
    assert abs(record["augmented_norm"] - row["selection_norm"]) < 1e-7
array_path = out / "endpoints" / chosen[0]["path_key"] / "endpoint_arrays.npz"
original = array_path.read_bytes()
array_path.unlink()
try:
    tamper = endpoint.verify_protocol(out, source, design)
    assert not tamper["passed"], "missing arrays incorrectly pass verification"
    assert not array_path.exists(), "verification recreated missing evidence"
finally:
    array_path.write_bytes(original)
# A wrong record path must not let a valid alias trigger canonical recreation.
record_path = array_path.with_name("endpoint.json")
record_bytes = record_path.read_bytes()
alias = out / "test_alias_arrays.npz"
alias.write_bytes(original)
array_path.unlink()
changed = json.loads(record_bytes)
changed["arrays_file"] = alias.relative_to(out).as_posix()
record_path.write_text(json.dumps(changed))
try:
    tamper = endpoint.verify_protocol(out, source, design)
    assert not tamper["passed"], "wrong array path incorrectly passes verification"
    assert not array_path.exists(), "wrong array path caused canonical evidence recreation"
finally:
    record_path.write_bytes(record_bytes)
    array_path.write_bytes(original)
    alias.unlink()
report = endpoint.verify_protocol(out, source, design)
assert report["passed"]
status = endpoint.execute_run(out, source, design, resume=True, path_keys=keys)
assert status["status"] == "completed"
print(json.dumps({"passed": True, "paths": keys, "checks": report,
                  "missing_evidence_rejected_without_recreation": True}, indent=2))

