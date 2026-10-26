"""Read-only tool for the recorded OXTR qualification dataset; no network calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path

METHODS = ("ordinary", "fibonacci", "phi", "alpha", "beta", "gamma", "epsilon", "shuffled")


def audit_recorded_measurements(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "universal-informatics/qualification-results/v1" or data.get("study_id") != "oxtr-qrc-01":
        raise ValueError("This tool only accepts the recorded OXTR qualification dataset.")
    windows = data.get("windows", [])
    if len(windows) != 2 or len({w["id"] for w in windows}) != 2:
        raise ValueError("Expected two distinct DNA samples.")
    if any(w["end"] - w["start"] != 256 for w in windows):
        raise ValueError("Each recorded sample must contain 256 DNA bases.")
    expected = {(w["id"], method) for w in windows for method in METHODS}
    records = data.get("records", [])
    pairs = [(r.get("window"), r.get("variant")) for r in records]
    if len(pairs) != 16 or set(pairs) != expected:
        raise ValueError("Missing or duplicate sample/method combinations.")
    for record in records:
        for field in ("ideal_observables", "gpu_observables", "hardware_observables"):
            values = record.get(field, [])
            if len(values) != 21 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Invalid atom measurement probabilities.")
        shots, valid = record.get("shots"), record.get("hardware_valid_shots")
        if type(shots) is not int or shots != 100 or type(valid) is not int or not 0 < valid <= shots:
            raise ValueError("Invalid recorded measurement counts.")
        if record.get("hardware_state") != "COMPLETED":
            raise ValueError("A hardware task is not recorded as complete.")
    for field in ("gpu_job_id", "hardware_task_id"):
        ids = [r.get(field) for r in records]
        if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != 16:
            raise ValueError("Missing or duplicate provider job identifiers.")
    return {
        "tool": "audit_recorded_measurements",
        "version": "0.1.0",
        "study_id": data["study_id"],
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "purpose": data["purpose"],
        "dna_samples": 2,
        "bases_per_sample": 256,
        "measurement_preparations": list(METHODS),
        "programs_per_provider": 16,
        "requested_hardware_measurements": sum(r["shots"] for r in records),
        "measurements_starting_with_all_atoms_present": sum(r["hardware_valid_shots"] for r in records),
        "integrity_checks": "passed",
        "biological_prediction_evaluated": False,
        "finding": "The saved dataset contains two DNA samples, each prepared eight ways, with GPU and Aquila measurements. It contains no binding-prediction accuracy test.",
        "verification_scope": "Checks the saved record structure and recorded job identifiers. Does not contact providers or independently authenticate their receipts.",
        "new_provider_jobs": 0,
        "new_provider_spend_usd": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_recorded_measurements(args.dataset), indent=2))
