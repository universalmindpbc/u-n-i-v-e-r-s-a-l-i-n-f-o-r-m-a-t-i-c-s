import copy
import json
from pathlib import Path
import tempfile
import unittest
from audit_tool import audit_recorded_measurements

SOURCE = Path(__file__).resolve().parents[3] / "Informatics/quantum/data/oxtr-qrc-qualification.json"


class RecordedMeasurementsTest(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(SOURCE.read_text())

    def rejects(self, mutate):
        data = copy.deepcopy(self.data)
        mutate(data)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                audit_recorded_measurements(path)

    def test_existing_receipts_and_scope(self):
        result = audit_recorded_measurements(SOURCE)
        self.assertEqual(result["programs_per_provider"], 16)
        self.assertEqual(result["measurements_starting_with_all_atoms_present"], 1541)
        self.assertFalse(result["biological_prediction_evaluated"])
        self.assertEqual(result["new_provider_spend_usd"], 0)

    def test_rejects_duplicate_method(self):
        self.rejects(lambda d: d["records"].__setitem__(0, d["records"][1]))

    def test_rejects_missing_program(self):
        self.rejects(lambda d: d["records"].pop())

    def test_rejects_nan(self):
        self.rejects(lambda d: d["records"][0]["gpu_observables"].__setitem__(0, float("nan")))

    def test_rejects_impossible_count(self):
        self.rejects(lambda d: d["records"][0].__setitem__("hardware_valid_shots", 101))

    def test_rejects_pending_job(self):
        self.rejects(lambda d: d["records"][0].__setitem__("hardware_state", "QUEUED"))

    def test_rejects_reused_receipt(self):
        self.rejects(lambda d: d["records"][0].__setitem__("gpu_job_id", d["records"][1]["gpu_job_id"]))


if __name__ == "__main__":
    unittest.main()
