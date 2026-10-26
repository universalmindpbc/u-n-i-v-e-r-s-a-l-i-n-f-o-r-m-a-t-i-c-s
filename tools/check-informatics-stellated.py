"""Verify summary counts against independent node sets and unchanged source data."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Informatics/pilot-v1'
BASE = 'a3f19f52c9ec819895998309046d2a5a4a6a2af1'
spec = importlib.util.spec_from_file_location('stellated', ROOT / 'research/informatics/pilot/sync_stellated.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
expected = {9: (4, 1, True), 18: (4, 1, True), 27: (12, 3, False),
            36: (4, 1, False), 45: (20, 5, False)}
dataset = json.loads((DATA / 'dataset.json').read_text())
before_dataset = json.loads(subprocess.check_output(['git', 'show', f'{BASE}:Informatics/pilot-v1/dataset.json'], cwd=ROOT))
rows = list(csv.DictReader((DATA / 'records.csv').open(encoding='utf-8-sig')))
for record in dataset['records']:
    m = record['modulus']
    native = json.loads((DATA / f'm{m}.json').read_text())
    before = json.loads(subprocess.check_output(['git', 'show', f'{BASE}:Informatics/pilot-v1/m{m}.json'], cwd=ROOT))
    row = next(row for row in rows if int(row['modulus']) == m)
    if m in expected:
        summary = module.summarize(native)
        assert native['other_geometry_summary'] == record['other_geometry_summary'] == summary
        assert (summary['all_alpha_count'], summary['primary_gamma_count'], summary['native_beta_doubling']) == expected[m]
        assert summary['native_beta_defined'] is True
        assert int(row['stellated_count_all_alpha']) == summary['all_alpha_count']
        assert int(row['stellated_count_primary_gamma']) == summary['primary_gamma_count']
        with zipfile.ZipFile(DATA / 'm1-m48-data.zip') as z:
            assert z.read(f'm{m}.json') == (DATA / f'm{m}.json').read_bytes()
        native.pop('other_geometry_summary')
        record.pop('other_geometry_summary')
    else:
        assert 'other_geometry_summary' not in native
        assert row['stellated_count_all_alpha'] == row['stellated_count_primary_gamma'] == ''
    native.pop('m9_sequence_views', None)
    record.pop('m9_sequence_views', None)
    native.pop('beta_m9_view', None)
    record.pop('beta_m9_view', None)
    native.pop('native_beta_observation', None)
    record.pop('native_beta_observation', None)
    assert native == before, f'Existing M{m} content changed'
dataset.pop('m9_sequence_family_audit', None)
dataset.pop('beta_m9_family_audit', None)
assert dataset == before_dataset, 'Existing aggregate data changed'
# The comparison record M42 has a defined Beta sequence but no native doubling.
m42 = json.loads((DATA / 'm42.json').read_text())
assert m42['beta']['interval_residues'] == [4, 26, 10, 2] * 3
assert m42['beta']['native_doubling'] is False
assert module.summarize(m42)['all_alpha_count'] == 8
print(json.dumps({'passed': True, 'family_counts': expected, 'other_records_unchanged': 43,
                  'existing_data_preserved': True, 'csv_json_zip_consistent': True,
                  'm42_beta_present_without_doubling': True}, indent=2))
