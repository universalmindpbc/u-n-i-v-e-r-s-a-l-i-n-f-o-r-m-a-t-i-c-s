"""Check exact-count publication, export parity, and unchanged other records."""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = root / 'Informatics/pilot-v1'
baseline = sys.argv[1] if len(sys.argv) > 1 else '6f7d1e21d3f091fe9ffe1dd251a4156e3da74800'
dataset = json.loads((data / 'dataset.json').read_text())
publication = json.loads((data / 'm9-probability.json').read_text())
native = json.loads((data / 'm9.json').read_text())
assert publication == native['probability']
assert publication['method']['gamma_values'] == [9, 3, 3, 9, 6, 6]
assert publication['method']['beta_values'] == [4, 8, 7, 5, 1, 2]
assert publication['method']['beta_start_interval_clock_positions'] == [24, 4]
assert all(publication['method'][k] is False for k in (
    'allow_rotated_gamma_positions', 'allow_reordered_gamma_values',
    'allow_rotated_beta_cycle', 'allow_reversed_beta_cycle'))
assert isinstance(publication['counts']['all_arrangements'], str)
assert Fraction(int(publication['counts']['matching_arrangements']), int(publication['counts']['all_arrangements'])) == Fraction(publication['result']['fraction']) == Fraction(263, 51305410956800)
assert publication['display'] == 'Approximately 1 in 195.1 billion'
assert publication['result']['one_in_approx'] == float(Fraction(51305410956800, 263))
assert publication['interpretation']['selection_adjustment_applied'] is False
publications = {m: json.loads((data / f'm{m}-probability.json').read_text()) for m in (9, 18)}
m18 = publications[18]
assert m18['method']['gamma_values'] == [18, 3, 3, 18, 15, 15]
assert m18['method']['beta_values'] == [4, 8, 16, 14, 10, 2]
assert m18['method']['beta_start_interval_clock_positions'] == [24, 4]
assert Fraction(int(m18['counts']['matching_arrangements']), int(m18['counts']['all_arrangements'])) == Fraction(m18['result']['fraction']) == Fraction(491, 2885929366320000)
assert m18['display'] == 'Approximately 1 in 5.88 trillion'
assert m18['counts']['qualifying_triple_combinations'] == 74
assert m18['result']['one_in_approx'] == float(Fraction(2885929366320000, 491))
assert dataset['probability_scope']['selected_record_ids'] == ['fibonacci-m9-p24', 'fibonacci-m18-p24']
for record in dataset['records']:
    m = record['modulus']
    before = json.loads(subprocess.check_output(['git', 'show', f"{baseline}:Informatics/pilot-v1/m{m}.json"], cwd=root))
    current = json.loads((data / f"m{m}.json").read_text())
    aggregate = dict(current)
    aggregate.pop('research_register')
    assert aggregate == record
    current.pop('beta_m9_view', None)
    current.pop('native_beta_observation', None)
    current.pop('other_geometry_summary', None)
    if m in (9, 18):
        assert current['probability'] == publications[m]
        if m == 9:
            assert current == before, 'M9 changed'
    elif m in (27, 36, 45):
        assert current['probability'] is None
        assert current['probability_scope']['status'] == 'Not selected'
    else:
        assert 'probability' not in current
    for target in (current, before):
        target.pop('probability', None)
        target.pop('probability_scope', None)
    assert current == before, f"M{m}: pre-existing geometry or research data changed"
rows = list(csv.DictReader((data / 'records.csv').open(encoding='utf-8-sig')))
assert len([r for r in rows if r['probability']]) == 2
assert {int(r['modulus']) for r in rows if r['probability_status'] == 'Not selected'} == {27, 36, 45}
row = next(r for r in rows if r['modulus'] == '9')
assert row['probability_exact'] == publication['result']['fraction']
assert row['probability_method'] == publication['method']['id']
assert float(row['probability_one_in']) == publication['result']['one_in_approx']
for m, p in publications.items():
    row = next(r for r in rows if int(r['modulus']) == m)
    assert row['probability_exact'] == p['result']['fraction']
    assert float(row['probability_one_in']) == p['result']['one_in_approx']
    assert row['probability_status'] == 'Calculated'
    assert all(p['method'][k] is False for k in (
        'allow_rotated_gamma_positions', 'allow_reordered_gamma_values',
        'allow_rotated_beta_cycle', 'allow_reversed_beta_cycle'))
    with zipfile.ZipFile(data / 'm1-m48-data.zip') as bundle:
        for name in ('dataset.json', f'm{m}.json', f'm{m}-probability.json', f'm{m}-probability.py', 'records.csv', 'm27.json', 'm36.json', 'm45.json'):
            assert bundle.read(name) == (data / name).read_bytes()
        with tempfile.TemporaryDirectory(prefix='m9-m18-probability-check-') as temp:
            for name in (f'm{m}.json', f'm{m}-probability.py'):
                Path(temp, name).write_bytes(bundle.read(name))
            audit = json.loads(subprocess.check_output([sys.executable, str(Path(temp, f'm{m}-probability.py'))], cwd=temp))
            assert audit['probabilities']['joint_exact_gamma_fixed_beta_phase']['exact_fraction'] == p['result']['fraction']
            assert audit['mathematical_input_sha256'] == p['verification']['mathematical_input_sha256']
            assert audit['verification'] == p['verification']['checks']
    assert (data / f'm{m}-probability.py').read_bytes() == (root / f'research/informatics/probability/m{m}p24_exact_shuffle.py').read_bytes()
print(json.dumps({'passed': True, 'exact_count_checks_per_record': 3, 'original_native_alignment': True,
                  'probability_records': ['M9 P24', 'M18 P24'], 'unscored_family_records': [27,36,45],
                  'existing_geometry_and_research_unchanged': True, 'json_csv_zip_consistent': True,
                  'calculations_run_from_download_bundle': True}, indent=2))
