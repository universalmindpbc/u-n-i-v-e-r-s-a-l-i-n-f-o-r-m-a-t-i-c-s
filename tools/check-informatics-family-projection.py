"""Independent whole-integer Fibonacci check of all reduced family views."""
import csv
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Informatics/pilot-v1'
BASE = 'e3b51518634ea79579268262245e53f46d7e0803'
audit = json.loads((DATA / 'm9-sequence-family-audit.json').read_text())
reference = audit['reference_cycles']
route = [0, 8, 20, 12, 4, 16, 0]
assert [r['modulus'] for r in audit['records']] == list(range(9, 1002, 9))
for row in audit['records']:
    m, p, repeats = row['modulus'], row['pisano_period'], row['repeat_count']
    assert p == 24 * repeats
    f0, f1, native = 0, 1, []
    for _ in range(p):
        native.append(f0 % m)
        f0, f1 = f1, f0 + f1
    assert (f0 % m, f1 % m) == (0, 1)
    reduced = [v % 9 or 9 for v in native]
    assert reduced == reference['alpha'] * repeats
    nodes = [i for i, value in enumerate(native) if value % 3 == 0]
    assert [reduced[i] for i in nodes] == reference['gamma'] * repeats
    doubled = native + native
    beta = [sum(doubled[a + 1:b]) % m % 9 for a, b in zip(nodes, nodes[1:] + [p])]
    assert beta == reference['beta'] * repeats
    for block in range(repeats):
        assert [reduced[24 * block + i] for i in route] == reference['epsilon']
    assert all(row['checks'].values()) and row['original_alignment_preserved']

dataset = json.loads((DATA / 'dataset.json').read_text())
csv_rows = {int(row['modulus']): row for row in csv.DictReader((DATA / 'records.csv').open(encoding='utf-8-sig'))}
with zipfile.ZipFile(DATA / 'm1-m48-data.zip') as bundle:
    assert json.loads(bundle.read('m9-sequence-family-audit.json')) == audit
    for record in dataset['records']:
        m = record['modulus']
        standalone = json.loads((DATA / f'm{m}.json').read_text())
        before = json.loads(subprocess.check_output(['git', 'show', f'{BASE}:Informatics/pilot-v1/m{m}.json'], cwd=ROOT))
        views = standalone.pop('m9_sequence_views', None)
        assert standalone == before, f'Native M{m} data changed'
        assert views == record.get('m9_sequence_views')
        if m % 9:
            assert views is None
            continue
        assert set(views) == {'alpha', 'beta', 'gamma', 'epsilon'}
        repeats = record['pisano_period'] // 24
        for kind, view in views.items():
            assert view['repeat_count'] == repeats and view['original_alignment_preserved']
            assert view['display_cycle'] == reference[kind]
            expected = reference[kind] * repeats if kind != 'epsilon' else reference[kind][:-1] * repeats + [9]
            assert view['reduced_values'] == expected
            assert json.loads(csv_rows[m][kind + '_m9_values']) == expected
            for fmt in ('svg', 'png'):
                relative = view['reduced_' + fmt].removeprefix('pilot-v1/')
                assert bundle.read(relative) == (DATA / relative).read_bytes()
            if kind != 'epsilon' or record['epsilon'] or record.get('epsilon_path'):
                for fmt in ('svg', 'png'):
                    path = f'Informatics/pilot-v1/views/m{m}/{kind}.{fmt}'
                    assert (ROOT / path).read_bytes() == subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT)
        for block in views['epsilon']['block_traversals']:
            assert [v % 9 or 9 for v in block['native_values']] == block['reduced_values'] == reference['epsilon']
        assert views['epsilon']['native_full_circle_comparison'] == record['epsilon_m9_comparison']
        assert bundle.read(f'm{m}.json') == (DATA / f'm{m}.json').read_bytes()
# M27 demonstrates why the cycle construction is recorded separately.
m27 = next(r for r in dataset['records'] if r['modulus'] == 27)
assert m27['epsilon'] is None
assert m27['epsilon_m9_comparison']['reduced_sequence_identical'] is False
assert m27['m9_sequence_views']['epsilon']['reference_cycle_match'] is True
print(json.dumps({'passed': True, 'family_members': 111, 'current_family_records': 5,
                  'sequences_per_record': 4, 'native_data_and_images_unchanged': True,
                  'epsilon_cycle_and_native_assessments_separate': True,
                  'json_csv_zip_consistent': True}, indent=2))
