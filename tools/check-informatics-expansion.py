"""Validate all 1001 published records, compact calculations and family views."""
import csv, hashlib, json, subprocess, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/informatics/pilot'))
from build_pilot import validate
DATA = ROOT / 'Informatics/pilot-v1'
index = json.loads((DATA / 'index.json').read_text())
manifest = json.loads((DATA / 'dataset.json').read_text())
assert manifest == json.loads((DATA / 'm1-m1001.json').read_text())
assert [r['modulus'] for r in index['records']] == list(range(1, 1002))
assert len({r['id'] for r in index['records']}) == 1001
assert manifest['records'] == index['records']
assert sum(r['m9_family'] for r in index['records']) == 111
register = json.loads((DATA / 'research-register.json').read_text())
assert len(register['publications']) == 14 and len(register['hero_publication_ids']) == 4
rows = {int(r['modulus']): r for r in csv.DictReader((DATA / 'm1-m1001.csv').open(encoding='utf-8-sig'))}
assert len(rows) == 1001
audit = json.loads((DATA / 'm9-sequence-family-audit.json').read_text())
reference = audit['reference_cycles']
assets = set(); intervals = 0
with zipfile.ZipFile(DATA / 'm1-m1001-data.zip') as z:
    for item in index['records']:
        m = item['modulus']; path = DATA / f'm{m}.json'; raw = path.read_bytes(); d = json.loads(raw)
        assert z.read(f'pilot-v1/m{m}.json') == raw
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        validate(d)
        assert d['family']['m9'] == (m % 9 == 0) == item['m9_family']
        assert d['verification']['python'] == d['verification']['wolfram'] == 'Passed'
        assert d['research_register'] == register
        assert all(t['wss_score'] is None and not t['evidence'] for t in d['research']['tiers'])
        assert d['research']['background_literature']['publication_ids'] == [p['id'] for p in register['publications']]
        assert json.loads(rows[m]['alpha']) == d['alpha']['display_values']
        if m <= 48:
            old = json.loads(subprocess.check_output(['git', 'show', f'e59431f:Informatics/pilot-v1/m{m}.json'], cwd=ROOT))
            for key in ('alpha', 'beta', 'gamma', 'epsilon', 'formation_catalogue', 'geometry_summary', 'probability', 'm9_sequence_views'):
                assert d.get(key) == old.get(key), (m, key)
        else:
            a, p = d['alpha']['residues'], d['pisano_period']
            for f in d['formation_catalogue']:
                for interval in f['beta']['intervals']:
                    lo, hi = interval['from_index'], interval['to_index']
                    if hi <= lo: hi += p
                    total = sum(a[i % p] for i in range(lo + 1, hi))
                    assert total == interval['sum'] and total % m == interval['residue']
                    intervals += 1
        for v in d['views']:
            for ext in ('svg', 'png'): assets.add(v[ext])
        if m % 9 == 0:
            assert set(d['m9_sequence_views']) == {'alpha', 'beta', 'gamma', 'epsilon'}
            repeats = d['pisano_period'] // 24
            for kind, view in d['m9_sequence_views'].items():
                expected = reference[kind] * repeats if kind != 'epsilon' else reference[kind][:-1] * repeats + [9]
                assert view['reduced_values'] == expected
                assert view['display_cycle'] == reference[kind]
                assert view['repeat_count'] == repeats and view['original_alignment_preserved']
                assert json.loads(rows[m][kind + '_reduced_m9']) == expected
                for ext in ('svg', 'png'): assets.add(view['reduced_' + ext])
            for block in d['m9_sequence_views']['epsilon']['block_traversals']:
                assert [d['alpha']['display_values'][i] for i in block['alpha_indices']] == block['native_values']
                assert [v % 9 or 9 for v in block['native_values']] == reference['epsilon']
            assert d['other_geometry_summary']['all_alpha_count'] == d['pisano_period'] // 6
        else: assert not d.get('m9_sequence_views')
        if m not in (9, 18): assert d.get('probability') is None
    assert json.loads(z.read('pilot-v1/dataset.json')) == manifest
for asset in assets:
    file = ROOT / 'Informatics' / asset
    assert file.is_file() and file.stat().st_size > 100, asset
    if file.suffix == '.png': assert file.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    else:
        import xml.etree.ElementTree as ET
        assert ET.parse(file).getroot().tag.endswith('svg')
proof = json.loads((DATA / 'm1001-verification.json').read_text())
assert proof['passed'] and proof['records_checked'] == 1001 and proof['comparisons_passed'] == 8008
result = {'passed': True, 'records': 1001, 'm9_family': 111, 'sequence_views_per_family_member': 4,
          'wolfram_comparisons': 8008, 'reconstructed_formation_intervals': intervals,
          'image_assets': len(assets), 'original_48_mathematics_preserved': True,
          'json_csv_bundle_consistent': True, 'unassigned_wss_tiers': 3003}
print(json.dumps(result, indent=2))
