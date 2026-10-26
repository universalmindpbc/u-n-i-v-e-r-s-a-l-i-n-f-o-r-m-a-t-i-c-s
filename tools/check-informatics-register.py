"""Verify shared literature, export integrity, and unchanged geometry/WSS."""
import argparse
import csv
import io
import json
import subprocess
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', help='Optional Git revision for unchanged-data verification')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
data_dir = root / 'Informatics/pilot-v1'
register = json.loads((root / 'research/informatics/research-register.json').read_text())
papers = register['publications']
ids = [p['id'] for p in papers]
assert len(set(ids)) == len(papers) == 14
assert len({p['doi'] for p in papers}) == 14
assert register['hero_publication_ids'] == ['E03', 'E06', 'E08', 'E11']
assert register['wss']['score'] is None
assert json.loads((data_dir / 'research-register.json').read_text()) == register
js = (root / 'Informatics/science-studies.js').read_text()
browser_register = json.loads(js.split('window.InformaticsResearchRegister = ', 1)[1].split(';\nwindow.InformaticsStudies', 1)[0])
assert browser_register == register
dataset = json.loads((data_dir / 'dataset.json').read_text())
assert dataset['research_register'] == register
assert [r['modulus'] for r in dataset['records']] == list(range(1, 49))
for record in dataset['records']:
    m = record['modulus']
    standalone = json.loads((data_dir / f'm{m}.json').read_text())
    assert standalone.pop('research_register') == register
    assert standalone == record
    refs = record['research']['background_literature']
    assert refs['publication_ids'] == ids
    assert refs['hero_publication_ids'] == register['hero_publication_ids']
    assert refs['relationship'] == 'background_context'
    assert refs['modulus_relevance_status'] == 'Not yet assessed'
    assert record['schema_version'] == '2.2.0'
    assert all(t['wss_score'] is None and t['evidence'] == [] for t in record['research']['tiers'])
    if args.baseline:
        previous = json.loads(subprocess.check_output(['git', 'show', f'{args.baseline}:Informatics/pilot-v1/m{m}.json'], cwd=root))
        current = json.loads(json.dumps(record))
        current['schema_version'] = previous['schema_version']
        del current['research']['background_literature']
        assert current == previous, f'M{m}: existing observation or research status changed'
with zipfile.ZipFile(data_dir / 'm1-m48-data.zip') as bundle:
    assert json.loads(bundle.read('dataset.json')) == dataset
    assert json.loads(bundle.read('research-register.json')) == register
    assert bundle.read('research_publications.csv') == (data_dir / 'research_publications.csv').read_bytes()
    assert bundle.read('records.csv') == (data_dir / 'records.csv').read_bytes()
rows = list(csv.DictReader((data_dir / 'records.csv').open(encoding='utf-8-sig')))
assert len(rows) == 48 and all(json.loads(row['background_publication_ids']) == ids for row in rows)
publications = list(csv.DictReader((data_dir / 'research_publications.csv').open(encoding='utf-8-sig')))
assert [p['id'] for p in publications] == ids
assert [p['id'] for p in publications if p['hero']] == register['hero_publication_ids']
print(json.dumps({'passed': True, 'records': 48, 'publications': 14, 'hero_subset': 4,
                  'unassigned_wss_tiers': 144, 'browser_json_csv_zip_consistent': True,
                  'geometry_unchanged_from': args.baseline}, indent=2))
