"""Extend the approved Atlas to M1001; one independently readable JSON per record.

The manifest indexes all records. The ZIP contains every record, shared evidence,
and CSV. Formation intervals reference the record's Alpha rather than duplicating
its values for every rotation. This representation is losslessly reconstructible.
"""
import argparse, copy, csv, hashlib, json, zipfile
from pathlib import Path
from build_pilot import record, validate
from render_pilot import draw_record
from sync_stellated import summarize
from sync_beta_projection import projection, draw as draw_beta, family_audit
from sync_family_projection import audit, projections, draw as draw_family

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Informatics/pilot-v1'
VERSION = 'm1-m1001-v1'
INTERVAL_RULE = ('For each formation interval, walk clockwise from from_index + 1 '
                 'to to_index, excluding both endpoints and wrapping modulo P. '
                 'Read values from alpha.residues. Endpoints, sum and residue are retained.')

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')

def sha(values):
    return hashlib.sha256(','.join(map(str, values)).encode()).hexdigest()

def signature(d):
    b = d['beta'] or {}
    return [d['modulus'], d['pisano_period'], sha(d['alpha']['residues']),
            sha(d['gamma']['alpha_indices']), sha(b.get('interval_sums', [])),
            sha(b.get('interval_residues', [])),
            sha([i for p in d['gamma']['triangle_paths'] for i in p[:-1]]),
            sha([i for p in d['geometry_summary']['shapes'][1]['alpha_paths'] for i in p[:-1]])]

def compact(d):
    for f in d['formation_catalogue']:
        b = f['beta']
        b.pop('start_interval_values', None)
        for interval in b['intervals']:
            # Prove the compact endpoints recover exactly the removed material.
            lo, hi = interval['from_index'], interval['to_index']
            if hi <= lo: hi += d['pisano_period']
            ids = [i % d['pisano_period'] for i in range(lo + 1, hi)]
            assert ids == interval.pop('interior_indices')
            assert [d['alpha']['residues'][i] for i in ids] == interval.pop('interior_values')
    d['formation_interval_encoding'] = {'version': 'alpha-endpoints-v1', 'rule': INTERVAL_RULE}

def build(render=False):
    register = json.loads((OUT / 'research-register.json').read_text())
    reference = json.loads((OUT / 'm9.json').read_text())['research']['background_literature']
    family = audit()
    beta_audit = family_audit()
    beta_audit['scope'] = 'Arithmetic audit of all 111 multiples of nine in the published M1–M1001 Atlas.'
    write(OUT / 'm9-sequence-family-audit.json', family)
    write(OUT / 'beta-m9-family-audit.json', beta_audit)
    signatures = []
    for m in range(1, 1002):
        path = OUT / f'm{m}.json'
        if m <= 48:
            d = json.loads(path.read_text())
        elif path.exists():
            d = json.loads(path.read_text())
        else:
            d = record(m)
            validate(d)
            d.update(schema_version='2.3.0', dataset_version=VERSION)
            d['research']['background_literature'] = copy.deepcopy(reference)
            d['research_register'] = register
            d['verification']['wolfram_scope'] = 'Pending independent core arithmetic verification.'
            # Drawing produces view metadata used by the projection records.
            draw_record(d, OUT)
            if m % 9 == 0:
                d['other_geometry_summary'] = summarize(d)
                d['beta_m9_view'] = projection(d)
                d['m9_sequence_views'] = projections(d, family)
                draw_beta(d, d['beta_m9_view'], OUT)
                for kind in ('alpha', 'gamma', 'epsilon'):
                    draw_family(d, kind, d['m9_sequence_views'][kind], OUT)
                d['probability'] = None
                d['probability_scope'] = {'status': 'Not selected', 'reason': 'Current native shuffle comparisons are M9 P24 and M18 P24.'}
            compact(d)
        d['family'] = {'m9': m % 9 == 0, 'label': 'M9 family' if m % 9 == 0 else None}
        write(path, d)
        signatures.append(signature(d))
        if m % 25 == 0: print(f'Generated and Python-checked through M{m}', flush=True)
    write(OUT / 'm1001-core-signatures.json', signatures)
    return signatures

def publish():
    index = {'version': VERSION, 'modulus_range': [1, 1001], 'record_count': 1001,
             'm9_family_count': 111, 'records': [],
             'research_register': {'id': 'oxtr-phase1-literature',
                                  'file': 'pilot-v1/research-register.json'}}
    summaries = []
    for m in range(1, 1002):
        path = OUT / f'm{m}.json'
        d = json.loads(path.read_text())
        item = {k: d[k] for k in ('id', 'name', 'modulus', 'pisano_period')}
        item.update(file=f'pilot-v1/m{m}.json', m9_family=m % 9 == 0,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        index['records'].append(item)
        b = d['beta']
        row = {'record_id': d['id'], 'modulus': m, 'pisano_period': d['pisano_period'],
               'm9_family': m % 9 == 0, 'alpha': d['alpha']['display_values'],
               'beta': b['display_cycle'] if b else None, 'gamma': d['gamma']['clockwise_values'],
               'epsilon': d['epsilon']['traversal_values'] if d['epsilon'] else None,
               'native_doubling': b['native_doubling'] if b else None,
               'm9_repeat_count': d.get('m9_sequence_views', {}).get('alpha', {}).get('repeat_count'),
               'python': d['verification']['python'], 'wolfram': d['verification']['wolfram'],
               'wss_score': None, 'research_status': 'Not yet assessed',
               'background_publications': d['research']['background_literature']['publication_ids'],
               'hero_publications': d['research']['background_literature']['hero_publication_ids'],
               'json': f'https://staging.universalmind.tv/informatics/pilot-v1/m{m}.json'}
        for kind in ('alpha', 'beta', 'gamma', 'epsilon'):
            row[kind + '_reduced_m9'] = d.get('m9_sequence_views', {}).get(kind, {}).get('reduced_values')
        for shape in d['geometry_summary']['shapes']:
            row[shape['name'] + '_count'] = shape['count']
        row['stellated_all_alpha'] = d.get('other_geometry_summary', {}).get('all_alpha_count')
        row['stellated_primary_gamma'] = d.get('other_geometry_summary', {}).get('primary_gamma_count')
        summaries.append(row)
    write(OUT / 'index.json', index)
    manifest = {**index, 'schema_version': '3.0.0', 'representation': 'record-manifest',
                'description': 'Complete M1–M1001 index. Resolve each record file for its full native sequences, calculation details, geometry and research context. The data bundle contains all 1001 JSON records.',
                'record_file_base': '../', 'bundle': 'm1-m1001-data.zip',
                'formation_interval_rule': INTERVAL_RULE,
                'research_register': json.loads((OUT / 'research-register.json').read_text()),
                'm9_sequence_family_audit': json.loads((OUT / 'm9-sequence-family-audit.json').read_text())}
    write(OUT / 'm1-m1001.json', manifest)
    legacy = OUT / 'm1-m48-archive.json'
    if not legacy.exists():
        legacy.write_bytes((OUT / 'dataset.json').read_bytes())
    write(OUT / 'dataset.json', manifest)
    with (OUT / 'm1-m1001.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in summaries)
    (OUT / 'M1-M1001-README.txt').write_text(
        'Universal Informatics — M1–M1001\n\n'
        'm1-m1001.json is the complete manifest. Each record file contains full native Alpha, Beta, Gamma, geometry, verification and research metadata. This ZIP includes all 1001 records under pilot-v1/. The first 48 retain their approved mathematical content.\n\n'
        'For new records, redundant formation interior arrays are represented by endpoints into alpha.residues. ' + INTERVAL_RULE + '\n\n'
        'M9 family: 111 multiples of nine through M999. All four reduced sequence views retain original alignment; Epsilon uses the reference route inside each 24-position block. Native full-circle Epsilon qualification is separate.\n\n'
        '14 publications, including four HERO studies, provide shared background context. Modulus-specific WSS and biological significance remain unassessed. Only M9 and M18 have selected native random-shuffle probabilities.\n\n'
        'Images are available through each JSON view URL on the Atlas; this data archive does not duplicate image assets. dataset.json and m1-m1001.json are current manifest entry points. m1-m48-archive.json and m1-m48-data.zip retain the previous M1–M48 snapshot.\n')
    from export_complete_csv import export
    export()
    with zipfile.ZipFile(OUT / 'm1-m1001-data.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for m in range(1, 1002): z.write(OUT / f'm{m}.json', f'pilot-v1/m{m}.json')
        for name in ('dataset.json', 'index.json', 'm1-m1001.json', 'm1-m1001.csv', 'M1-M1001-README.txt', 'M1-M1001-CSV-README.txt', 'm1-m1001-images.csv', 'research-publications.csv', 'research-register.json',
                     'm9-sequence-family-audit.json', 'beta-m9-family-audit.json', 'm1001-core-signatures.json',
                     'm1001-verification.json', 'm9-probability.json', 'm18-probability.json'):
            if (OUT / name).exists(): z.write(OUT / name, 'pilot-v1/' + name)
        for p in sorted((ROOT / 'research/informatics/pilot').glob('*.py')): z.write(p, 'source/' + p.name)
        z.write(OUT / 'explorer.js', 'source/atlas-geometry.js')
    print('Published 1001 record manifest, CSV and complete JSON/source bundle.', flush=True)
    if (OUT / 'oxtr-tfbs-v1/catalogue.json').exists():
        from package_oxtr_catalogue import publish
        publish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish-only', action='store_true')
    args = parser.parse_args()
    if not args.publish_only: build()
    publish()
