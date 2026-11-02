"""Build one complete JSON and one complete, lossless CSV from published data.

Only tracked public Informatics data enters the export. Credentials, worker
state and local research directories never enter this inventory. gzip is the
transport; the website saves one .json.gz or .csv.gz file for decompression.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'Informatics'
OUT = SITE / 'exports'
sys.path.insert(0, str(ROOT / 'research/informatics/pilot'))
from csv_records import rows

SCHEMA = 'universal-informatics/complete-dataset/v1'
RULE = 'All published experiment revisions enter the complete JSON and CSV. Earlier revisions remain available.'
TEXT_DATA = {'.pdb', '.xyz', '.cube', '.fasta', '.fa', '.wl'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def inventory():
    tracked = subprocess.check_output(['git', 'ls-files', '-z', 'Informatics'], cwd=ROOT).decode().split('\0')
    sources, assets = [], []
    for name in sorted(filter(None, tracked)):
        p = ROOT / name
        rel = p.relative_to(SITE).as_posix()
        # Staging review packages are pending proposals, not approved research.
        # Keep them out of the public corpus and all complete-dataset consumers.
        if rel.startswith(('exports/', 'review/')):
            continue
        # These CSV files are exact alternate serialisations of the JSON files
        # already included. The full record values remain in both exports.
        duplicate_csv = '/csv-records/' in rel or '/csv-shared/' in rel
        if rel.endswith(('.json', '.json.gz')) or (p.suffix == '.csv' and not duplicate_csv) or p.suffix in TEXT_DATA:
            fmt = 'json' if rel.endswith(('.json', '.json.gz')) else 'csv' if p.suffix == '.csv' else 'text'
            sources.append({'id': rel, 'source_path': rel, 'sha256': sha(p), 'format': fmt})
        elif p.suffix in {'.bcif', '.png', '.jpg', '.jpeg', '.svg', '.pdf'}:
            assets.append({'source_path': rel, 'sha256': sha(p), 'url': 'https://universalmind.tv/informatics/' + rel})
    digest = hashlib.sha256(json.dumps({'sources': sources, 'assets': assets}, sort_keys=True).encode()).hexdigest()
    return {'schema': SCHEMA, 'dataset_version': digest, 'sources': sources, 'assets': assets}


def record(source):
    p = SITE / source['source_path']
    if source['format'] == 'json':
        raw = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
        data = json.loads(raw)
    elif source['format'] == 'csv':
        with p.open(encoding='utf-8-sig', newline='') as f:
            data = list(csv.reader(f))
    else:
        data = p.read_text()
    return {**source, 'data': data}


def safe_value(value):
    # Reversible spreadsheet formula protection. escaped=1 means remove exactly
    # the first apostrophe when reconstructing the original value.
    escape = bool(value) and ((bool(value.lstrip()) and value.lstrip()[0] in '=+-@') or value[0] in '\t\r\n')
    return ("'" + value, 1) if escape else (value, 0)


def csv_rows(writer, ident, data):
    for field, kind, value, _ in rows(data):
        value, escaped = safe_value(value)
        writer.writerow([ident, field, kind, value, escaped])


def compressed_text(path):
    return io.TextIOWrapper(gzip.GzipFile(filename='', mode='wb', fileobj=path.open('wb'), mtime=0), encoding='utf-8', newline='')


def transport_parts(path):
    parts = []
    with path.open('rb') as f:
        while block := f.read(48 * 1024 * 1024):
            digest = hashlib.sha256(block).hexdigest()
            target = OUT / ('part-' + digest + '.bin')
            target.write_bytes(block)
            parts.append({'file': target.name, 'bytes': len(block), 'sha256': digest})
    return parts


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = inventory()
    metadata = {k: v for k, v in manifest.items() if k != 'sources'}
    metadata.update(title='Universal Informatics complete public dataset', source_count=len(manifest['sources']),
                    experiment_rule=RULE,
                    scope='All published structured datasets and scientific text files. Images, molecular binary structures and PDFs retain file links and hashes; website code and media playback are not data records.')
    jpath, cpath = OUT / 'informatics-complete.json.gz', OUT / 'informatics-complete.csv.gz'
    with compressed_text(jpath) as jf, compressed_text(cpath) as cf:
        writer = csv.writer(cf)
        writer.writerow(['record_id', 'field', 'type', 'value', 'escaped'])
        csv_rows(writer, '__metadata__', metadata)
        jf.write('{"metadata":')
        json.dump(metadata, jf, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        jf.write(',"records":[')
        for i, source in enumerate(manifest['sources']):
            item = record(source)
            if i: jf.write(',')
            json.dump(item, jf, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            csv_rows(writer, source['id'], item)
        jf.write(']}\n')
    manifest['downloads'] = {}
    for fmt, p in [('json', jpath), ('csv', cpath)]:
        h = hashlib.sha256(); size = 0
        with gzip.open(p, 'rb') as f:
            while chunk := f.read(1024 * 1024): h.update(chunk); size += len(chunk)
        manifest['downloads'][fmt] = {'file': p.name, 'bytes': p.stat().st_size,
                                      'uncompressed_bytes': size, 'sha256': sha(p), 'uncompressed_sha256': h.hexdigest(), 'parts': transport_parts(p)}
        p.unlink()
    text=json.dumps(manifest, indent=2) + '\n'
    (OUT / 'manifest.json').write_text(text)
    (OUT / ('manifest-' + manifest['dataset_version'] + '.json')).write_text(text)
    print(json.dumps({'sources': len(manifest['sources']), 'downloads': manifest['downloads']}))


def check():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    current = inventory()
    assert (OUT / ('manifest-' + manifest['dataset_version'] + '.json')).read_bytes() == (OUT / 'manifest.json').read_bytes(), 'Versioned manifest differs from the current dataset'
    if any(manifest[k] != current[k] for k in current):
        raise ValueError('Complete exports are stale. Run python3 tools/export-informatics.py, validate, and commit the updated exports.')
    for item in manifest['downloads'].values():
        combined = hashlib.sha256()
        for part in item['parts']:
            p = OUT / part['file']
            if sha(p) != part['sha256']: raise ValueError('Export checksum mismatch: ' + str(p))
            with p.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''): combined.update(block)
        if combined.hexdigest() != item['sha256']: raise ValueError('Combined export checksum mismatch')
    verify_experiments()
    print(json.dumps({'passed': True, 'sources': len(current['sources']), 'version': current['dataset_version']}))


def verify_experiments():
    index = json.loads((SITE / 'quantum/experiments/index.json').read_text())
    for item in index['experiments']:
        p = SITE / item['json']
        c = SITE / item['csv']
        if sha(p) != item['json_sha256'] or sha(c) != item['csv_sha256']:
            raise ValueError('Experiment revision changed without a new registered revision: ' + item['id'])
    registered = {i['json'] for i in index['experiments']}
    for p in (SITE / 'quantum/experiments').glob('*/r*/experiment.json'):
        if p.relative_to(SITE).as_posix() not in registered:
            raise ValueError('Unregistered experiment revision: ' + str(p))
    # Each completed currently displayed study must point to its archived data.
    for p in (SITE / 'quantum/data').glob('*.json'):
        data = json.loads(p.read_text())
        if data.get('schema') == 'universal-informatics/qualification-results/v1':
            matches = [x for x in index['experiments'] if x['source_sha256'] == sha(p)]
            if not matches: raise ValueError('Completed measurements need a JSON and CSV experiment revision: ' + p.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    check() if args.check else build()
