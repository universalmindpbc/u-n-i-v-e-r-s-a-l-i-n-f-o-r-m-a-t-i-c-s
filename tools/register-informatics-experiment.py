"""Archive a completed, public experiment as an immutable JSON/CSV revision."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('exports', ROOT/'tools/export-informatics.py')
exports = importlib.util.module_from_spec(spec); spec.loader.exec_module(exports)

def register(bundle, revision):
    ident = bundle['experiment_id']
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,80}', ident) or not re.fullmatch(r'r[1-9][0-9]*', revision):
        raise ValueError('Invalid experiment or revision identifier')
    if bundle.get('schema') != 'universal-informatics/experiment/v1' or bundle.get('status') != 'completed':
        raise ValueError('A completed, public experiment record is required')
    for key in ['question', 'methods', 'measurements', 'field_guide', 'source_sha256']:
        if not bundle.get(key): raise ValueError('Missing experiment field: ' + key)
    text = json.dumps({**bundle, 'revision': revision}, indent=2, allow_nan=False) + '\n'
    directory = exports.SITE/'quantum/experiments'/ident/revision
    index = directory.parent.parent/'index.json'
    lockname = 'informatics-registry-' + hashlib.sha256(str(index.resolve()).encode()).hexdigest() + '.lock'
    with (Path(tempfile.gettempdir())/lockname).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists(): raise ValueError('Revision already exists; create a new revision')
        catalogue = json.loads(index.read_text()) if index.exists() else {'schema':'universal-informatics/experiments/v1', 'experiments':[]}
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory.parent, prefix='.register-') as temp:
            prepared = Path(temp)/revision; prepared.mkdir()
            jp, cp = prepared/'experiment.json', prepared/'experiment.csv'
            jp.write_text(text, encoding='utf-8')
            with cp.open('w', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f); w.writerow(['record_id', 'field', 'type', 'value', 'escaped'])
                exports.csv_rows(w, ident, json.loads(text))
            entry={'id':ident,'revision':revision,'title':bundle['question'],
                'json':(directory/'experiment.json').relative_to(exports.SITE).as_posix(),
                'csv':(directory/'experiment.csv').relative_to(exports.SITE).as_posix(),
                'json_sha256':exports.sha(jp),'csv_sha256':exports.sha(cp),'source_sha256':bundle['source_sha256']}
            catalogue['experiments'].append(entry)
            staged_index=Path(temp)/'index.json';staged_index.write_text(json.dumps(catalogue,indent=2)+'\n')
            os.rename(prepared,directory)
            try: os.replace(staged_index,index)
            except Exception:
                shutil.rmtree(directory);raise
            # A process death between the two renames leaves an orphan revision;
            # export validation rejects it rather than publishing partial records.
            return entry

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path); p.add_argument('--revision', required=True)
    a = p.parse_args(); print(json.dumps(register(json.loads(a.bundle.read_text()), a.revision)))
