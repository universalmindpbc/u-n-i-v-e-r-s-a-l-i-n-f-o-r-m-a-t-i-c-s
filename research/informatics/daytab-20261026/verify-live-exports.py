"""Hash every published export part without storing another local copy."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = 'https://staging.universalmind.tv/informatics/exports/'
manifest = json.loads((ROOT / 'Informatics/exports/manifest.json').read_text())
parts = [part for download in manifest['downloads'].values() for part in download['parts']]

def verify(part):
    digest = hashlib.sha256()
    count = 0
    with subprocess.Popen(['curl', '--fail', '--silent', '--show-error', '--max-time', '180', BASE + part['file']], stdout=subprocess.PIPE) as response:
        while block := response.stdout.read(1024 * 1024):
            count += len(block)
            digest.update(block)
        assert response.wait() == 0, part['file']
    assert digest.hexdigest() == part['sha256'], part['file']
    assert count == (ROOT / 'Informatics/exports' / part['file']).stat().st_size
    print('Verified', part['file'], flush=True)
    return {'file': part['file'], 'bytes': count, 'sha256': digest.hexdigest()}

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    checked = list(pool.map(verify, parts))
report = {'passed': True, 'parts': checked, 'totalBytes': sum(p['bytes'] for p in checked)}
out = HERE / 'staging-qa'
out.mkdir(exist_ok=True)
(out / 'complete-export-verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': True, 'parts': len(checked), 'totalBytes': report['totalBytes']}))
