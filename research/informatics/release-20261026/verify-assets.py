import concurrent.futures
import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
changed = subprocess.check_output(['git', 'diff', '--name-only', '017f6d1aa1f1688745a9e40c6e0cc421238d19ac', 'HEAD', '--', 'Informatics'], cwd=ROOT, text=True).splitlines()
# Reading pages and archive bytes are covered by verify-crawler.mjs.
paths = [p for p in changed if '/read/' not in p and not p.endswith(('.php', '.htaccess', '.bin'))]
versions = {'Informatics/quantum/biology.js': 'clean-3', 'Informatics/quantum/study.js': 'clean-3',
            'Informatics/quantum/biology.css': 'unified-2', 'Informatics/library.js': '18'}

def check(path):
    version = versions.get(path, 'release-20261026')
    url = 'https://universalmind.tv/' + path.replace('Informatics/', 'informatics/', 1) + '?v=' + version
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'UniversalInformatics-ReleaseVerification/1.0'}), timeout=60) as response:
        body = response.read()
    raw = hashlib.sha256(body).hexdigest()
    if path.endswith('.html'):
        body = re.sub(rb'<script type="module" src="https://static\.cloudflareinsights\.com/beacon\.min\.js/[^<]+</script>\n', b'', body)
    actual = hashlib.sha256(body).hexdigest()
    assert actual == hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), path
    return {'path': path, 'url': url, 'sha256': actual, 'rawSHA256': raw, 'matches': True,
            'normalization': 'Existing analytics script only' if raw != actual else None}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(check, paths))
report = {'passed': True, 'sourceCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'files': records}
(OUT / 'assets-live.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': True, 'assets': len(records)}))
