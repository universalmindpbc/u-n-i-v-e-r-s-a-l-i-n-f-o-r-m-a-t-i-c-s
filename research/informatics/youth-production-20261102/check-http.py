"""Verify exact approved public bytes after every upload has completed."""
import concurrent.futures, hashlib, json, re, sys, urllib.request, urllib.error
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / 'research/informatics/youth-production-20261102'
SOURCE = json.loads((OUT / 'source-equivalence.json').read_text())
ORIGIN = 'https://universalmind.tv'
ROOT_FILES = {'llms.txt', 'read/index.html', 'read/resources.json', 'sitemaps/informatics-publications.xml'}

def check(item):
    path = item['path']
    url = ORIGIN + '/' + path.replace('Informatics/', 'informatics/', 1)
    if path == 'Informatics/library.html':
        url += '?embedded=1'
    elif path == 'Informatics/library.css':
        url += '?v=reading-background-20261102'
    request = urllib.request.Request(url, headers={'User-Agent': 'UniversalMind-Staging-Verification/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = response.read()
            final_url, status = response.url, response.status
    except urllib.error.HTTPError as error:
        if path.startswith('Informatics/review/') and error.code == 403:
            rule = (ROOT / 'Informatics/review/.htaccess').read_text()
            assert 'RewriteRule ^ - [F,L]' in rule and 'staging' in rule
            return {'path': path, 'url': url, 'status': 403, 'passed': True,
                    'expectedRestriction': 'Historical candidate review provenance remains staging-only under the unchanged production access rule.'}
        return {'path': path, 'url': url, 'passed': False, 'error': str(error)}
    except Exception as error:
        return {'path': path, 'url': url, 'passed': False, 'error': str(error)}
    analytics = 0
    if hashlib.sha256(data).hexdigest() != item['sha256'] and path.endswith('.html'):
        data, analytics = re.subn(rb'<script type="module" src="https://static\.cloudflareinsights\.com/beacon\.min\.js/[^" ]+" integrity="[^"]+" data-cf-beacon=\'[^\']*\' crossorigin="anonymous"></script>\n', b'', data)
    actual = hashlib.sha256(data).hexdigest()
    return {'path': path, 'url': url, 'finalUrl': final_url, 'status': status,
            'expected': item['sha256'], 'actual': actual, 'bytes': len(data),
            'hostAnalyticsRemoved': analytics, 'passed': status == 200 and actual == item['sha256']}

items = [item for item in SOURCE['files'] if item['path'].startswith('Informatics/') or item['path'] in ROOT_FILES]
previous = []
if '--retry-failed' in sys.argv:
    previous = [item for item in json.loads((OUT / 'production-http.json').read_text())['files'] if item['passed']]
    passed_paths = {item['path'] for item in previous}
    items = [item for item in items if item['path'] not in passed_paths]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = sorted(previous + list(pool.map(check, items)), key=lambda item: item['path'])
report = {'passed': all(item['passed'] for item in results), 'origin': ORIGIN,
          'approvedStaging': SOURCE['approvedStaging'], 'files': results}
(OUT / 'production-http.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': report['passed'], 'files': len(results),
                  'mismatches': [item for item in results if not item['passed']]}))
assert report['passed']
