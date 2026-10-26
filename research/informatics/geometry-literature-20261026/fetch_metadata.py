"""Retrieve public bibliographic metadata; retain the source snapshot."""
import concurrent.futures
import json
import urllib.parse
import urllib.request
import urllib.error
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOIS = ['10.1038/eye.1998.142', '10.1038/nn.2824', '10.1038/s41593-017-0050-8',
        '10.22331/q-2021-10-19-564', '10.1038/s41467-026-76090-6',
        '10.1021/acs.nanolett.6c03195', '10.1038/s41467-024-45584-6',
        '10.1103/PhysRevA.83.042330']

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Universal-Informatics-Research/1.0'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))

def one(doi):
    path = ROOT / ('metadata-' + doi.replace('/', '_') + '.json')
    if path.exists():
        return json.loads(path.read_text())
    crossref = get('https://api.crossref.org/works/' + urllib.parse.quote(doi, safe=''))['message']
    query = urllib.parse.urlencode({'query': 'DOI:' + doi, 'format': 'json', 'resultType': 'core'})
    pmc = get('https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + query)['resultList']['result']
    data = {'doi': doi, 'crossref': crossref, 'europepmc': pmc, 'retrievedAt': '2026-10-26'}
    path.write_text(json.dumps(data, indent=2) + '\n')
    return data

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for d in pool.map(one, DOIS):
            c=d['crossref']
            print(json.dumps({'doi':d['doi'],'title':c.get('title'),'authors':c.get('author'),'date':c.get('published'),'count':c.get('is-referenced-by-count'),'pmc':[{k:r.get(k) for k in ['id','pmcid','doi','abstractText','isOpenAccess']} for r in d['europepmc']]}, ensure_ascii=False))
