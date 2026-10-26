#!/usr/bin/env python3
"""Screen the public catalogue against Europe PMC. Never infer paywalls from misses."""
import concurrent.futures as cf
import datetime as dt
import difflib
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/informatics/fulltext-access-20261023'
CACHE = Path('/private/tmp/informatics-open-access-cache-20261023')
BASE = 'https://www.ebi.ac.uk/europepmc/webservices/rest/'
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
records = json.loads(Path(sys.argv[1]).read_text())['records']

def fetch(query):
    key = hashlib.sha256(query.encode()).hexdigest()
    path = CACHE / (key + '.json')
    if path.exists(): return json.loads(path.read_text())
    url = BASE + 'search?' + urllib.parse.urlencode({'query': query, 'format':'json', 'resultType':'core', 'pageSize':1000})
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'UniversalInformatics-ResearchAccessAudit/1.0'})
            with urllib.request.urlopen(req, timeout=40) as response: data = json.load(response)
            path.write_text(json.dumps(data))
            return data
        except Exception as e:
            if attempt == 2: return {'error':str(e), 'query':query}
            time.sleep(2 * (attempt + 1))

def identifiers(record):
    pmid = record.get('pmid')
    pmcid = None
    for text in [record['id'], *record.get('aliases', []), *record.get('urls', [])]:
        m = re.search(r'(?:^pmid-|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)', text)
        if m and not pmid: pmid = m.group(1)
        m = re.search(r'PMC\d+', text)
        if m: pmcid = m.group()
    return str(pmid) if pmid else None, (record.get('doi') or '').strip().lower(), pmcid

def norm(text): return re.sub(r'[^a-z0-9]+', '', html.unescape(text).lower())
def similarity(a,b): return difflib.SequenceMatcher(None,norm(a),norm(b)).ratio()

queries = []
for r in records:
    pmid,doi,pmcid = identifiers(r)
    parts = ([f'(EXT_ID:{pmid} AND SRC:MED)'] if pmid else []) + ([f'DOI:"{doi}"'] if doi else []) + ([f'PMCID:{pmcid}'] if pmcid else [])
    if parts: queries.append('('+' OR '.join(parts)+')')
chunks = [' OR '.join(queries[i:i+18]) for i in range(0,len(queries),18)]
pool = cf.ThreadPoolExecutor(max_workers=3)
found = []
errors = []
for i,data in enumerate(pool.map(fetch,chunks)):
    found.extend(data.get('resultList',{}).get('result',[]))
    if data.get('error'): errors.append(data)
    if (i+1)%10 == 0: print(f'Identifier batches {i+1}/{len(chunks)}',flush=True)
by_pmid = {r.get('pmid') or r['id']:r for r in found if r.get('source')=='MED'}
by_doi = {(r.get('doi') or '').lower():r for r in found if r.get('doi')}
by_pmc = {r['pmcid']:r for r in found if r.get('pmcid')}
matched = {}
pending = []
for r in records:
    pmid,doi,pmcid = identifiers(r)
    hit = by_pmid.get(pmid) or by_doi.get(doi) or by_pmc.get(pmcid)
    if hit: matched[r['id']] = (hit, 'identifier', similarity(r['title'],hit.get('title','')))
    else: pending.append(r)

def title_lookup(r):
    title = re.sub(r'[^\w\s-]',' ',html.unescape(r['title']))
    title = ' '.join(title.split())
    data = fetch('TITLE:"'+title+'"')
    candidates = data.get('resultList',{}).get('result',[])
    ranked = sorted([(similarity(r['title'],c.get('title','')),c) for c in candidates],key=lambda x:x[0],reverse=True)
    for score,c in ranked:
        year = c.get('pubYear')
        agrees = not (year and r.get('year')) or abs(int(year)-int(r['year']))<=1
        if score>=.94 and agrees: return r['id'],(c,'title-and-year',score),None
    return r['id'],None,data.get('error')

for i,(record_id,hit,error) in enumerate(pool.map(title_lookup,pending)):
    if hit: matched[record_id] = hit
    if error: errors.append({'id':record_id,'error':error})
    if (i+1)%40 == 0: print(f'Title checks {i+1}/{len(pending)}',flush=True)
pool.shutdown()

audit=[]
for r in records:
    item={'id':r['id'],'title':r['title'],'collections':r['collections'],'doi':r.get('doi'),'originalUrls':r.get('urls',[])}
    if r['id'] not in matched:
        item.update(status='not-confirmed',reason='No exact identifier or high-confidence title/year match in Europe PMC. Other repositories and publisher access remain to be checked.')
    else:
        hit,method,score=matched[r['id']]
        links=hit.get('fullTextUrlList',{}).get('fullTextUrl',[])
        free=[x for x in links if x.get('availabilityCode') in ('OA','F') or x.get('availability') in ('Open access','Free')]
        pmc=hit.get('pmcid')
        available=bool(pmc or hit.get('inEPMC')=='Y' or free)
        conflict=method=='identifier' and score<.65
        item.update(status='identity-review' if conflict else 'free-fulltext-reported' if available else 'no-free-fulltext-found',matchedBy=method,titleSimilarity=round(score,4),matchedTitle=hit.get('title'),pmid=hit.get('pmid'),pmcid=pmc,doi=hit.get('doi') or r.get('doi'),isOpenAccess=hit.get('isOpenAccess')=='Y',license=hit.get('license'),fullTextLocations=links,abstractAvailable=bool(hit.get('abstractText')),metadataSource=BASE+'article/'+hit['source']+'/'+hit['id']+'?format=json&resultType=core',reuseStatus='license-review-required')
        if pmc: item['repositoryUrl']='https://pmc.ncbi.nlm.nih.gov/articles/'+pmc+'/'
        if pmc and hit.get('isOpenAccess')=='Y' and not conflict: item['openFulltextApi']=BASE+pmc+'/fullTextXML'
    audit.append(item)
summary={'recordsScreened':len(audit),'matched':len(matched),'statuses':dict(Counter(x['status'] for x in audit)),'openAccessInEuropePMC':sum(x.get('isOpenAccess',False) and x['status']!='identity-review' for x in audit),'withAbstract':sum(x.get('abstractAvailable',False) for x in audit),'errors':len(errors)}
report={'checkedAt':dt.datetime.now(dt.timezone.utc).isoformat(),'catalogueSha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),'method':'All public records checked using DOI, PMID or PMCID, followed by title/year matching when needed. Europe PMC supplies PubMed and repository records plus publisher and Unpaywall full-text links. Free availability is reported by the index; it is not an independent publisher-page content check or a reuse permission. Unmatched records are unresolved, not presumed paywalled. No approved quotations or scores changed.','summary':summary,'errors':errors,'records':audit}
(OUT/'access-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary),flush=True)
