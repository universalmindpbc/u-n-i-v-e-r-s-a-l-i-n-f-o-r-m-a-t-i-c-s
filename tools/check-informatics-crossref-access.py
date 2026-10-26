#!/usr/bin/env python3
"""Check unresolved DOI records for publisher-deposited access/licence metadata."""
import concurrent.futures as cf,datetime as dt,difflib,hashlib,html,json,re,time,urllib.parse,urllib.request
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/informatics/fulltext-access-20261023';CACHE=Path('/private/tmp/informatics-crossref-access-20261023');CACHE.mkdir(exist_ok=True)
records=[r for r in json.loads((OUT/'access-audit.json').read_text())['records'] if r['status']=='not-confirmed' and r.get('doi')]
def norm(s):return re.sub('[^a-z0-9]','',html.unescape(s).lower())
def check(r):
 out={'id':r['id'],'doi':r['doi'],'title':r['title']};url='https://api.crossref.org/works/'+urllib.parse.quote(r['doi'],safe='');cache=CACHE/(hashlib.sha256(r['doi'].encode()).hexdigest()+'.json')
 try:
  if cache.exists():d=json.loads(cache.read_text())
  else:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UniversalInformatics-ResearchAccessAudit/1.0'}),timeout=25) as res:d=json.load(res)
   cache.write_text(json.dumps(d))
  m=d['message'];title=' '.join(m.get('title',[]));sim=difflib.SequenceMatcher(None,norm(r['title']),norm(title)).ratio()
  licenses=m.get('license',[]);links=m.get('link',[])
  open_licenses=[x for x in licenses if 'creativecommons.org/' in x.get('URL','').lower() and x.get('start',{}).get('timestamp',0)<=time.time()*1000]
  out.update(status='identity-review' if sim<.75 else 'open-license-reported' if open_licenses else 'no-open-license-in-metadata',matchedTitle=title,titleSimilarity=round(sim,4),licenses=licenses,fullTextLinks=links,metadataSource=url,publisher=m.get('publisher'),landingPage=m.get('resource',{}).get('primary',{}).get('URL'))
 except Exception as e:out.update(status='unresolved',error=str(e),metadataSource=url)
 return out
results=[]
with cf.ThreadPoolExecutor(max_workers=3) as pool:
 for i,r in enumerate(pool.map(check,records)):
  results.append(r)
  if (i+1)%30==0:print(f'Crossref DOI checks {i+1}/{len(records)}',flush=True)
report={'checkedAt':dt.datetime.now(dt.timezone.utc).isoformat(),'method':'Unresolved Europe PMC records with a DOI checked against the Crossref work record. A deposited Creative Commons licence is access evidence requiring publisher/version confirmation, not a verified article download. No public records changed.','summary':dict(Counter(r['status'] for r in results)),'records':results}
(OUT/'crossref-access.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report['summary']),flush=True)
