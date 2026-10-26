#!/usr/bin/env python3
"""Read-only Europe PMC literature discovery and identifier verification; no secrets."""
import urllib.request,urllib.parse,json,time,re,concurrent.futures,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/informatics/library'; CACHE=Path(os.environ.get('INFORMATICS_SOURCE_CACHE','/private/tmp/informatics-library-audit'));CACHE.mkdir(parents=True,exist_ok=True);meta=json.loads((CACHE/'metadata.json').read_text()) if (CACHE/'metadata.json').exists() else {}
ENDPOINT='https://www.ebi.ac.uk/europepmc/webservices/rest/search'
def search(query,size=100):
 url=ENDPOINT+'?'+urllib.parse.urlencode({'query':query,'format':'json','resultType':'core','pageSize':size})
 for attempt in range(3):
  try:
   with urllib.request.urlopen(url,timeout=45) as r:return json.load(r)
  except Exception:
   if attempt==2:raise
   time.sleep(2)
# Replay the recorded query set, retaining its explicit search horizon and discovery limits.
queries=json.loads((OUT/'search-log.json').read_text());log=[]
for item in queries:
 d=search(item['query'],size=1000 if item['name'].startswith('vision-') and item.get('retrieved',0)>100 else 100)
 rs=d.get('resultList',{}).get('result',[])
 (CACHE/(item['name']+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2))
 log.append({**item,'hitCount':d['hitCount'],'retrieved':len(rs)})
 for r in rs:meta['pmid:'+r['id']]=r
 print(item['name'],d['hitCount'],len(rs),flush=True)
records=json.loads((ROOT/'Informatics/library-data/archive-records.json').read_text());terms=set()
for r in records:
 for url in r['urls']:
  m=re.search(r'pubmed.ncbi.nlm.nih.gov/(\d+)',url)
  if m:terms.add('EXT_ID:'+m[1]);continue
  m=re.search(r'PMC\d+',url)
  if m:terms.add('PMCID:'+m[0]);continue
 if r.get('doi'):terms.add('DOI:"'+r['doi']+'"')
terms.update('EXT_ID:'+r['pmid'] for r in json.loads((OUT/'curated-additions.json').read_text()))
terms=sorted(terms)
for i in range(0,len(terms),30):
 d=search('('+' OR '.join(terms[i:i+30])+') AND SRC:MED',size=1000)
 for r in d.get('resultList',{}).get('result',[]):meta['pmid:'+r['id']]=r
 print('archive identifiers',min(i+30,len(terms)),'/',len(terms),flush=True)
# Explicit supplied pathways and papers identified during web searches.
extra=['38040763','42615937','42881739','39004412','40547443','40655918','36332459','41074971','41229901','38362742','39060246','41020112','39680432','35427760','37982547','36267832','37560036','36278141','39098435','38327501','36427257','35329068']
q='('+' OR '.join('EXT_ID:'+x for x in extra)+' OR DOI:"10.1038/s41467-019-08503-8" OR DOI:"10.3389/fnmol.2022.1055344" OR DOI:"10.1038/s41380-025-03038-5" OR DOI:"10.1177/09567976241260247" OR DOI:"10.1002/brb3.70035") AND SRC:MED'
for r in search(q,1000).get('resultList',{}).get('result',[]):meta['pmid:'+r['id']]=r
(CACHE/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
(CACHE/'search-replay-log.json').write_text(json.dumps(log,indent=2)+'\n')
print('metadata records',len(meta),flush=True)
