import urllib.request,urllib.parse,json,concurrent.futures,re,difflib,time,os
from pathlib import Path
P=Path(os.environ.get('INFORMATICS_SOURCE_CACHE','/private/tmp/informatics-library-audit'));R=Path(__file__).resolve().parents[1];data=json.loads((R/'Informatics/library-data/library.json').read_text())['records'];out={};errors=[]
hosts=['sciencedirect.com','academic.oup.com','nature.com','springer.com','tandfonline.com','voices.no','jamanetwork.com','wiley.com','sagepub.com','springerpub.com','cambridge.org','pubs.rsc.org','ieeexplore.ieee.org','psycnet.apa.org','scirp.org','ingentaconnect.com','pnas.org','aps.org','doi.org','jmir.org','mdpi.com','cell.com','plos.org','bmj.com','karger.com','ncbi.nlm.nih.gov','frontiersin.org','acs.org','onlinelibrary.wiley.com','hindawi.com','sciencemag.org','science.org']
norm=lambda s:re.sub('[^a-z0-9]','',s.lower())
records=[r for r in data if not r.get('pmid') and len(r['title'])>30 and len(r['title'])<500 and any(h in urllib.parse.urlsplit(r['urls'][0]).netloc for h in hosts)]
def get(r):
 url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query.title':r['title'],'rows':3,'select':'DOI,title,type,container-title,published,author,update-to'})
 for attempt in range(2):
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'UniversalInformaticsResearchMigration/1.0 (bibliographic verification)'})
   with urllib.request.urlopen(req,timeout=30) as f:d=json.load(f)
   rs=d['message']['items'];ranked=sorted([(difflib.SequenceMatcher(None,norm(r['title']),norm(x.get('title',[''])[0])).ratio(),x) for x in rs],key=lambda x:x[0],reverse=True)
   if ranked and ranked[0][0]>=.88:return r['id'],{'score':round(ranked[0][0],3),'metadata':ranked[0][1]}
   return r['id'],None
  except Exception as e:
   if attempt:errors.append({'id':r['id'],'error':type(e).__name__});return r['id'],None
   time.sleep(1)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for i,(id,m) in enumerate(pool.map(get,records),1):
  if m:out[id]=m
  if i%40==0:print('Crossref',i,'/',len(records),'matched',len(out),flush=True)
(P/'crossref.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));(P/'crossref-errors.json').write_text(json.dumps(errors));print('Crossref complete',len(out),'/',len(records),'errors',len(errors),flush=True)
# Resolve matched DOIs in MED for methods and retraction flags where available.
meta=json.loads((P/'metadata.json').read_text());terms=[x['metadata']['DOI'] for x in out.values()]
for i in range(0,len(terms),35):
 q='('+' OR '.join('DOI:"'+x+'"' for x in terms[i:i+35])+') AND SRC:MED'
 u='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urllib.parse.urlencode({'query':q,'format':'json','resultType':'core','pageSize':1000})
 try:
  with urllib.request.urlopen(u,timeout=45) as f:d=json.load(f)
  for r in d['resultList']['result']:meta['pmid:'+r['id']]=r
 except Exception as e:print('DOI enrichment retry required',i,type(e).__name__,flush=True)
(P/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))

by_source={e['sourceText']:out[r['id']] for r in data if r['id'] in out for e in r.get('archiveEntries',[])}
(P/'crossref-by-source.json').write_text(json.dumps(by_source,ensure_ascii=False,indent=2)+'\n')
