"""Verify selected identifiers and cache accessible primary full texts for review."""
import concurrent.futures, hashlib, json, re, time, urllib.parse, urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
CACHE=Path('/private/tmp/um-daytab-20261026')
IDS=json.loads((HERE/'selected-pmids.json').read_text())
records={pmid:{**entry['metadata'],'source':'https://europepmc.org/article/MED/'+pmid} for pmid,entry in json.loads((CACHE/'primary-metadata.json').read_text()).items()}
def get(url,path):
 if path.exists():return path.read_bytes()
 for attempt in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UniversalInformatics-LiteratureAudit/1.0'}),timeout=35) as response:data=response.read()
   path.write_bytes(data);return data
  except Exception:
   if attempt==2:raise
   time.sleep(2)
def verify(pmid):
 r=records[pmid];result={'pmid':pmid,'doi':r.get('doi'),'primaryURL':r['source'],'pmidMatches':str(r['id'])==pmid}
 if not r.get('doi'):
  result['doiStatus']='No DOI supplied by the PubMed-indexed primary record'
  return result
 try:
  b=get('https://api.crossref.org/works/'+urllib.parse.quote(r['doi'],safe=''),CACHE/(pmid+'-crossref.json'))
  c=json.loads(b)['message'];result.update({'doiMatches':c['DOI'].lower()==r['doi'].lower(),'crossrefTitle':c.get('title'),'type':c.get('type'),'updateTo':c.get('update-to',[]),'relation':c.get('relation',{}),'crossrefSHA256':hashlib.sha256(b).hexdigest()})
 except Exception as e:result['metadataError']=str(e)
 if r.get('pmcid'):
  try:
   b=get('https://www.ebi.ac.uk/europepmc/webservices/rest/'+r['pmcid']+'/fullTextXML',CACHE/(r['pmcid']+'.xml'))
   result.update({'fullTextCached':True,'pmcid':r['pmcid'],'fullTextSHA256':hashlib.sha256(b).hexdigest()})
  except Exception as e:result['fullTextError']=str(e)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 results=[]
 for r in ex.map(verify,IDS):
  results.append(r);print(json.dumps({k:v for k,v in r.items() if k in ['pmid','doiMatches','fullTextCached','metadataError','fullTextError']}),flush=True)
(HERE/'selected-source-verification.json').write_text(json.dumps(results,indent=2)+'\n')
