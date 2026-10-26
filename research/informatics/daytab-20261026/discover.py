"""Read-only primary-source discovery and duplicate screening. No quality scores."""
import concurrent.futures, datetime, hashlib, html, json, re, time, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = Path('/private/tmp/um-daytab-20261026')
CACHE.mkdir(exist_ok=True)
ENDPOINT = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
QUERIES = {
 'auditory': '(TITLE_ABS:"binaural beat" OR TITLE_ABS:"binaural beats" OR TITLE_ABS:isochronic OR TITLE_ABS:"auditory entrainment" OR TITLE_ABS:"brainwave entrainment" OR TITLE_ABS:"brain wave entrainment") AND SRC:MED',
 'emdr-rct': '(TITLE_ABS:EMDR OR TITLE_ABS:"eye movement desensitization" OR TITLE_ABS:"eye movement desensitisation") AND (PUB_TYPE:"Randomized Controlled Trial" OR TITLE_ABS:random*) AND SRC:MED',
 'mantra-rct': '(TITLE_ABS:mantra OR TITLE_ABS:mantram OR TITLE_ABS:"transcendental meditation" OR TITLE_ABS:"Kirtan Kriya") AND (PUB_TYPE:"Randomized Controlled Trial" OR TITLE_ABS:random*) AND SRC:MED',
 'mindfulness-rct': '(TITLE_ABS:mindfulness OR TITLE_ABS:MBSR) AND PUB_TYPE:"Randomized Controlled Trial" AND SRC:MED sort_cited:y',
 'genomics': '(TITLE_ABS:meditation OR TITLE_ABS:mindfulness OR TITLE_ABS:MBSR OR TITLE_ABS:mantra OR TITLE_ABS:EMDR OR TITLE_ABS:"eye movement desensitization" OR TITLE_ABS:binaural OR TITLE_ABS:isochronic) AND (TITLE_ABS:"gene expression" OR TITLE_ABS:transcriptom* OR TITLE_ABS:"RNA sequencing" OR TITLE_ABS:methylation OR TITLE_ABS:epigen* OR TITLE_ABS:genom*) AND SRC:MED',
}

def fetch(query, cursor='*'):
 url=ENDPOINT+'?'+urllib.parse.urlencode({'query':query,'format':'json','resultType':'core','pageSize':1000,'cursorMark':cursor})
 for attempt in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UniversalInformatics-LiteratureAudit/1.0'}),timeout=60) as r:return json.load(r),url
  except Exception:
   if attempt==2:raise
   time.sleep(2)

def collect(item):
 name,query=item
 # Mindfulness uses a declared top-1000 citation-ranked discovery window.
 cache=CACHE/(name+'.json')
 if cache.exists():d=json.loads(cache.read_text());url=ENDPOINT+'?'+urllib.parse.urlencode({'query':query})
 else:
  d,url=fetch(query);cache.write_text(json.dumps(d))
 rs=d.get('resultList',{}).get('result',[])
 return name,rs,{'name':name,'query':query,'url':url,'hits':d['hitCount'],'retrieved':len(rs),'completeQueryRetrieval':len(rs)==d['hitCount'],'sourceSHA256':hashlib.sha256(cache.read_bytes()).hexdigest()}

def normalized(s):return re.sub(r'[^a-z0-9]','',str(s or '').casefold())
existing=json.loads((ROOT/'research/informatics/release-20261026/library-download.json').read_text())['records']
dois={(r.get('doi') or '').lower() for r in existing}-{''}
titles={normalized(r['title']) for r in existing}
pmids=set()
for r in existing:
 if r.get('pmid'):pmids.add(str(r['pmid']))
 for value in [r['id'],*r.get('aliases',[]),*r.get('urls',[])]:
  pmids.update(re.findall(r'(?:pmid-|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)',value))

records={};logs=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for name,rs,log in ex.map(collect,QUERIES.items()):
  logs.append(log);print(json.dumps(log),flush=True)
  for r in rs:
   item=records.setdefault(r['id'],{'metadata':r,'queries':[]})
   item['queries'].append(name)

screen=[]
for pmid,entry in records.items():
 r=entry['metadata'];abstract=html.unescape(re.sub(r'</?[A-Za-z][^>]*>',' ',r.get('abstractText','')))
 existingMatch=pmid in pmids or (r.get('doi') or '').lower() in dois or normalized(r['title']) in titles
 pubtypes=r.get('pubTypeList',{}).get('pubType',[])
 text=r['title']+' '+abstract
 screen.append({'pmid':pmid,'doi':r.get('doi'),'title':r['title'],'year':r.get('pubYear'),'journal':r.get('journalInfo',{}).get('journal',{}).get('title'),'abstract':abstract,'queries':entry['queries'],'alreadyIndexed':existingMatch,'pubTypes':pubtypes,'randomizedPublicationType':'Randomized Controlled Trial' in pubtypes,'randomizationMention':bool(re.search(r'randomi[sz]',text,re.I)),'molecularMention':bool(re.search(r'gene expression|transcriptom|methylation|epigen|RNA.seq',text,re.I)),'blindingMention':bool(re.search(r'blind|mask',abstract,re.I)),'protocolOrReview':bool(re.search(r'protocol|systematic review|meta.analysis',r['title'],re.I)),'pmcid':r.get('pmcid'),'openAccess':r.get('isOpenAccess')=='Y','citations':r.get('citedByCount',0),'source':'https://europepmc.org/article/MED/'+pmid})

(CACHE/'primary-metadata.json').write_text(json.dumps(records))
(HERE/'primary-screening.json').write_text(json.dumps({'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queries':logs,'baselineRecords':len(existing),'records':screen},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'uniqueRetrieved':len(screen),'alreadyIndexed':sum(x['alreadyIndexed'] for x in screen),'newRCTPublicationType':sum(x['randomizedPublicationType'] and not x['alreadyIndexed'] and not x['protocolOrReview'] for x in screen),'newMolecularCandidates':sum(x['molecularMention'] and not x['alreadyIndexed'] for x in screen)}),flush=True)
