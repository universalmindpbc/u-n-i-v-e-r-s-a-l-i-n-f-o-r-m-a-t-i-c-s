#!/usr/bin/env python3
"""Retrieve indexed papers through the documented PMC Cloud Service, by version."""
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/informatics/fulltext-expansion-20261023'
STORE = Path('/Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/i n f o r m a t i c s/research')
CACHE = Path('/private/tmp/informatics-pmc-cloud-20261023')
BASE = 'https://pmc-oa-opendata.s3.amazonaws.com/'
for p in (OUT,CACHE): p.mkdir(parents=True,exist_ok=True)

def fetch(url):
    p=CACHE/hashlib.sha256(url.encode()).hexdigest()
    if p.exists(): return p.read_bytes()
    req=urllib.request.Request(url,headers={'User-Agent':'UniversalInformatics-ResearchAccessAudit/1.0'})
    with urllib.request.urlopen(req,timeout=35) as response: data=response.read()
    p.write_bytes(data)
    return data

def text(node): return ' '.join(' '.join(node.itertext()).split()) if node is not None else ''

def retrieve(r):
    out={'id':r['id'],'pmcid':r['pmcid'],'title':r['title']}
    try:
        listing=ET.fromstring(fetch(BASE+'?'+urllib.parse.urlencode({'list-type':'2','prefix':r['pmcid']+'.','max-keys':1000})))
        keys=[x.text for x in listing.findall('.//{*}Key') if x.text.endswith('.json')]
        versions=[json.loads(fetch(BASE+k)) for k in keys]
        out['versions']=[{k:m.get(k) for k in ('version','title','doi','pmid','license_code','is_manuscript','is_retracted')} for m in versions]
        allowed=[m for m in versions if not m.get('is_retracted') and (str(m.get('license_code','')).lower().replace(' ','-') in ('cc-by','cc0') or (m.get('is_manuscript') and m.get('license_code')=='TDM'))]
        if not allowed:return {**out,'status':'rights-review-required' if versions else 'not-in-cloud-dataset'}
        allowed.sort(key=lambda m:(m.get('license_code')=='TDM',m.get('is_manuscript',False),-m.get('version',0)))
        m=allowed[0]
        expected=(r.get('doi') or '').lower(); actual=(m.get('doi') or '').lower()
        if expected and actual and expected!=actual:return {**out,'status':'identity-review-required','cloudDOI':actual}
        url=m['xml_url'].replace('s3://pmc-oa-opendata/',BASE)
        data=fetch(url)
        checksum=urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get('md5',[None])[0]
        if checksum and hashlib.md5(data).hexdigest()!=checksum:raise ValueError('PMC metadata checksum mismatch')
        article=ET.fromstring(data)
        for element in article.iter():element.tag=element.tag.split('}')[-1]
        body=article.find('body')
        if body is None:return {**out,'status':'no-body'}
        sections=[]
        # Every paragraph occurs once, even inside nested sections.
        parent={child:node for node in body.iter() for child in node}
        for i,p in enumerate(body.iter('p')):
            node=p; headings=[]
            while node in parent:
                node=parent[node]
                if node.tag=='sec':headings.append(text(node.find('title')))
            value=text(p)
            if value:sections.append({'id':p.get('id') or f'paragraph-{i+1}','sectionPath':list(reversed(headings)),'text':value})
        if sum(len(p['text']) for p in sections)<1000:return {**out,'status':'body-too-short-for-automatic-retention'}
        rights=[text(p) for p in article.findall('./front/article-meta/permissions')]
        tdm=m['license_code']=='TDM'
        rights_text=' '.join(rights)
        if not tdm and re.search(r'non.?commercial|does not permit commercial|CC BY.NC',rights_text,re.I):return {**out,'status':'conflicting-license-review-required','rights':rights}
        folder=STORE/('text-mining-manuscripts-20261023' if tdm else 'open-fulltext-20261023')/r['pmcid']
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'article.xml').write_bytes(data)
        (folder/'cloud-metadata.json').write_text(json.dumps(m,indent=2)+'\n')
        policy={'license':m['license_code'],'use':'Private text mining' if tdm else 'Attribution-preserving reuse','publicFulltextRedistribution':False,'providerUpload':False,'modelTraining':False,'basis':'https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/','articleRights':rights}
        (folder/'passages.json').write_text(json.dumps({'recordId':r['id'],'pmcid':r['pmcid'],'version':m['version'],'title':m['title'],'source':url,'rights':policy,'paragraphs':sections},ensure_ascii=False,indent=2)+'\n')
        return {**out,'status':'text-mining-manuscript-cached' if tdm else 'open-fulltext-cached','version':m['version'],'license':m['license_code'],'source':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'paragraphs':len(sections),'localDirectory':str(folder),'rights':policy}
    except Exception as e:return {**out,'status':'retrieval-error','error':str(e)}

if __name__=='__main__':
    audit=json.loads((ROOT/'research/informatics/fulltext-access-20261023/access-audit.json').read_text())['records']
    baseline=json.loads((ROOT/'research/informatics/fulltext-access-20261023/fulltext-manifest.json').read_text())['records']
    done={r['pmcid'] for r in baseline if r['status']=='fulltext-cached'}
    if len(sys.argv)>1:audit+=json.loads(Path(sys.argv[1]).read_text())['records']
    candidates={r['pmcid']:r for r in audit if r.get('pmcid') and r['pmcid'] not in done}
    rows=[]
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        for i,row in enumerate(pool.map(retrieve,candidates.values())):
            rows.append(row)
            if (i+1)%10==0: print(f'PMC cloud checks {i+1}/{len(candidates)}',flush=True)
    report={'checkedAt':dt.datetime.now(dt.timezone.utc).isoformat(),'baselineRetained':115,'method':'Version-specific official PMC Cloud metadata and JATS. CC BY/CC0 and author manuscripts explicitly tagged TDM are retained separately. NC/ND/unknown licences are held for review. Retractions, identifier conflicts and missing bodies are excluded. TDM manuscripts remain private; no full-text redistribution, provider upload, embedding or model training performed.','summary':dict(Counter(r['status'] for r in rows)),'records':rows}
    (OUT/'pmc-cloud-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['summary']),flush=True)
