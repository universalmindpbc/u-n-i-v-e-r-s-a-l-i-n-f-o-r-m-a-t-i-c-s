#!/usr/bin/env python3
"""Retrieve API-provided OA XML; retain only texts with a permissive article licence."""
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/informatics/fulltext-access-20261023'
STORE=Path('/Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/i n f o r m a t i c s/research/open-fulltext-20261023')
STORE.mkdir(parents=True,exist_ok=True)
audit=json.loads((OUT/'access-audit.json').read_text())
candidates=[r for r in audit['records'] if r.get('openFulltextApi') and r.get('license')=='cc by']

def retrieve(record):
    out={'id':record['id'],'pmcid':record['pmcid'],'source':record['openFulltextApi']}
    previous=STORE/record['pmcid']/'article.xml'
    for attempt in range(3):
        try:
            if previous.exists():data=previous.read_bytes()
            else:
                with urllib.request.urlopen(urllib.request.Request(record['openFulltextApi'],headers={'User-Agent':'UniversalInformatics-ResearchAccessAudit/1.0'}),timeout=40) as res:data=res.read()
            root=ET.fromstring(data)
            permissions=root.findall('./front/article-meta/permissions')
            license_nodes=[n for p in permissions for n in p.iter() if n.tag.split('}')[-1]=='license']
            links=[v for n in license_nodes for e in n.iter() for k,v in e.attrib.items() if k.endswith('href')]
            license_text=' '.join(' '.join(n.itertext()) for n in license_nodes)
            links += re.findall(r'https?://creativecommons\.org/[^\s<>\)]+',license_text)
            allowed=[u for u in links if re.search(r'creativecommons\.org/licenses/by/(?:[1-4]\.0|2\.5)/?',u)]
            restrictions=bool(re.search(r'non.?commercial|no.?deriv|CC BY.NC|CC BY.ND|does not permit commercial',license_text,re.I))
            if restrictions:return {**out,'status':'license-review-required','licenseLinks':links,'licenseText':license_text}
            explicit_by=bool(re.search(r'Creative Commons Attribution License|Creative Commons Attribution 4\.0|\(CC BY\)',license_text,re.I)) and not re.search(r'non.?commercial|no.?deriv|CC BY.NC|CC BY.ND',license_text,re.I)
            if not allowed and not explicit_by:return {**out,'status':'license-review-required','licenseLinks':links,'licenseText':license_text}
            license_value=allowed[0] if allowed else 'CC BY (version not specified in article)'
            # Verify the returned archive identity before retaining its text.
            article_ids=[n.text for n in root.findall('./front/article-meta/article-id') if n.attrib.get('pub-id-type') in ('pmc','pmcid')]
            if article_ids and not any(str(v).removeprefix('PMC')==record['pmcid'].removeprefix('PMC') for v in article_ids):
                return {**out,'status':'identity-mismatch'}
            body=root.find('body')
            if body is None:return {**out,'status':'no-article-body'}
            sections=[]
            for i,sec in enumerate(body.findall('sec')):
                title=' '.join(sec.find('title').itertext()) if sec.find('title') is not None else 'Section '+str(i+1)
                paragraphs=[' '.join(' '.join(n.itertext()).split()) for n in sec.iter() if n.tag.split('}')[-1] in ('p','table-wrap')]
                sections.append({'id':sec.get('id') or 'section-'+str(i+1),'title':title,'paragraphs':paragraphs})
            if not sections:sections=[{'id':'body','title':'Article body','paragraphs':[' '.join(' '.join(n.itertext()).split()) for n in body.findall('.//p')]}]
            if not any(sec['paragraphs'] for sec in sections):return {**out,'status':'no-body-paragraphs'}
            credit=[' '.join(' '.join(n.itertext()).split()) for n in permissions]
            title=' '.join(root.find('./front/article-meta/title-group/article-title').itertext())
            folder=STORE/record['pmcid'];folder.mkdir(exist_ok=True)
            (folder/'article.xml').write_bytes(data)
            (folder/'passages.json').write_text(json.dumps({'recordId':record['id'],'pmcid':record['pmcid'],'title':title,'source':record['openFulltextApi'],'license':license_value,'copyrightAndLicense':credit,'processing':'Article text extracted from JATS XML; spacing normalized. Original XML retained. Not a scientific appraisal.','sections':sections},ensure_ascii=False,indent=2)+'\n')
            return {**out,'status':'fulltext-cached','license':license_value,'copyrightAndLicense':credit,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'sections':len(sections),'paragraphs':sum(len(s['paragraphs']) for s in sections),'localDirectory':str(folder)}
        except Exception as e:
            if attempt==2:return {**out,'status':'retrieval-error','error':str(e)}
            time.sleep(2*(attempt+1))

results=[]
with cf.ThreadPoolExecutor(max_workers=3) as pool:
    for i,r in enumerate(pool.map(retrieve,candidates)):
        results.append(r)
        if (i+1)%20==0:print(f'Open texts checked {i+1}/{len(candidates)}',flush=True)
summary=dict(Counter(r['status'] for r in results))
report={'checkedAt':dt.datetime.now(dt.timezone.utc).isoformat(),'method':'Only Europe PMC OA API results identified as CC BY were fetched. Storage requires a CC BY URL or explicit Creative Commons Attribution licence statement in the article permissions and usable body content. Original XML, attribution and paragraph records are retained locally. NC/ND/unspecified terms remain for review. No public full-text publication, embeddings, model training or API upload has occurred.','summary':summary,'records':results}
(OUT/'fulltext-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary),flush=True)
