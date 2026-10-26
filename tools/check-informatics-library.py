#!/usr/bin/env python3
"""Check source migration coverage and evidence boundaries before publication."""
import json,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'Informatics/library-data';AUDIT=ROOT/'research/informatics/library'
library=json.loads((DATA/'library.json').read_text());records=library['records'];archive=json.loads((DATA/'archive-records.json').read_text())
assert len({r['id'] for r in records})==len(records)
assert library['summary']['records']==len(records)
originals=collections.Counter((r['collections'][0],r['sourceText']) for r in archive)
migrated=collections.Counter((e['collection'],e['sourceText']) for r in records for e in r.get('archiveEntries',[]))
assert originals==migrated,'Original citation entries lost or duplicated during metadata merging'
for domain,count in [('sound',144),('vision',19),('language',73)]:
 source=json.loads((DATA/f'{domain}-source.json').read_text());audit=json.loads((AUDIT/'extraction-audit.json').read_text())[domain]
 assert len(source['pages'])==count and [p['page'] for p in source['pages']]==list(range(1,count+1))
 assert not audit['unassignedLinks']
 for p in source['pages']:
  assert p['text'].strip()
  assert 'internetofhappiness' not in re.sub(r'\s+','',p['text']).lower()
  for figure in p['figures']:assert (ROOT/'Informatics'/figure).is_file(),figure
for r in records:
 assert r['collections'] and set(r['collections'])<= {'sound','vision','language'}
 assert r['stars'] in [None,1,2,3]
 assert r['urls'] and all(u.startswith(('https://','http://')) for u in r['urls'])
 assert all(1<=p<= {'sound':144,'vision':19,'language':73}[e['collection']] for e in r.get('archiveEntries',[]) for p in e['pages'])
 if r.get('identityConflict') or r.get('retracted'):assert not r['stars']
 if r['stars'] and r['stars']>1:assert r.get('pmid') and 'randomized' in r['ratingBasis'].lower()
 if r['origin']=='update':
  assert all(r.get(k) for k in ['measurement','finding','limitation','design','relation'])
  assert r.get('published','9999')<='2026-10-21'
  assert 'full-paper appraisal pending' in r['verification'].lower()
  assert not re.search(r'\b(?:should|could|would)\b',r['finding']+' '+r['limitation'],re.I)
byid={r['id']:r for r in records}
assert byid['pmid-19995444']['stars']==1,'A recommendation for a future RCT is not a completed RCT'
assert byid['pmid-17284128']['stars']==2,'Medication-arm blinding cannot be transferred to EMDR'
assert 'Gene expression' not in byid['pmid-41974356']['categories'],'IL-6 protein is not IL6 expression'
assert 'Gene expression' in byid['pmid-40749778']['categories'] and 'vision' in byid['pmid-40749778']['collections']
assert not any(r.get('pmid')=='39240701' and r['origin']=='update' for r in records)
for filename in ['library.html','library.js','library-launcher.js']:
 assert not re.search(r'\b(?:should|could|would)\b',(ROOT/'Informatics'/filename).read_text(),re.I),filename
result={'passed':True,'sourcePages':236,'sourceEntriesPreserved':len(archive),'records':len(records),'unassignedSourceLinks':0,'updates':sum(r['origin']=='update' for r in records),'identityConflictRecords':sum(bool(r.get('identityConflict')) for r in records),'checks':['source text retention','every source page and figure','citation link coverage','star and modality boundaries','dated appraisal provenance','retraction exclusion','language preference']}
(AUDIT/'data-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
