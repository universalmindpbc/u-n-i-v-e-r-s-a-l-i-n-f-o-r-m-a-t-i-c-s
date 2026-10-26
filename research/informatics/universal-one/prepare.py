"""Build versioned retrieval input from the reader export and rights catalogue.

Usage: python prepare.py RECORDS_JSON CATALOGUE_JSON REPO_ROOT OUTPUT_DIRECTORY
The output contains research text. Keep it outside the public website and Git.
"""
import sys, json, hashlib, pathlib, re, collections, os, subprocess

records_file, catalogue_file, repo, output = map(pathlib.Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('build-routing-catalogue.py')),str(repo)],check=True)
records = json.loads(records_file.read_text())['records']
catalogue = json.loads(catalogue_file.read_text())['records']
version = os.environ.get('UO_DATASET_VERSION', 'research-20261024-v2')
digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
lookup = {}
for record in records:
    for key in [record['id'], *record.get('aliases', []), record.get('doi'), record.get('pmid')]:
        if key: lookup[str(key).lower()] = record['id']
passages, objects, matched = [], [], set()
def add(record_id, kind, section, text, url, rights):
    if not text.strip(): return
    sha = digest(text)
    ident = f'{record_id}:{kind}:{digest(section + text)[:20]}'
    passages.append(dict(id=ident, publication_id=record_id, kind=kind, section=section,
                         text=text, source_url=url, rights=rights, sha256=sha, version=version))

for record in records:
    # Keep the approved source fields intact; no generated replacement summaries.
    parts = [record['title'], record.get('summary', ''), record.get('limitation', ''),
             *[q if isinstance(q,str) else json.dumps(q, ensure_ascii=False) for q in record.get('quotes', [])]]
    if record.get('evidence'): parts.append(json.dumps(record['evidence'], ensure_ascii=False))
    if record.get('studyDesignAssessment'): parts.append(json.dumps(record['studyDesignAssessment'], ensure_ascii=False))
    add(record['id'], 'approved-record', 'Library record', '\n\n'.join(filter(None,parts)),
        next(iter(record.get('urls', [])), ''), {'basis':'Approved library text; quotation attribution retained'})

unmatched = []
for item in catalogue:
    if item['permissionClass'] != 'open-license': continue
    ident = next((lookup[str(item.get(k,'')).lower()] for k in ['recordId','doi','pmid']
                  if str(item.get(k,'')).lower() in lookup), None)
    if not ident: unmatched.append(item['recordId']); continue
    source = json.loads(pathlib.Path(item['passagesPath']).read_text())
    units = []
    for section in source.get('sections', []):
        for i, p in enumerate(section.get('paragraphs', [])):
            units.append((section.get('title') or section.get('id') or 'Article', p if isinstance(p,str) else p['text']))
    for i,p in enumerate(source.get('paragraphs', [])):
        units.append((' / '.join(p.get('sectionPath', [])) if isinstance(p,dict) else f"{item.get('textUnit','paragraph')} {i+1}", p['text'] if isinstance(p,dict) else p))
    # Pack adjacent paragraphs without changing their wording. Very long paragraphs
    # split at sentence/space boundaries. Hashes identify the exact retrieved text.
    packed, section, buf = [], '', ''
    for label, text in units:
        if not isinstance(text,str) or not text.strip(): continue
        if buf and (label != section or len(buf)+len(text)>2800):
            packed.append((section,buf)); buf=''
        section=label or 'Article'
        while len(text)>3200:
            end=text.rfind(' ',0,3000)
            if end<1000: end=3000
            if buf: packed.append((section,buf));buf=''
            packed.append((section,text[:end]));text=text[end:]
        buf += ('\n\n' if buf else '') + text
    if buf: packed.append((section,buf))
    for label,text in packed: add(ident,'full-text',label,text,item['source'],item['rights'])
    matched.add(ident)
    for field,suffix in [('articlePath',pathlib.Path(item['articlePath']).suffix),('passagesPath','.json')]:
        p=pathlib.Path(item[field]); sha=hashlib.sha256(p.read_bytes()).hexdigest()
        objects.append({'path':str(p),'key':f'{version}/{ident}/{sha}{suffix}','sha256':sha,'rights':item['rights']})

passages=list({p['id']:p for p in passages}.values())
atlas=[]
index=json.loads((repo/'Informatics/pilot-v1/index.json').read_text())
for r in index['records']:
    p=repo/'Informatics'/r['file']; data=json.loads(p.read_text())
    atlas.append({'id':r['id'],'modulus':r['modulus'],'data':data,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest=dict(version=version, publications=len(records), fullTextArticles=len(matched), passages=len(passages),
              atlasRecords=len(atlas), withheldManuscripts=sum(r['permissionClass']=='text-mining-manuscript' for r in catalogue),
              unmatched=unmatched, recordsSHA256=hashlib.sha256(records_file.read_bytes()).hexdigest(),
              embeddingModel='text-embedding-3-small',embeddingDimensions=512,
              collections=dict(collections.Counter(c for r in records for c in r['collections'])))
assert len(records)==len({r['id'] for r in records})
assert len(matched)==sum(r['permissionClass']=='open-license' for r in catalogue) and not unmatched
for name,value in [('publications',records),('passages',passages),('atlas',atlas),('objects',objects),('manifest',manifest)]:
    (output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False))
print(json.dumps(manifest,indent=2))
