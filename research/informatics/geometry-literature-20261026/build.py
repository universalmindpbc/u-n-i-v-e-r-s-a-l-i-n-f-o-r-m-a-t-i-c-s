"""Add verified geometry sources to existing reader datasets, without duplicates."""
import copy
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'Informatics/library-data'
read = lambda p: json.loads(p.read_text())
write = lambda p, d: p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
curation = read(HERE / 'curation.json')
stamp = curation['reviewedAt']
records = []
for r in curation['records']:
    meta_path = HERE / ('metadata-' + str(r['doi']).replace('/', '_') + '.json')
    meta = read(meta_path) if meta_path.exists() else {}
    c = meta.get('crossref', {})
    matches = [x for x in meta.get('europepmc', []) if str(x.get('doi', '')).lower() == str(r['doi']).lower()]
    pm = next((x for x in matches if str(x.get('id', '')).isdigit()), {})
    title = c.get('title', [r.get('title')])[0]
    assert title
    dates = c.get('published', {}).get('date-parts', [[]])[0]
    published = '-'.join(str(v).zfill(2 if i else 4) for i, v in enumerate(dates)) if dates else r['published']
    authors = '; '.join(' '.join(filter(None, [a.get('given'), a.get('family')])) for a in c.get('author', [])) or r['authors']
    urls = [r['url']]
    if r['doi']: urls.append('https://doi.org/' + r['doi'])
    if pm: urls.append('https://pubmed.ncbi.nlm.nih.gov/' + pm['id'] + '/')
    if pm.get('pmcid'): urls.append('https://pmc.ncbi.nlm.nih.gov/articles/' + pm['pmcid'] + '/')
    sources = list(dict.fromkeys(urls))
    record = dict(id=r['id'], title=title, doi=r['doi'], pmid=pm.get('id'),
        journal=c.get('container-title', [r.get('journal')])[0], year=dates[0] if dates else r['year'],
        published=published, authors=authors, urls=sources, collections=[r['collection']],
        categories=['Vision geometry', 'Brain & nervous system'] if r['collection']=='vision' else ['Quantum computing', 'Geometry and information'],
        targets=[], origin='geometry-literature-20261026', priority=1, kind=r.get('kind','Journal article'),
        design=r['design'], summary=r['summary'], finding=r['summary'], limitation=r['limitation'],
        citationCount=c.get('is-referenced-by-count'),
        citationSnapshot={'count':c.get('is-referenced-by-count'),'source':'Crossref' if c else 'Not available', 'retrievedAt':stamp},
        studyDesignAssessment={'category':'peer-reviewed' if r['peerReviewed'] else 'preprint' if r.get('kind')=='Preprint' else 'technical-article',
          'peerReviewed':r['peerReviewed'],'basis':r['design'] + '. This is not a clinical efficacy trial.', 'sources':sources, 'reviewedAt':stamp},
        evidence={'evidenceType':r['design'],'evidenceClass':r['evidenceClass'],'measurement':r['mechanism'],
          'relationshipNote':r['researchLead'],'clinicalEvidence':False,
          'reviewDepth':'Publisher or author abstract and bibliographic metadata; selected explanatory sections where available.',
          'hypothesisBoundary':'Connections to M9 P24, Alpha, Beta, Gamma and Epsilon remain research hypotheses. Shared geometry does not establish a shared physical mechanism.'},
        jadad={'status':'not-applicable','score':None,'reason':'Not a human randomised clinical trial.'},
        quotes=[], archiveEntries=[], verifiedAt=stamp,
        sourceProvenance={'origin':'Vision Atlas citations' if r['collection']=='vision' else 'Verified follow-up to Gemini research notes',
          'notionPage':None if r['collection']=='vision' else curation['stimulus'], 'reviewedAt':stamp})
    if 'nature.com' in r['url']:
        record.update(publisherFamily='nature', publisherFamilyLabel='Nature Portfolio')
    records.append(record)

access = read(DATA / 'publication-access.json')
fulltext = read(DATA / 'fulltext-availability.json')
vision = read(DATA / 'vision-expansion.json')
fib = read(DATA / 'fibonacci-expansion.json')
ids = {r['id'] for r in records}
for target, collection in [(vision,'vision'),(fib,'general')]:
    target['records'] = [r for r in target['records'] if r['id'] not in ids] + [r for r in records if r['collections']==[collection]]
    target['reviewedAt'] = stamp
for r in records:
    key = re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', r['journal']).lower())
    a = {'urls':r['urls'],'doi':r['doi'],'pmid':r['pmid'],'publicationKey':'doi:'+r['doi'].lower() if r['doi'] else 'id:'+r['id'], 'journalKey':key,'journalLabel':r['journal']}
    access['records'][r['id']] = a
    if r['collections']==['vision']: vision['publicationAccess'][r['id']] = a
    fulltext['records'][r['id']] = {'status':'curated-record','retrievalAvailable':False,'url':r['urls'][0],
      'checkedAt':stamp,'indexedContent':'Verified metadata and original research summary are indexed. Full article text is not indexed in this update.'}

def paper(r, topic):
    p = copy.deepcopy(r)
    p.update(id=r['collections'][0]+'-'+topic+'-'+r['id'],recordId=r['id'],sourcePages=[])
    return p

def topic(ident, title, context):
    included = [r for r, c in zip(records,curation['records']) if c['topic']==ident]
    count = sum(r['studyDesignAssessment']['peerReviewed'] for r in included)
    return {'id':ident,'title':title,'context':context,'trustScore':1 if count>=3 else None,
      'trustAssessment':{'status':'assessed' if count>=3 else 'below-minimum','peerReviewedArticles':count,
        'basis':'At least three peer-reviewed articles.' if count>=3 else 'Fewer than three peer-reviewed articles; no topic grade.',
        'scope':'Bibliographic threshold only. It does not validate an Atlas hypothesis or imply clinical efficacy.'},
      'papers':[paper(r,ident) for r in included]}

v = read(DATA / 'vision-edition.json')
v['chapters'] = [c for c in v['chapters'] if c['title']!='Geometry of vision']
v['chapters'].append({'number':6,'title':'Geometry of vision','topics':[topic('vision-geometry','Retinal mosaics, cortical maps and visual space',
  'The three source papers cited in Vision Atlas. They examine different levels of visual organisation: retinal anatomy, a model of cortical maps, and human brain activity during gaze. These findings motivate geometric questions; the proposed tetrahedral and M9 P24 connections remain hypotheses.')]})
v['introduction']['paragraphs']=['Vision research includes eye movement desensitization and reprocessing (EMDR), alongside foundational research on retinal organisation, visual maps and spatial coding. Clinical intervention evidence and geometric mechanisms are assessed separately.']
g = read(DATA / 'general-edition.json')
g['chapters'] = [c for c in g['chapters'] if c['title']!='Quantum Computing: geometry, stability and error correction']
existing = next((t for c in g['chapters'] for t in c['topics'] if t['id']=='fibonacci-quantum-information'), None)
if existing is None:
    # Preserve the source topic when rerunning after it moved into the new chapter.
    old = read(DATA / 'general-edition.json')
    existing = next(t for c in old['chapters'] for t in c['topics'] if t['id']=='fibonacci-quantum-information')
for c in g['chapters']: c['topics']=[t for t in c['topics'] if t['id']!='fibonacci-quantum-information']
g['chapters'].append({'number':10,'title':'Quantum Computing: geometry, stability and error correction','topics':[
    existing,
    topic('quantum-tilings','Quasicrystals and Fibonacci tilings','A mathematical connection between tilings and quantum error correction. The research status remains visible alongside the result.'),
    topic('quantum-error-correction','Honeycomb codes and heavy-hex hardware','Lattice layout, measurement schedules and hardware control play different roles. These sources make those mechanisms explicit.'),
    topic('quantum-neutral-atoms-materials','Neutral atoms and geometric quantum materials','Research on Rydberg-atom measurements and molecular kagome lattices provides candidate methods and physical comparisons.'),
    topic('quantum-fractal-codes','Fractal structure and information protection','Code properties provide a testable mathematical starting point for investigating fractal organisation.')
]})
for edition in [v,g]:
    topics=[t for c in edition['chapters'] for t in c['topics']]
    papers=[p for t in topics for p in t['papers']]
    edition.update(topicCount=len(topics),paperCount=len(papers),uniquePaperCount=len({p.get('doi') or p.get('recordId') or p['id'] for p in papers}))
    for c in edition['chapters']:
        for i,t in enumerate(c['topics'],1): t['number']=str(c['number'])+'.'+str(i)
for f,d in [('vision-expansion',vision),('fibonacci-expansion',fib),('publication-access',access),('fulltext-availability',fulltext),('vision-edition',v),('general-edition',g)]: write(DATA/(f+'.json'),d)
write(HERE/'records.json',{'schemaVersion':'2026-10-26','records':records})
print(json.dumps({'added':len(records),'vision':3,'quantum':7,'visionUnique':v['uniquePaperCount'],'generalUnique':g['uniquePaperCount']}))
