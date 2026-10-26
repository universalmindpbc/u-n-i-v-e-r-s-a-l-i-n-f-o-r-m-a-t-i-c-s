#!/usr/bin/env python3
"""Resolve publication identity, retain source provenance, and assemble the public library."""
import json,re,html,hashlib,collections,difflib,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CACHE=Path(os.environ.get('INFORMATICS_SOURCE_CACHE','/private/tmp/informatics-library-audit'));OUT=ROOT/'Informatics/library-data';AUDIT=ROOT/'research/informatics/library'
meta=json.loads((CACHE/'metadata.json').read_text());bydoi={r['doi'].lower():r for r in meta.values() if r.get('doi')};bypmc={r['pmcid']:r for r in meta.values() if r.get('pmcid')}
crossref=json.loads((CACHE/'crossref-by-source.json').read_text()) if (CACHE/'crossref-by-source.json').exists() else {}
clean=lambda t:html.unescape(re.sub(r'</?[A-Za-z][^>]*>','',t or '')).strip()
norm=lambda t:re.sub(r'[^a-z0-9]','',clean(t).lower())
crossref={norm(k):v for k,v in crossref.items()}
crossref_titles={norm(v['metadata']['title'][0]):v['metadata'] for v in crossref.values()}
def resolve(r):
 if r.get('pmid'):return meta.get('pmid:'+r['pmid'])
 if r.get('doi') and r['doi'].lower() in bydoi:return bydoi[r['doi'].lower()]
 for u in r.get('urls',[]):
  m=re.search(r'pubmed.ncbi.nlm.nih.gov/(\d+)',u)
  if m and 'pmid:'+m[1] in meta:return meta['pmid:'+m[1]]
  m=re.search(r'PMC\d+',u)
  if m and m[0] in bypmc:return bypmc[m[0]]
 return None

def applymeta(r,m):
 types=m.get('pubTypeList',{}).get('pubType',[]);joined=' '.join(types).lower();ab=clean(m.get('abstractText',''));mesh=json.dumps(m.get('meshHeadingList',{}))
 r.update({'title':clean(m['title']).rstrip('.'),'pmid':m['id'],'doi':m.get('doi'),'year':int(m.get('pubYear',0)) or None,'published':m.get('firstPublicationDate'),'authors':clean(m.get('authorString','')),'journal':m.get('journalInfo',{}).get('journal',{}).get('title'),'publicationTypes':types,'verifiedAt':'2026-10-21','publicationStatus':'PubMed-indexed publication'})
 r['urls']=list(dict.fromkeys(['https://pubmed.ncbi.nlm.nih.gov/'+m['id']+'/']+(['https://doi.org/'+m['doi']] if m.get('doi') else [])+r.get('urls',[])))
 retracted='retracted publication' in joined or m.get('isRetracted')=='Y'
 r['retracted']=retracted
 r['kind']='Retracted publication' if retracted else 'Protocol' if re.search(r'\bprotocol\b',r['title'],re.I) else 'Review / synthesis' if any(x in joined for x in ['review','meta-analysis']) else 'Journal article'
 # A recommendation for a future randomized trial is not the current study design.
 rct='randomized controlled trial' in joined or bool(re.search(r'\brandomi[sz]ed.{0,65}\btrial',r['title'],re.I))
 methodmatch=re.search(r'<h4>Methods?</h4>(.*?)(?:<h4>|$)',m.get('abstractText',''),re.I|re.S)
 methods=r['title']+' '+(clean(methodmatch[1]) if methodmatch else '')
 double=bool(re.search(r'\bdouble[ -](?:blind|masked)',methods,re.I) or 'Double-Blind Method' in mesh)
 single=bool(re.search(r'\bsingle[ -](?:blind|masked)',methods,re.I) or 'Single-Blind Method' in mesh)
 # Retractions, protocols and non-research editorial material do not receive a trust star.
 eligible=not retracted and r['kind']!='Protocol' and 'Journal Article' in types and not any(x in types for x in ['Editorial','Comment','News','Letter','Published Erratum'])
 r['stars']=(3 if rct and double else 2 if rct and single else 1) if eligible else None
 r['ratingBasis']='Indexed randomized trial and double-blind methods.' if r['stars']==3 else 'Indexed randomized trial and single-blind methods.' if r['stars']==2 else 'Indexed journal paper; blinded-RCT criteria are not confirmed in the retrieved record.' if r['stars']==1 else 'No star rating: protocol, retracted publication or non-research material.'
 r['ratingScope']='Study-design shorthand, not a risk-of-bias or certainty assessment.'
 if m['id']=='17284128':
  r['stars']=2;r['ratingBasis']='Randomized comparison with blind outcome raters. Double blinding applies to the medication/placebo comparison, not the EMDR arm; the psychotherapy comparison is scored for assessor blinding.'
 r['verification']='Publication identity and indexed methods checked; full-paper appraisal pending' if r.get('origin')=='archive' else 'Abstract and publication metadata reviewed; full-paper appraisal pending'
 if retracted:r['verification']='Retracted publication: retained only for provenance; excluded from evidence synthesis'
 return r

def applycrossref(r,m):
 dates=m.get('published',{}).get('date-parts',[[None]])[0]
 r.update({'title':clean(m['title'][0]).rstrip('.'),'doi':m['DOI'],'year':dates[0],'authors':'; '.join(' '.join(filter(None,[a.get('given'),a.get('family')])) for a in m.get('author',[])),'journal':clean(m.get('container-title',[''])[0]),'publicationTypes':[m.get('type','')],'publicationStatus':'Crossref publication identity checked','kind':'Journal article' if m.get('type')=='journal-article' else 'Publication record','verifiedAt':'2026-10-21','stars':None,'ratingBasis':'Bibliographic identity matched in Crossref; research content and blinding require article-level verification.','verification':'Crossref title and DOI checked; article appraisal pending'})
 r['urls']=list(dict.fromkeys(['https://doi.org/'+m['DOI']]+r['urls']))
 if m.get('update-to'):
  r['verification']='Publisher update relationship requires review; excluded from star assignment'
 return r

banners=json.loads((CACHE/'banner-map.json').read_text());keys=json.loads((CACHE/'banner-keys.json').read_text());gradehash={keys[0]:3,keys[1]:2,keys[2]:1};gradepages={}
for k,h in banners.items():
 d,p,name=k.split(':')
 if h in gradehash and int(p)>1:gradepages[(d,int(p))]=gradehash[h]
archive=json.loads((OUT/'archive-records.json').read_text());merged={};ledger=[];mismatches=[]
for r in archive:
 m=resolve(r);cr=crossref.get(norm(r['sourceText']),{}).get('metadata') or crossref_titles.get(norm(r['title']));entry={**r['provenance'][0],'title':r['title'],'originalNote':r['originalNote'],'sourceText':r['sourceText'],'urls':r['urls'][:],'sourceDesign':r['sourceDesign'],'designStars':r['designStars'],'section':r['sourceSection'],'sectionPage':r['sourceSectionPage'],'sectionGrade':gradepages.get((r['collections'][0],r['sourceSectionPage'])),'historicalCitationCount':r['historicalCitationCount']}
 if not m and cr:m=bydoi.get(cr['DOI'].lower())
 r['archiveEntries']=[entry];r['sourceTitle']=r['title'];r['sourceGrade']=entry['sectionGrade']
 match=difflib.SequenceMatcher(None,norm(r['title']),norm(m['title'])).ratio() if m else 0
 if m and match>=.53:
  applymeta(r,m);r['id']='pmid-'+m['id'];ledger.append({'record':r['id'],'source':r['sourceTitle'],'matchedTitle':r['title'],'similarity':round(match,3)})
 elif m:
  conflict={'source':r['sourceTitle'],'metadata':clean(m['title']),'pmid':m['id'],'similarity':round(match,3)}
  mismatches.append(conflict);r['identityConflict']=conflict;r['publicationStatus']='Publication identity pending';r['verification']='Original title and linked publication disagree; reconciliation pending';r['ratingBasis']='Unrated: original citation identity conflict.'
 elif cr and difflib.SequenceMatcher(None,norm(r['title']),norm(cr['title'][0])).ratio()>=.88:
  applycrossref(r,cr);r['id']='doi-'+hashlib.sha256(cr['DOI'].lower().encode()).hexdigest()[:16];ledger.append({'record':r['id'],'source':r['sourceTitle'],'matchedTitle':r['title'],'sourceDatabase':'Crossref'})
 else:r['kind']='Source record';r['publicationStatus']='Publication identity pending'
 key=r['id']
 if key in merged:
  existing=merged[key];existing['archiveEntries']+=r['archiveEntries'];existing['provenance']+=r['provenance']
  for field in ['collections','targets','categories','urls']:existing[field]=list(dict.fromkeys(existing[field]+r[field]))
  existing['priority']=min(existing['priority'],r['priority'])
 else:merged[key]=r

curated=json.loads((AUDIT/'curated-additions.json').read_text())
for c in curated:
 m=resolve(c)
 if not m:raise ValueError('Missing curated metadata '+c['pmid'])
 if m.get('firstPublicationDate','9999')>'2026-10-21':raise ValueError('Future publication '+c['pmid'])
 rid='pmid-'+c['pmid'];old=merged.get(rid,{})
 r={**old,**c,'id':rid,'origin':'update','urls':old.get('urls',[]),'archiveEntries':old.get('archiveEntries',[]),'provenance':old.get('provenance',[]),'priority':0 if any(x in c['categories'] for x in ['Gene expression','Epigenetics','Genetics & genomics']) else 1,'originalNote':old.get('originalNote',''),'sourceDesign':old.get('sourceDesign'),'sourceGrade':old.get('sourceGrade'),'relation':c.get('relation','Intervention-level evidence; specific-word effects not isolated' if c['collections']==['language'] else 'See measured outcome and limitations')}
 r['collections']=list(dict.fromkeys(old.get('collections',[])+c['collections']));applymeta(r,m)
 if r['retracted']:raise ValueError('Curated retraction '+rid)
 merged[rid]=r

# Preserve the portal's existing editorial summaries as source material, separate from fresh appraisal.
portal=json.loads((CACHE/'portal.json').read_text())
for layer in portal['layers']:
 domain={'emdr':'vision','mantra':'language','sound':'sound'}[layer['id']]
 for p in layer['heroStudies']:
  if p.get('internal'):continue
  q={'urls':[p['url']],'doi':p.get('doi')};m=resolve(q)
  rid='pmid-'+m['id'] if m else 'portal-'+hashlib.sha256(p['url'].encode()).hexdigest()[:12]
  note={'title':p['title'],'finding':p['finding'],'limitation':p['limitation'],'url':p['url'],'source':'I AM portal research record','sourceReviewedAt':portal['reviewedAt']}
  if rid in merged:merged[rid].setdefault('portalNotes',[]).append(note);continue
  r={'id':rid,'title':p['title'],'collections':[domain],'origin':'portal','urls':[p['url']],'doi':p.get('doi'),'year':p['year'],'journal':p['journal'],'kind':'Portal source record','portalNotes':[note],'originalNote':'','archiveEntries':[],'provenance':[],'sourceDesign':p['type'],'stars':None,'ratingBasis':'Publication and methods require confirmation.','verification':'Portal curation retained; independent source appraisal pending','categories':['Clinical & self-report'] if domain!='sound' else ['Brain & nervous system','Clinical & self-report'],'targets':[],'priority':1,'measurement':'See the portal source note; molecular outcomes are not inferred.','publicationStatus':'Publication identity pending'}
  if m:applymeta(r,m)
  merged[rid]=r

figures=json.loads((CACHE/'figure-map.json').read_text())
# Exclude legacy promotional artwork and unrelated decorative photographs from the public reader.
exclude={'1f9604e02ab4.jpg','6c457a16b50d.jpg','5eb58a7464bc.png','c620c80d131e.jpg','e9be14c3a2a1.jpg','409af82cb9f0.jpg'}
for domain in ['sound','vision','language']:
 path=OUT/f'{domain}-source.json';data=json.loads(path.read_text())
 for p in data['pages']:
  p['figures']=[f for f in figures.get(f'{domain}:{p["page"]}',[]) if Path(f).name not in exclude]
  p['sectionGrade']=gradepages.get((domain,p['page']))
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
for filename in exclude:
 p=OUT/'figures'/filename
 if p.exists():p.unlink()
# Scientific table and figure captions previously detected alongside the narrow grade banners.
from pypdf import PdfReader
original=Path(os.environ.get('INFORMATICS_PDF_DIR','/Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/universal light/i  a m/a s s e t s/r e s e a r c h'))
for domain,label in [('sound','Sound'),('vision','EMDR')]:
 reader=PdfReader(original/f'Universal Mind [ I o H ] - {label} Medical Research.pdf');path=OUT/f'{domain}-source.json';data=json.loads(path.read_text())
 for key,h in banners.items():
  d,page,name=key.split(':')
  if d!=domain or h in gradehash:continue
  im=reader.pages[int(page)-1].images['/'+name];filename=h+Path(im.name).suffix;(OUT/'figures'/filename).write_bytes(im.data)
  f='library-data/figures/'+filename
  if f not in data['pages'][int(page)-1]['figures']:data['pages'][int(page)-1]['figures'].append(f)
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

records=list(merged.values());records.sort(key=lambda r:(r['priority'],r['origin']!='update',-(r.get('year') or 0),r['title']))
summary={'records':len(records),'sourceEntries':len(archive),'sourcePages':236,'updatedRecords':len(curated),'portalNotes':sum(len(r.get('portalNotes',[])) for r in records),'indexedRecords':sum(bool(r.get('pmid')) for r in records),'retractedRecords':sum(bool(r.get('retracted')) for r in records),'collections':{d:sum(d in r['collections'] for r in records) for d in ['sound','vision','language']},'ratings':dict(collections.Counter(str(r['stars']) for r in records))}
policy={'articleStars':{'3':'Explicitly double-blind randomized controlled trial','2':'Explicitly single-blind randomized controlled trial','1':'Other indexed journal paper with research content','unrated':'Publication/design not confirmed, protocol, editorial material or retracted article'},'originalTopicGrades':{'3':'Source section reports at least three double-blind randomized trials plus supporting peer-reviewed trials','2':'Source section reports at least three randomized trials plus supporting peer-reviewed trials','1':'Source section reports at least three peer-reviewed trials'},'note':'Stars describe study design. They do not measure certainty, risk of bias, clinical relevance or replication. Imported topic grades and design labels are preserved separately. PubMed indexing is not a quality endorsement.','searchScope':'Curated update searched on 21 October 2026, covering publications from 2022 through that date, with selected earlier foundations. This is not a systematic review. Full-paper appraisal remains open.','targetPolicy':'Target tags support discovery. A mention does not establish direct expression measurement or an interaction with OXTR.','sources':'Original curator excerpts and notes are preserved as historical source material; unverified claims are not adopted as current conclusions.'}
(OUT/'library.json').write_text(json.dumps({'schemaVersion':'1.0','reviewedAt':'2026-10-21','summary':summary,'policy':policy,'records':records},ensure_ascii=False,indent=2)+'\n')
(OUT/'search-log.json').write_text((AUDIT/'search-log.json').read_text())
(AUDIT/'publication-identity-audit.json').write_text(json.dumps({'summary':summary,'matched':ledger,'unresolvedTitleMismatches':mismatches,'excludedRecentPapers':[{'pmid':'39240701','reason':'Retracted publication; excluded from the curated update.'}],'excludedLegacyArtwork':sorted(exclude)},ensure_ascii=False,indent=2)+'\n')
# Keep only the metadata used by published records for reproducible audits; abstracts stay in the local discovery cache.
(AUDIT/'verified-publications.json').write_text(json.dumps([{k:r.get(k) for k in ['id','title','pmid','doi','journal','published','publicationTypes','stars','ratingBasis','retracted']} for r in records if r.get('pmid')],ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,indent=2));print('title mismatches',len(mismatches))
