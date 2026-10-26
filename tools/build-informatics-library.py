#!/usr/bin/env python3
"""Migrate the supplied dossiers into a traceable, loss-preserving research collection."""
import json,re,hashlib,unicodedata,collections,os
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit,unquote
ROOT=Path(__file__).resolve().parents[1]
AUDIT=Path(os.environ.get('INFORMATICS_SOURCE_CACHE','/private/tmp/informatics-library-audit'))
OUT=ROOT/'Informatics/library-data';OUT.mkdir(parents=True,exist_ok=True)
LABEL={'sound':'Sound','vision':'EMDR','language':'Mantra'}
FIRST={'sound':6,'vision':3,'language':4}

def normal(s): return unicodedata.normalize('NFKC',s).replace('\u00ad','').strip()
def tidy(s): return re.sub(r'\s+',' ',normal(s)).strip()
def lines(page):
 groups=[]
 for w in sorted(page['words'],key=lambda w:(round(w['top'],1),w['x0'])):
  if w['top']>=775:continue
  if groups and abs(groups[-1][0]-w['top'])<2:groups[-1][1].append(w)
  else:groups.append([w['top'],[w]])
 rows=[]
 for y,ws in groups:
  ws.sort(key=lambda w:w['x0']);t=tidy(' '.join(w['text'] for w in ws))
  if any(brand in re.sub(r'\s+','',t).lower() for brand in ['internetofhappiness','theepigeneticsofhappiness']):continue
  rows.append({'text':t,'top':round(y,2),'page':page['page'],'bold':sum(len(w['text']) for w in ws if 'Bold' in w['fontname'])>sum(len(w['text']) for w in ws)*.7,'italic':sum(len(w['text']) for w in ws if 'Italic' in w['fontname'])>sum(len(w['text']) for w in ws)*.5,'size':round(max(w['size'] for w in ws),1)})
 return rows

def canon(url):
 url=unquote(url.strip()).replace(' ','')
 m=re.search(r'(?:pubmed/|pubmed.ncbi.nlm.nih.gov/)(\d+)',url)
 if m:return 'https://pubmed.ncbi.nlm.nih.gov/'+m[1]+'/'
 m=re.search(r'(PMC\d+)',url,re.I)
 if m:return 'https://pmc.ncbi.nlm.nih.gov/articles/'+m[1].upper()+'/'
 try:
  p=urlsplit(url);return urlunsplit((p.scheme.lower(),p.netloc.lower(),p.path.rstrip('/'),p.query,''))
 except ValueError:return url

def doi(url):
 m=re.search(r'(10\.\d{4,9}/[^?#\s]+)',unquote(url),re.I)
 return re.sub(r'/(?:full|abstract|pdf)$','',m[1]).rstrip('/.,') if m else None

TARGETS={
 'OXTR':r'\bOXTR\b|oxytocin receptor','OXT':r'\boxytocin\b|\bOXT\b','DRD2':r'\bDRD2\b|dopamine D2',
 'SLC6A4':r'\bSLC6A4\b|5-HTTLPR|serotonin transporter','NR3C1':r'\bNR3C1\b|glucocorticoid receptor',
 'DOCK2':r'\bDOCK2\b','IL6':r'\bIL[-\s]?6\b|interleukin[-\s]?6','NMDA':r'\bNMDA\b|\bGRIN[12][ABCD]?\b',
 'TERRA':r'\bTERRA\b|telomeric repeat.containing','NOS3':r'\bNOS3\b|\beNOS\b','NAD+':r'\bNAD\+|nicotinamide adenine',
 'Telomerase':r'telomerase|\bTERT\b|\bTERC\b','CD38':r'\bCD38\b','COL1A1':r'\bCOL1A1\b','NOS1':r'\bNOS1\b|\bnNOS\b',
 'FKBP5':r'\bFKBP5\b','BDNF':r'\bBDNF\b|brain.derived neurotrophic','CRHR2':r'\bCRHR2\b','TNF':r'\bTNF\b|tumor necrosis|tumour necrosis','SNCA':r'\bSNCA\b|alpha.synuclein','FOXP2':r'\bFOXP2\b'}
CATEGORIES={
 'Gene expression':r'gene expression|transcriptom|transcriptional|\bmRNA\b|RNA.seq|microRNA|\bmiRNA',
 'Epigenetics':r'epigen|methylation|histone|chromatin',
 'Genetics & genomics':r'\bgenom|\bgenetic|\bgenotype|polymorphism|\bSNP|\ballele|\bDNA\b',
 'Molecular biomarkers':r'cytokine|interleukin|\bIL[-\s]?6\b|cortisol|oxytocin|telomer|inflammat|dopamine|serotonin|\bBDNF\b',
 'Heart rate & autonomic':r'heart rate|\bHRV\b|vagus|vagal|autonomic|blood pressure|baroreflex|parasympathetic',
 'Brain & nervous system':r'\bEEG\b|\bfMRI\b|brain|neural|neuron|cortex|neuroplast|cognitive|cognition',
 'Clinical & self-report':r'PTSD|depression|anxiety|symptom|survey|questionnaire|self.report|pain|mental health|well.being',
 'Methods & context':r'quantum|fibonacci|golden ratio|phi ratio|hypothes|model|blockchain|homeless|policy|statistics'}

def classify(r):
 t=r['title']+' '+r.get('originalNote','')+' '+r.get('finding','')
 r['targets']=[key for key,p in TARGETS.items() if re.search(p,t,re.I)]
 r['categories']=[key for key,p in CATEGORIES.items() if re.search(p,t,re.I)] or ['Methods & context']
 r['priority']=0 if any(c in r['categories'] for c in ['Gene expression','Epigenetics','Genetics & genomics']) else 1
 r['measurement']=r.get('measurement') or 'Not independently appraised; topic tags describe the source text.'

allrecords=[];audit={};sources=[]
for domain,label in LABEL.items():
 pages=json.loads((AUDIT/f'{domain}-pages.json').read_text());sourcepages=[];chunks=[];chunk=[];section='';sectionpage=None
 def finish():
  global chunk
  if chunk:chunks.append((list(chunk),section,sectionpage));chunk=[]
 for p in pages:
  rows=lines(p)
  links=[]
  for link in p['links']:
   u=link.get('uri','')
   if not u or link.get('top',999)>=775 or not u.startswith(('http://','https://')):continue
   links.append({'url':u,'top':link.get('top',0),'bottom':link.get('bottom',0)})
  sourcepages.append({'page':p['page'],'text':'\n'.join(r['text'] for r in rows),'links':list(dict.fromkeys(x['url'] for x in links)),'figures':[]})
  if p['page']<FIRST[domain]:continue
  for row_index,row in enumerate(rows):
   row['links']=[x['url'] for x in links if x['top']-3<=row['top']<=x['bottom']+3]
   # Printed URLs can lack a PDF hyperlink annotation.
   if not row['links']:
    printed=row['text']
    if printed.startswith(('http://','https://')) and printed.endswith('-') and row_index+1<len(rows) and re.fullmatch(r'[A-Za-z0-9/?.=&_%+\-]+',rows[row_index+1]['text']):printed+=rows[row_index+1]['text']
    row['links']=re.findall(r'https?://[^\s]+',printed)
   if row['bold'] and row['size']>=11:
    finish()
    if sectionpage==p['page']:section+=' '+row['text']
    else:section=row['text'];sectionpage=p['page']
    continue
   if re.match(r'^[—–_\-]{2,}$',row['text']):finish();continue
   chunk.append(row)
   # The source often omits divider rules between consecutive citations.
   if re.match(r'^citations\s*:',row['text'],re.I):finish()
 finish()
 records=[]
 for rows,section,sectionpage in chunks:
  urls=list(dict.fromkeys(u for r in rows for u in r.get('links',[]) if not any(x in u.lower() for x in ['internetofhappiness','universalmind.tv','drive.google.com','docs.google.com'])))
  # A cited publication remains a record even when the PDF has no link.
  if not urls and not (any(re.match(r'^citations\s*:',r['text'],re.I) for r in rows) and any(r['bold'] and not r['italic'] for r in rows)):continue
  # Citation chunks retain full context when a bibliography contains several links.
  firstlink=next((j for j,r in enumerate(rows) if r.get('links')),len(rows))
  title_rows=[]
  for row in reversed(rows[:firstlink]):
   is_title=row['bold'] and not row['italic'] and row['size']<11 and not row['text'].startswith(('http','www.'))
   if is_title:title_rows.insert(0,row['text'])
   elif title_rows:break
  bold=title_rows
  title=tidy(' '.join(bold))
  if not title or len(title)>650:
   title=next((r['text'] for r in rows if not r['text'].startswith(('http','www.','citations:'))),'Source bibliography')[:260]
  notes=[];last_note=None
  for row in rows:
   if not row['italic']:last_note=None;continue
   if last_note and row['page']==last_note['page'] and row['top']-last_note['top']<19:notes[-1]+=' '+row['text']
   else:notes.append(row['text'])
   last_note=row
  method=next((r['text'] for r in rows if re.search(r'^(?:(?:semi[- ]|non[- ])?randomi[sz]ed|peer.review|double blind|single blind|systematic review|meta.analysis|clinical trial|observational)',r['text'],re.I)),None)
  citation=next((r['text'] for r in rows if r['text'].lower().startswith('citations:')),None)
  journal=None
  for j,row in enumerate(rows):
   if row.get('links') and j>0 and not rows[j-1]['bold'] and not rows[j-1]['italic'] and not rows[j-1]['text'].startswith(('http','www.')):journal=rows[j-1]['text'];break
  normurls=list(dict.fromkeys(canon(u) for u in urls))
  identity=normurls[0] if normurls else 'title:'+title.casefold();rid='src-'+hashlib.sha256((domain+'|'+identity+'|'+title).encode()).hexdigest()[:12]
  r={'id':rid,'title':title,'collections':[domain],'origin':'archive','urls':normurls,'doi':next((doi(u) for u in urls if doi(u)),None),'journal':journal,'year':None,'kind':'Source record' if len(urls)<4 else 'Source bibliography','originalNote':'\n\n'.join(notes),'sourceText':'\n'.join(r['text'] for r in rows),'sourceDesign':method,'historicalCitationCount':citation,'sourceSection':section,'sourceSectionPage':sectionpage,'provenance':[{'collection':domain,'pages':sorted(set(r['page'] for r in rows))}],'verification':'Imported source; article appraisal pending','stars':None,'ratingBasis':'Unrated until publication and study design are checked.','designStars':3 if method and 'double blind' in method.lower() else 2 if method and 'single blind' in method.lower() else None,'sourceGrade':None}
  classify(r);records.append(r)
 sourcepath=OUT/f'{domain}-source.json';sourcepath.write_text(json.dumps({'collection':domain,'label':label,'pages':sourcepages},ensure_ascii=False,indent=2)+'\n')
 bodylinks=set(canon(u) for p in sourcepages[FIRST[domain]-1:] for u in p['links'] if not any(x in u.lower() for x in ['internetofhappiness','universalmind.tv','drive.google.com','docs.google.com']))
 used=set(u for r in records for u in r['urls'])
 audit[domain]={'pages':len(pages),'records':len(records),'bodyLinks':len(bodylinks),'recordLinks':len(used),'unassignedLinks':sorted(bodylinks-used)}
 allrecords.extend(records)
 sources.append({'id':domain,'label':label,'pages':len(pages),'source':f'library-data/{domain}-source.json'})
(OUT/'archive-records.json').write_text(json.dumps(allrecords,ensure_ascii=False,indent=2)+'\n')
(ROOT/'research/informatics/library/extraction-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
