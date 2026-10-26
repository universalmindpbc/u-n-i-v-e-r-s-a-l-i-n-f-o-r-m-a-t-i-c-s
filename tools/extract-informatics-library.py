#!/usr/bin/env python3
"""Extract original pages, positioned text, links and figures; never execute PDF content."""
import os,json,hashlib
from pathlib import Path
import pdfplumber
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(os.environ.get('INFORMATICS_SOURCE_CACHE','/private/tmp/informatics-library-audit'))
SOURCE=Path(os.environ.get('INFORMATICS_PDF_DIR','/Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/universal light/i  a m/a s s e t s/r e s e a r c h'))
FIGURES=ROOT/'Informatics/library-data/figures'
CACHE.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
banners={};keys=[];figures={};manifest=[]
for domain,label in [('sound','Sound'),('vision','EMDR'),('language','Mantra')]:
 source=SOURCE/f'Universal Mind [ I o H ] - {label} Medical Research.pdf'
 reader=PdfReader(source);pages=[]
 with pdfplumber.open(source) as pdf:
  for i,page in enumerate(pdf.pages):
   pages.append({'page':i+1,'width':page.width,'height':page.height,'text':page.extract_text(x_tolerance=1) or '', 'words':page.extract_words(x_tolerance=1,extra_attrs=['fontname','size']),'links':[{k:v for k,v in a.items() if k in ['uri','top','bottom','x0','x1']} for a in page.hyperlinks]})
   for im in page.images:
    if im['top']<=100 or im['width']<30 or im['height']<20:continue
    try:image=reader.pages[i].images['/'+im['name']]
    except KeyError:continue
    digest=hashlib.sha256(image.data).hexdigest()[:12]
    if im['width']/max(im['height'],1)>5:
     if digest not in keys:keys.append(digest)
     banners[f'{domain}:{i+1}:{im["name"]}']=digest
    elif i>0:
     name=digest+Path(image.name).suffix;(FIGURES/name).write_bytes(image.data)
     figures.setdefault(f'{domain}:{i+1}',[]).append('library-data/figures/'+name)
 (CACHE/f'{domain}-pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2)+'\n')
 manifest.append({'collection':domain,'file':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pages':len(pages),'extraction':'pdfplumber positioned text and hyperlinks; pypdf embedded images','excludedPresentation':'Legacy branding, footers and promotional artwork are removed from the public reader.'})
 print(domain,len(pages),'pages',flush=True)
for name,data in [('banner-map',banners),('banner-keys',keys),('figure-map',figures)]:
 (CACHE/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
(ROOT/'research/informatics/library/source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
