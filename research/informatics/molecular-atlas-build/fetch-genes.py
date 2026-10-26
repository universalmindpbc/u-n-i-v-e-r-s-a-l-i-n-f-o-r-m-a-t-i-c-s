"""Build the Atlas gene catalogue from Ensembl annotation and UCSC GRCh38 DNA."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip, hashlib, json, urllib.request, time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Informatics/molecular/data/genes'
CACHE=ROOT/'research/informatics/molecular-atlas-build/gene-sources'
OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
SYMBOLS=['OXTR','SLC6A4','DRD2','GRIN1','GRIN2B','IL6','NR3C1','DOCK2','CD38']
def get(url):
 for attempt in range(3):
  try:
   req=urllib.request.Request(url,headers={'Accept':'application/json','User-Agent':'UniversalInformatics/1.0'})
   with urllib.request.urlopen(req,timeout=45) as r:return json.load(r)
  except Exception:
   if attempt==2:raise
   time.sleep(1)
def save(name,data):
 with gzip.open(CACHE/name,'wt') as f:json.dump(data,f)
def one(symbol):
 lookup='https://rest.ensembl.org/lookup/symbol/homo_sapiens/'+symbol+'?expand=1&content-type=application/json'
 g=get(lookup);assert g['display_name']==symbol and g['assembly_name']=='GRCh38' and g['biotype']=='protein_coding'
 canonical=g['canonical_transcript'].split('.')[0]
 t=next(t for t in g['Transcript'] if t['id']==canonical)
 chrom=g['seq_region_name'];start=g['start'];end=g['end'];strand=g['strand']
 source=f'https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr{chrom};start={start-1};end={end}'
 dna=get(source);assert dna['start']==start-1 and dna['end']==end
 seq=dna['dna'].upper();assert len(seq)==end-start+1 and set(seq)<=set('ACGTN')
 if strand==-1:seq=seq.translate(str.maketrans('ACGTN','TGCAN'))[::-1]
 sequrl='https://rest.ensembl.org/sequence/id/'+g['id']+'?type=genomic&content-type=application/json'
 independent=get(sequrl);assert independent['seq'].upper()==seq, symbol+' sequence source mismatch'
 exons=sorted(t['Exon'],key=lambda e:e['start'],reverse=strand==-1)
 def offset(p):return p-start if strand==1 else end-p
 row={'symbol':symbol,'ensembl':g['id'],'version':g['version'],'assembly':'GRCh38','chromosome':chrom,'start':start,'end':end,'strand':strand,'length':len(seq),'canonicalTranscript':g['canonical_transcript'],'transcriptStart':t['start'],'transcriptEnd':t['end'],'tss':t['start'] if strand==1 else t['end'],'exons':[{'number':i+1,'id':e['id'],'start':e['start'],'end':e['end'],'offset':offset(e['start'] if strand==1 else e['end']),'length':e['end']-e['start']+1} for i,e in enumerate(exons)],'sha256':hashlib.sha256(seq.encode()).hexdigest(),'retrieved':datetime.now(timezone.utc).isoformat(),'annotationSource':lookup,'sequenceSource':source,'crossCheckSource':sequrl,'sequenceCrossCheck':'Exact agreement between UCSC hg38 and Ensembl GRCh38 genomic sequence','geneSource':'https://www.ensembl.org/Homo_sapiens/Gene/Summary?g='+g['id'],'sequenceOrientation':'Gene strand, 5-prime to 3-prime; reverse complement of reference plus strand for minus-strand genes','coordinateConvention':'1-based, inclusive genomic coordinates'}
 assert all(0<=e['offset']<len(seq) and e['offset']+e['length']<=len(seq) for e in row['exons'])
 (OUT/(symbol.lower()+'.json')).write_text(json.dumps({**row,'sequence':seq},separators=(',',':'))+'\n')
 save(symbol.lower()+'-annotation.json.gz',g);save(symbol.lower()+'-dna.json.gz',dna);save(symbol.lower()+'-crosscheck.json.gz',independent)
 print(symbol,len(seq),'bp',len(exons),'exons; cross-check passed',flush=True)
 return row
with ThreadPoolExecutor(3) as pool:rows=list(pool.map(one,SYMBOLS))
(OUT/'index.json').write_text(json.dumps({'schema':'universal-informatics/gene-reference/v1','genes':rows},indent=2)+'\n')
