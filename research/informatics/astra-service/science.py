"""Bounded OXTR follow-through. No provider submission or model-written code."""
from __future__ import annotations
import csv,gzip,hashlib,json,os,re,time
from pathlib import Path
from itertools import combinations
import numpy as np
from Bio.Seq import Seq
from scipy.integrate import solve_ivp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
import requests

VERSION="oxtr-follow-through-v1"
VARIANTS=['ordinary','fibonacci','phi','alpha','beta','gamma','epsilon','shuffled']
ROOT=Path(__file__).resolve().parents[3]
WEB=ROOT/'Informatics/quantum/data'
SCI=Path(os.environ.get('INFORMATICS_RESEARCH_ROOT','/Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/i n f o r m a t i c s'))
SOURCE=SCI/'research/quantum-lab/oxtr-qrc-01'
CACHE=Path(os.environ.get('ASTRA_DATA_ROOT','/private/tmp/astra-research-data'))
PAIRS=list(combinations(range(6),2))
BITS=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
OBS=np.column_stack([BITS]+[(BITS[:,i]*BITS[:,j])[:,None] for i,j in PAIRS])
XS=sum(np.eye(64)[np.arange(64)^(1<<i)] for i in range(6))
PHI=(1+5**.5)/2

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');temp.replace(p)
def load(p):return json.loads(Path(p).read_text())
def overlaps(start,end,peaks,pad=0):return any(start<b+pad and end>a-pad for a,b in peaks)

def prepare(progress=lambda *a:None):
 """Freeze chromosome-separated rows before computing any prediction score."""
 CACHE.mkdir(parents=True,exist_ok=True)
 bed=SOURCE/'source/ENCFF651DQL.bed.gz';peaks={}
 with gzip.open(bed,'rt') as f:
  for line in f:
   c,a,b,*_=line.split();peaks.setdefault(c,[]).append((int(a),int(b)))
 for p in peaks.values():p.sort()
 source=SCI/'research/oxtr-tfbs-v1/private-source-cache'
 dna=load(source/'ref-dna-expanded.json')
 specs=[]
 for chrom,split in [('chr1','development'),('chr2','validation')]:
  chosen=[(a,b) for a,b in peaks[chrom] if 10_000_000<a<20_000_000 and b-a<=512][:32]
  if len(chosen)!=32:raise ValueError('Insufficient development peak coverage')
  for a,b in chosen:
   s=((a+b)//2)-128;specs.append({'chrom':chrom,'start':s,'end':s+256,'split':split,'label':1})
   for offset in [1024,2048,4096,8192,16384]:
    c=s+offset
    if not overlaps(c,c+256,peaks[chrom],500) and all(not(r['chrom']==chrom and c<r['end'] and c+256>r['start']) for r in specs):
     specs.append({'chrom':chrom,'start':c,'end':c+256,'split':split,'label':0});break
   else:raise ValueError('No matching background window')
 for start in range(((8740380+255)//256)*256,8779628-255,256):
  specs.append({'chrom':'chr3','start':start,'end':start+256,'split':'OXTR test','label':int(overlaps(start,start+256,peaks['chr3']))})
 # Duplicated/overlapping training observations cannot masquerade as independent samples.
 for chrom in ['chr1','chr2']:
  group=sorted((r for r in specs if r['chrom']==chrom),key=lambda r:r['start'])
  if any(a['end']>b['start'] for a,b in zip(group,group[1:])):raise ValueError('Development windows overlap; revise frozen sampling recipe')
 references={'chr3':dna};sources=[{'kind':'GRCh38 DNA','sha256':digest(source/'ref-dna-expanded.json'),'source':'https://api.genome.ucsc.edu/getData/sequence'}, {'kind':'ENCODE peaks','sha256':digest(bed),'source':'https://www.encodeproject.org/files/ENCFF651DQL/'},{'kind':'JASPAR motif library','sha256':digest(source/'jaspar2026-vertebrates.jaspar'),'source':'https://jaspar.elixir.no/matrix/MA0139.2/'}]
 for chrom in ['chr1','chr2']:
  group=[r for r in specs if r['chrom']==chrom];start=min(r['start'] for r in group);end=max(r['end'] for r in group)
  if end-start>4_000_000:raise ValueError('DNA request exceeds four-million-base bound')
  target=CACHE/f'GRCh38-{chrom}-{start}-{end}.json'
  url=f'https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom};start={start};end={end}'
  if not target.exists():
   progress('evidence',f'Retrieving public reference DNA for {chrom}.')
   r=requests.get(url,timeout=90);r.raise_for_status();value=r.json()
   if value.get('start')!=start or value.get('end')!=end or value.get('genome')!='hg38' or value.get('chrom')!=chrom or len(value.get('dna',''))!=end-start:raise ValueError('Reference coordinates do not match')
   save(target,value)
  references[chrom]=load(target);sources.append({'kind':'GRCh38 DNA '+chrom,'sha256':digest(target),'source':url})
 lines=(source/'jaspar2026-vertebrates.jaspar').read_text().splitlines();at=next(i for i,s in enumerate(lines) if s.startswith('>MA0139.2\t'))
 counts=np.array([[float(v) for v in re.findall(r'\d+',s)] for s in lines[at+1:at+5]])
 pwm=np.log2((counts+.5)/(counts.sum(axis=0)+2)/.25);floor=pwm.min(axis=0).sum();span=pwm.max(axis=0).sum()-floor;width=pwm.shape[1]
 rows=[]
 for row in specs:
  ref=references[row['chrom']];seq=ref['dna'][row['start']-ref['start']:row['end']-ref['start']].upper()
  if len(seq)!=256 or set(seq)-set('ACGT'):raise ValueError('Ambiguous DNA: do not silently change the frozen cohort')
  row={**row,'id':f"{row['chrom']}-{row['start']}-{row['end']}",'dna_sha256':hashlib.sha256(seq.encode()).hexdigest(),**features(seq,pwm,floor,span,width)};rows.append(row)
 spec={'version':VERSION,'analysis_code_sha256':digest(__file__),'assay':'ENCSR932ZMX','factor':'CTCF','cell':'GM23338 iPSCs','assembly':'GRCh38','sampling':'32 peak-centred and 32 nearby non-peak windows per development chromosome; all 152 fixed OXTR windows for evaluation. Non-peak means no peak reported in this assay, not proven absence.','qc':['Antibody characterization exemption','Low read depth','Bottlenecking and library complexity warnings'],'sources':sources,'rows':rows,'selection_frozen_before_scores':True,'model_selection':'Use chr1 for learning, chr2 to choose a fixed candidate, then evaluate chr3/OXTR once. No OXTR labels enter fitting or candidate selection.','primary_measure':'Average precision: how often regions ranked as likely binding regions match the laboratory peak map.','limits':['Small pilot with five OXTR peak regions; descriptive results, not an independent confirmation.','Local background selection and one cell context limit generalisation.','No accessibility-matched negative set or ENCODE blacklist filtering in this pilot.','Additional hardware and population-wide inference are outside this task.']}
 save(CACHE/'cohort.json',spec);return spec

def features(seq,pwm,floor,span,width):
 letters={'A':0,'C':1,'G':2,'T':3};candidates=[]
 for p in range(256-width+1):
  s=seq[p:p+width]
  for strand,text in [(1,s),(-1,str(Seq(s).reverse_complement()))]:
   ix=np.array([letters[c] for c in text]);score=(pwm[ix,np.arange(width)].sum()-floor)/span
   if score>=.7:candidates.append((p,float(score),strand))
 hits=[]
 for h in sorted(candidates,key=lambda x:(-x[1],x[0],-x[2])):
  if all(abs(h[0]-a[0])>=width for a in hits):hits.append(h)
 hits.sort();gaps=np.diff([h[0] for h in hits]);bins=np.array([1,3,6,9,14,22,56]);fibs=np.array([1,2,3,5,8,13,21,34,55]);den=max(1,len(gaps))
 hist=np.histogram(gaps,bins=bins)[0]/den;fh=np.zeros(6);ph=np.zeros(6)
 for gap in gaps:
  b=np.searchsorted(bins,gap,side='right')-1
  if 0<=b<6:fh[b]+=max(0.,1.-np.min(abs(fibs-gap))/2.)/den
 for g1,g2 in zip(gaps,gaps[1:]):
  ratio=max(g1,g2)/min(g1,g2);b=np.searchsorted(bins,min(g1,g2),side='right')-1
  if 0<=b<6:ph[b]+=max(0.,1.-abs(ratio-PHI)/(.1*PHI))/max(1,len(gaps)-1)
 scores=np.zeros(12)
 for p,score,strand in hits:scores[min(11,int((p+width/2)/256*12))]=max(scores[min(11,int((p+width/2)/256*12))],score)
 context=[sum(c in 'GC' for c in seq)/256,seq.count('CG')/255,min(len(hits)/16,1),sum(h[2]==1 for h in hits)/max(1,len(hits)),float(np.mean(gaps)/256) if len(gaps) else 0,max([h[1] for h in hits],default=0)]
 return {'motif_hits':hits,'gaps':gaps.tolist(),'vector':np.r_[scores,hist,context].tolist(),'fibonacci_vector':np.r_[scores,fh,context].tolist(),'phi_vector':np.r_[scores,ph,context].tolist()}

def fit_candidates(x,y,train,validation):
 candidates=[('logistic C='+str(c),make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=20261025))) for c in [.1,1,10]]
 candidates += [('random forest',RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=3,random_state=20261025,n_jobs=1))]
 scored=[]
 for name,model in candidates:
  model.fit(x[train],y[train]);scored.append((float(average_precision_score(y[validation],model.predict_proba(x[validation])[:,1])),name,model))
 return max(scored,key=lambda r:r[0]),[{'name':n,'validation_average_precision':s} for s,n,_ in scored]

def analyse(cohort):
 rows=cohort['rows'];y=np.array([r['label'] for r in rows]);train=np.array([r['split']=='development' for r in rows]);val=np.array([r['split']=='validation' for r in rows]);test=np.array([r['split']=='OXTR test' for r in rows])
 baseline=np.array([r['vector'] for r in rows]);extra=np.array([r['fibonacci_vector'][12:18]+r['phi_vector'][12:18] for r in rows]);results={};predictions=[]
 for name,x in [('ordinary motif and spacing',baseline),('with Fibonacci and Phi Ratio',np.c_[baseline,extra])]:
  chosen,scores=fit_candidates(x,y,train,val);v,modelname,model=chosen;p=model.predict_proba(x[test])[:,1]
  results[name]={'selected_model':modelname,'validation_candidates':scores,'validation_average_precision':v,'OXTR_average_precision':float(average_precision_score(y[test],p)),'OXTR_roc_auc':float(roc_auc_score(y[test],p))}
  predictions.append(p)
 # Resample contiguous genomic blocks, not repeated shots or individual neighbours.
 groups=np.array([(r['start']-8740380)//4096 for r in np.array(rows,dtype=object)[test]]);rng=np.random.default_rng(20261025);diff=[]
 for _ in range(1000):
  take=np.concatenate([np.where(groups==g)[0] for g in rng.choice(np.unique(groups),len(np.unique(groups)),replace=True)])
  if len(np.unique(y[test][take]))<2:continue
  diff.append(average_precision_score(y[test][take],predictions[1][take])-average_precision_score(y[test][take],predictions[0][take]))
 intervals=[float(v) for v in np.quantile(diff,[.025,.975])]
 delta=results['with Fibonacci and Phi Ratio']['OXTR_average_precision']-results['ordinary motif and spacing']['OXTR_average_precision']
 counts={s:{'windows':sum(r['split']==s for r in rows),'peak_windows':sum(r['split']==s and r['label']==1 for r in rows)} for s in ['development','validation','OXTR test']}
 table=[{'sample':r['id'],'start':r['start'],'end':r['end'],'reported_CTCF_peak':r['label'],'ordinary_score':float(a),'Fibonacci_Phi_score':float(b),'motif_hits':len(r['motif_hits']),'fibonacci_gap_signal':sum(r['fibonacci_vector'][12:18]),'phi_gap_signal':sum(r['phi_vector'][12:18])} for r,a,b in zip([r for r in rows if r['split']=='OXTR test'],*predictions)]
 return {'version':VERSION,'counts':counts,'models':results,'difference_average_precision':delta,'block_bootstrap_95_interval':intervals,'bootstrap_valid_replicates':len(diff),'verdict':'exploratory_increase' if intervals[0]>0 else 'exploratory_decrease' if intervals[1]<0 else 'inconclusive','positive_result_rule':'The added features improve average precision and the block-resampled interval remains above zero. This small pilot still needs a fresh independent dataset.','negative_result_rule':'An interval below zero is discouraging for this recipe; an interval spanning zero leaves benefit unresolved.','test_prevalence':float(y[test].mean()),'predictions':table,'limits':cohort['limits'],'quantum_prediction_status':'Not evaluated: the two DNA samples with saved Aquila measurements do not overlap the five CTCF peak regions. They cannot train or test a binding predictor.'}

def prior_measurements():
 d=load(WEB/'oxtr-qrc-qualification.json');m=lambda r,k:float(np.mean(r[k][:6]));rows=[]
 for r in d['records']:
  base=next(x for x in d['records'] if x['window']==r['window'] and x['variant']=='ordinary')
  rows.append({'sample':r['window'],'representation':r['variant'],'simulation_mean':m(r,'gpu_observables'),'Aquila_mean':m(r,'hardware_observables'),'reference_mean':m(r,'ideal_observables'),'Aquila_change_percentage_points':100*(m(r,'hardware_observables')-m(base,'hardware_observables'))})
 wf=load(SOURCE/'results/wolfram-comparison.json');physical=load(SOURCE/'results/physical-model.json')
 if not wf.get('passed') or wf.get('dimension')!=[64,64] or not wf.get('hermitian'):raise ValueError('Saved Wolfram template verification failed')
 for r in d['records']:
  for field in ['gpu_observables','hardware_observables','ideal_observables']:
   if len(r[field])!=21 or any(not 0<=v<=1 for v in r[field]):raise ValueError('Invalid recorded observable')
 return {'sha256':digest(WEB/'oxtr-qrc-qualification.json'),'wolfram':{'mode':'cached_template_verification','program':wf['program_id'],'passed':True,'maximum_probability_error':wf['maximum_probability_error'],'receipt_sha256':digest(SOURCE/'results/wolfram-comparison.json'),'physical_model_sha256':digest(SOURCE/'results/physical-model.json'),'scope':'Saved ordinary-input program; no new remote Wolfram call or new Hamiltonian validation claimed.'},'rows':rows,'new_Aquila_jobs':0,'new_qBraid_jobs':0,'meaning':'Fraction of atoms excited under the programmed pulse; not binding probabilities. All provider measurements here are reused from 25 October 2026.'}

def execute(progress=lambda *a:None):
 progress('evidence','Checking the existing measurements and public binding evidence.')
 prior=prior_measurements();cohort=prepare(progress)
 progress('classical','Learning from chromosome 1, choosing the method on chromosome 2, then testing OXTR on chromosome 3.')
 analysis=analyse(cohort)
 result={'version':VERSION,'cohort':{k:v for k,v in cohort.items() if k!='rows'},'analysis':analysis,'prior_quantum':prior,'spend':{'new_Aquila_USD':0,'new_qBraid_credits':0},'next_action':{'state':'hardware_disabled_by_owner','text':'The agent completed the available classical test. New matched quantum measurements remain unsubmitted under today’s no-Aquila instruction. Larger independent genomic validation remains necessary before claiming a biological benefit.'}}
 save(CACHE/'analysis.json',result)
 return result

if __name__=='__main__':
 result=execute(lambda stage,msg:print(stage,msg,flush=True));print(json.dumps({k:v for k,v in result['analysis'].items() if k!='predictions'},indent=2))
