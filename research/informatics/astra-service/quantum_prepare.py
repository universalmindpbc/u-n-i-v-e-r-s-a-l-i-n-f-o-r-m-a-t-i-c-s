"""Compile a matched positive/background pair and run CUDA-Q locally. No provider API calls."""
import json,hashlib,os,re
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.integrate import solve_ivp
import cudaq
ROOT=Path(__file__).resolve().parents[3]
SCI=Path(os.environ.get('INFORMATICS_RESEARCH_ROOT','/Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/i n f o r m a t i c s'))
CACHE=Path(os.environ.get('ASTRA_DATA_ROOT','/private/tmp/astra-research-data'))
OUT=CACHE/'prepared-quantum';OUT.mkdir(parents=True,exist_ok=True)
SPEC=json.loads((ROOT/'Informatics/quantum/data/oxtr-qrc-study.json').read_text())
VARIANTS=['ordinary','fibonacci','phi','alpha','beta','gamma','epsilon','shuffled']
PAIRS=list(combinations(range(6),2));BITS=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
OBS=np.column_stack([BITS]+[(BITS[:,i]*BITS[:,j])[:,None] for i,j in PAIRS]);XS=sum(np.eye(64)[np.arange(64)^(1<<i)] for i in range(6))
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')
def sha(data):return hashlib.sha256(data).hexdigest()
def weights():
    a = SPEC['atlas']
    alpha = a['alpha']['residues']
    templates = {'alpha': alpha, 'beta':a['beta']['interval_residues']*4,
                 'gamma':[alpha[i] for path in a['gamma']['triangle_paths'] for i in path],
                 'epsilon':[alpha[i] for i in a['epsilon']['alpha_indices']]}
    result = {}
    for name, values in templates.items():
        v = np.interp(np.linspace(0,len(values)-1,24), np.arange(len(values)), values)
        centered = v-v.mean()
        result[name] = 1+.25*centered/max(abs(centered))
    for name in ['ordinary','fibonacci','phi','shuffled']:
        result[name] = np.ones(24)
    save('weights.json', {k:v.tolist() for k,v in result.items()})
    return result

def schedule(row,variant,w):
    key = variant+'_vector' if variant in ['fibonacci','phi'] else 'vector'
    x=np.array(row[key])
    if variant=='shuffled':
        x=x[np.random.default_rng(20261025).permutation(24)]
    x=np.clip(x*w[variant]/1.25,0,1)
    # Continuous linear interpolation through 24 slot centres. Times in microseconds.
    t=np.round(np.r_[0,.2+(np.arange(24)+.5)*3.6/24,4.],3)
    detuning=np.round(np.r_[4*(2*x[0]-1),4*(2*x-1),4*(2*x[-1]-1)]*1e6/.2)*.2/1e6
    return t,detuning

def reference(t,d,interaction):
    knots=np.unique(np.r_[t,.2,3.8])
    psi=np.zeros(64,dtype=complex);psi[0]=1
    def rhs(s,y):
        omega=np.interp(s,[0,.2,3.8,4],[0,2,2,0])
        delta=np.interp(s,t,d)
        return -1j*(omega/2*(XS@y)+(interaction-delta*BITS.sum(axis=1))*y)
    for left,right in zip(knots,knots[1:]):
        sol=solve_ivp(rhs,[left,right],psi,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.02)
        assert sol.success
        psi=sol.y[:,-1]
    assert abs(np.vdot(psi,psi)-1)<1e-8
    return psi

def steps(t,d,dt):
    knots=np.unique(np.r_[t,.2,3.8]);result=[]
    for left,right in zip(knots,knots[1:]):
        count=int(np.ceil((right-left)/dt));delta=(right-left)/count
        for i in range(count):
            mid=left+(i+.5)*delta
            omega=np.interp(mid,[0,.2,3.8,4],[0,2,2,0])
            result.append((delta,omega,np.interp(mid,t,d)))
    return result

def approximate(schedule,interaction):
    psi=np.zeros(64,dtype=complex);psi[0]=1
    for dt,o,d in schedule:
        c,s=np.cos(o*dt/4),-1j*np.sin(o*dt/4)
        for i in range(6):psi=c*psi+s*psi[np.arange(64)^(1<<i)]
        psi*=np.exp(-1j*(interaction-d*BITS.sum(axis=1))*dt)
        for i in range(6):psi=c*psi+s*psi[np.arange(64)^(1<<i)]
    return psi

def kernel(schedule,coupling):
    k=cudaq.make_kernel();q=k.qalloc(6);prev=0.
    for dt,o,d in schedule:
        half=o*dt/2
        for i in range(6):k.rx(float(prev+half),q[i])
        for i in range(6):k.r1(float(d*dt),q[i])
        for v,(i,j) in zip(coupling,PAIRS):k.cr1(float(-v*dt),q[i],q[j])
        prev=half
    for i in range(6):k.rx(float(prev),q[i])
    return k

def qasm_state(program):
    psi=np.zeros(64,dtype=complex);psi[0]=1;count=0
    for line in program.splitlines():
        line=line.strip()
        if not line or line.startswith(('//','OPENQASM','include','qreg')):continue
        m=re.fullmatch(r'(rx|rz)\(([^)]+)\) var0\[(\d)\];',line)
        if m:
            gate,angle,i=m.groups();a=float(angle);i=int(i)
            if gate=='rx':psi=np.cos(a/2)*psi-1j*np.sin(a/2)*psi[np.arange(64)^(1<<i)]
            else:psi*=np.exp(-1j*a/2*(1-2*BITS[:,i]))
        else:
            m=re.fullmatch(r'cx var0\[(\d)\], var0\[(\d)\];',line)
            if not m:raise RuntimeError('Unvalidated exported instruction: '+line)
            i,j=map(int,m.groups());indices=np.arange(64)^((BITS[:,i].astype(int))<<j);psi=psi[indices]
        count+=1
    return psi,count
def main():
 cohort=json.loads((CACHE/'cohort.json').read_text());datahash=sha((CACHE/'cohort.json').read_bytes());codehash=sha(Path(__file__).read_bytes())
 templatehash=sha((ROOT/'Informatics/quantum/data/oxtr-qrc-study.json').read_bytes());physicalpath=SCI/'research/quantum-lab/oxtr-qrc-01/results/physical-model.json';physicalhash=sha(physicalpath.read_bytes())
 target=OUT/'manifest.json'
 if target.exists():
  old=json.loads(target.read_text())
  if old.get('cohort_sha256')==datahash and old.get('compiler_sha256')==codehash and old.get('template_sha256')==templatehash and old.get('physical_sha256')==physicalhash and all((OUT/(r['id']+'.qasm')).exists() and sha((OUT/(r['id']+'.qasm')).read_bytes())==r['qasm_sha256'] for r in old['cases']):
   print('Reused verified local CUDA-Q calculations.',flush=True);return
 test=[r for r in cohort['rows'] if r['split']=='OXTR test'];positive=next(r for r in test if r['label']==1)
 negative=next(r for r in test if r['label']==0 and r['start']>=positive['end']+512)
 chosen=[positive,negative];physical=json.loads((SCI/'research/quantum-lab/oxtr-qrc-01/results/physical-model.json').read_text())
 p=physical['positions_m'];coupling=np.array(physical['coupling_rad_per_us']);interaction=sum(v*BITS[:,i]*BITS[:,j] for v,(i,j) in zip(coupling,PAIRS));w=weights();cudaq.set_target('qpp-cpu');records=[]
 for row in chosen:
  for variant in VARIANTS:
   name=row['id']+'-'+variant;t,d=schedule(row,variant,w);psi=reference(t,d,interaction);target_probs=abs(psi)**2
   for dt in [.01,.005,.0025]:
    ss=steps(t,d,dt);approx=approximate(ss,interaction);err=float(np.max(abs((abs(approx)**2-target_probs)@OBS)))
    if err<.0001:break
   if err>=.001:raise ValueError('Numerical convergence failed')
   k=kernel(ss,coupling);state=np.asarray(cudaq.get_state(k));qasm=cudaq.translate(k,format='openqasm2');exported,gates=qasm_state(qasm)
   export_error=float(np.max(abs((abs(exported)**2-target_probs)@OBS)));implementation_error=float(np.max(abs(abs(state)**2-abs(approx)**2)))
   if export_error>=.001 or implementation_error>=1e-9:raise ValueError('CUDA-Q numerical check failed')
   (OUT/(name+'.qasm')).write_text(qasm)
   ahs={'braketSchemaHeader':{'name':'braket.ir.ahs.program','version':'1'},'setup':{'ahs_register':{'sites':p,'filling':[1]*6}},'hamiltonian':{'drivingFields':[{'amplitude':{'time_series':{'times':[0,2e-7,3.8e-6,4e-6],'values':[0,2000000,2000000,0]},'pattern':'uniform'},'phase':{'time_series':{'times':[0,4e-6],'values':[0,0]},'pattern':'uniform'},'detuning':{'time_series':{'times':(t*1e-6).tolist(),'values':(d*1e6).tolist()},'pattern':'uniform'}}],'localDetuning':[]}}
   save(name+'.ahs.json',ahs)
   records.append({'id':name,'sample':row['id'],'reported_CTCF_peak':row['label'],'representation':variant,'execution':'cudaq-cpu','reference_observables':(target_probs@OBS).tolist(),'cudaq_observables':((abs(state)**2)@OBS).tolist(),'max_export_observable_error':export_error,'max_implementation_probability_error':implementation_error,'qasm_gate_count':gates,'qasm_sha256':sha(qasm.encode()),'ahs_sha256':sha((OUT/(name+'.ahs.json')).read_bytes()),'pulse':{'times_us':t.tolist(),'detuning_rad_per_us':d.tolist()}})
   print('CUDA-Q CPU checked',name,flush=True)
 save('manifest.json',{'version':'oxtr-matched-pair-preparation-v1','cohort_sha256':datahash,'compiler_sha256':codehash,'template_sha256':templatehash,'physical_sha256':physicalhash,'template_id':SPEC['template']['id'],'physical_model':physical,'cases':records,'scope':'One reported CTCF peak window and one nearby non-peak window, each prepared eight ways. Local numerical preparation only; two samples cannot establish binding-prediction accuracy.','Wolfram':'Reuses the previously checked Hamiltonian family. No fresh Wolfram service evaluation.','qBraid':{'submitted':False,'reason':'No new paid credit allowance; local CUDA-Q execution used.'},'Aquila':{'submitted':False,'reason':'Owner disabled new Aquila submissions today.','requires':'Fresh device validation and explicit allowance before submission.'}})
if __name__=='__main__':main()
