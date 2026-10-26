"""Freeze label-blind OXTR inputs and validate six-site CUDA-Q qualification programs.

No cloud submission occurs here. This batch qualifies the instrument/encoding route;
it is not a fitted occupancy predictor or the held-out biological benchmark.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.integrate import solve_ivp
import cudaq

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent.parent
OUT = ROOT / 'results'
SOURCE = ROOT / 'source'
WEB = Path('/private/tmp/um-quantum-lab-20261025/Informatics/quantum/data')
SPEC = json.loads((WEB / 'oxtr-qrc-study.json').read_text())
VARIANTS = ['ordinary', 'fibonacci', 'phi', 'alpha', 'beta', 'gamma', 'epsilon', 'shuffled']
PAIRS = list(combinations(range(6), 2))
BITS = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(float)
OBS = np.column_stack([BITS] + [(BITS[:, i] * BITS[:, j])[:, None] for i, j in PAIRS])
XS = sum(np.eye(64)[np.arange(64) ^ (1 << i)] for i in range(6))
PHI = (1 + 5 ** .5) / 2

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def inputs():
    dna_file = BASE / 'research/oxtr-tfbs-v1/private-source-cache/ref-dna-expanded.json'
    motif_file = BASE / 'research/oxtr-tfbs-v1/private-source-cache/jaspar2026-vertebrates.jaspar'
    dna = json.loads(dna_file.read_text())
    lines = motif_file.read_text().splitlines()
    at = next(i for i, s in enumerate(lines) if s.startswith('>MA0139.2\t'))
    counts = np.array([[float(v) for v in re.findall(r'\d+', s)] for s in lines[at+1:at+5]])
    pwm = np.log2((counts + .5) / (counts.sum(axis=0) + 2) / .25)
    floor, span = pwm.min(axis=0).sum(), pwm.max(axis=0).sum() - pwm.min(axis=0).sum()
    width = pwm.shape[1]
    letters = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    rows = []
    # Frozen nonoverlapping genomic tiling; complete windows only.
    for start in range(((8740380 + 255)//256)*256, 8779628-255, 256):
        seq = dna['dna'][start-dna['start']:start-dna['start']+256].upper()
        if len(seq) != 256 or any(c not in letters for c in seq):
            continue
        candidates = []
        for p in range(256-width+1):
            ids = np.array([letters[c] for c in seq[p:p+width]])
            for strand, ix in [(1, ids), (-1, 3-ids[::-1])]:
                score = (pwm[ix, np.arange(width)].sum()-floor)/span
                if score >= .7:
                    candidates.append((p, float(score), strand))
        # Nonmaximum suppression collapses overlapping motif hits, independent of labels.
        hits = []
        for hit in sorted(candidates, key=lambda x: (-x[1], x[0], -x[2])):
            if all(abs(hit[0]-h[0]) >= width for h in hits):
                hits.append(hit)
        hits.sort()
        gaps = np.diff([h[0] for h in hits])
        bins = np.array([1, 3, 6, 9, 14, 22, 56])
        hist = np.histogram(gaps, bins=bins)[0] / max(1, len(gaps))
        fibhist = np.zeros(6)
        fibs = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55])
        for gap in gaps:
            b = np.searchsorted(bins, gap, side='right')-1
            if 0 <= b < 6:
                fibhist[b] += max(0., 1.-np.min(abs(fibs-gap))/2.)/max(1, len(gaps))
        phihist = np.zeros(6)
        ratios = []
        for g1, g2 in zip(gaps, gaps[1:]):
            ratio = max(g1, g2)/min(g1, g2)
            ratios.append(float(ratio))
            b = np.searchsorted(bins, min(g1, g2), side='right')-1
            if 0 <= b < 6:
                phihist[b] += max(0., 1.-abs(ratio-PHI)/(.1*PHI))/max(1, len(gaps)-1)
        scores = np.zeros(12)
        for p, score, strand in hits:
            b = min(11, int((p+width/2)/256*12))
            scores[b] = max(scores[b], score)
        context = [sum(c in 'GC' for c in seq)/256, seq.count('CG')/255,
                   min(len(hits)/16, 1), sum(h[2]==1 for h in hits)/max(1, len(hits)),
                   float(np.mean(gaps)/256) if len(gaps) else 0,
                   max([h[1] for h in hits], default=0)]
        vector = np.r_[scores, hist, context]
        rows.append({'id':f'chr3-{start}-{start+256}', 'start':start, 'end':start+256,
                     'motif_hits':hits, 'gaps':gaps.tolist(), 'native_gap_ratios':ratios,
                     'vector':vector.tolist(), 'fibonacci_vector':np.r_[scores,fibhist,context].tolist(),
                     'phi_vector':np.r_[scores,phihist,context].tolist()})
    # Most motif-rich two windows, ties in genomic order. No occupancy input is read.
    eligible = sorted([r for r in rows if len(r['motif_hits'])>=3], key=lambda r:(-len(r['motif_hits']),r['start']))
    if len(eligible)<2:
        raise RuntimeError('Insufficient multi-motif windows for the gap-ratio qualification.')
    selected = eligible[:2]
    save('inputs.json', {'selection':'Two highest motif-count OXTR tiling windows; genomic-order tie break; occupancy labels excluded.',
                        'windows_screened':len(rows), 'eligible_windows':len(eligible), 'motif':'MA0139.2',
                        'motif_width':width, 'relative_pwm_threshold':.7, 'pseudocount':.5,
                        'dna_source_sha256':sha(dna_file.read_bytes()), 'motif_library_sha256':sha(motif_file.read_bytes()),
                        'rows':selected})
    return selected

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

def setup():
    d = json.loads((SOURCE/'aquila-device.json').read_text())
    cap = json.loads(d['deviceCapabilities'])
    resolution = cap['paradigm']['lattice']['geometry']['positionResolution']
    p = np.array([[9e-6*np.sin(i*np.pi/3),9e-6*np.cos(i*np.pi/3)] for i in range(6)])
    p -= p.min(axis=0)
    p = np.round(p/resolution)*resolution
    c6 = cap['paradigm']['rydberg']['c6Coefficient']
    coupling = np.array([c6/np.linalg.norm(p[i]-p[j])**6/1e6 for i,j in PAIRS])
    interaction = sum(v*BITS[:,i]*BITS[:,j] for v,(i,j) in zip(coupling,PAIRS))
    return cap,p,coupling,interaction

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
    OUT.mkdir(exist_ok=True);(OUT/'programs').mkdir(exist_ok=True)
    rows=inputs();w=weights();cap,p,coupling,interaction=setup();cudaq.set_target('qpp-cpu')
    save('physical-model.json',{'positions_m':p.tolist(),'c6':cap['paradigm']['rydberg']['c6Coefficient'],
         'coupling_rad_per_us':coupling.tolist(),'pairs':PAIRS,'units':'time microseconds in numerical reference; AHS uses seconds and rad/s','basis':'site 0 is least-significant bit',
         'hamiltonian_dimension':64,'hermitian':bool(np.allclose(XS,XS.T))})
    records=[]
    for row in rows:
        for variant in VARIANTS:
            name=row['id']+'-'+variant;t,d=schedule(row,variant,w);ref=reference(t,d,interaction);target=abs(ref)**2
            convergence=[]
            for dt in [.02,.01,.005,.0025]:
                ss=steps(t,d,dt);ap=approximate(ss,interaction);err=float(np.max(abs((abs(ap)**2-target)@OBS)))
                convergence.append({'max_dt_us':dt,'steps':len(ss),'max_observable_error':err})
                if err<.0001 and len(convergence)>1:break
            assert err<.001
            k=kernel(ss,coupling);state=np.asarray(cudaq.get_state(k));impl=float(np.max(abs(abs(state)**2-abs(ap)**2)));assert impl<1e-9
            qasm=cudaq.translate(k,format='openqasm2');exported,gate_count=qasm_state(qasm)
            export_error=float(np.max(abs((abs(exported)**2-target)@OBS)));assert export_error<.001
            qpath=OUT/'programs'/(name+'.qasm');qpath.write_text(qasm)
            ahs={'braketSchemaHeader':{'name':'braket.ir.ahs.program','version':'1'},
                 'setup':{'ahs_register':{'sites':p.tolist(),'filling':[1]*6}},
                 'hamiltonian':{'drivingFields':[{'amplitude':{'time_series':{'times':[0,2e-7,3.8e-6,4e-6],'values':[0,2000000,2000000,0]},'pattern':'uniform'},
                    'phase':{'time_series':{'times':[0,4e-6],'values':[0,0]},'pattern':'uniform'},
                    'detuning':{'time_series':{'times':(t*1e-6).tolist(),'values':(d*1e6).tolist()},'pattern':'uniform'}}],'localDetuning':[]}}
            apath=OUT/'programs'/(name+'.ahs.json');apath.write_text(json.dumps(ahs,indent=2)+'\n')
            record={'id':name,'window':row['id'],'variant':variant,'convergence':convergence,'cudaq_cpu_max_probability_error':impl,
                    'export_max_observable_error':export_error,'qasm_gate_count':gate_count,'qasm_sha256':sha(qpath.read_bytes()),
                    'ahs_sha256':sha(apath.read_bytes()),'ideal_probabilities':target.tolist(),'ideal_observables':(target@OBS).tolist()}
            records.append(record);save('qualification.json',{'complete':False,'records':records})
            print(name,'verified',round(export_error,8),gate_count,flush=True)
    differences=[]
    for row in rows:
        a=next(r for r in records if r['window']==row['id'] and r['variant']=='gamma')
        b=next(r for r in records if r['window']==row['id'] and r['variant']=='epsilon')
        differences.append({'window':row['id'],'gamma_epsilon_max_observable_difference':float(np.max(abs(np.array(a['ideal_observables'])-b['ideal_observables'])))})
    save('qualification.json',{'complete':True,'at':datetime.now(timezone.utc).isoformat(),'execution':'CUDA-Q qpp-cpu plus independent SciPy reference',
         'study_version':'1.1.0-qualification','purpose':'Model and route qualification only; no biological prediction result',
         'records':records,'distinguishability':differences,'gpu_run_completed':False,'qpu_run_completed':False})

if __name__=='__main__':main()
