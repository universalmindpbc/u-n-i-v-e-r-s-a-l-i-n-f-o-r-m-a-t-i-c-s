"""Submit the frozen qualification circuits within verified free qBraid allowance."""
import argparse, json, hashlib, stat
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from qbraid.runtime import QbraidProvider
from qbraid_core.services.runtime.schemas import JobRequest, Program

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; GPU=OUT/'gpu'; GPU.mkdir(exist_ok=True)
KEY=ROOT.parent.parent.parent/'.private/quantum/qbraid.key'
DEVICE='qbraid:qbraid:sim:gpu-sv'
def write(path,data,exclusive=False):
    with path.open('x' if exclusive else 'w') as f:json.dump(data,f,indent=2);f.write('\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['submit','status']);args=p.parse_args()
    assert stat.S_IMODE(KEY.stat().st_mode)==0o600
    key=KEY.read_text().strip();client=QbraidProvider(api_key=key).client
    q=json.loads((OUT/'qualification.json').read_text());assert q['complete']
    cases=[{'id':'ordering-probe','qasm':'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[6];\nx q[0];\n','shots':8}]
    cases += [{'id':r['id'],'qasm':(OUT/'programs'/(r['id']+'.qasm')).read_text(),'shots':100,'record':r} for r in q['records']]
    try:
        if args.mode=='submit':
            proof=json.loads((OUT/'free-access-check.json').read_text())
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(proof['recorded_at_utc'])).total_seconds()
            assert 0<=age<3600 and proof['usage_based_billing'] is False and proof['free_circuits_remaining']>=len(cases)
            assert not list(GPU.glob('*-intent.json')), 'Existing submission intent; inspect outcomes instead of resubmitting.'
            device=client.get_device(DEVICE).model_dump(mode='json');assert device['status'].upper()=='ONLINE'
            for case in cases:
                digest=hashlib.sha256(case['qasm'].encode()).hexdigest()
                if 'record' in case:assert digest==case['record']['qasm_sha256']
                request=JobRequest(deviceQrn=DEVICE,program=Program(format='qasm2',data=case['qasm']),shots=case['shots'],
                       name='OXTR qualification '+case['id'],tags={'study':'oxtr-qrc-01','phase':'qualification','source_sha256':digest},
                       runtimeOptions={'precision':'double','timeout':60,'seed':20261025})
                write(GPU/(case['id']+'-intent.json'),{'id':case['id'],'sha256':digest,'shots':case['shots'],'at':datetime.now(timezone.utc).isoformat()},True)
                job=client.create_job(request).model_dump(mode='json');write(GPU/(case['id']+'-submission.json'),job,True)
                print(case['id'],job['jobQrn'],job['status'],flush=True)
            return
        summary=[]
        bits=((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
        obs=np.column_stack([bits]+[(bits[:,i]*bits[:,j])[:,None] for i in range(6) for j in range(i+1,6)])
        for case in cases:
            submission=GPU/(case['id']+'-submission.json')
            if not submission.exists():continue
            job=json.loads(submission.read_text());meta=client.get_job(job['jobQrn']).model_dump(mode='json');write(GPU/(case['id']+'-status.json'),meta)
            row={'id':case['id'],'job_id':job['jobQrn'],'status':meta['status'],'cost_credits':meta['cost']}
            assert float(meta['cost'] or 0)==0
            if meta['status'].upper()=='COMPLETED':
                raw=client.get_job_result(job['jobQrn']).model_dump(mode='json');write(GPU/(case['id']+'-result.json'),raw)
                counts=raw['resultData'].get('measurementCounts',raw['resultData'].get('measurement_counts'));assert sum(counts.values())==case['shots']
                assert float(raw['cost'])==0
                if case['id']=='ordering-probe':
                    row['passed']=counts=={'000001':8}
                else:
                    frequency=np.array([counts.get(format(i,'06b'),0) for i in range(64)])/case['shots']
                    measured=frequency@obs;ideal=np.array(case['record']['ideal_observables'])
                    tolerance=6*np.sqrt(ideal*(1-ideal)/case['shots'])+2/case['shots']
                    row.update(passed=bool(np.all(abs(measured-ideal)<=tolerance)),measured_observables=measured.tolist(),
                               ideal_observables=ideal.tolist(),tolerance=tolerance.tolist(),counts=counts,shots=case['shots'])
                assert row['passed'],case['id']
            summary.append(row);print(case['id'],row['status'],row.get('passed'),flush=True)
        write(OUT/'gpu-qualification.json',{'at':datetime.now(timezone.utc).isoformat(),'complete':len(summary)==len(cases) and all(r['status'].upper()=='COMPLETED' for r in summary),
              'purpose':'Numerical/instrument-route qualification; no occupancy prediction claim','cases':summary,'cost_credits':sum(float(r['cost_credits'] or 0) for r in summary)})
    except Exception as e:raise SystemExit(type(e).__name__+': '+str(e).replace(key,'[REDACTED]')) from None
if __name__=='__main__':main()
