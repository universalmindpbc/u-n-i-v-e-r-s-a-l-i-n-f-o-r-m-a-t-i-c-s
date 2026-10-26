"""Validate frozen qualification programs against saved live Aquila capabilities; no submission."""
import json,hashlib
from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace
from braket.ir.ahs import Program
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.device_schema.quera import QueraDeviceCapabilities
import numpy as np
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'
cap=QueraDeviceCapabilities.parse_raw(json.loads((ROOT/'source/aquila-device.json').read_text())['deviceCapabilities'])
device=SimpleNamespace(properties=cap);g=cap.paradigm.rydberg.rydbergGlobal;geo=cap.paradigm.lattice.geometry
q=json.loads((OUT/'qualification.json').read_text());assert q['complete'];records=[]
for r in q['records']:
 path=OUT/'programs'/(r['id']+'.ahs.json');original=Program.parse_raw(path.read_text());p=AnalogHamiltonianSimulation.from_ir(original).discretize(device).to_ir();sites=np.array(p.setup.ahs_register.sites,dtype=float)
 assert len(sites)==6
 assert all(np.linalg.norm(a-b)>=float(geo.spacingRadialMin) for i,a in enumerate(sites) for b in sites[i+1:])
 ys=sorted(set(sites[:,1]));assert all(b-a>=float(geo.spacingVerticalMin)-1e-15 for a,b in zip(ys,ys[1:]))
 assert np.ptp(sites[:,0])<=float(cap.paradigm.lattice.area.width)
 assert np.ptp(sites[:,1])<=float(cap.paradigm.lattice.area.height)
 f=p.hamiltonian.drivingFields[0];maxdiff=0
 for field,bounds,slew in [('amplitude',g.rabiFrequencyRange,g.rabiFrequencySlewRateMax),('detuning',g.detuningRange,g.detuningSlewRateMax),('phase',g.phaseRange,None)]:
  ts=getattr(f,field).time_series;t=np.array(ts.times,dtype=float);v=np.array(ts.values,dtype=float)
  assert t[0]==0 and t[-1]<=float(g.timeMax)
  assert min(np.diff(t))>=float(g.timeDeltaMin)-1e-16
  assert min(v)>=float(bounds[0]) and max(v)<=float(bounds[1])
  if slew is not None:assert max(abs(np.diff(v)/np.diff(t)))<=float(slew)
  old=getattr(original.hamiltonian.drivingFields[0],field).time_series
  # Numerically equivalent serialization, not a changed physical model.
  td=float(max(abs(t-np.array(old.times,dtype=float))));vd=float(max(abs(v-np.array(old.values,dtype=float))))
  assert td<1e-20 and vd<1e-7
  maxdiff=max(maxdiff,vd)
 assert float(f.amplitude.time_series.values[0])==float(f.amplitude.time_series.values[-1])==0
 assert np.max(abs(sites-np.array(original.setup.ahs_register.sites,dtype=float)))<1e-18
 action=p.json();target=OUT/'programs'/(r['id']+'.ahs.discretized.json');target.write_text(action)
 records.append({'id':r['id'],'action_sha256':hashlib.sha256(action.encode()).hexdigest(),'file':target.name,'qasm_sha256':r['qasm_sha256'],'maximum_serialization_field_difference_rad_s':maxdiff,'device_constraints_passed':True})
report={'passed':True,'tasks':len(records),'shots_per_task':100,'total_shots':100*len(records),'provider_price_usd_per_task':.3,'provider_price_usd_per_shot':.01,'quantum_fee_usd':round(len(records)*1.3,2),'existing_total_spend_usd':1.3,'required_cumulative_cap_usd':round(1.3+len(records)*1.3,2),'price_source':'https://aws.amazon.com/braket/pricing/','records':records,'purpose':'Qualification batch; biological prediction evaluation remains a later phase','hardware_submitted':False}
(OUT/'hardware-qualification-manifest.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='records'})
