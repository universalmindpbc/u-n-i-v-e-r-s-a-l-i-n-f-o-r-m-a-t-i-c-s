"""Reproducible native Fibonacci pilot. Native data with a separately labelled M9 comparison."""
import json, math, hashlib, argparse
from reduction import compare
from epsilon_path import reference_path, path_comparison
from epsilon_assessment import assessment
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1]
VERSION='m1-m24-v1'
SCOPE='Shapes in the primary Gamma node set (native residues divisible by 3, including zero). All regular polygons on the period circle are catalogued separately with their native values; their existence alone is a property of the plotting grid.'
EPS_RULE='A regular six-node formation must support the reference unicursal route through only 3, 6 and 9, containing all three. Rotations and reflections of the route are tested; reversed or cyclically shifted traversals are deduplicated.'
def period(m):
 a,b=0,1%m;seq=[]
 while True:
  seq.append(a);a,b=b,(a+b)%m
  if (a,b)==(0,1%m):return seq
  if len(seq)>m*m:raise AssertionError('Finite-state return bound exceeded')
def minimal(seq):
 return next((seq[:k] for k in range(1,len(seq)+1) if len(seq)%k==0 and seq==seq[:k]*(len(seq)//k)),[])
def canon(seq):return min((seq[i:]+seq[:i] for i in range(len(seq))),default=[])
def beta_for(a,nodes,m):
 if len(nodes)<2:return None
 p=len(a);sums=[];intervals=[]
 for lo,hi in zip(nodes,nodes[1:]+[nodes[0]+p]):
  inds=[j%p for j in range(lo+1,hi)];vs=[a[j] for j in inds];sums.append(sum(vs));intervals.append({'from_index':lo,'to_index':hi%p,'interior_indices':inds,'interior_values':vs,'sum':sum(vs),'residue':sum(vs)%m})
 b=[s%m for s in sums];mi=minimal(b);doubling=len(set(mi))>1 and all((2*v)%m==mi[(i+1)%len(mi)] for i,v in enumerate(mi))
 return {'interval_sums':sums,'interval_residues':b,'display_cycle':[v or m for v in b],'intervals':intervals,'minimal_cycle':mi,'native_doubling':doubling,'recognised_doubling_cycle':canon(mi) if doubling else None,'rotation_to_recognised':next((i for i in range(len(mi)) if mi[i:]+mi[:i]==canon(mi)),None) if doubling else None,'m9_doubling_cycle_present':len(mi)==6 and canon(mi)==[1,2,4,8,7,5],'start_value':b[0] or m,'start_interval_clock_positions':[nodes[0] or p,nodes[1] or p],'start_interval_values':intervals[0]['interior_values'],'circle_positions':m,'cycle_length':len(b),'endpoint_rule':'Exclude endpoints, include wraparound; raw residues used for sums. Empty interiors sum to zero.'}
def polygons(p,k):return [[i+j*(p//k) for j in range(k)] for i in range(p//k)] if p%k==0 else []
def eps_for(nodes,display):
 routes={};base=[0,2,5,3,1,4]
 if len(nodes)!=6:return []
 for sign in (1,-1):
  for offset in range(6):
   path=[nodes[(offset+sign*j)%6] for j in base];vals=[display[j] for j in path]
   if set(vals)!={3,6,9}:continue
   variants=[tuple(path[i:]+path[:i]) for i in range(6)]+[tuple(list(reversed(path))[i:]+list(reversed(path))[:i]) for i in range(6)]
   key=min(variants)
   if key not in routes:routes[key]={'alpha_indices':path+[path[0]],'traversal_values':vals+[vals[0]],'three_step_repeat':vals[:3]==vals[3:] and len(set(vals[:3]))==3}
 return list(routes.values())
def chevron(beta,m):
 # Exact M9 W chord-angle pattern on the native value circle, allowing scale,
 # rotation and reflection. Shape matching is independent of doubling arithmetic.
 if not beta:return False
 b=minimal(beta['interval_residues'])
 if len(b)!=6 or len(set(b))!=6:return False
 template=[1,2,4,8,7,5];ds=[Fraction((template[(i+1)%6]-template[i])%9,9) for i in range(6)]
 actual=[Fraction((b[(i+1)%6]-b[i])%m,m) for i in range(6)]
 return any(actual==[(s*v)%1 for v in ds[j:]+ds[:j]] for s in (1,-1) for j in range(6))
def record(m):
 a=period(m);p=len(a);display=[v or m for v in a];nodes=[i for i,v in enumerate(a) if v%3==0]
 ts=polygons(p,3);hs=polygons(p,6);primary_t=[t for t in ts if set(t)<=set(nodes)];primary_h=[h for h in hs if set(h)<=set(nodes)]
 b=beta_for(a,nodes,m);eps=[];catalog=[];notable=[]
 for k,sets in [(3,ts),(6,hs)]:
  for inds in sets:
   vals=[display[i] for i in inds];rotation=360*inds[0]/p;ident=f'{k}-nodes-offset-{inds[0]}'
   item={'id':ident,'shape':'Hexagon' if k==6 else 'Equilateral triangle','alpha_indices':inds,'clock_positions':[i or p for i in inds],'native_values':vals,'rotation_degrees':rotation,'north_south':k==6 and inds[0]==0,'in_primary_gamma':set(inds)<=set(nodes),'uniform_values':len(set(vals))==1,'minimal_value_cycle':minimal(vals),'beta':beta_for(a,inds,m)}
   if k==6:
    item['epsilon_routes']=eps_for(inds,display)
    for route in item['epsilon_routes']:
     route.update({'formation_id':ident,'clock_positions':[i or p for i in route['alpha_indices']],'rotation_degrees':rotation});eps.append(route)
   reasons=[]
   if item['uniform_values']:reasons.append('All nodes have the same native value')
   if 1<len(minimal(vals))<k:reasons.append('Repeating native-value pattern')
   if set(vals)=={3,6,9}:reasons.append('Contains only 3, 6 and 9, with all three present')
   if item['beta'] and item['beta']['native_doubling']:reasons.append('Interior sums form a native doubling cycle')
   if reasons:notable.append({'formation_id':ident,'shape':item['shape'],'values':vals,'rotation_degrees':rotation,'reasons':reasons,'python_status':'Passed','wolfram_status':'Pending','significance':'Not yet assessed'})
   catalog.append(item)
 eps.sort(key=lambda e:(not e['three_step_repeat'],e['rotation_degrees'],e['alpha_indices']))
 # Keep the approved reference traversal when it exists.
 if m==9:eps.sort(key=lambda e:e['alpha_indices']!=[0,8,20,12,4,16,0])
 ep=eps[0] if eps else None
 if ep:ep={**ep,'status':'Proposed geometric traversal; physical interpretation not assessed.'}
 def sh(name,sets,paths,rotations):
  return {'name':name,'status':'Present' if sets else 'Absent','count':len(sets),'orientation':('North–south' if rotations==[0] else ' / '.join(f'{x:g}°' for x in rotations)) if sets else None,'rotation_degrees':rotations[0] if len(rotations)==1 else None,'rotations_degrees':rotations,'alpha_paths':paths}
 shapes=[sh('Equilateral triangle',primary_t,[x+[x[0]] for x in primary_t],[360*x[0]/p for x in primary_t]),sh('Hexagon',primary_h,[x+[x[0]] for x in primary_h],[360*x[0]/p for x in primary_h]),sh('2D stellated octahedron',primary_h,[z+[z[0]] for x in primary_h for z in (x[::2],x[1::2])],[360*x[0]/p for x in primary_h]),sh('Chevron',[1] if chevron(b,m) else [],[],[0] if chevron(b,m) else []),sh('Unicursal hexagram',sorted(set(x['formation_id'] for x in eps)),[x['alpha_indices'] for x in eps],sorted(set(x['rotation_degrees'] for x in eps)))]
 for x in shapes:
  if x['name']=='Equilateral triangle' and x['rotations_degrees']==[0,60]:x['orientation']='North / south'
 rid=f'fibonacci-m{m}-p{p}'
 result={'schema_version':'2.0.0','dataset_version':VERSION,'id':rid,'name':f'Fibonacci M{m}P{p}','modulus':m,'pisano_period':p,'conventions':{'index_origin':0,'clock_labels':[p]+list(range(1,p)),'orientation':'clockwise from north','zero_display':m,'alpha_circle_positions':p,'beta_circle_positions':m,'node_rule':'Native residues divisible by 3, including zero; preserves the approved M6/M9/M12 reference selection. Additional regular formations of all native values are catalogued separately.','m1_convention':'Modulo 1 has period 1 and residue 0.'},'alpha':{'residues':a,'display_values':display},'gamma':{'alpha_indices':nodes,'clock_positions':[i or p for i in nodes],'clockwise_values':[display[i] for i in nodes],'triangle_paths':[t+[t[0]] for t in primary_t]},'beta':b,'epsilon':ep,'epsilon_routes':eps,'epsilon_status':'Present' if ep else 'Null: no tested regular six-node reference traversal contains only 3, 6 and 9 with all three present.','geometry_summary':{'rule_version':'native-pilot-v1','scope':SCOPE,'circle_role':'Pisano-period plotting framework','shapes':shapes,'other_orientations_status':'Present' if any(any(r not in (0,60) for r in s['rotations_degrees']) for s in shapes) else 'Absent','epsilon_rule':EPS_RULE},'formation_catalogue':catalog,'notable_gamma_patterns':notable,'research':{'focus':'OXTR gene expression','biological_significance_status':'Not yet assessed','statistical_significance_status':'Not yet assessed','quantum_annealing_status':'Not yet performed','tiers':[{'name':name,'range':[lo,hi],'wss_score':None,'evidence':[]} for name,lo,hi in [('Creates cellular life',8,10),('Supports cellular life',5,7),('Creates elemental life',1,4)]]},'verification':{'python':'Passed','wolfram':'Pending','scope':'Native period, sequences, regular polygon index spacing and reference Epsilon routes; no physical or biological validation.'}}
 result['reduction_to_m9']=compare(result)
 result['epsilon_m9_comparison']=path_comparison(result)
 path=reference_path(result)
 if path:result['epsilon_path']=path
 result['epsilon_assessment']=assessment(result)
 return result
def validate(d):
 m=d['modulus'];a=d['alpha']['residues'];p=len(a)
 # Independent unreduced integer Fibonacci sequence checks each value and return.
 f0,f1=0,1
 for i in range(p+1):
  if i<p:assert f0%m==a[i]
  if 0<i<p:assert (f0%m,f1%m)!=(0,1%m)
  if i==p:assert (f0%m,f1%m)==(0,1%m)
  f0,f1=f1,f0+f1
 for x in d['formation_catalogue']:
  ids=x['alpha_indices'];k=len(ids);assert len(set(ids))==k
  assert all((ids[(i+1)%k]-ids[i])%p==p//k for i in range(k))
 for ep in d['epsilon_routes']:
  assert set(ep['traversal_values'])=={3,6,9}
  assert len(set(ep['alpha_indices'][:-1]))==6 and ep['alpha_indices'][0]==ep['alpha_indices'][-1]
 if m<9:assert d['epsilon'] is None
 if m in (6,9,12):
  expected={6:([6,3,3,6,3,3],[4,2,4,2,4,2]),9:([9,3,3,9,6,6],[4,8,7,5,1,2]),12:([12,3,9,12,3,9],[4,2,10,8,10,2])}[m]
  assert p==24 and d['gamma']['clockwise_values']==expected[0] and d['beta']['display_cycle']==expected[1]

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(ROOT/'data'/VERSION));args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 records=[]
 for m in range(1,25):
  d=record(m);validate(d);records.append(d);(out/f'm{m}.json').write_text(json.dumps(d,indent=2)+'\n')
 (out/'dataset.json').write_text(json.dumps({'schema_version':'2.0.0','dataset_version':VERSION,'records':records},indent=2)+'\n')
 print(json.dumps([{'m':d['modulus'],'p':d['pisano_period'],'gamma':d['gamma']['clockwise_values'],'beta':d['beta']['display_cycle'] if d['beta'] else None,'hexagons':d['geometry_summary']['shapes'][1]['count'],'epsilon':bool(d['epsilon']),'other_formations':len(d['formation_catalogue'])} for d in records],indent=2))
if __name__=='__main__':main()
