"""Build reproducible M1–M48 candidate records, preserving prior verification evidence."""
import json,hashlib
from pathlib import Path
from build_pilot import record,validate
from render_pilot import draw_record
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/m1-m48-v1';OLD=ROOT/'data/m1-m24-v1'
def wl(x):
 if isinstance(x,list):return '{'+','.join(wl(v) for v in x)+'}'
 if isinstance(x,bool):return 'True' if x else 'False'
 return str(x)
def main():
 OUT.mkdir(exist_ok=True);records=[];expected=[]
 for m in range(1,49):
  d=record(m);validate(d);d['dataset_version']='m1-m48-v1'
  if m<=24:d['verification']=json.loads((OLD/f'm{m}.json').read_text())['verification']
  else:
   proof_path=OUT/'wolfram-batch-result.json';proof=json.loads(proof_path.read_text()) if proof_path.exists() else {};text=' '.join(x.get('text','') for x in proof.get('content',[]));passed=bool(proof) and not proof.get('isError') and text.count('True')==264 and 'False' not in text
   d['verification'].update({'wolfram':'Passed' if passed else 'Pending','wolfram_scope':'Native period, complete Alpha, primary Gamma indices, Beta sums/residues, primary triangles/hexagons and exact north-oriented M9 reference path comparison. Extended catalogue, native Epsilon search and Chevron remain Python-checked.','evidence':'wolfram-batch-result.json'})
  draw_record(d,OUT);records.append(d);(OUT/f'm{m}.json').write_text(json.dumps(d,indent=2)+'\n')
  if m>=25:
   b=d['beta'];sh=d['geometry_summary']['shapes'];c=d['epsilon_m9_comparison']
   expected.append([m,d['pisano_period'],d['alpha']['residues'],d['gamma']['alpha_indices'],b['interval_sums'] if b else [],b['interval_residues'] if b else [],[x[:-1] for x in sh[0]['alpha_paths']],[x[:-1] for x in sh[1]['alpha_paths']],c['alpha_indices'],c['reduced_m9_values'],c['combined_eligible']])
 (OUT/'dataset.json').write_text(json.dumps({'schema_version':'2.1.0','dataset_version':'m1-m48-v1','records':records},indent=2)+'\n')
 (OUT/'index.json').write_text(json.dumps({'version':'m1-m48-v1','records':[{'id':d['id'],'name':d['name'],'modulus':d['modulus'],'pisano_period':d['pisano_period'],'file':f"pilot-v1/m{d['modulus']}.json"} for d in records]},indent=2)+'\n')
 code='expected='+wl(expected)+''';Table[Module[{p,a,nodes,sums,tri,hex,route,values,eligible,result},p=SelectFirst[Range[m^2],Mod[Fibonacci[#],m]==0&&Mod[Fibonacci[#+1],m]==1&];a=Table[Mod[Fibonacci[n],m],{n,0,p-1}];nodes=Flatten[Position[Mod[a,3],0]]-1;sums=If[Length[nodes]<2,{},Table[Sum[a[[Mod[j,p]+1]],{j,nodes[[k]]+1,If[k<Length[nodes],nodes[[k+1]],nodes[[1]]+p]-1}],{k,Length[nodes]}]];tri=If[Mod[p,3]==0,Select[Table[Range[i,p-1,p/3],{i,0,p/3-1}],SubsetQ[nodes,#]&],{}];hex=If[Mod[p,6]==0,Select[Table[Range[i,p-1,p/6],{i,0,p/6-1}],SubsetQ[nodes,#]&],{}];route=If[Mod[p,6]==0,p {0,2,5,3,1,4,0}/6,{}];values=If[Length[route]>0,Map[Replace[Mod[#,9],0->9]&,a[[route+1]]],{}];eligible=Length[route]>0&&route/p=={0,1/3,5/6,1/2,1/6,2/3,0}&&values=={9,3,6,9,3,6,9};result={m,p,a,nodes,sums,Mod[sums,m],tri,hex,route,values,eligible};{m,MapThread[SameQ,{result,expected[[m-24]]}]}],{m,25,48}]'''
 if proof:assert passed and hashlib.sha256(code.encode()).hexdigest()==proof['verification_code_sha256'], 'Wolfram evidence must match this exact verification input'
 (OUT/'verify_batch.wl').write_text(code)
 print(json.dumps([{'m':d['modulus'],'p':d['pisano_period'],'native_epsilon':bool(d['epsilon']),'exact_m9_path':d['epsilon_m9_comparison']['combined_eligible'],'primary_hexagons':d['geometry_summary']['shapes'][1]['count']} for d in records if d['modulus']>=25]))
if __name__=='__main__':main()
