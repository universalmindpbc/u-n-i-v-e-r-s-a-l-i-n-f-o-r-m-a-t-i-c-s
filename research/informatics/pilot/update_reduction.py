import json,sys
from pathlib import Path
from reduction import compare
from build_pilot import period,beta_for
out=Path(__file__).resolve().parents[1]/'data/m1-m24-v1'
d=json.loads((out/'dataset.json').read_text())
for r in d['records']:
 r['reduction_to_m9']=compare(r)
 (out/f"m{r['modulus']}.json").write_text(json.dumps(r,indent=2)+'\n')
(out/'dataset.json').write_text(json.dumps(d,indent=2)+'\n')
audit=[]
for m in range(9,1002,9):
 a=period(m);nodes=[i for i,v in enumerate(a) if v%3==0]
 r={'modulus':m,'pisano_period':len(a),'alpha':{'residues':a},'gamma':{'clockwise_values':[a[i] for i in nodes],'alpha_indices':nodes},'beta':beta_for(a,nodes,m)}
 c=compare(r);assert all(c['checks'].values()),m
 audit.append({'modulus':m,'period':len(a),'repeat_count':c['repeat_count'],'checks':c['checks']})
(out/'m9-reduction-audit.json').write_text(json.dumps({'scope':'111 multiples of 9 in M1–M1001; arithmetic comparison only, not full Atlas records','proof':'For m=9k, reducing modulo m then modulo 9 equals reducing modulo 9. Divisibility by 3 preserves Gamma selection; interior summation commutes with reduction. The minimal M9 period is 24, so each reduced full period is a repetition of it.','records':audit},indent=2)+'\n')
print('Updated 24 comparisons; all 111 divisible-by-nine checks passed.')
