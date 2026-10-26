"""Reproducible geometry for the first M9/P24 homepage proof."""
import json, math
from pathlib import Path
root=Path(__file__).resolve().parents[2]
a,b=0,1
alpha=[]
while True:
    alpha.append(a)
    a,b=b,(a+b)%9
    if (a,b)==(0,1): break
nodes=[i for i,v in enumerate(alpha) if v in (0,3,6)]
path=[0,8,20,12,4,16,0]
sums=[sum(alpha[j%24] for j in range(lo+1,hi)) for lo,hi in zip(nodes,nodes[1:]+[24])]
def point(i,n,r=240):
    return [round(400+r*math.sin(2*math.pi*i/n),9),round(400-r*math.cos(2*math.pi*i/n),9)]
record=dict(modulus=9,period=len(alpha),alpha=alpha,display_alpha=[v or 9 for v in alpha],node_indices=nodes,unicursal_indices=path,unicursal_labels=[alpha[i%24] or 9 for i in path],triangles=[[0,8,16,0],[4,12,20,4]],interval_sums=sums,beta=[v%9 for v in sums],doubling=[1,2,4,8,7,5,1],alpha_points=[point(i,24) for i in range(24)],doubling_points=[point(i,9) for i in range(9)],path_status='Proposed path; no electron-motion or quantum-annealing result claimed.')
assert record['period']==24
assert record['unicursal_labels']==[9,3,6,9,3,6,9]
assert record['beta']==[4,8,7,5,1,2]
assert all((2*x)%9==y for x,y in zip(record['doubling'],record['doubling'][1:]))
(root/'Informatics/geometry.json').write_text(json.dumps(record,indent=2)+'\n')
print('M9/P24, continuous 9-3-6-9-3-6 path, and six-step doubling cycle verified.')
