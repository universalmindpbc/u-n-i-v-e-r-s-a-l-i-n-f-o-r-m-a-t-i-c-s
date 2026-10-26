"""Native modulus views. Same coordinates and colours as the approved references."""
import json, hashlib, argparse
from pathlib import Path
import drawing
from epsilon_path import draw_path
from drawing import Drawing,pt,BLUE,PINK,GREEN,YELLOW,FG,MUTED
from build_pilot import record,validate,VERSION,ROOT

def draw_record(d,out):
 m,p=d['modulus'],d['pisano_period'];a=d['alpha']['display_values'];b=d['beta'];gam=d['gamma'];eps=d['epsilon'];combined_eps=d.get('epsilon_m9_comparison') if d.get('epsilon_m9_comparison',{}).get('combined_eligible') else None
 directory=out/f'views/m{m}';directory.mkdir(parents=True,exist_ok=True);drawing.OUT=directory
 def alpha(g,combined=False):
  g.circle((400,365),238,BLUE)
  for i,v in enumerate(a):
   g.circle(pt(i,p),2,BLUE,fill=BLUE);g.text(v,pt(i,p,267),(14 if p>40 else 18) if m<=24 else max(6,min(18,1100//p)))
   if not combined:g.text(i or p,pt(i,p,291),(9 if p>40 else 10) if m<=24 else max(5,min(10,900//p)),MUTED)
 def gamma(g,labels=True):
  # Fill all triangles first so one fill cannot erase another triangle's edge.
  for path in gam['triangle_paths']:g.line([pt(i,p) for i in path[:-1]],PINK,2,True,'#261525')
  for path in gam['triangle_paths']:g.line([pt(i,p) for i in path],PINK,2)
  for i in gam['alpha_indices']:
   g.circle(pt(i,p),4,PINK,fill=PINK)
   if labels:g.text(a[i],pt(i,p,270),18 if m<=24 else max(7,min(18,1100//p)));g.text(i or p,pt(i,p,292),10,MUTED)
 def beta(g,labels=True):
  if not b:
   g.text('No interval sequence',(400,365),22,MUTED);return
  points=[pt(i,m) for i in b['interval_residues']+[b['interval_residues'][0]]]
  g.line(points,YELLOW,3)
  if labels:
   for i in range(m):g.text(i or m,pt(i,m,270),20,YELLOW if i in b['interval_residues'] else MUTED)
   for x,y in zip(points,points[1:]):
    if x!=y:g.arrow(x,y,YELLOW)
   g.circle(points[0],5,YELLOW,fill=YELLOW);g.text('start',pt(b['interval_residues'][0],m,297),11,MUTED)
 def epsilon(g,labels=True,source=None):
  source=source or eps
  points=[pt(i,p) for i in source['alpha_indices']];g.line(points,FG,3)
  for x,y in zip(points,points[1:]):g.arrow(x,y,FG)
  if labels:
   for step,i in enumerate(eps['alpha_indices'][:-1],1):
    g.circle(pt(i,p),4,FG,fill=FG);g.text(a[i],pt(i,p,270),24);g.text(f'{step:02}',pt(i,p,292),11,MUTED)
 def combined(g):
  gamma(g,False)
  for s in d['geometry_summary']['shapes']:
   if s['name']=='Hexagon':
    for path in s['alpha_paths']:g.line([pt(i,p) for i in path],GREEN,2)
  alpha(g,True)
  if combined_eps:epsilon(g,False,combined_eps)
  if b:beta(g,False)
 entries=[('alpha','Alpha',alpha,'Native Fibonacci residues · clockwise from north'),('beta','Beta',beta,'Interior sums · original start preserved'),('gamma','Gamma',gamma,'Primary nodes · native residues divisible by three')]
 if eps:entries.append(('epsilon','Epsilon',epsilon,'Proposed unicursal route · arrows show traversal'))
 entries.append((f'fibonacci-m{m}p{p}',d['name'],combined,'Combined layers · period and native-value circles'))
 views=[]
 for key,title,fn,note in entries:
  g=Drawing(title,note);g.circle((400,365),238,'#34434e');fn(g)
  g.text(f'M{m} / P{p}',(400,32),12,MUTED);g.text(title,(400,690),24);g.text(note,(400,723),14,MUTED)
  if key=='alpha':g.text(f'North is clock position {p}. Remainder zero is displayed as {m}.',(400,752),12,MUTED)
  if key=='beta' and b:
   values=b['minimal_cycle'];shown=' → '.join(map(str,values)) if m<=24 or len(values)<=14 else f'{len(values)} values · full sequence in the adjacent panel and data export';g.text(shown,(400,752),16 if m<=24 or len(values)<=14 else 12,FG if m<=24 or len(values)<=14 else MUTED)
  if key=='gamma' and not gam['triangle_paths']:g.text('No equilateral triangle within this primary node set.',(400,752),12,MUTED)
  if key=='epsilon':g.text(' → '.join(map(str,eps['traversal_values'])),(400,752),16)
  g.save(key);views.append({'id':key,'name':title,'svg':f'pilot-v1/views/m{m}/{key}.svg','png':f'pilot-v1/views/m{m}/{key}.png'})
 d['views']=views
 draw_path(d,out)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(ROOT/'data'/VERSION));args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 # Verification evidence is durable and checked before marking core values passed.
 proof=json.loads((ROOT/'data'/VERSION/'wolfram-core-result.json').read_text());assert not proof.get('isError')
 text=' '.join(x.get('text','') for x in proof['content']);assert text.count('True')==192 and 'False' not in text
 records=[]
 for m in range(1,25):
  d=record(m);validate(d);draw_record(d,out)
  d['verification'].update({'wolfram':'Passed','wolfram_scope':'24 native periods, complete Alpha, primary Gamma indices, Beta interval sums/residues, primary regular triangles and hexagons. Extended catalogue, Chevron and Epsilon remain Python-checked only.','evidence':'wolfram-core-result.json'})
  records.append(d);(out/f'm{m}.json').write_text(json.dumps(d,indent=2)+'\n')
 (out/'dataset.json').write_text(json.dumps({'schema_version':'2.0.0','dataset_version':VERSION,'records':records},indent=2)+'\n')
 (out/'index.json').write_text(json.dumps({'version':VERSION,'records':[{'id':d['id'],'name':d['name'],'modulus':d['modulus'],'pisano_period':d['pisano_period'],'file':f"pilot-v1/m{d['modulus']}.json"} for d in records]},indent=2)+'\n')
 (out/'checksums.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='checksums.json'},indent=2)+'\n')
 print(f'{len(records)} records; {len(list(out.rglob("*.png")))} PNG and SVG pairs; 192 Wolfram core comparisons passed.')
if __name__=='__main__':main()
