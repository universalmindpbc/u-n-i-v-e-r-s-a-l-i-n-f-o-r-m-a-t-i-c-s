"""Generate five M9/P24 SVG/PNG views and a versioned research record.
Run with Python + Pillow. Coordinates are clockwise, with index 0 at north.
No biological significance or physical motion is inferred by these drawings.
"""
import json, math, html
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Informatics/views'
OUT.mkdir(exist_ok=True)
d=json.loads((ROOT/'Informatics/geometry.json').read_text())
BG='#080d12'; FG='#e2e8ec'; MUTED='#7d8d99'; BLUE='#5ca8ff'; PINK='#ff84c5'; GREEN='#62d998'; YELLOW='#ffea42'
font='/System/Library/Fonts/Supplemental/Arial.ttf'
def pt(i,n,r=238):
 a=2*math.pi*i/n
 return (400+r*math.sin(a),365-r*math.cos(a))
class Drawing:
 def __init__(self,title,desc):
  self.image=Image.new('RGB',(800,800),BG);self.draw=ImageDraw.Draw(self.image)
  self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800" role="img"><title>{html.escape(title)}</title><desc>{html.escape(desc)}</desc>']
 def line(self,points,color,width=2,close=False,fill=None):
  p=list(points)
  if close:
   if fill:self.draw.polygon(p,fill=fill)
   p=p+[p[0]]
  self.draw.line(p,fill=color,width=width)
  coords=' '.join(f'{x:.3f},{y:.3f}' for x,y in p)
  self.svg.append(f'<polyline points="{coords}" stroke="{color}" stroke-width="{width}" fill="{fill or "none"}" stroke-linejoin="round"/>')
 def circle(self,p,r,color,width=1,fill=None):
  x,y=p;self.draw.ellipse((x-r,y-r,x+r,y+r),outline=color,width=width,fill=fill)
  self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{color}" stroke-width="{width}" fill="{fill or "none"}"/>')
 def text(self,t,p,size=18,color=FG):
  self.draw.text(p,str(t),font=ImageFont.truetype(font,size),fill=color,anchor='mm')
  self.svg.append(f'<text x="{p[0]}" y="{p[1]}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}" text-anchor="middle" dominant-baseline="central">{html.escape(str(t))}</text>')
 def arrow(self,a,b,color):
  x=a[0]+.56*(b[0]-a[0]);y=a[1]+.56*(b[1]-a[1]);ang=math.atan2(b[1]-a[1],b[0]-a[0])
  self.line([(x-10*math.cos(ang-.4),y-10*math.sin(ang-.4)),(x,y),(x-10*math.cos(ang+.4),y-10*math.sin(ang+.4))],color,2)
 def save(self,key):
  (OUT/f'{key}.svg').write_text(''.join(self.svg)+'</svg>\n')
  self.image.save(OUT/f'{key}.png')
def alpha(g,combined=False):
 g.circle((400,365),238,BLUE)
 for i,v in enumerate(d['display_alpha']):
  g.circle(pt(i,24),2,BLUE,fill=BLUE)
  g.text(v,pt(i,24,267),17 if combined else 20)
  if not combined:g.text(i or 24,pt(i,24,294),10,MUTED)
def gamma(g,labels=True):
 for indices in d['triangles']:
  g.line([pt(i,24) for i in indices[:-1]],PINK,2,True,'#261525')
 for indices in d['triangles']:
  g.line([pt(i,24) for i in indices],PINK,2)
 for i in d['node_indices']:
  g.circle(pt(i,24),5,PINK,fill=PINK)
  if labels:g.text(d['display_alpha'][i],pt(i,24,270),24)
def epsilon(g,labels=True):
 points=[pt(i,24) for i in d['unicursal_indices']]
 g.line(points,FG,3)
 for a,b in zip(points,points[1:]):g.arrow(a,b,FG)
 if labels:
  for step,i in enumerate(d['unicursal_indices'][:-1],1):
   g.circle(pt(i,24),4,FG,fill=FG);g.text(d['display_alpha'][i],pt(i,24,270),24);g.text(f'{step:02}',pt(i,24,290),11,MUTED)
def beta(g,labels=True):
 points=[pt(i,9) for i in d['beta']+[d['beta'][0]]]
 g.line(points,YELLOW,3)
 if labels:
  for i in range(9):g.text(i or 9,pt(i,9,270),24,YELLOW if i in d['doubling'] else MUTED)
  for a,b in zip(points,points[1:]):g.arrow(a,b,YELLOW)
  g.circle(pt(d['beta'][0],9),5,YELLOW,fill=YELLOW)
  g.text('start',pt(d['beta'][0],9,301),11,MUTED)
views=[
 ('alpha','Alpha','24 Fibonacci residues','F(n + 2) = F(n + 1) + F(n) mod 9','Clock positions 24, 1–23 clockwise. Remainder 0 is displayed as 9.'),
 ('beta','Beta','Embedded doubling sequence','4 → 8 → 7 → 5 → 1 → 2 → 4','Cyclic equivalent: 1 → 2 → 4 → 8 → 7 → 5. Start preserved at 4.'),
 ('gamma','Gamma','Nodal geometry','9 · 3 · 3 · 9 · 6 · 6','Clockwise node order. Two intersecting equilateral triangles.'),
 ('epsilon','Epsilon','Unicursal hexagram','9 → 3 → 6 → 9 → 3 → 6 → 9','Arrows show traversal; small numbers show visit order.'),
 ('fibonacci-m9p24','Fibonacci M9P24','Combined geometry','Alpha · Beta · Gamma · Epsilon','24-position Alpha and nine-position Beta share a circle.')]
for key,title,subtitle,seq,note in views:
 g=Drawing(title,note);g.circle((400,365),238,'#34434e')
 if key=='alpha':alpha(g)
 elif key=='beta':beta(g)
 elif key=='gamma':gamma(g)
 elif key=='epsilon':epsilon(g)
 else:
  gamma(g,False);g.line([pt(i,6) for i in range(6)],GREEN,2,True);alpha(g,True);epsilon(g,False);beta(g,False)
 g.text('M9 / P24',(400,32),12,MUTED);g.text(title,(400,690),24);g.text(subtitle,(400,721),14,MUTED);g.text(seq,(400,750),15);g.text(note,(400,779),11,MUTED);g.save(key)
record={
 'schema_version':'1.1.0','id':'fibonacci-m9-p24','name':'Fibonacci M9P24','modulus':9,'pisano_period':24,
 'conventions':{'index_origin':0,'clock_labels':[24]+list(range(1,24)),'orientation':'clockwise from north','zero_display':9,'alpha_circle_positions':24,'beta_circle_positions':9},
 'alpha':{'residues':d['alpha'],'display_values':d['display_alpha']},
 'beta':{'interval_sums':d['interval_sums'],'interval_residues':d['beta'],'display_cycle':d['beta'],'rotation_to_display':0,'recognised_doubling_cycle':d['doubling'][:-1],'rotation_to_recognised':4,'start_value':d['beta'][0],'start_interval_clock_positions':[24,4],'start_interval_values':[1,1,2],'endpoint_rule':'Exclude Gamma nodes; include wraparound interval.','circle_positions':9,'cycle_length':6},
 'gamma':{'alpha_indices':d['node_indices'],'clock_positions':[i or 24 for i in d['node_indices']],'clockwise_values':[d['display_alpha'][i] for i in d['node_indices']],'triangle_paths':d['triangles']},
 'epsilon':{'alpha_indices':d['unicursal_indices'],'clock_positions':[i or 24 for i in d['unicursal_indices']],'traversal_values':d['unicursal_labels'],'status':'Proposed geometric traversal; physical interpretation not assessed.'},
 'views':[{'id':key,'name':title,'description':subtitle,'svg':f'views/{key}.svg?v=3','png':f'views/{key}.png?v=3'} for key,title,subtitle,seq,note in views],
 'research':{'focus':'OXTR gene expression','biological_significance_status':'Not yet assessed','statistical_significance_status':'Not yet assessed','quantum_annealing_status':'Not yet performed'},
 'verification':{'python':'Period, node spacing, triangle spacing, and doubling recurrence checked by generator.','wolfram':'Reference notebook source available; this generator does not execute Wolfram.'}}
assert len(d['alpha'])==24 and d['node_indices']==list(range(0,24,4))
assert all((d['alpha'][i]+d['alpha'][(i+1)%24])%9==d['alpha'][(i+2)%24] for i in range(24))
assert d['beta'][4:]+d['beta'][:4]==d['doubling'][:-1]
assert all((2*a)%9==b for a,b in zip(d['doubling'],d['doubling'][1:]))
assert all((t[j+1]-t[j])%24==8 for t in d['triangles'] for j in range(3))
from shape_summary import shape_summary
record['geometry_summary']=shape_summary(record)
(ROOT/'Informatics/m9p24.json').write_text(json.dumps(record,indent=2)+'\n')
print('Five SVG/PNG views and M9/P24 JSON generated; mathematical checks passed.')
