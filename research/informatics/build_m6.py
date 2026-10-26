"""Native-modulus M6 comparison; zero display is 6, never a mod9 projection."""
import json, math, html
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Informatics/views/m6'
OUT.mkdir(parents=True,exist_ok=True)
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

a,b=0,1
values=[]
while True:
 values.append(a); a,b=b,(a+b)%6
 if (a,b)==(0,1):break
n=len(values); display=[v or 6 for v in values]
nodes=[i for i,v in enumerate(values) if v%3==0]
sums=[sum(values[j%n] for j in range(lo+1,hi)) for lo,hi in zip(nodes,nodes[1:]+[n])]
beta=[v%6 for v in sums]
triangles=[[0,8,16,0],[4,12,20,4]]
assert not {3,6,9}.issubset(display) # Epsilon requires all three values; M1–M8 are ineligible.
assert n==24 and nodes==[0,4,8,12,16,20]
assert sums==[4,8,10,14,10,8] and beta==[4,2,4,2,4,2]
assert all((2*beta[i])%6==beta[(i+1)%6] for i in range(6))
assert all((values[i]+values[(i+1)%n])%6==values[(i+2)%n] for i in range(n))
assert all((t[j+1]-t[j])%n==n//3 for t in triangles for j in range(3))
def draw_alpha(g):
 g.circle((400,365),238,BLUE)
 for i,v in enumerate(display):
  g.text(v,pt(i,n,267),20);g.text(i or n,pt(i,n,294),10,MUTED)
def draw_gamma(g,labels=True):
 for t in triangles:g.line([pt(i,n) for i in t[:-1]],PINK,2,True,'#261525')
 for t in triangles:g.line([pt(i,n) for i in t],PINK,2)
 if labels:
  for i in nodes:g.circle(pt(i,n),5,PINK,fill=PINK);g.text(display[i],pt(i,n,270),24)
def draw_beta(g,labels=True):
 points=[pt(v,6) for v in beta+[beta[0]]];g.line(points,YELLOW,3)
 if labels:
  for i in range(6):g.text(i or 6,pt(i,6,270),24,YELLOW if i in beta else MUTED)
  g.text('start 4',pt(4,6,310),12,YELLOW)
  g.text('Repeated traversal: 4 ↔ 2',(400,365),18,YELLOW)
views=[('alpha','Alpha','24 native Fibonacci residues','Clock position 24 is north; zero is displayed as 6.'),('beta','Beta','4 → 2 → 4 → 2 → 4 → 2','Two-step doubling cycle modulo 6; repeated edges overlap.'),('gamma','Gamma','6 · 3 · 3 · 6 · 3 · 3','Same nodal positions as M9; different native values.'),('fibonacci-m6p24','Fibonacci M6P24','Alpha · Beta · Gamma','Alpha uses 24 positions; Beta uses six positions.')]
for key,title,seq,note in views:
 g=Drawing(title,note);g.circle((400,365),238,'#34434e')
 if key=='alpha':draw_alpha(g)
 elif key=='beta':draw_beta(g)
 elif key=='gamma':draw_gamma(g)
 else:
  draw_gamma(g,False);g.line([pt(i,6) for i in range(6)],GREEN,2,True);draw_alpha(g);draw_beta(g,False)
 g.text('M6 / P24',(400,32),12,MUTED);g.text(title,(400,690),24);g.text(seq,(400,735),17);g.text(note,(400,779),11,MUTED);g.save(key)
d={
 'schema_version':'1.1.0','id':'fibonacci-m6-p24','name':'Fibonacci M6P24','modulus':6,'pisano_period':n,
 'conventions':{'index_origin':0,'clock_labels':[24]+list(range(1,24)),'orientation':'clockwise from north','zero_display':6,'alpha_circle_positions':24,'beta_circle_positions':6,'node_rule':'Native residues divisible by 3, including zero; no modulo9 projection.'},
 'alpha':{'residues':values,'display_values':display},
 'beta':{'interval_sums':sums,'interval_residues':beta,'display_cycle':beta,'recognised_doubling_cycle':[2,4],'rotation_to_recognised':1,'start_value':4,'start_interval_clock_positions':[24,4],'start_interval_values':[1,1,2],'endpoint_rule':'Exclude Gamma nodes; include wraparound interval.','circle_positions':6,'cycle_length':2,'m9_doubling_cycle_present':False},
 'gamma':{'alpha_indices':nodes,'clock_positions':[i or 24 for i in nodes],'clockwise_values':[display[i] for i in nodes],'triangle_paths':triangles},
 'epsilon':None,
 'epsilon_status':'Null: requires sequential mapping of all three values 3, 6 and 9. M1–M8 cannot supply 9.',
 'views':[{'id':key,'name':title,'description':note,'svg':f'views/m6/{key}.svg?v=2','png':f'views/m6/{key}.png?v=2'} for key,title,seq,note in views],
 'research':{'focus':'OXTR gene expression','biological_significance_status':'Not yet assessed','statistical_significance_status':'Not yet assessed','quantum_annealing_status':'Not yet performed'},
 'verification':{'python':'Exact state recurrence, node positions, triangle spacing and native doubling checked.','wolfram':'Independent kernel confirmed period24, residues, nodes, interval sums, Beta on 2026-10-16.'}}
from shape_summary import shape_summary
d['geometry_summary']=shape_summary(d)
(ROOT/'Informatics/m6p24.json').write_text(json.dumps(d,indent=2)+'\n')
print('M6/P24 verified; four SVG and PNG views generated.')

# Replace legacy Epsilon URLs with an explicit null notice; no traversal remains.
g=Drawing('Epsilon — null','M6 cannot supply 9.');g.text('M6 / P24',(400,280),20,MUTED);g.text('Epsilon — null',(400,365),30);g.text('Requires 3, 6 and 9; M6 cannot supply 9.',(400,420),18,MUTED);g.save('epsilon')
