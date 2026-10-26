import math, html
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
OUT=Path(".")
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
