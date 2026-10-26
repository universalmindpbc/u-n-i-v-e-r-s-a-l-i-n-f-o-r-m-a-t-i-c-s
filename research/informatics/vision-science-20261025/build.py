import json,re,html
from pathlib import Path
root=Path(__file__).resolve().parents[3]
folder=Path(__file__).resolve().parent
source=json.loads((folder/'notion-source.json').read_text())
text=source['text'].split('<content>')[1].split('</content>')[0]
paras=[]
for raw in text.split('\n\n'):
 raw=raw.strip().removeprefix('- ')
 if raw and raw not in ['---','<p />'] and raw not in paras:paras.append(raw)
assert len(paras)==17
links={
 'retina':('https://www.nature.com/articles/eye1998142','The photoreceptor mosaic','Eye · 1998'),
 'cortex':('https://www.nature.com/articles/nn.2824','Retinal origin of orientation maps in visual cortex','Nature Neuroscience · 2011'),
 'space':('https://www.nature.com/articles/s41593-017-0050-8','Hexadirectional coding of visual space in human entorhinal cortex','Nature Neuroscience · 2018'),
}
def clean(p):
 p=re.sub(r'\\\[.*?\\\]','',p).strip()
 def link(m):
  label=m.group(1)
  key='retina' if 'Eye' in label else 'cortex' if 'Retinal origin' in label else 'space'
  return f'<a href="{links[key][0]}" target="_blank" rel="noopener noreferrer">{label}</a>'
 p=re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',link,p)
 return p.replace('<b>','<strong>').replace('</b>','</strong>').replace('<i>','<em>').replace('</i>','</em>')
def paragraph(i):return f'<p data-source-paragraph="{i}">{clean(paras[i])}</p>'
def pub(key):
 url,title,journal=links[key]
 return f'<a class="vs-publication" href="{url}" target="_blank" rel="noopener noreferrer"><span>{journal}</span><strong>{title}</strong><i aria-hidden="true">↗</i></a>'
content='''<section class="vision-science" id="vision-science" aria-labelledby="vision-science-title">
<div class="vs-masthead"><span>VISION ATLAS / RESEARCH STIMULUS</span><span>01 — 03</span></div>
<div class="vs-intro"><div><p class="vs-kicker">THE SCIENCE BEHIND THE GEOMETRY</p><h2 id="vision-science-title">Hexagons &amp;<br><em>equilateral triangles.</em></h2><div class="vs-mini-geometry" aria-hidden="true"></div></div><div class="vs-opening">'''+paragraph(0)+paragraph(1)+paragraph(2)+'''</div></div>
<nav class="vs-chapter-nav" aria-label="Explore the science"><a href="#vs-retina"><span>01</span> Retina <b aria-hidden="true">↓</b></a><a href="#vs-cortex"><span>02</span> Visual cortex <b aria-hidden="true">↓</b></a><a href="#vs-space"><span>03</span> Spatial coding <b aria-hidden="true">↓</b></a></nav>'''
for n,(key,title,subtitle,ids) in enumerate([
 ('retina','Hexagonal Packing in the Retina','Optimal Sampling',[4,5,6]),
 ('cortex','Moiré Interference Patterns in the Visual Cortex','',[8,9,10]),
 ('space','Hexadirectional Coding of Space','',[12,13,14])],1):
 controls=''
 if key=='retina':controls='<div class="vs-toggles" role="group" aria-label="Retinal geometry"><button type="button" data-vs-layer="cells" aria-pressed="true">Cells</button><button type="button" data-vs-layer="triangles" aria-pressed="false">Triangles</button><button type="button" data-vs-layer="hexagons" aria-pressed="false">Hexagons</button></div>'
 if key=='cortex':controls='<label class="vs-slider" for="vs-grid-rotation"><span>Rotate the overlapping grid <output id="vs-rotation-value" for="vs-grid-rotation">5°</output></span><input id="vs-grid-rotation" type="range" min="0" max="12" value="5" step="0.5"></label>'
 content+=f'<article class="vs-chapter vs-{key}" id="vs-{key}" aria-labelledby="vs-{key}-title"><figure class="vs-figure"><div class="vs-figure-heading"><span>0{n}</span><span>{["CELL PACKING","OVERLAPPING MOSAICS","SIXFOLD SYMMETRY"][n-1]}</span></div><div class="vs-diagram" data-vs-diagram="{key}"></div>{controls}<figcaption>{["Illustrative geometry · cell centres and their neighbours","Illustrative moiré model · two overlapping lattices","Illustrative spatial grid · six directions, 60° apart"][n-1]}</figcaption></figure><div class="vs-chapter-copy"><p class="vs-kicker">0{n} / {["RETINA","VISUAL CORTEX","SPATIAL CODING"][n-1]}</p><h3 id="vs-{key}-title">{title}</h3>'
 if subtitle:content+=f'<p class="vs-subtitle">{subtitle}</p>'
 content+=''.join(paragraph(i) for i in ids)+pub(key)+'</div></article>'
content+='<section class="vs-summary" aria-labelledby="vs-summary-title"><div class="vs-summary-symbol" aria-hidden="true"></div><div><p class="vs-kicker">THE CONNECTING IDEA</p><h3 id="vs-summary-title">Summary of the Geometry</h3>'+paragraph(16)+'</div></section><div class="vs-colophon"><span>VISION ATLAS</span><span>Research stimulus · Geometry, vision &amp; EMDR</span><a href="#vision-science">Back to the science ↑</a></div></section>'
(folder/'content.html').write_text(content)
(folder/'source-paragraphs.json').write_text(json.dumps({str(i):clean(paras[i]) for i in [0,1,2,4,5,6,8,9,10,12,13,14,16]},ensure_ascii=False,indent=2)+'\n')
p=root/'Informatics/vision-science.js'
p.write_text('/* Vision Atlas: content supplied in Notion, 25 October 2026. */\n(()=>{\nconst host=document.getElementById("area-vision");if(!host||document.getElementById("vision-science"))return;\nhost.insertAdjacentHTML("beforeend",'+json.dumps(content,ensure_ascii=False)+');\n'+(folder/'interactions.js').read_text()+'\n})();\n')
print('Built Vision explainer from 13 unique source paragraphs; duplicated Notion blocks removed.')
