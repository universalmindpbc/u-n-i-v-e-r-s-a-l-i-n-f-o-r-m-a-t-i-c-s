const ns='http://www.w3.org/2000/svg',section=host.querySelector('.vision-science');
function svg(label){const s=document.createElementNS(ns,'svg');s.setAttribute('viewBox','0 0 440 400');s.setAttribute('role','img');s.setAttribute('aria-label',label);return s;}
function el(name,attrs,parent){const n=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);parent.append(n);return n;}
function polygon(cx,cy,r,angle=0){return Array.from({length:6},(_,i)=>{const a=(angle+i*60)*Math.PI/180;return [cx+r*Math.cos(a),cy+r*Math.sin(a)].join(',');}).join(' ');}
function circle(parent,x,y,r,attrs={}){return el('circle',{cx:x,cy:y,r,...attrs},parent);}
function grid(parent,spacing,callback){for(let row=-7;row<=7;row++)for(let col=-7;col<=7;col++){const x=220+spacing*(col+(row%2)/2),y=200+spacing*Math.sqrt(3)/2*row;if(x>24&&x<416&&y>26&&y<374)callback(x,y,row,col);}}
function frame(s){circle(s,220,200,166,{fill:'none',stroke:'currentColor','stroke-opacity':'.12','stroke-width':'.8'});el('path',{d:'M28 200h16 M396 200h16 M220 14v16 M220 370v16',stroke:'currentColor','stroke-opacity':'.3'},s);}
const retina=svg('Idealised close packing of cells with equilateral triangle and hexagonal neighbourhood overlays');frame(retina);
const cellGroup=el('g',{'data-vs-cells':'','fill':'none','stroke':'#82dfb5','stroke-width':'1','stroke-opacity':'.55'},retina);
grid(cellGroup,44,(x,y)=>{if(Math.hypot(x-220,y-200)<155)circle(cellGroup,x,y,21);});
const triGroup=el('g',{'data-vs-triangles':'',visibility:'hidden','fill':'none','stroke':'#ec9cc9','stroke-width':'1.3'},retina);
grid(triGroup,44,(x,y)=>{if(Math.hypot(x-220,y-200)>=155)return;for(const[dx,dy]of [[44,0],[22,44*Math.sqrt(3)/2],[-22,44*Math.sqrt(3)/2]]){if(Math.hypot(x+dx-220,y+dy-200)<155)el('path',{d:`M${x} ${y}l${dx} ${dy}`},triGroup);}});
const hexGroup=el('g',{'data-vs-hexagons':'',visibility:'hidden','fill':'none','stroke':'#e7c98f','stroke-width':'1.1'},retina);
grid(hexGroup,44,(x,y)=>{if(Math.hypot(x-220,y-200)<135)el('polygon',{points:polygon(x,y,44/Math.sqrt(3),30)},hexGroup);});
const centres=el('g',{fill:'#d9f6e8'},retina);grid(centres,44,(x,y)=>{if(Math.hypot(x-220,y-200)<155)circle(centres,x,y,2);});
el('polygon',{points:polygon(220,200,44),fill:'#82dfb5','fill-opacity':'.08',stroke:'#82dfb5','stroke-width':'1.7'},retina);circle(retina,220,200,4,{fill:'#f6f7f4'});
section.querySelector('[data-vs-diagram="retina"]').append(retina);
for(const button of section.querySelectorAll('[data-vs-layer]'))button.addEventListener('click',()=>{for(const b of section.querySelectorAll('[data-vs-layer]'))b.setAttribute('aria-pressed',String(b===button));const kind=button.dataset.vsLayer;cellGroup.setAttribute('stroke-opacity',kind==='cells'?'.55':'.1');triGroup.setAttribute('visibility',kind==='triangles'?'visible':'hidden');hexGroup.setAttribute('visibility',kind==='hexagons'?'visible':'hidden');});
const moire=svg('Two idealised hexagonal point lattices: change their relative rotation to explore a moiré pattern');frame(moire);
const defs=el('defs',{},moire),clip=el('clipPath',{id:'vs-moire-clip'},defs);circle(clip,220,200,161);
const mask=el('g',{'clip-path':'url(#vs-moire-clip)'},moire),on=el('g',{fill:'#82dfb5','fill-opacity':'.85'},mask),off=el('g',{fill:'#ed94c8','fill-opacity':'.8',transform:'rotate(5 220 200)'},mask);
for(let row=-12;row<=12;row++)for(let col=-12;col<=12;col++){const x=220+22*(col+(row%2)/2),y=200+22*Math.sqrt(3)/2*row;circle(on,x,y,2.6);circle(off,220+(x-220)*1.035,200+(y-200)*1.035,2.6);}
section.querySelector('[data-vs-diagram="cortex"]').append(moire);
section.querySelector('#vs-grid-rotation').addEventListener('input',e=>{const angle=e.target.value;off.setAttribute('transform',`rotate(${angle} 220 200)`);section.querySelector('#vs-rotation-value').value=angle+'°';});
const space=svg('An idealised triangular spatial grid with six directions separated by sixty degrees');frame(space);
const mesh=el('g',{fill:'none',stroke:'#8caee2','stroke-width':'.7','stroke-opacity':'.35'},space);
grid(mesh,54,(x,y)=>{if(Math.hypot(x-220,y-200)<135)el('path',{d:`M${x} ${y}l54 0 -27 ${54*Math.sqrt(3)/2}Z`},mesh);});
grid(space,54,(x,y)=>{if(Math.hypot(x-220,y-200)<155)circle(space,x,y,3,{fill:'#9dc2ed','fill-opacity':'.55'});});
el('polygon',{points:polygon(220,200,108),fill:'#9dc2ed','fill-opacity':'.04',stroke:'#a4c6ed','stroke-width':'1.7'},space);
for(let i=0;i<6;i++){const a=i*Math.PI/3,x=220+108*Math.cos(a),y=200+108*Math.sin(a);el('path',{d:`M220 200L${x} ${y}`,stroke:'#a4c6ed','stroke-width':'1.3'},space);circle(space,x,y,6,{fill:'#e0ecff'});}
circle(space,220,200,8,{fill:'#f5f4e8'});el('path',{d:'M256 200A36 36 0 0 1 238 231.18',fill:'none',stroke:'#e7c98f','stroke-width':'1.5'},space);const t=el('text',{x:270,y:229,fill:'#e7c98f','font-size':'13'},space);t.textContent='60°';section.querySelector('[data-vs-diagram="space"]').append(space);
for(const target of section.querySelectorAll('.vs-mini-geometry,.vs-summary-symbol')){const s=svg('');s.removeAttribute('role');s.setAttribute('aria-hidden','true');s.setAttribute('viewBox','90 70 260 260');el('polygon',{points:polygon(220,200,108,30),fill:'none',stroke:'currentColor','stroke-width':'1'},s);for(const points of ['220,92 313.53,254 126.47,254','220,308 126.47,146 313.53,146'])el('polygon',{points,fill:'none',stroke:'currentColor','stroke-width':'.7','stroke-opacity':'.6'},s);target.append(s);}
section.addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]');if(!a)return;const target=section.querySelector(a.getAttribute('href'))||section;if(target){e.preventDefault();target.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});const heading=target.querySelector('h2,h3');if(heading){heading.setAttribute('tabindex','-1');heading.focus({preventScroll:true});}}});
