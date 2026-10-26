const ns='http://www.w3.org/2000/svg';
const node=(tag,attrs={},text)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
const line=(svg,x1,y1,x2,y2,color='#e9d86b',extra={})=>svg.append(node('line',{x1,y1,x2,y2,stroke:color,'stroke-width':1.4,...extra}));
const circle=(svg,cx,cy,r,extra={})=>svg.append(node('circle',{cx,cy,r,fill:'none',stroke:'#52666b',...extra}));
const text=(svg,x,y,t,extra={})=>svg.append(node('text',{x,y,fill:'#e9d86b','font-size':11,'text-anchor':'middle',...extra},t));
const notes={
 fibonacci:'Successive Fibonacci ratios approach the Phi Ratio. This is an exact limiting relationship. It does not by itself specify a physical arrangement of atoms.',
 modular:'M9 P24 means nine residue classes in a 24-step cycle. The Atlas places period positions on a circle; Gamma and Epsilon visit six equally spaced positions. M1–M1001 lets us compare modular representations. These are mathematical coordinates, not nine physical spatial dimensions.',
 odom:'Odom’s construction gives MP/MN = φ exactly: M and N bisect two sides of an equilateral triangle, and the ray MN meets its circumcircle at P. A regular tetrahedron has equilateral faces, so this construction applies within each face.',
 honeycomb:'Hales proved that regular hexagonal tiling minimizes boundary length when the plane is divided into equal-area cells. This is a geometric efficiency theorem; it does not establish quantum stability or minimum energy for every physical system.',
 quantum:'Working hypothesis: does the hexagonal structure associated with these representations contribute to quantum coherence? A direct test varies pulse ordering and atom geometry independently, with matched controls and a coherence measurement. The Nature study supports Fibonacci-driven quantum effects; it does not establish this geometric mechanism.'
};
for(const button of document.querySelectorAll('[data-hypothesis]'))button.onclick=()=>{
 document.querySelectorAll('[data-hypothesis]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
 document.querySelector('#hypothesis-detail').textContent=notes[button.dataset.hypothesis];
};
document.querySelector('#hypothesis-detail').textContent=notes.fibonacci;
const icon=key=>document.querySelector(`[data-hypothesis-icon="${key}"]`);
const fib=icon('fibonacci');text(fib,90,31,'1 · 1 · 2 · 3 · 5 · 8 · 13');line(fib,25,47,151,47);text(fib,90,73,'Fₙ₊₁ / Fₙ → φ',{'font-size':17});
const odom=icon('odom'),r=36,cx=90,cy=43,dx=Math.sqrt(3)*r/2;
circle(odom,cx,cy,r);odom.append(node('polygon',{points:`${cx},${cy-r} ${cx-dx},${cy+r/2} ${cx+dx},${cy+r/2}`,fill:'none',stroke:'#a3ead1','stroke-width':1.5}));
const mx=cx-dx/2,nx=cx+dx/2,my=cy-r/4,px=cx+Math.sqrt(r*r-(my-cy)**2);
line(odom,mx,my,px,my);for(const[x,label]of[[mx,'M'],[nx,'N'],[px,'P']]){circle(odom,x,my,2,{fill:'#e9d86b',stroke:'#e9d86b'});text(odom,x,my-6,label,{'font-size':8});}text(odom,90,88,'MP / MN = φ',{'font-size':10});
const honey=icon('honeycomb'),radius=17;
for(const[x,y]of[[90,45],[90+Math.sqrt(3)*radius,45],[90-Math.sqrt(3)*radius,45],[90+Math.sqrt(3)*radius/2,45-1.5*radius],[90-Math.sqrt(3)*radius/2,45-1.5*radius],[90+Math.sqrt(3)*radius/2,45+1.5*radius],[90-Math.sqrt(3)*radius/2,45+1.5*radius]]){
 honey.append(node('polygon',{points:Array.from({length:6},(_,i)=>`${x+radius*Math.sin(i*Math.PI/3)},${y-radius*Math.cos(i*Math.PI/3)}`).join(' '),fill:'none',stroke:'#e9d86b','stroke-width':1,'stroke-opacity':.65}));
}
const q=icon('quantum');line(q,17,45,61,45,'#e9d86b',{'stroke-dasharray':'3 5'});circle(q,104,44,29,{stroke:'#e9d86b','stroke-dasharray':'3 5'});text(q,104,54,'?',{'font-size':31});
try{
 const response=await fetch('data/oxtr-qrc-study.json');if(!response.ok)throw Error();const {atlas}=await response.json(),svg=icon('modular');
 const point=i=>[90+35*Math.sin(i*Math.PI/12),43-35*Math.cos(i*Math.PI/12)];circle(svg,90,43,35);
 for(let i=0;i<24;i++){const[x,y]=point(i);circle(svg,x,y,1,{fill:'#52666b',stroke:'none'});}
 for(const path of atlas.gamma.triangle_paths)svg.append(node('polyline',{points:path.map(i=>point(i).join(',')).join(' '),fill:'none',stroke:'#e9d86b','stroke-width':1.4}));
 for(const i of atlas.gamma.alpha_indices){const[x,y]=point(i);circle(svg,x,y,2.3,{fill:'#e9d86b',stroke:'none'});}
}catch{icon('modular').append(node('text',{x:90,y:45,fill:'#a9bbc4','font-size':10,'text-anchor':'middle'},'Geometry unavailable'));}
