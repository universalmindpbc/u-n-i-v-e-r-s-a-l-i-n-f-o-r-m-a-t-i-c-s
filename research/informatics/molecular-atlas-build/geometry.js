// Unit circumradius constructions. Coordinates are mathematical, not biological.
export const PHI=(1+Math.sqrt(5))/2;
const add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),mul=(a,s)=>a.map(x=>x*s);
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),norm=a=>mul(a,1/Math.hypot(...a));
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const tetra=[[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]].map(norm);
export const geometryNames={tetrahedron:'Tetrahedron',cube:'Cube',octahedron:'Octahedron',dodecahedron:'Dodecahedron',icosahedron:'Icosahedron',stellated:'Stellated octahedron','truncated-icosahedron':'Truncated icosahedron',odom:'Odom · tetrahedral study',phi:'Phi sphere'};
export function geometry(name){
 let vertices=[],edges=[],segments=[],points=[],note='',ratio=null;
 if(name==='tetrahedron')vertices=tetra;
 if(name==='cube')for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])vertices.push([x,y,z]);
 if(name==='octahedron')vertices=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
 if(name==='icosahedron')for(const a of [-1,1])for(const b of [-PHI,PHI])vertices.push([0,a,b],[a,b,0],[b,0,a]);
 if(name==='dodecahedron'){
  for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])vertices.push([x,y,z]);
  for(const a of [-1/PHI,1/PHI])for(const b of [-PHI,PHI])vertices.push([0,a,b],[a,b,0],[b,0,a]);
 }
 if(name==='truncated-icosahedron'){
  // Truncate every icosahedron edge at one third from each endpoint.
  const ico=geometry('icosahedron');
  for(const [i,j] of ico.edges){
   const a=ico.vertices[i],b=ico.vertices[j];
   vertices.push(mul(add(mul(a,2),b),1/3),mul(add(a,mul(b,2)),1/3));
  }
 }
 if(vertices.length){
  vertices=vertices.map(norm);let min=Infinity;
  for(let i=0;i<vertices.length;i++)for(let j=i+1;j<vertices.length;j++)min=Math.min(min,Math.hypot(...sub(vertices[i],vertices[j])));
  for(let i=0;i<vertices.length;i++)for(let j=i+1;j<vertices.length;j++)if(Math.abs(Math.hypot(...sub(vertices[i],vertices[j]))-min)<1e-7)edges.push([i,j]);
  note='Regular solid · uniform circumradius. Position and scale are exploratory.';
 }
 if(name==='truncated-icosahedron')note='Archimedean solid · 12 pentagons and 20 hexagons. The ideal geometric form associated with buckminsterfullerene (C₆₀). Position and scale are exploratory.';
 if(name==='stellated'){
  vertices=[...tetra,...tetra.map(v=>mul(v,-1))];
  for(const offset of [0,4])for(let i=0;i<4;i++)for(let j=i+1;j<4;j++)edges.push([i+offset,j+offset]);
  note='Two interpenetrating regular tetrahedra. Their shared volume is an octahedron.';
 }
 if(name==='odom'){
  const faces=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]];
  for(const f of faces){
   const [a,b,c]=f.map(i=>tetra[i]);const centre=mul(add(add(a,b),c),1/3),r=Math.hypot(...sub(a,centre));
   const u=norm(sub(a,centre)),normal=norm(cross(sub(b,a),sub(c,a))),v=cross(normal,u);
   for(let i=0;i<3;i++)segments.push({a:f.map(k=>tetra[k])[i],b:f.map(k=>tetra[k])[(i+1)%3],color:0x91a4ad});
   for(let i=0;i<80;i++){
    const p=t=>add(centre,mul(add(mul(u,Math.cos(t)),mul(v,Math.sin(t))),r));
    segments.push({a:p(i*2*Math.PI/80),b:p((i+1)*2*Math.PI/80),color:0x50636e});
   }
   const p=mul(add(a,b),.5),q=mul(add(a,c),.5),dir=norm(sub(q,p)),mid=mul(add(p,q),.5);
   const half=Math.sqrt(r*r-dot(sub(mid,centre),sub(mid,centre)));
   const left=sub(mid,mul(dir,half)),right=add(mid,mul(dir,half));
   segments.push({a:left,b:right,color:0xc49c61},{a:p,b:q,color:0xf2c77d});points.push(p,q,right);
   ratio=Math.hypot(...sub(q,p))/Math.hypot(...sub(right,q));
  }
  note='Odom’s planar construction repeated on four tetrahedron faces. Midpoint segment ÷ adjacent extension = φ. The tetrahedral arrangement is our exploratory 3D extension.';
 }
 if(name==='phi'){
  for(let i=0;i<89;i++){const y=1-2*(i+.5)/89,r=Math.sqrt(1-y*y),angle=i*2*Math.PI*(1-1/PHI);points.push([r*Math.cos(angle),y,r*Math.sin(angle)]);}
  note='89 points distributed on a sphere with a Phi Ratio angular step: 360° × (1 − 1/φ). A mathematical construction.';
 }
 for(const [i,j] of edges)segments.push({a:vertices[i],b:vertices[j],color:name==='stellated'&&i>=4?0x85c5db:0xd5b57e});
 return {name:geometryNames[name],vertices,edges,segments,points:points.length?points:vertices,note,ratio};
}
export function transformPoint(point,{centre=[0,0,0],radius=1,scale=1,rotation=0,offset=0}={}){
 const t=rotation*Math.PI/180,[x,y,z]=point;
 return [centre[0]+radius*(scale*(x*Math.cos(t)+z*Math.sin(t))+offset),centre[1]+radius*scale*y,centre[2]+radius*scale*(-x*Math.sin(t)+z*Math.cos(t))];
}
