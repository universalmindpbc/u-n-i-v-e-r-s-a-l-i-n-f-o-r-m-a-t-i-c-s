import {Color} from 'molstar/lib/mol-util/color/index.js';
import {MeshBuilder} from 'molstar/lib/mol-geo/geometry/mesh/mesh-builder.js';
import {Mesh} from 'molstar/lib/mol-geo/geometry/mesh/mesh.js';
import {addCylinder} from 'molstar/lib/mol-geo/geometry/mesh/builder/cylinder.js';
import {addSphere} from 'molstar/lib/mol-geo/geometry/mesh/builder/sphere.js';
import {addEllipsoid} from 'molstar/lib/mol-geo/geometry/mesh/builder/ellipsoid.js';
import {Torus} from 'molstar/lib/mol-geo/primitive/torus.js';
import {Shape} from 'molstar/lib/mol-model/shape.js';
import {ShapeRepresentation} from 'molstar/lib/mol-repr/shape/representation.js';
import {Mat4} from 'molstar/lib/mol-math/linear-algebra.js';
import {OrderedSet} from 'molstar/lib/mol-data/int.js';

// Schematic basis functions and teaching guides, independent of the DFT fields.
export const teachingProvenance={
 schema:'universal-informatics/quantum-teaching/v1',
 model:'Idealised phenol ring with a peptide attachment cue and schematic p orbitals.',
 interpretation:'The colours illustrate the opposite signs of p orbitals in one bonding combination. The complete aromatic pi system occupies several molecular orbitals. Guides and speed are symbolic, not electron trajectories or a time-dependent calculation.',
 geometry:'Regular six-carbon ring, 1.4 model units per edge; ellipsoidal p lobes and a schematic shared pi band. Dimensions do not represent a fitted electronic field. Ring hydrogens are implicit.',
 source:'https://openbooks.lib.msu.edu/oclue/chapter/chapter-8-conjugated-compounds-and-aromaticity/'
};
export const teachingRing=Array.from({length:6},(_,i)=>[1.4*Math.cos(i*Math.PI/3),0,1.4*Math.sin(i*Math.PI/3)]);

export async function createQuantumTeaching(engine,onChange,onPick,saved={}){
 const plugin=engine.plugin,shapes=[],viewport=document.querySelector('#viewport');
 let lesson=saved.lesson||2,labels=saved.labels!==false,guides=saved.guides!==false,speed=saved.speed||1,phase=saved.phase||0,playing=false,disposed=false,frame=0,last=0;
 await engine.clearScene('Interactive Quantum teaching model. Drag to rotate; click a carbon or orbital.');
 async function mesh(name,build,alpha=1){
  const state=MeshBuilder.createState(2048,1024),colors=[],names=[];
  const group=(color,label)=>{state.currentGroup=colors.length;colors.push(color);names.push(label);};
  build(state,group);
  const shape=Shape.create(name,{},MeshBuilder.getMesh(state),i=>Color(colors[i]),()=>1,i=>names[i]);
  const repr=ShapeRepresentation(()=>shape,Mesh.Utils);
  await plugin.runTask(repr.createOrUpdate({alpha,metalness:.22,roughness:.3,doubleSided:true},{}));
  plugin.canvas3d.add(repr);shapes.push(repr);return repr;
 }
 const bond=(s,a,b,r=.085)=>addCylinder(s,a,b,1,{radiusTop:r,radiusBottom:r,radialSegments:16});
 await mesh('Carbon framework',(s,g)=>{
  g(0xd8e5ec,'Aromatic carbon framework · ring hydrogens implicit');
  teachingRing.forEach((p,i)=>{addSphere(s,p,.23,3);bond(s,p,teachingRing[(i+1)%6]);});
  bond(s,teachingRing[0],[2.15,0,0]);bond(s,teachingRing[3],[-2.35,0,0]);addSphere(s,[-2.35,0,0],.21,3);bond(s,[-2.35,0,0],[-3.1,0,-.28]);
  g(0xe88379,'Phenolic oxygen');addSphere(s,[2.55,0,0],.24,3);bond(s,[2.15,0,0],[2.55,0,0]);
  g(0xeef3f5,'Hydroxyl hydrogen');bond(s,[2.55,0,0],[3.13,0,0],.065);addSphere(s,[3.13,0,0],.15,3);
 });
 const lobes=await mesh('Six schematic p orbitals',(s,g)=>{
  for(let i=0;i<6;i++)for(const sign of [1,-1]){
   g(sign>0?0x8fcfdf:0xe8b66d,`Carbon ${i+1} · schematic p-orbital ${sign>0?'negative':'positive'} sign`);
   const p=[teachingRing[i][0],sign*.75,teachingRing[i][2]];
   addEllipsoid(s,p,[0,1,0],[1,0,0],[.56,.56,.76],3);
  }
 },.48);
 const bands=await mesh('Shared pi system · schematic',(s,g)=>{
  for(const sign of [1,-1]){
   g(sign>0?0x8fcfdf:0xe8b66d,'Shared pi system · schematic delocalisation');
   const transform=Mat4.fromRotation(Mat4(),Math.PI/2,[1,0,0]);Mat4.setTranslation(transform,[0,sign*.78,0]);
   MeshBuilder.addPrimitive(s,transform,Torus({radius:1.4,tube:.31,radialSegments:24,tubularSegments:96}));
  }
 },.26);
 const tracks=await mesh('Symbolic motion guides',(s,g)=>{
  for(const sign of [1,-1]){
   g(sign>0?0x9fe6f3:0xf4d296,'Teaching guide · not an electron path');
   for(let i=0;i<72;i+=2)bond(s,[1.4*Math.cos(i*Math.PI/36),sign*1.04,1.4*Math.sin(i*Math.PI/36)],[1.4*Math.cos((i+1)*Math.PI/36),sign*1.04,1.4*Math.sin((i+1)*Math.PI/36)],.015);
  }
 });
 const markers=await mesh('Moving teaching markers',(s,g)=>{
  for(const sign of [1,-1])for(let i=0;i<3;i++){
   const t=i*Math.PI*2/3+(sign<0?.35:0);g(sign>0?0xadeeff:0xffdfaa,'Symbolic attention marker · not an electron');addSphere(s,[1.4*Math.cos(t),sign*1.04,1.4*Math.sin(t)],.075,2);
  }
 });
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('teaching-labels');svg.setAttribute('aria-hidden','true');viewport.append(svg);
 const callouts=[
  {text:'Carbon framework',p:teachingRing[3],dx:-52,dy:-30,step:1},
  {text:'p orbital',p:[.7,1.3,-1.21],dx:38,dy:-22,step:2},
  {text:'Shared π system',p:[.7,-.8,1.21],dx:35,dy:36,step:3},
  {text:'OH',p:[2.8,0,0],dx:9,dy:-25,step:1},
  {text:'To peptide',p:[-2.7,0,-.14],dx:-12,dy:32,step:1}
 ];
 for(const item of callouts){const node=document.createElementNS(svg.namespaceURI,'g');node.innerHTML='<path/><circle r="2.5"/><text/>';node.querySelector('text').textContent=item.text;svg.append(node);item.node=node;}
 function labelPositions(){
  if(disposed)return;svg.style.display=labels?'':'none';if(!labels)return;
  const canvas=viewport.querySelector('canvas'),w=viewport.clientWidth,h=viewport.clientHeight;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
  for(const item of callouts){
   const p=plugin.canvas3d.camera.project([0,0,0,0],item.p),x=p[0]*w/canvas.width,y=h-p[1]*h/canvas.height;
   item.node.style.display=lesson>=item.step&&p[2]>=0&&p[2]<=1?'':'none';
   const length=item.text.length*(w<760?5.5:6.6),endX=Math.min(w-length-12,Math.max(12,x+item.dx)),endY=Math.min(h-85,Math.max(w<760?150:70,y+item.dy));
   item.node.querySelector('path').setAttribute('d',`M${x},${y} L${endX},${endY-4}`);item.node.querySelector('circle').setAttribute('cx',x);item.node.querySelector('circle').setAttribute('cy',y);const text=item.node.querySelector('text');text.setAttribute('x',endX);text.setAttribute('y',endY-8);
  }
 }
 function refresh(){
  lobes.setState({visible:lesson>=2});bands.setState({visible:lesson===3});tracks.setState({visible:guides&&lesson>=2});markers.setState({visible:guides&&lesson>=2});
  for(const repr of shapes)plugin.canvas3d.update(repr,false);plugin.canvas3d.syncVisibility();plugin.canvas3d.commit(true);labelPositions();onChange?.(snapshot());
 }
 function setPhase(value){phase=((value%1)+1)%1;markers.setState({transform:Mat4.fromRotation(Mat4(),phase*Math.PI*2,[0,1,0])});plugin.canvas3d.update(markers,false);plugin.canvas3d.commit(true);}
 function pause(){playing=false;cancelAnimationFrame(frame);onChange?.(snapshot());}
 function play(){if(disposed||playing)return;if(lesson===1)lesson=2;guides=true;playing=true;last=performance.now();refresh();
  const step=now=>{if(disposed||!playing)return;const dt=Math.min(100,now-last);last=now;setPhase(phase+dt*speed/12000);frame=requestAnimationFrame(step);};frame=requestAnimationFrame(step);
 }
 function setLesson(next){lesson=Math.max(1,Math.min(3,Number(next)));if(lesson===1)pause();refresh();}
 function setLabels(value){labels=Boolean(value);refresh();}
 function setGuides(value){guides=Boolean(value);if(!guides)pause();refresh();}
 function setSpeed(value){speed=Math.max(.25,Math.min(2,Number(value)));onChange?.(snapshot());}
 function setFocus(){
  const w=viewport.clientWidth,h=viewport.clientHeight,mobile=w<600,d=Math.max(7.6,7.6/(2*Math.tan(Math.PI/8)*(w/h))*1.15),shift=mobile?1:.15;
  const target=[0,.91*shift,-.42*shift],position=[0,target[1]+d*.42,target[2]+d*.91];
  const snap={...plugin.canvas3d.camera.getSnapshot(),position,target,up:[0,1,0],radius:18,radiusMax:35};plugin.canvas3d.camera.setState(snap,350);engine.rememberCamera(snap);
 }
 function snapshot(){return {kind:'quantum',view:'teaching',lesson,labels,guides,speed,phase,playing,ringCarbons:6,pOrbitals:6,construction:teachingProvenance.schema};}
 const draw=plugin.canvas3d.didDraw.subscribe(labelPositions);
 const picked=plugin.behaviors.interaction.click.subscribe(({current:c})=>{if(!disposed&&c.loci.kind==='shape-loci'&&c.loci.groups?.length)onPick?.(c.loci.shape.getLabel(OrderedSet.getAt(c.loci.groups[0].ids,0),0));});
 function dispose(){disposed=true;pause();draw.unsubscribe();picked.unsubscribe();svg.remove();for(const repr of shapes){plugin.canvas3d.remove(repr);repr.destroy();}}
 setPhase(phase);setFocus();refresh();
 return {setFocus,setLesson,setLabels,setGuides,setSpeed,setPhase,play,pause,snapshot,dispose,provenance:teachingProvenance};
}
