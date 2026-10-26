import { Color } from 'molstar/lib/mol-util/color/index.js';
import { MeshBuilder } from 'molstar/lib/mol-geo/geometry/mesh/mesh-builder.js';
import { Mesh } from 'molstar/lib/mol-geo/geometry/mesh/mesh.js';
import { addCylinder } from 'molstar/lib/mol-geo/geometry/mesh/builder/cylinder.js';
import { addSphere } from 'molstar/lib/mol-geo/geometry/mesh/builder/sphere.js';
import { Shape } from 'molstar/lib/mol-model/shape.js';
import { ShapeRepresentation } from 'molstar/lib/mol-repr/shape/representation.js';
import { Mat4 } from 'molstar/lib/mol-math/linear-algebra.js';
import { OrderedSet } from 'molstar/lib/mol-data/int.js';
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms.js';
import { createVolumeRepresentationParams } from 'molstar/lib/mol-plugin-state/helpers/volume-representation-params.js';
import { Volume } from 'molstar/lib/mol-model/volume.js';

const radius={C:.76,N:.71,O:.66,S:1.05,H:.31};
const elementColor={O:0xe88379,N:0x8aabdf,S:0xe8ce78,H:0xe6e8e9};
const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
export async function createInteraction(engine,kind,data,onChange,onPick){
 const plugin=engine.plugin,shapes=[];let progress=1,focus='contact',planes=true,disposed=false;
 let playing=false,frame=0,last=0,field='homo',volumeData,surfaces=[],alpha=.56,iso=.025;
 async function mesh(name,build,opacity=1){
  const state=MeshBuilder.createState(2048,1024),colors=[],labels=[];
  const group=(color,label)=>{state.currentGroup=colors.length;colors.push(color);labels.push(label);};
  build(state,group);const shape=Shape.create(name,{},MeshBuilder.getMesh(state),i=>Color(colors[i]),()=>1,i=>labels[i]);
  const repr=ShapeRepresentation(()=>shape,Mesh.Utils);await plugin.runTask(repr.createOrUpdate({alpha:opacity,metalness:.18,roughness:.3,doubleSided:true},{}));
  plugin.canvas3d.add(repr);shapes.push(repr);return repr;
 }
 async function atomMesh(name,atoms,carbonColor){
  return mesh(name,(s,g)=>{
   for(const a of atoms){g(elementColor[a.element]||carbonColor,`${a.chain==='L'?'OXT':'OXTR'} · ${a.comp} ${a.residue} · ${a.name}`);addSphere(s,a.position,a.element==='H'?.14:.28,2);}
   for(let i=0;i<atoms.length;i++)for(let j=i+1;j<atoms.length;j++){
    const a=atoms[i],b=atoms[j],d=distance(a.position,b.position);
    if(a.chain!==b.chain||d<.5||d>(radius[a.element]+radius[b.element]) *1.22)continue;
    const mid=mix(a.position,b.position,.5);
    for(const [from,to,c] of [[a.position,mid,elementColor[a.element]||carbonColor],[mid,b.position,elementColor[b.element]||carbonColor]]){
     g(c,`${a.comp}${a.residue} ${a.name} — ${b.comp}${b.residue} ${b.name}`);addCylinder(s,from,to,1,{radiusTop:.105,radiusBottom:.105,radialSegments:10});
    }
   }
  });
 }
 async function ringPlane(ring,color,name){
  return mesh(name,(s,g)=>{g(color,name);for(let i=0;i<6;i++)MeshBuilder.addTriangle(s,ring.center,ring.points[i],ring.points[(i+1)%6]);},.20);
 }
 function camera(center,r,normal){
  const mobileQuantum=innerWidth<760&&kind==='quantum',n=normal||[.7,.35,1],len=Math.hypot(...n),position=center.map((v,i)=>v+n[i]/len*r*3.3*(mobileQuantum?1.3:1));
  const snap=plugin.managers.camera.getFocusSphereSnapshot({center,radius:r},{extraRadius:1});
  snap.position=position;snap.target=[...center];snap.up=[0,1,0];if(mobileQuantum){snap.position[1]+=r*.5;snap.target[1]+=r*.5;}snap.radius=r*3;snap.radiusMax=r*7;
  plugin.canvas3d.camera.setState(snap,350);engine.rememberCamera(snap);
 }
 let moving,ringLigand,fullPocket,ringPocket,tyrPlane,phePlane,contacts;
 const pickSubscription=plugin.behaviors.interaction.click.subscribe(({current:c})=>{
  if(disposed||c.loci.kind!=='shape-loci'||!c.loci.groups?.length)return;
  const i=OrderedSet.getAt(c.loci.groups[0].ids,0);onPick?.(c.loci.shape.getLabel(i,0));
 });
 if(kind==='docking'){
  await engine.render('molecular-surface');await engine.setTargetOpacity(.06);await engine.toggle('ligand',false);
  moving=await atomMesh('OXT · deposited ligand',data.ligand,0xe8b66d);
  ringLigand=await atomMesh('OXT · Tyr2 side chain',data.ligand.filter(a=>a.residue===2&&!['N','CA','C','O'].includes(a.name)),0xe8b66d);
  fullPocket=await atomMesh('OXTR · pocket contacts',data.pocket,0x8fcfdf);
  ringPocket=await atomMesh('OXTR · aromatic contact',data.pocket.filter(a=>(a.residue===291&&!['N','CA','C','O'].includes(a.name))||(a.residue===316&&['C','O'].includes(a.name))),0x8fcfdf);
  tyrPlane=await ringPlane(data.rings[0],0xe8b66d,'OXT Tyr2 aromatic plane');
  phePlane=await ringPlane(data.rings[1],0x8fcfdf,'OXTR Phe291 aromatic plane');
  contacts=await mesh('Bound-pose measurements',(s,g)=>{
   const pairs=data.rings[0].points.flatMap(a=>data.rings[1].points.map(b=>[a,b]));pairs.sort((a,b)=>distance(...a)-distance(...b));
   for(const [a,b,color,label] of [[...pairs[0],0xe8b66d,'Closest ring-carbon contact'],[...data.hydrogenBond,0xa3e2cd,'Tyr2 OH · L316 backbone O']]){
    for(let i=0;i<12;i+=2){g(color,label);addCylinder(s,mix(a,b,i/12),mix(a,b,(i+1)/12),1,{radiusTop:.045,radiusBottom:.045,radialSegments:8});}
   }
  });contacts.setState({pickable:false});
 }else{
  await loadField('homo');
 }
 function setProgress(value){
  progress=Math.max(0,Math.min(1,Number(value)));
  if(kind!=='docking')return;
  const offset=data.approachVector.map(v=>v*(1-progress));
  const transform=Mat4.fromTranslation(Mat4(),offset);
  moving.setState({transform});ringLigand.setState({transform});tyrPlane.setState({transform,visible:planes});phePlane.setState({visible:planes});
  contacts.setState({visible:progress===1});for(const repr of [moving,ringLigand,tyrPlane,phePlane,contacts])plugin.canvas3d.update(repr,false);plugin.canvas3d.syncVisibility();plugin.canvas3d.commit(true);onChange?.(snapshot());
 }
 function setFocus(next){
  focus=next;
  if(kind==='quantum'){
   const atoms=next==='ring'?data.atoms.filter(a=>a.group==='Tyr2'):data.atoms;
   const lo=[0,1,2].map(i=>Math.min(...atoms.map(a=>a.position[i]))),hi=[0,1,2].map(i=>Math.max(...atoms.map(a=>a.position[i]))),center=mix(lo,hi,.5);
   const r=Math.max(...atoms.map(a=>distance(a.position,center)))+1.4;camera(center,r,[.15,-.6,1]);return;
  }
  moving.setState({visible:next!=='contact'});ringLigand.setState({visible:next==='contact'});fullPocket.setState({visible:next!=='contact'});ringPocket.setState({visible:next==='contact'});
  plugin.canvas3d.syncVisibility();
  const a=data.rings[0],b=data.rings[1],center=mix(a.center,b.center,.5);
  if(next==='complex')camera(mix(data.receptorCenter,data.ligandCenter.map((v,i)=>v+data.approachVector[i]*.5),.55),31,[-1,.15,.45]);
  else if(next==='pocket')camera(data.ligandCenter,15,a.normal.map((v,i)=>v+b.normal[i]*.6));
  else camera(center,innerWidth<760?8:6,a.normal.map((v,i)=>v+b.normal[i]*.8));
 }
 function pause(){playing=false;cancelAnimationFrame(frame);onChange?.(snapshot());}
 function play(){
  if(kind!=='docking'||disposed)return;
  if(progress>=.999)setProgress(0);playing=true;last=performance.now();
  const step=now=>{if(disposed||!playing)return;const dt=Math.min(now-last,80);last=now;setProgress(progress+dt/12000);if(progress>=1)pause();else frame=requestAnimationFrame(step);};
  frame=requestAnimationFrame(step);onChange?.(snapshot());
 }
 async function loadField(next){
  field=next;if(volumeData){await plugin.state.data.build().delete(volumeData.ref).commit();volumeData=null;surfaces=[];}
  const file=next==='density'?'density.cube':data.orbitals.find(o=>o.id===next)?.file;
  if(!file)throw Error('This orbital is unavailable.');
  const response=await fetch('data/interactions/'+file);if(!response.ok)throw Error('The calculated field did not load.');
  const raw=await response.text();
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(raw))),x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==data.files[file])throw Error('The calculated field checksum does not match its source record.');
  volumeData=await plugin.builders.data.rawData({data:raw,label:next==='density'?'Calculated electron density':next.toUpperCase()});
  const parsed=await plugin.dataFormats.get('cube').parse(plugin,volumeData);
  const tree=plugin.build();
  for(const sign of next==='density'?[1]:[1,-1]){
   const params=createVolumeRepresentationParams(plugin,parsed.volume.data,{type:'isosurface',typeParams:{isoValue:Volume.IsoValue.absolute(iso*sign),alpha,tryUseGpu:false,quality:'high'},color:'uniform',colorParams:{value:Color(next==='density'?0xb7d8cc:sign>0?0xe8b66d:0x8fcfdf)}});
   surfaces.push(tree.to(parsed.volume).apply(StateTransforms.Representation.VolumeRepresentation3D,params).selector);
  }
  await tree.commit();plugin.canvas3d.commit(true);
 }
 async function fieldStyle({opacity=alpha,level=iso}={}){
  alpha=opacity;iso=level;const tree=plugin.build();
  surfaces.forEach((r,i)=>tree.to(r).update(StateTransforms.Representation.VolumeRepresentation3D,old=>({...old,type:{...old.type,params:{...old.type.params,alpha,isoValue:Volume.IsoValue.absolute(iso*(i===1?-1:1))}}})));
  await tree.commit();
 }
 function setPlanes(value){planes=value;tyrPlane?.setState({visible:planes});phePlane?.setState({visible:planes});plugin.canvas3d.syncVisibility();}
 function snapshot(){return kind==='docking'?{kind,progress,playing,focus,planes}:{kind,focus,field,opacity:alpha,isovalue:iso};}
 function dispose(){disposed=true;pause();pickSubscription.unsubscribe();for(const s of shapes){plugin.canvas3d.remove(s);s.destroy();}}
 setFocus(kind==='docking'?'contact':'pair');if(kind==='docking')setProgress(1);
 function geometrySphere(){return focus==='complex'?undefined:{center:focus==='contact'?mix(data.rings[0].center,data.rings[1].center,.5):data.ligandCenter,radius:focus==='contact'?7:15};}
 return {setProgress,setFocus,pause,play,setPlanes,loadField,fieldStyle,snapshot,geometrySphere,dispose};
}
