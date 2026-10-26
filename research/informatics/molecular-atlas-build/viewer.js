import {PluginContext} from 'molstar/lib/mol-plugin/context.js';
import {DefaultPluginSpec} from 'molstar/lib/mol-plugin/spec.js';
import {MolScriptBuilder as MS} from 'molstar/lib/mol-script/language/builder.js';
import {Script} from 'molstar/lib/mol-script/script.js';
import {StructureSelection,StructureElement,StructureProperties} from 'molstar/lib/mol-model/structure.js';
import {dnaModel,dnaModelSource,complement} from './dna.js';
export {dnaModelSource};
export {createInteraction} from './interactions.js';
export {createQuantumTeaching,teachingProvenance,teachingRing} from './quantum-teaching.js';
import {Color} from 'molstar/lib/mol-util/color/index.js';
import {MeshBuilder} from 'molstar/lib/mol-geo/geometry/mesh/mesh-builder.js';
import {Mesh} from 'molstar/lib/mol-geo/geometry/mesh/mesh.js';
import {addCylinder} from 'molstar/lib/mol-geo/geometry/mesh/builder/cylinder.js';
import {addSphere} from 'molstar/lib/mol-geo/geometry/mesh/builder/sphere.js';
import {Shape} from 'molstar/lib/mol-model/shape.js';
import {ShapeRepresentation} from 'molstar/lib/mol-repr/shape/representation.js';
import {PluginCommands} from 'molstar/lib/mol-plugin/commands.js';
import {geometry,transformPoint,PHI,geometryNames} from './geometry.js';
export {geometry,PHI,geometryNames};
const chainExpr=chains=>MS.struct.generator.atomGroups({'chain-test':MS.core.set.has([MS.set(...chains),MS.ammp('label_asym_id')])});
export async function create(parent,onPick){
 const canvas=document.createElement('canvas');parent.append(canvas);canvas.tabIndex=0;canvas.setAttribute('aria-label','Interactive molecular structure. Drag to rotate, scroll to zoom. Keyboard: arrows rotate, plus and minus zoom, R resets.');
 const plugin=new PluginContext(DefaultPluginSpec());await plugin.init();
 if(!await plugin.initViewerAsync(canvas,parent))throw new Error('WebGL is unavailable. Try a browser with hardware acceleration enabled.');
 const small=innerWidth<700;
 plugin.canvas3d.setProps({transparentBackground:true,camera:{manualReset:true,helper:{axes:{name:'off',params:{}}},mode:'perspective'},cameraFog:{name:'off',params:{}},renderer:{backgroundColor:Color(0x0b1014),ambientIntensity:.48,light:[{inclination:150,azimuth:320,color:Color(0xe5eff4),intensity:.7},{inclination:65,azimuth:105,color:Color(0x98c9e0),intensity:.35}],highlightColor:Color(0x79d6f4),selectColor:Color(0xe7b76e)},postprocessing:{occlusion:{name:small?'off':'on',params:{samples:16,multiScale:{name:'off',params:{}},radius:5,bias:.8,blurKernelSize:15,resolutionScale:1,color:Color(0x000000)}},outline:{name:'off',params:{}},shadow:{name:'off',params:{}}},multiSample:{mode:'off',sampleLevel:2},trackball:{animate:{name:'off',params:{}}}});
 let target,structure,context,ligand,repr,ligandRepr,contextRepr,overlay,current,defaultCamera,clipRadius,representation='cartoon',geneData,geneWindow,geneAtomicModel;
 const resize=new ResizeObserver(()=>plugin.canvas3d.requestResize());resize.observe(parent);
 plugin.behaviors.interaction.click.subscribe(({current:c})=>{
  if(c.loci.kind!=='element-loci'||!c.loci.elements.length)return;
  const loc=StructureElement.Loci.getFirstLocation(c.loci);
  if(loc&&geneData){const chain=StructureProperties.chain.auth_asym_id(loc),residue=StructureProperties.residue.auth_seq_id(loc),i=chain==='A'?residue-1:geneWindow.sequence.length-residue;onPick({kind:'gene',offset:geneWindow.start+i,base:geneWindow.sequence[i],complement:complement[geneWindow.sequence[i]],position:geneWindow.position(i),atom:StructureProperties.atom.label_atom_id(loc),element:StructureProperties.atom.type_symbol(loc),chain});return;}
  if(loc)onPick({chain:StructureProperties.chain.auth_asym_id(loc),labelChain:StructureProperties.chain.label_asym_id(loc),residue:StructureProperties.residue.auth_seq_id(loc),name:StructureProperties.atom.label_comp_id(loc),seqId:StructureProperties.residue.label_seq_id(loc)});
 });
 async function render(type=representation){
  representation=type;
  if(repr)await plugin.state.data.build().delete(repr.ref).commit();
  const chainPalette={asymId:'auth',palette:{name:'colors',params:{list:{kind:'set',colors:[Color(0xd9b27a),Color(0xc7d6de)]}}}};
  const color=geneData?(type==='ball-and-stick'?'element-symbol':'chain-id'):'uniform';
  const colorParams=geneData?(type==='ball-and-stick'?{carbonColor:{name:'chain-id',params:chainPalette},saturation:-.2,lightness:.3}:chainPalette):{value:Color(0xc7d6de)};
  repr=await plugin.builders.structure.representation.addRepresentation(target,{type,typeParams:{quality:small?'medium':'high',alpha:type==='molecular-surface'?(geneData?.94:.88):1,metalness:.28,roughness:.28,sizeFactor:type==='ball-and-stick'?(geneData?.28:.20):geneData?.4:.26,resolution:small?(geneData?1.1:1.3):.8,...(geneData&&type==='cartoon'?{visuals:['polymer-trace','nucleotide-ring']}:{} )},color,colorParams});
  plugin.canvas3d.commit(true);
 }
 function removeOverlay(){if(overlay){plugin.canvas3d.remove(overlay);overlay.destroy();overlay=null;}}
 async function loadGene(g,offset=0,keepCamera=false,type='cartoon'){
  const camera=keepCamera?plugin.canvas3d.camera.getSnapshot():null;removeOverlay();
  await plugin.clear();target=structure=ligand=context=repr=ligandRepr=contextRepr=null;
  geneData=g;geneAtomicModel=dnaModel(g,offset);geneWindow=geneAtomicModel.window;current=null;clipRadius=null;
  const data=await plugin.builders.data.rawData({data:geneAtomicModel.pdb,label:g.symbol+' · modelled B-DNA'},{state:{isGhost:true}});
  const trajectory=await plugin.builders.structure.parseTrajectory(data,'pdb'),model=await plugin.builders.structure.createModel(trajectory);
  structure=await plugin.builders.structure.createStructure(model,{name:'model',params:{}});
  target=await plugin.builders.structure.tryCreateComponentFromExpression(structure,chainExpr(['A','B']),'gene');
  if(!target?.obj?.data?.elementCount)throw new Error('The atomic DNA model did not load.');
  await render(type);
  if(camera)plugin.canvas3d.camera.setState(camera,0);else{
   plugin.canvas3d.camera.setState({position:[0,55,235],target:[0,0,0],up:small?[-.95,.31,0]:[-.18,1,0],radius:geneWindow.sphere.radius,radiusMax:geneWindow.sphere.radius*5},0);
   const s=plugin.managers.camera.getFocusSphereSnapshot(geneWindow.sphere,{extraRadius:2}),factor=small?1.45:.86,shift=s.up.map(x=>small?x*35:0);s.position=s.position.map((x,i)=>s.target[i]+(x-s.target[i])*factor+shift[i]);s.target=s.target.map((x,i)=>x+shift[i]);plugin.canvas3d.requestCameraReset({snapshot:s,durationMs:0});defaultCamera=s;
  }
  canvas.setAttribute('aria-label','Interactive modelled DNA molecular structure. Drag to rotate; click a base pair. Arrow keys rotate, plus and minus zoom, R resets.');
  return {start:geneWindow.start,count:geneWindow.sequence.length,atoms:target.obj.data.elementCount};
 }
 async function load(record){
  geneData=geneWindow=geneAtomicModel=null;canvas.setAttribute('aria-label','Interactive protein structure. Drag to rotate, scroll to zoom. Arrow keys rotate, plus and minus zoom, R resets.');
  removeOverlay();await plugin.clear();target=structure=ligand=context=repr=ligandRepr=contextRepr=null;clipRadius=null;current=record;
  const data=await plugin.builders.data.download({url:'data/'+record.pdb.toLowerCase()+'.bcif',isBinary:true},{state:{isGhost:true}});
  const trajectory=await plugin.builders.structure.parseTrajectory(data,'mmcif');const model=await plugin.builders.structure.createModel(trajectory);
  structure=await plugin.builders.structure.createStructure(model,{name:'model',params:{}});
  target=await plugin.builders.structure.tryCreateComponentFromExpression(structure,chainExpr(record.chains),'target');
  if(!target?.obj?.data?.elementCount)throw new Error('The selected molecular chains are unavailable.');
  await render('cartoon');
  if(record.ligandChains?.length)ligand=await plugin.builders.structure.tryCreateComponentFromExpression(structure,chainExpr(record.ligandChains),'ligand');
  else if(record.ligands?.length)ligand=await plugin.builders.structure.tryCreateComponentFromExpression(structure,MS.struct.generator.atomGroups({'residue-test':MS.core.set.has([MS.set(...record.ligands),MS.ammp('label_comp_id')])}),'ligand');
  if(ligand?.obj?.data?.elementCount)ligandRepr=await plugin.builders.structure.representation.addRepresentation(ligand,{type:'ball-and-stick',typeParams:{sizeFactor:.30,metalness:.25,roughness:.25},color:'uniform',colorParams:{value:Color(0xe6ac62)}});
  if(record.contextChains?.length){context=await plugin.builders.structure.tryCreateComponentFromExpression(structure,chainExpr(record.contextChains),'context');if(context?.obj?.data?.elementCount)contextRepr=await plugin.builders.structure.representation.addRepresentation(context,{type:'cartoon',typeParams:{alpha:.35},color:'uniform',colorParams:{value:Color(0x688998)}});if(contextRepr)await PluginCommands.State.ToggleVisibility(plugin,{state:plugin.state.data,ref:contextRepr.ref});}
  plugin.canvas3d.commit(true);
  plugin.canvas3d.camera.setState({radiusMax:structure.obj.data.boundary.sphere.radius*5},0);
  plugin.managers.camera.focusSphere(target.obj.data.boundary.sphere,{durationMs:0,extraRadius:4});
  plugin.canvas3d.commit(true);
  plugin.managers.camera.orientAxes([target.obj.data],0);
  plugin.canvas3d.commit(true);
  await new Promise(r=>setTimeout(r,120));
  const upright=plugin.canvas3d.camera.getSnapshot(),d=upright.position.map((x,i)=>x-upright.target[i]),u=upright.up;
  upright.up=[d[1]*u[2]-d[2]*u[1],d[2]*u[0]-d[0]*u[2],d[0]*u[1]-d[1]*u[0]];
  const upLength=Math.hypot(...upright.up);upright.up=upright.up.map(x=>x/upLength);plugin.canvas3d.camera.setState(upright,0);
  defaultCamera=plugin.canvas3d.camera.getSnapshot();
  return {atoms:target.obj.data.elementCount,ligand:!!ligandRepr,context:!!contextRepr,radius:target.obj.data.boundary.sphere.radius};
 }
 async function loadFragment(pdb){
  geneData=geneWindow=geneAtomicModel=null;removeOverlay();await plugin.clear();target=structure=ligand=context=repr=ligandRepr=contextRepr=null;current={pdb:'7QVM fragment'};clipRadius=null;
  const raw=await plugin.builders.data.rawData({data:pdb,label:'Tyr2–Phe291 capped side-chain pair'});
  const trajectory=await plugin.builders.structure.parseTrajectory(raw,'pdb'),model=await plugin.builders.structure.createModel(trajectory);
  structure=await plugin.builders.structure.createStructure(model,{name:'model',params:{}});
  target=await plugin.builders.structure.tryCreateComponentStatic(structure,'all');
  await render('ball-and-stick');plugin.managers.camera.focusSphere(target.obj.data.boundary.sphere,{durationMs:0,extraRadius:3});defaultCamera=plugin.canvas3d.camera.getSnapshot();
  canvas.setAttribute('aria-label','Interactive calculated electronic structure. Drag to rotate. Arrow keys rotate, plus and minus zoom.');
  return {atoms:target.obj.data.elementCount};
 }
 async function clearScene(label){geneData=geneWindow=geneAtomicModel=null;removeOverlay();await plugin.clear();target=structure=ligand=context=repr=ligandRepr=contextRepr=null;current={pdb:'Quantum teaching model'};clipRadius=null;canvas.setAttribute('aria-label',label);}
 async function setTargetOpacity(alpha){if(!repr)return;await plugin.state.data.build().to(repr).update(old=>({...old,type:{...old.type,params:{...old.type.params,alpha}}})).commit();}
 async function geometryLayer(name,opts={}){
  removeOverlay();if(!name||(!target&&!geneWindow))return;
  const g=geometry(name),s=opts.sphere||geneWindow?.sphere||target.obj.data.boundary.sphere;const tf={centre:s.center,radius:s.radius,scale:opts.scale??1.2,rotation:opts.rotation??0,offset:opts.offset??0};
  const state=MeshBuilder.createState(2048,1024);const colors=[];let group=0;const width=s.radius*.003;
  for(const segment of g.segments){state.currentGroup=group++;colors.push(segment.color);addCylinder(state,transformPoint(segment.a,tf),transformPoint(segment.b,tf),1,{radiusTop:width,radiusBottom:width,radialSegments:8});}
  for(const p of g.points){state.currentGroup=group++;colors.push(0xe5bd7c);addSphere(state,transformPoint(p,tf),width*2.4,1);}
  const mesh=MeshBuilder.getMesh(state);const shape=Shape.create(g.name,g,mesh,i=>Color(colors[i]),()=>1,()=>g.name);
  overlay=ShapeRepresentation(()=>shape,Mesh.Utils);await plugin.runTask(overlay.createOrUpdate({alpha:opts.opacity??.7,ignoreLight:true,doubleSided:true},g));overlay.setState({pickable:false});plugin.canvas3d.add(overlay);
  clipRadius=Math.max(s.radius+4,s.radius*(tf.scale+Math.abs(tf.offset))+4);plugin.canvas3d.camera.setState({radius:clipRadius,radiusMax:Math.max(geneWindow?.sphere.radius||structure.obj.data.boundary.sphere.radius,clipRadius)},0);return g;
 }
 async function toggle(which,visible){const r=which==='ligand'?ligandRepr:contextRepr;if(!r)return;const cell=plugin.state.data.cells.get(r.ref);if(Boolean(cell.state.isHidden)===visible)await PluginCommands.State.ToggleVisibility(plugin,{state:plugin.state.data,ref:r.ref});}
 function focusResidue(chain,seqId){
  if(!structure)return;const expression=MS.struct.generator.atomGroups({'chain-test':MS.core.rel.eq([MS.ammp('label_asym_id'),chain]),'residue-test':MS.core.rel.eq([MS.ammp('label_seq_id'),seqId])});
  const selection=Script.getStructureSelection(expression,structure.obj.data),loci=StructureSelection.toLociWithSourceUnits(selection);
  if(!loci.elements.length)return false;plugin.managers.interactivity.lociSelects.deselectAll();plugin.managers.interactivity.lociSelects.select({loci});plugin.managers.camera.focusLoci(loci,{durationMs:400,minRadius:9,extraRadius:5});return true;
 }
 function reset(){if(defaultCamera)plugin.canvas3d.camera.setState({...defaultCamera,...(overlay&&clipRadius?{radius:clipRadius,radiusMax:Math.max(defaultCamera.radiusMax,clipRadius)}:{})},300);plugin.managers.interactivity.lociSelects.deselectAll();}
 function zoom(factor){const snap=plugin.canvas3d.camera.getSnapshot();snap.position=snap.position.map((x,i)=>snap.target[i]+(x-snap.target[i])*factor);plugin.managers.camera.setSnapshot(snap,100);}
 function rotate(dx,dy){const s=plugin.canvas3d.camera.getSnapshot(),v=s.position.map((x,i)=>x-s.target[i]);const x=v[0]*Math.cos(dx)+v[2]*Math.sin(dx),z=-v[0]*Math.sin(dx)+v[2]*Math.cos(dx);s.position=[x+s.target[0],v[1]*Math.cos(dy)-z*Math.sin(dy)+s.target[1],v[1]*Math.sin(dy)+z*Math.cos(dy)+s.target[2]];plugin.managers.camera.setSnapshot(s,100);}
 canvas.addEventListener('keydown',e=>{const key=e.key;if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','r','R'].includes(key)){e.preventDefault();if(key==='r'||key==='R')reset();else if(key==='+'||key==='='||key==='-')zoom(key==='-'?1.15:.87);else rotate(key==='ArrowLeft'?-.15:key==='ArrowRight'?.15:0,key==='ArrowUp'?-.15:key==='ArrowDown'?.15:0);}});
 function spin(on){plugin.canvas3d.setProps({trackball:{animate:on?{name:'spin',params:{speed:.12}}:{name:'off',params:{}}}});}
 async function screenshot(options={}){const helper=plugin.helpers.viewportScreenshot,previous=helper.behaviors.values.value;helper.behaviors.values.next({...previous,transparent:options.transparent??false,...(options.size?{resolution:{name:'custom',params:{width:options.size,height:options.size}}}:{})});try{return await helper.getImageDataUri();}finally{helper.behaviors.values.next(previous);}}
 function observed(chain){
  const ids=new Set();if(!target)return [];
  for(const unit of target.obj.data.units){if(unit.kind!==0)continue;const a=unit.model.atomicHierarchy;
   for(const e of unit.elements){const ci=a.chainAtomSegments.index[e];if(a.chains.label_asym_id.value(ci)!==chain)continue;ids.add(a.residues.label_seq_id.value(a.residueAtomSegments.index[e]));}
  }return [...ids];
 }
 function snapshot(){return {level:geneData?'gene':'protein',...(geneData?{gene:geneData.symbol,windowOffset:geneWindow.start,windowBases:geneWindow.sequence.length,atomCount:geneAtomicModel.atoms.length,modelSource:dnaModelSource}:{pdb:current?.pdb}),representation,camera:plugin.canvas3d.camera.getSnapshot()};}
 function dispose(){removeOverlay();resize.disconnect();plugin.dispose();}
 return {load,loadGene,loadFragment,clearScene,setTargetOpacity,rememberCamera:snapshot=>{defaultCamera=snapshot;},render,geneCoordinates:()=>geneAtomicModel?.pdb,geometryLayer,toggle,focusResidue,reset,zoom,rotate,spin,screenshot,snapshot,observed,dispose,plugin};
}
