'use strict';
window.createInteractionUI=function({enqueue,geometryChanged,showEvidence,download}){
 const $=s=>document.querySelector(s);
 const panel=document.createElement('section');panel.className='interaction-controls';panel.hidden=true;
 $('.inspector').prepend(panel);
 const legend=document.createElement('div');legend.className='interaction-legend';legend.hidden=true;$('.scene').append(legend);
 const quantumSwitch=document.createElement('div');quantumSwitch.className='quantum-switch';quantumSwitch.hidden=true;quantumSwitch.setAttribute('role','group');quantumSwitch.setAttribute('aria-label','Quantum mode');quantumSwitch.innerHTML='<button data-quantum-view="teaching">Teaching</button><button data-quantum-view="calculated">Calculated</button>';$('.scene').append(quantumSwitch);
 const teachingNote=document.createElement('p');teachingNote.className='teaching-note';teachingNote.hidden=true;teachingNote.textContent='Symbolic teaching model. Moving markers guide the explanation; they do not trace electron trajectories.';$('.scene').append(teachingNote);
 let mode,model,source,engine,styleTimer,active=false,quantumView=new URLSearchParams(location.search).get('quantum')==='calculated'?'calculated':'teaching',teachingState={},calculatedState={};
 quantumSwitch.querySelectorAll('button').forEach(b=>b.onclick=()=>enqueue(()=>active&&mode==='quantum'?loadQuantumView(b.dataset.quantumView):null));
 const labels={'homo-minus-1':'HOMO − 1',homo:'HOMO',lumo:'LUMO',density:'Electron density'};
 async function json(file){const response=await fetch('data/interactions/'+file);if(!response.ok)throw Error('The interaction model did not load.');return response.json();}
 function progress(state){if(mode!=='docking'||!active)return;
  $('#dock-progress').value=String(Math.round(state.progress*1000));$('#dock-progress-label').textContent=Math.round(state.progress*100)+'%';
  $('#dock-play').textContent=state.playing?'Pause':'Play docking';$('#dock-play').setAttribute('aria-pressed',String(state.playing));
  $('#dock-state').textContent=state.progress===1?'Bound pose · experimental coordinates':state.progress<.01?'Separated · illustrative start':'Approach · illustrative motion';
  $('#dock-metrics').hidden=state.progress<1;document.body.dataset.dockingProgress=String(state.progress);document.body.dataset.dockingPlaying=String(state.playing);
 }
 function pick(text){$('#selection').hidden=false;$('#selection-label').textContent=text;$('#selection-sequence').hidden=true;}
 async function load(next,currentEngine,record){
  dispose();mode=next;engine=currentEngine;active=true;panel.hidden=false;legend.hidden=false;
  if(mode==='docking'){
   source=await json('contact.json');await engine.load(record);
   panel.innerHTML=`<h2 class="eyebrow">OXT · OXTR</h2><h3>The molecular encounter</h3><p class="interaction-intro">Explore oxytocin at its receptor, from the complete complex to the aromatic contact.</p>
    <div class="interaction-segments" role="group" aria-label="Docking focus"><button data-contact-focus="complex">Complex</button><button data-contact-focus="pocket">Pocket</button><button data-contact-focus="contact" aria-pressed="true">Rings</button></div>
    <h4>DOCKING SEQUENCE</h4><div class="dock-actions"><button id="dock-play" aria-pressed="false">Play docking</button><button id="dock-bound">Bound pose</button></div>
    <div class="field-label"><label for="dock-progress">Approach</label><output id="dock-progress-label">100%</output></div><input id="dock-progress" type="range" min="0" max="1000" value="1000" step="1"><p id="dock-state" class="model-state">Bound pose · experimental coordinates</p>
    <label class="interaction-check"><input type="checkbox" id="dock-planes" checked> Ring planes</label>
    <div class="field-label"><label for="dock-opacity">Receptor opacity</label><output id="dock-opacity-label">6%</output></div><input id="dock-opacity" type="range" min="0" max="70" value="6" step="1">
    <div id="dock-metrics" class="contact-metrics"><div><span>Closest ring carbons</span><strong>${source.measurements.closestRingCarbonAngstrom.toFixed(2)} <small>Å</small></strong></div><div><span>Plane angle</span><strong>${source.measurements.ringPlaneAngleDegrees.toFixed(1)}<small>°</small></strong></div><div><span>Tyr2 OH · L316 O</span><strong>${source.measurements.hydrogenBondHeavyAtomDistanceAngstrom.toFixed(2)} <small>Å</small></strong></div></div>
    <p class="model-caption">Tilted Tyr2–Phe291 contact. Gold dashes measure a gap, not a chemical bond or an electron path. Pale green dashes mark the Tyr2–Leu316 hydrogen-bond contact.</p>`;
   legend.innerHTML='<span><i class="ligand-dot"></i>OXT · Tyr2</span><span><i class="receptor-dot"></i>OXTR · Phe291</span><span class="legend-sub">GOLD DASHES · DISTANCE GUIDE</span>';
   model=await MolecularEngine.createInteraction(engine,mode,source,progress,pick);
   panel.querySelectorAll('[data-contact-focus]').forEach(b=>b.onclick=()=>{model.setFocus(b.dataset.contactFocus);geometryChanged();panel.querySelectorAll('[data-contact-focus]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));});
   $('#dock-play').onclick=()=>{if(model.snapshot().playing)model.pause();else{if(model.snapshot().progress>=.999){model.setFocus('complex');geometryChanged();panel.querySelectorAll('[data-contact-focus]').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.contactFocus==='complex')));}model.play();}};
   $('#dock-bound').onclick=()=>{model.pause();model.setProgress(1);model.setFocus('contact');geometryChanged();panel.querySelectorAll('[data-contact-focus]').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.contactFocus==='contact')));};
   $('#dock-progress').oninput=e=>{const value=Number(e.target.value)/1000;model.pause();if(value<1){model.setFocus('complex');geometryChanged();panel.querySelectorAll('[data-contact-focus]').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.contactFocus==='complex')));}model.setProgress(value);};
   $('#dock-planes').onchange=e=>model.setPlanes(e.target.checked);
   $('#dock-opacity').oninput=e=>{const n=Number(e.target.value);$('#dock-opacity-label').textContent=n+'%';clearTimeout(styleTimer);styleTimer=setTimeout(()=>enqueue(()=>active?engine.setTargetOpacity(n/100):null),60);};
  }else{
   source=await json('quantum.json');await loadQuantumView(quantumView);
  }
  if(mode==='docking'){$('#pdb-link').href='https://www.rcsb.org/structure/7QVM';$('#pdb-link').innerHTML='PDB 7QVM <span>↗</span>';}
  if(mode==='docking')$('#method-summary').textContent='Cryo-EM · 3.25 Å · engineered complex';
  if(mode==='docking')$('#structure-status').textContent='7QVM · OXT–OXTR · experimental bound pose';
  return {atoms:mode==='docking'?source.ligand.length+source.pocket.length:source.atoms.length};
 }

 function teachingProgress(state){
  if(!active||mode!=='quantum'||quantumView!=='teaching')return;
  panel.querySelectorAll('[data-teaching-step]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.teachingStep)===state.lesson)));
  $('#teaching-explanation').textContent=[null,'Six carbon atoms form the aromatic ring. The OH group identifies the phenol part of tyrosine.','Each ring carbon contributes a p orbital. Their overlap forms the aromatic π system.','The ring π system extends across the six carbons. The bands are a schematic summary of this shared system.'][state.lesson];
  $('#teaching-play').textContent=state.playing?'Pause explanation':'Animate explanation';$('#teaching-play').setAttribute('aria-pressed',String(state.playing));$('#teaching-labels').checked=state.labels;$('#teaching-guides').checked=state.guides;$('#teaching-speed').value=state.speed;$('#teaching-speed-label').textContent=state.speed.toFixed(2)+'×';
  document.body.dataset.teachingPlaying=String(state.playing);document.body.dataset.teachingStep=String(state.lesson);
 }
 function rememberQuantum(){if(mode!=='quantum'||!model)return;if(quantumView==='teaching')teachingState=model.snapshot();else calculatedState=model.snapshot();}
 async function loadQuantumView(next){
  rememberQuantum();model?.dispose();model=null;clearTimeout(styleTimer);quantumView=next;document.body.dataset.busy='true';document.body.dataset.quantumView=next;
  $('#selection').hidden=true;quantumSwitch.hidden=false;teachingNote.hidden=next!=='teaching';quantumSwitch.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.quantumView===next)));
  if(next==='teaching'){
   panel.innerHTML=`<h2 class="eyebrow">LEARN THE RING</h2><h3>From atoms to orbitals</h3>
    <div class="teaching-steps" role="group" aria-label="Teaching steps"><button data-teaching-step="1"><span>1</span>Carbon framework</button><button data-teaching-step="2"><span>2</span>Overlapping p orbitals</button><button data-teaching-step="3"><span>3</span>Shared π system</button></div>
    <p id="teaching-explanation" class="interaction-intro" aria-live="polite"></p>
    <button id="teaching-play" class="teaching-play" aria-pressed="false">Animate explanation</button>
    <label class="interaction-check"><input id="teaching-labels" type="checkbox" checked>Labels</label><label class="interaction-check"><input id="teaching-guides" type="checkbox" checked>Motion guides</label>
    <div class="field-label"><label for="teaching-speed">Teaching speed</label><output id="teaching-speed-label">1.00×</output></div><input id="teaching-speed" type="range" min="0.25" max="2" value="1" step="0.25">
    <p class="model-caption">The lobes and motion guides are schematic. Colour shows orbital sign, not electrical charge. Ring hydrogens are implicit.</p><button id="teaching-calculated" class="text-button">Explore the calculated orbitals ↗</button><button id="teaching-source" class="text-button">Teaching model &amp; assumptions ↗</button>`;
   model=await MolecularEngine.createQuantumTeaching(engine,teachingProgress,pick,teachingState);
   panel.querySelectorAll('[data-teaching-step]').forEach(b=>b.onclick=()=>model.setLesson(b.dataset.teachingStep));
   $('#teaching-play').onclick=()=>model.snapshot().playing?model.pause():model.play();$('#teaching-labels').onchange=e=>model.setLabels(e.target.checked);$('#teaching-guides').onchange=e=>model.setGuides(e.target.checked);$('#teaching-speed').oninput=e=>model.setSpeed(e.target.value);
   $('#teaching-calculated').onclick=()=>enqueue(()=>loadQuantumView('calculated'));$('#teaching-source').onclick=showEvidence;
   $('#target-name').textContent='Inside the aromatic ring';$('#target-context').textContent='OXT · Tyr2 · teaching model';$('#structure-status').textContent='Idealised ring · schematic orbitals';$('#method-summary').textContent='Teaching model · idealised geometry';$('#footer-context').textContent='SYMBOLIC TEACHING MODEL';$('#pdb-link').href=MolecularEngine.teachingProvenance.source;$('#pdb-link').innerHTML='Orbital teaching source <span>↗</span>';
   legend.innerHTML='<span><i class="receptor-dot"></i>Negative orbital sign</span><span><i class="ligand-dot"></i>Positive orbital sign</span><span class="legend-sub">COLOUR SHOWS ORBITAL SIGN</span>';
  }else{
   const response=await fetch('data/interactions/fragment.pdb');if(!response.ok)throw Error('The quantum fragment did not load.');await engine.loadFragment(await response.text());
   panel.innerHTML=`<h2 class="eyebrow">ELECTRONIC STRUCTURE</h2><h3>Inside the aromatic system</h3><p class="interaction-intro">Explore calculated orbitals around a Tyr2–Phe291 fragment pair.</p>
    <label class="field-label" for="quantum-field">Electronic field</label><select id="quantum-field"><option value="homo">HOMO · highest occupied</option><option value="homo-minus-1">HOMO − 1 · occupied</option><option value="lumo">LUMO · lowest unoccupied</option><option value="density">Total electron density</option></select>
    <div id="orbital-readout" class="orbital-readout"></div>
    <div class="interaction-segments" role="group" aria-label="Quantum focus"><button data-quantum-focus="pair" aria-pressed="true">Fragment pair</button><button data-quantum-focus="ring">Tyr2 ring</button></div>
    <div class="field-label"><label for="quantum-iso">Surface level</label><output id="quantum-iso-label">0.025</output></div><input id="quantum-iso" type="range" min="0.01" max="0.06" value="0.025" step="0.005"><p id="quantum-units" class="model-state">Orbital amplitude · Bohr⁻³ᐟ²</p>
    <div class="field-label"><label for="quantum-opacity">Orbital opacity</label><output id="quantum-opacity-label">56%</output></div><input id="quantum-opacity" type="range" min="15" max="90" value="56" step="1">
    <p id="quantum-explanation" class="model-caption"></p><button id="quantum-source" class="text-button">Model &amp; assumptions ↗</button><button id="quantum-download" class="text-button">Download field · CUBE ↓</button>`;
   model=await MolecularEngine.createInteraction(engine,mode,source,null,pick);
   if(calculatedState.field&&calculatedState.field!=='homo')await model.loadField(calculatedState.field);
   if(calculatedState.isovalue||calculatedState.opacity)await model.fieldStyle({level:calculatedState.isovalue,opacity:calculatedState.opacity});
   if(calculatedState.focus)model.setFocus(calculatedState.focus);
   const state=model.snapshot();$('#quantum-field').value=state.field;$('#quantum-iso').value=state.isovalue;$('#quantum-opacity').value=Math.round(state.opacity*100);$('#quantum-iso-label').textContent=state.isovalue.toFixed(3);$('#quantum-opacity-label').textContent=Math.round(state.opacity*100)+'%';panel.querySelectorAll('[data-quantum-focus]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.quantumFocus===state.focus)));fieldInfo();
   $('#quantum-field').onchange=e=>{const field=e.target.value;enqueue(async()=>{if(!active||mode!=='quantum')return;document.body.dataset.busy='true';$('#structure-status').textContent='Loading calculated field…';await model.loadField(field);fieldInfo();document.body.dataset.busy='false';});};
   panel.querySelectorAll('[data-quantum-focus]').forEach(b=>b.onclick=()=>{model.setFocus(b.dataset.quantumFocus);panel.querySelectorAll('[data-quantum-focus]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));});
   for(const id of ['quantum-iso','quantum-opacity'])$('#'+id).oninput=()=>{const iso=Number($('#quantum-iso').value),opacity=Number($('#quantum-opacity').value)/100;$('#quantum-iso-label').textContent=iso.toFixed(3);$('#quantum-opacity-label').textContent=Math.round(opacity*100)+'%';clearTimeout(styleTimer);styleTimer=setTimeout(()=>enqueue(()=>active&&mode==='quantum'?model.fieldStyle({level:iso,opacity}):null),100);};
   $('#quantum-source').onclick=showEvidence;
   $('#quantum-download').onclick=()=>enqueue(async()=>{const f=model.snapshot().field,filename=f+'.cube',response=await fetch('data/interactions/'+filename);if(!response.ok)throw Error('Field download failed.');download('OXTR-'+filename,await response.text(),'text/plain');});
   $('#target-name').textContent='The aromatic electron system';$('#target-context').textContent='Tyr2–Phe291 · calculated fragment pair';$('#method-summary').textContent='PBE0 / def2-SVP · gas-phase fragment';$('#footer-context').textContent='CALCULATED ORBITALS · STATIONARY MODEL';$('#pdb-link').href='https://www.rcsb.org/structure/7QVM';$('#pdb-link').innerHTML='PDB 7QVM <span>↗</span>';
  }
  const url=new URL(location.href);url.searchParams.set('quantum',next);history.replaceState(null,'',url);document.body.dataset.busy='false';
 }
 function fieldInfo(){
  const f=model.snapshot().field,o=source.orbitals.find(x=>x.id===f),density=f==='density';
  $('#orbital-readout').innerHTML=o?`<strong>${labels[f]}</strong><span>${o.energyEV.toFixed(2)} eV · ${o.occupation===0?'unoccupied':'2 electrons'}</span>`:`<strong>Electron density</strong><span>${source.electrons} electrons · complete fragment pair</span>`;
  legend.innerHTML=density?'<span><i class="density-dot"></i>Calculated electron density</span>':'<span><i class="ligand-dot"></i>Positive orbital sign</span><span><i class="receptor-dot"></i>Negative orbital sign</span>';
  $('#quantum-explanation').textContent=density?'The surface connects points with equal electron density. A lower surface level reveals more of the diffuse distribution.':'The two colours show the wavefunction’s signs. Nodes separate its lobes. These colours are not electrical charges or electron paths.';
  $('#quantum-units').textContent=density?'Electron density · electrons / Bohr³':'Orbital amplitude · Bohr⁻³ᐟ²';
  $('#structure-status').textContent='PBE0 / def2-SVP · '+labels[f]+' · calculated fragment model';document.body.dataset.quantumField=f;
 }
 function detail(view){
  if(!source)return '';
  if(mode==='docking')return `<span class="evidence-badge">${view==='sequence'?'MOLECULAR CONTACTS':'EXPERIMENTAL BOUND POSE'}</span><h3>Oxytocin at OXTR</h3><p>Oxytocin is a cyclic peptide with the sequence CYIQNCPLG–NH₂ and a Cys1–Cys6 disulfide. Tyr2 sits deep in the receptor pocket.</p><h4>AROMATIC CONTACT</h4><p>The Tyr2 phenol ring contacts receptor Phe291, which has one aromatic ring. Trp288 has a fused two-ring indole group, but this focused view isolates the Tyr2–Phe291 contact. The primary study places Trp288 in the receptor activation rearrangement; it does not describe the illustrated parallel Tyr2–Trp288 stack. Their deposited ring planes are tilted by ${source.measurements.ringPlaneAngleDegrees.toFixed(1)}°. This view does not classify the contact as a parallel π stack. Other contacts include Gln92 and Gln171, and the Tyr2 hydroxyl hydrogen bond to the backbone oxygen of Leu316.</p><dl><dt>Ring-centre distance</dt><dd>${source.measurements.centroidDistanceAngstrom.toFixed(3)} Å</dd><dt>Closest ring carbons</dt><dd>${source.measurements.closestRingCarbonAngstrom.toFixed(3)} Å</dd><dt>Tyr2 OH · Leu316 O</dt><dd>${source.measurements.hydrogenBondHeavyAtomDistanceAngstrom.toFixed(3)} Å</dd></dl><p>Gold dashes are a distance guide between the closest ring carbons, not a chemical bond or an electron path. Pale green dashes identify the Tyr2–Leu316 hydrogen-bond contact. Distances use the deposited heavy atoms. Ring angles use least-squares planes through the six ring carbons. Structural resolution and model uncertainty limit their interpretation.</p><h4>ANIMATION</h4><p class="evidence-note">The final bound pose uses 7QVM coordinates. The approach is a rigid, illustrative translation along an outward vector. It is not a molecular-dynamics trajectory, docking search, free-energy calculation or observed entry pathway. Only the bound pose displays contact measurements.</p><p>7QVM is an engineered human receptor complex determined by cryo-EM at 3.25 Å. G protein and antibody components are omitted from this view.</p><p><a href="https://www.rcsb.org/structure/7QVM" target="_blank" rel="noopener">RCSB · 7QVM ↗</a></p><p><a href="${source.publication}" target="_blank" rel="noopener">Waltenspühl et al., 2022 · structure and contact evidence ↗</a></p><p class="source-hash">Coordinate SHA-256: ${source.sourceSHA256}</p>`;
  if(quantumView==='teaching')return `<span class="evidence-badge">SYMBOLIC TEACHING MODEL</span><h3>From atoms to orbitals</h3><p>The regular phenol ring and peptide attachment cue help explain the tyrosine side chain. They are idealised teaching geometry, with ring hydrogens implicit.</p><h4>SIX P ORBITALS</h4><p>Each aromatic carbon contributes a p orbital perpendicular to the ring plane. The two lobes have opposite wavefunction signs. Their colours do not represent positive and negative electrical charges.</p><h4>THE SHARED π SYSTEM</h4><p>Aligned signs illustrate one bonding combination. The complete aromatic π system occupies several molecular orbitals. The bands summarise delocalisation; they are not a calculated density surface or the complete set of occupied orbitals.</p><h4>MOTION GUIDES</h4><p>Moving markers guide the explanation. Their number, route and speed are symbolic, not electron counts, electron trajectories or physical timescales. No stimulus or time-dependent calculation is being simulated.</p><h4>CALCULATED VIEW</h4><p>The separate Calculated mode retains the PBE0/def2-SVP fields for the capped Tyr2–Phe291 fragment pair. The teaching geometry does not replace those data.</p><p><a href="${MolecularEngine.teachingProvenance.source}" target="_blank" rel="noopener">Michigan State University · conjugation and aromaticity ↗</a></p>`;
  return `<span class="evidence-badge">CALCULATED ELECTRONIC STRUCTURE</span><h3>A local quantum model</h3><p>${source.model}</p><h4>CALCULATION</h4><dl><dt>Method</dt><dd>${source.method}</dd><dt>Software</dt><dd>${source.software}</dd><dt>Charge / spin</dt><dd>0 / singlet</dd><dt>Electrons / atoms</dt><dd>${source.electrons} / ${source.atoms.length}</dd><dt>SCF convergence</dt><dd>${source.scfConverged?'Converged':'Not converged'} · 10⁻⁹ Hartree tolerance</dd></dl><h4>ORBITALS</h4><p>HOMO is the highest occupied molecular orbital; HOMO − 1 is the occupied orbital immediately below it. LUMO is the lowest unoccupied orbital. The orbital energies are model quantities, not an optical excitation spectrum.</p>${source.orbitals.map(o=>`<p>${labels[o.id]} · ${o.energyEV.toFixed(3)} eV · occupation ${o.occupation}</p>`).join('')}<h4>PREPARATION &amp; ENVIRONMENT</h4><p>${source.preparation}</p><p>${source.environment}</p><p class="evidence-note">${source.interpretation}</p><p>${source.limitations}</p><p><a href="https://www.rcsb.org/structure/7QVM" target="_blank" rel="noopener">Source heavy-atom geometry · 7QVM ↗</a></p><p><a href="https://pyscf.org/user/dft.html" target="_blank" rel="noopener">PySCF · density functional theory ↗</a></p><p class="source-hash">Fragment SHA-256: ${source.fragmentSHA256}</p>`;
 }
 function dispose(){rememberQuantum();active=false;quantumSwitch.hidden=true;teachingNote.hidden=true;delete document.body.dataset.quantumView;delete document.body.dataset.teachingPlaying;clearTimeout(styleTimer);model?.dispose();model=null;panel.hidden=true;legend.hidden=true;$('#selection-sequence').hidden=false;}
 return {load,detail,dispose,pause:()=>model?.pause(),reset:()=>{model?.setFocus(mode==='docking'?'contact':'pair');if(mode==='docking')geometryChanged();},snapshot:()=>({model:{...model?.snapshot(),...(mode==='quantum'?{view:quantumView}:{}),...(mode==='docking'?{receptorOpacity:Number($('#dock-opacity').value)/100}:{})},provenance:mode==='quantum'&&quantumView==='teaching'?MolecularEngine.teachingProvenance:source}),get model(){return model;}};
};
