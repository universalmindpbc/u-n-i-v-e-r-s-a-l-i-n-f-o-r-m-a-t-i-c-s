import {BASE,BASIS,BOUNDS,VERSION,validate,isBaseline,pulse} from './model.mjs';
const $=s=>document.querySelector(s),params=new URLSearchParams(location.search),embedded=params.get('embedded')==='1';
document.body.classList.toggle('embedded',embedded);
const create=(tag,text,cls)=>{const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;};
const STORE='informatics.quantum.runs.v1';
let reference=null,current=null,runs=[],busy=false,draft=false,worker=null,workerTimer=null,requestId=null;
const stages={
 gpt:['Assist · GPT / AI','Assistance with formulation, code and explanation. Researcher review and independent numerical checks remain essential. The main Quantum Lab now offers GPT guidance. This optional numerical verification view does not call GPT.'],
 wolfram:['Model · Wolfram Quantum Framework','The original two-atom Hamiltonian was independently evaluated with Wolfram Quantum Framework. Its probabilities agreed with the continuous CPU reference within 8.2 × 10⁻⁹.'],
 spec:['Define · Versioned experiment','Preserve the Hamiltonian, units, initial state, pulse and observables. A new intentional run gets a new ID; a retry retains its submission identity.'],
 cpu:['Verify · Independent CPU','The recorded reference used SciPy. This page uses an independently implemented browser solver and repeats each calculation at half the time step. Both describe an ideal closed system.'],
 braket:['Route · Amazon Braket','The original analog programme was submitted to Aquila through Braket. The website does not yet submit cloud jobs. The last verified hardware status is recorded with its date.'],
 aquila:['Execute · QuEra Aquila','Neutral atoms evolve under a programmed Rydberg Hamiltonian. Physical measurements can differ from the ideal model through preparation, noise, loss and readout.'],
 cudaq:['Compile · NVIDIA CUDA-Q','CUDA-Q 0.16.0 generated the gate approximation. A 5,120-step circuit was checked against the continuous model and exported as OpenQASM 2.'],
 qbraid:['Dispatch · qBraid','The exported circuit was sent through the qBraid SDK to its managed GPU-SV simulator. The stored remote programme matched the submitted file hash.'],
 nvidia:['Simulate · NVIDIA GH200','The provider identified GH200 GPU hardware. The recorded 20,000 shots are samples from a classical GPU simulation, not quantum hardware measurements.'],
 compare:['Compare · Universal Informatics','Keep predictions, sampled outcomes, numerical error and physical measurements distinct. Compare the same experiment specification and report uncertainty and provenance.']
};
for(const b of document.querySelectorAll('[data-stage]'))b.onclick=()=>{
 document.querySelectorAll('[data-stage]').forEach(n=>n.setAttribute('aria-pressed',String(n===b)));
 const [title,body]=stages[b.dataset.stage];$('#stage-detail').replaceChildren(create('strong',title),document.createTextNode(body));
};
const methods=[
 ['How can short DNA fragments be ordered?','Annealing / DNA assembly','Represent read overlaps as a graph and an ordering objective. Compare exact or classical assembly with a validated binary optimisation mapping.','Output: read order and reconstruction accuracy. Start with small synthetic reads; sequencing errors and scaling need separate evaluation.'],
 ['Which reads belong to each haplotype?','Annealing / phasing','Express read compatibility and phase assignment as an optimisation problem, then compare with a classical phasing tool.','Output: phase assignment and switch error. Published QPU demonstrations use short synthetic examples; larger studies may use classical solvers.'],
 ['Can DNA-binding measurements be classified?','Annealing / machine learning','Use documented experimental features, a defined learning objective and a held-out test set. Benchmark conventional machine learning first.','Output: classification or ranking metrics. Dataset size and feature design matter; existing studies do not establish an OXTR-specific result.'],
 ['Which molecular contacts are compatible?','Neutral atoms / graph optimisation','Translate candidate contacts into a graph. Test an independent-set formulation with classical methods and an emulator before a device-compatible mapping.','Output: selected contacts and independently assessed pose quality. A graph solution alone does not validate binding or drug efficacy.']
];
for(const [title,route,model,limit] of methods){const article=create('article');article.append(create('p',route,'method-route'),create('h3',title),create('p',model),create('p',limit));$('#method-cards').append(article);}
async function get(url){const r=await fetch(url,{signal:AbortSignal.timeout(15000)});if(!r.ok)throw Error(`Unable to load ${url} (${r.status}).`);return r.json();}
get('data/papers.json').then(data=>{
 for(const p of data.papers){
  const a=create('article'),year=(p.published||p.published_online||p.first_posted).slice(0,4),citation=`${p.authors_short} · ${p.journal||p.venue} · ${year}${p.version?' · '+p.version:''}`;
  a.append(create('div',p.evidence,'paper-label'),create('h3',p.title),create('p',citation,'citation'),create('p',p.finding));
  const limit=create('p',undefined,'limit');limit.append(create('strong','Scope: '),document.createTextNode(p.limit));a.append(limit);
  const link=create('a','Read the paper ↗');link.href=p.url;link.target='_blank';link.rel='noopener noreferrer';a.append(link);$('#papers').append(a);
 }
}).catch(e=>$('#papers').append(create('p',e.message+' Reload to retry.')));
function getSpec(){return validate(Object.fromEntries(Object.keys(BOUNDS).map(k=>[k,Number($('#'+k).value)])));}
function setSpec(spec){for(const k of Object.keys(BOUNDS))$('#'+k).value=spec[k];draft=false;$('#draft-status').textContent='Parameters loaded. Run simulation to create a new run record.';}
function canUseCurrent(){return current&&!draft&&!busy;}
function controls(){
 $('#run').disabled=busy;$('#rerun').disabled=busy||!current;$('#baseline').disabled=busy;$('#time').disabled=busy||!current;
 for(const id of ['export','export-figure','prepare-request'])$('#'+id).disabled=!canUseCurrent();
 for(const k of Object.keys(BOUNDS))$('#'+k).disabled=busy;
 $('#import').disabled=busy;$('.import-button').setAttribute('aria-disabled',String(busy));$('#clear-history').disabled=busy;
 $('#run').firstChild.textContent=busy?'Calculating… ':'Run simulation ';
}
function save(){
 try{localStorage.setItem(STORE,JSON.stringify(runs));}
 catch{$('#draft-status').textContent='This browser cannot save the history. Export the run to keep it.';}
}
function validateStored(r){
 if(!r||r.schema!=='universal-informatics/quantum-run/v1'||r.model_version!==VERSION||typeof r.run_id!=='string'||r.run_id.length>100||!Number.isFinite(Date.parse(r.created_at)))throw Error('Invalid run.');
 validate(r.spec);
 if(!r.result||!Array.isArray(r.result.probabilities)||r.result.probabilities.length!==4||r.result.probabilities.some(p=>!Number.isFinite(p)||p<0||p>1.000001))throw Error('Invalid probabilities.');
 if(!Array.isArray(r.result.trace)||r.result.trace.length<2||r.result.trace.length>400||r.result.trace.some(x=>!Number.isFinite(x.t)||x.t<0||x.t>r.spec.duration_us+1e-8||!Array.isArray(x.p)||x.p.length!==4||x.p.some(p=>!Number.isFinite(p)||p<0||p>1.000001)))throw Error('Invalid trace.');
 return r;
}
try{const loaded=JSON.parse(localStorage.getItem(STORE)||'[]');if(Array.isArray(loaded))runs=loaded.slice(0,30).filter(r=>{try{validateStored(r);return true;}catch{return false;}});}catch{}
function history(){
 $('#history-count').textContent=String(runs.length);$('#history').replaceChildren();
 if(!runs.length)$('#history').append(create('p','No saved simulations yet.'));
 for(const run of runs){const b=create('button',`${new Date(run.created_at).toLocaleString()} · ${run.spec.distance_um} µm · ${run.spec.duration_us} µs · ${run.run_id.slice(0,8)}`);b.setAttribute('aria-pressed',String(current?.run_id===run.run_id));b.onclick=()=>{if(busy)return;current=run;setSpec(run.spec);controls();render();history();$('#run-status').textContent='Saved local run loaded. Rerun to reproduce the calculation.';};$('#history').append(b);}
}
history();
function percent(p){return p>0&&p<.00001?'<0.001%':(p*100).toFixed(2)+'%';}
function wilson(k,n){const p=k/n,z=1.959963984540054,den=1+z*z/n,mid=(p+z*z/(2*n))/den,half=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [Math.max(0,mid-half),Math.min(1,mid+half)];}
function svgNode(tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text)n.textContent=text;return n;}
function render(){
 if(!current)return;
 const {spec,result}=current,trace=result.trace;$('#time').max=trace.length-1;$('#time').value=trace.length-1;
 $('#distance-label').textContent=spec.distance_um.toFixed(2)+' µm';$('#duration-label').textContent=spec.duration_us;
 $('#metric-error').textContent=Number.isFinite(result.convergence)?result.convergence.toExponential(1):'—';
 $('#reference-check').textContent=reference?`Recorded validation: Wolfram vs independent CPU maximum probability difference ${reference.validation.wolfram_probability_error.toExponential(2)}; exported GPU circuit vs continuous model ${reference.validation.circuit_probability_error.toExponential(2)}. GPU intervals show sampling uncertainty, separate from numerical approximation error.`:'Archived reference data are unavailable; this browser calculation remains independent.';
 const svg=$('#pulse');svg.replaceChildren();
 for(const [key,color] of [['omega','#92ebce'],['delta','#86baff']]){
  const scale=key==='omega'?spec.omega_rad_us:spec.detuning_rad_us;
  const points=Array.from({length:161},(_,i)=>{const t=spec.duration_us*i/160,v=pulse(t,spec)[key];return `${i*600/160},${key==='omega'?70-60*v/scale:40-30*v/scale}`;}).join(' ');
  svg.append(svgNode('polyline',{points,fill:'none',stroke:color,'stroke-width':1.5}));
 }
 svg.append(svgNode('line',{id:'pulse-cursor',x1:600,x2:600,y1:0,y2:80,stroke:'#e9f5fb','stroke-width':1,'stroke-dasharray':'2 3'}));
 drawAt(trace.length-1);
}
function drawAt(index){
 if(!current)return;
 const {spec,result}=current,point=result.trace[index],end=index===result.trace.length-1,compare=end&&isBaseline(spec)&&reference;
 const time=point.t.toFixed(3)+' µs';$('#time-label').textContent=time;$('#time-value').textContent=time;$('#plot-state').textContent=end?'ENDPOINT':'SIMULATED TIME';
 $('#atom-a-ring').style.strokeDasharray=`${point.p[2]+point.p[3]} 1`;$('#atom-b-ring').style.strokeDasharray=`${point.p[1]+point.p[3]} 1`;
 $('#atom-a-prob').textContent='P(r) '+percent(point.p[2]+point.p[3]);$('#atom-b-prob').textContent='P(r) '+percent(point.p[1]+point.p[3]);
 const x=600*point.t/spec.duration_us;$('#pulse-cursor').setAttribute('x1',x);$('#pulse-cursor').setAttribute('x2',x);
 $('#gpu-key').hidden=!compare;$('#outcomes').replaceChildren();
 BASIS.forEach((state,i)=>{
  const row=create('div',undefined,'outcome'),label=create('div',undefined,'outcome-label');label.append(create('strong','|'+state+'⟩'),create('span',percent(point.p[i])));row.append(label);
  const bars=create('div',undefined,'bars'),track=create('div',undefined,'track'),bar=create('div',undefined,'bar');bar.style.width=Math.min(100,100*point.p[i])+'%';track.append(bar);bars.append(track);
  if(compare){const k=reference.gpu.counts[state],n=reference.gpu.shots,[lo,hi]=wilson(k,n),gt=create('div',undefined,'track gpu'),gb=create('div',undefined,'bar'),ci=create('div',undefined,'interval');gb.style.width=100*k/n+'%';ci.style.left=100*lo+'%';ci.style.width=Math.max(.1,100*(hi-lo))+'%';gt.append(gb,ci);bars.append(gt);row.append(bars,create('div',`${k.toLocaleString()} / ${n.toLocaleString()} · ${percent(k/n)} GPU · 95% interval ${(lo*100).toFixed(3)}–${(hi*100).toFixed(3)}%`,'outcome-meta'));}
  else row.append(bars,create('div',state.split('').map((v,j)=>`Atom ${j}: ${v==='g'?'ground':'excited'}`).join(' · '),'outcome-meta'));
  $('#outcomes').append(row);
 });
 $('#metric-shots').textContent=compare?'20,000':'—';
 $('#comparison-note').textContent=compare?'GPU-SV · NVIDIA GH200 · 24 October 2026. Whiskers: 95% marginal Wilson intervals. A displayed interval can exclude the reference; the recorded six-standard-error check passed. Zero rr counts do not imply zero probability.':end?'Custom parameters: no matching GPU or QPU samples are available. The display shows the selected browser calculation.':'Intermediate states are calculated simulation. Recorded endpoint samples appear only at the end of the original reference pulse.';
}
$('#time').oninput=()=>drawAt(Number($('#time').value));
async function simulate(spec,{saveRun=true,parent=null}={}){
 if(busy)return;
 let valid;try{valid=validate(spec);}catch(e){$('#run-status').textContent=e.message;return;}
 busy=true;requestId=crypto.randomUUID();const id=requestId;controls();$('#run-status').textContent='Solving the model twice to check numerical convergence…';
 try{
  worker=new Worker(new URL('./model.worker.js',import.meta.url),{type:'module'});
  const result=await new Promise((resolve,reject)=>{
   workerTimer=setTimeout(()=>reject(Error('The calculation timed out. No result was saved.')),30000);
   worker.onerror=()=>reject(Error('The calculation worker failed. Reload and retry.'));
   worker.onmessage=({data})=>{if(data.id!==id)return;if(data.error)reject(Error(data.error));else resolve(data.result);};
   worker.postMessage({id,spec:valid});
  });
  const bytes=new TextEncoder().encode(JSON.stringify({model_version:VERSION,spec:valid})),hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
  current={schema:'universal-informatics/quantum-run/v1',model_version:VERSION,run_id:id,parent_run_id:parent,created_at:new Date().toISOString(),execution:'browser-cpu',spec:valid,spec_sha256:hash,result};
  setSpec(valid);draft=false;
  $('#run-status').textContent=`${saveRun?'Run '+id.slice(0,8)+' complete':'Reference ready'} · ${result.steps.toLocaleString()} steps · norm error ${result.normError.toExponential(1)}.`;
  $('#draft-status').textContent=saveRun?'Simulation complete. Saved in this browser; export to preserve or share it.':'The reference is ready. Create a simulation to add your own run to the history.';
  if(saveRun){runs=[current,...runs].slice(0,30);save();}
  render();history();
 }catch(e){$('#run-status').textContent=e.message;}
 finally{clearTimeout(workerTimer);worker?.terminate();worker=null;busy=false;controls();}
}
$('#experiment-form').onsubmit=e=>{e.preventDefault();try{simulate(getSpec());}catch(error){$('#run-status').textContent=error.message;}};
$('#rerun').onclick=()=>current&&simulate(current.spec,{parent:current.run_id});
$('#baseline').onclick=()=>{setSpec(BASE);simulate(BASE,{saveRun:false});};
for(const k of Object.keys(BOUNDS))$('#'+k).oninput=()=>{draft=true;controls();$('#draft-status').textContent='Parameters changed. The chart still shows the last completed run. Run simulation to update it.';};
function download(name,data){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'}));const a=create('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('#export').onclick=()=>{if(canUseCurrent())download('quantum-run-'+current.run_id+'.json',current);};
$('#export-figure').onclick=()=>{
 if(!canUseCurrent())return;
 const svg=svgNode('svg',{xmlns:'http://www.w3.org/2000/svg',viewBox:'0 0 1100 720',width:1100,height:720});
 svg.append(svgNode('rect',{width:1100,height:720,fill:'#080e14'}));
 const label=(x,y,text,size=14,fill='#9bafbd')=>svg.append(svgNode('text',{x,y,fill,'font-family':'Arial, sans-serif','font-size':size},text));
 label(55,50,'UNIVERSAL INFORMATICS / QUANTUM LAB',13);label(55,104,'Two atoms. Four outcomes.',38,'#edf4f7');
 label(55,139,`${current.spec.distance_um} µm separation · ${current.spec.duration_us} µs pulse · Ω = ${current.spec.omega_rad_us} rad/µs · ±Δ = ${current.spec.detuning_rad_us} rad/µs`);
 const compare=reference&&isBaseline(current.spec);
 label(55,184,'Ideal CPU probabilities',13,'#86baff');if(compare)label(300,184,'Recorded GPU samples · 20,000 shots · 24 October 2026',13,'#92ebce');
 BASIS.forEach((state,i)=>{
  const y=235+i*86,p=current.result.probabilities[i];label(55,y+13,'|'+state+'⟩',22,'#edf4f7');
  svg.append(svgNode('rect',{x:135,y,width:690,height:12,fill:'#172735'}),svgNode('rect',{x:135,y,width:690*p,height:12,fill:'#86baff'}));label(855,y+12,percent(p),15,'#86baff');
  if(compare){const k=reference.gpu.counts[state],n=reference.gpu.shots,[lo,hi]=wilson(k,n);svg.append(svgNode('rect',{x:135,y:y+22,width:690,height:12,fill:'#172735'}),svgNode('rect',{x:135,y:y+22,width:690*k/n,height:12,fill:'#92ebce'}),svgNode('path',{d:`M${135+690*lo} ${y+19}v18M${135+690*hi} ${y+19}v18M${135+690*lo} ${y+28}H${135+690*hi}`,stroke:'#edf4f7',fill:'none'}));label(855,y+34,k.toLocaleString()+' / 20,000',13,'#92ebce');}
 });
 label(55,615,compare?'Whiskers: 95% marginal Wilson intervals. GPU data are classical simulation; no QPU measurements shown.':'Custom browser simulation. No matching GPU or QPU measurements are available.',12);
 label(55,641,'Closed system · initial gg · C₆ = 5.42 × 10⁻²⁴ rad m⁶/s · excludes noise and atom loss',12);
 label(55,667,'Model '+VERSION+' · Run '+current.run_id,11);label(55,689,'Created '+current.created_at+' · Step-halving difference '+current.result.convergence.toExponential(2),11);
 const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml'})),a=create('a');a.href=url;a.download='quantum-figure-'+current.run_id+'.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('#import').onchange=async e=>{
 const file=e.target.files[0];if(!file||busy)return;
 try{
  if(file.size>1_000_000)throw Error('Use a JSON experiment smaller than 1 MB.');
  const data=JSON.parse(await file.text());
  if(data.model_version!==VERSION)throw Error('This workspace imports rydberg-pair-v1 experiments. Other Hamiltonians need a separate validated template.');
  const spec=validate(data.spec||data.parameters);setSpec(spec);draft=true;controls();
  $('#draft-status').textContent='Imported parameters only. Run simulation to independently calculate and save a new result.';
 }catch(error){$('#draft-status').textContent='Import failed: '+error.message;}
 e.target.value='';
};
$('.import-button').onkeydown=e=>{if(!busy&&(e.key==='Enter'||e.key===' ')){e.preventDefault();$('#import').click();}};
$('#clear-history').onclick=()=>{if(busy)return;runs=[];save();history();$('#draft-status').textContent='Local run history cleared. Exported files remain on your device.';};
$('#prepare-request').onclick=()=>{
 if(!canUseCurrent())return;
 const shots=Number($('#shots').value);if(!Number.isInteger(shots)||shots<1||shots>1000){$('#request-status').textContent='Enter an integer shot count between 1 and 1,000.';return;}
 download('quantum-review-'+current.run_id+'.json',{schema:'universal-informatics/quantum-review/v1',request_id:crypto.randomUUID(),status:'LOCAL_DRAFT_NOT_SUBMITTED',funding:'Universal Informatics; allocation and authorisation pending',requested_provider:'quera-aquila',requested_shots:shots,hardware_compatibility:'Not yet validated for the requested provider',requested_at:new Date().toISOString(),experiment:current});
 $('#request-status').textContent='Review request downloaded. No job was submitted and no funds were reserved.';
};
window.addEventListener('pagehide',()=>{worker?.terminate();clearTimeout(workerTimer);});
// Embedding preserves the active Informatics experience and its media state.
document.addEventListener('click',e=>{
 const a=e.target.closest('[data-global-area]');if(embedded&&a){e.preventDefault();parent.postMessage({type:'quantum:navigate',area:a.dataset.globalArea},location.origin);}
});
get('data/reference.json?v=3').then(data=>{
 reference=data;const h=data.hardware;
 const snapshot=$('#hardware-snapshot');
 if(h.last_verified_status==='COMPLETED'&&h.results_available){
  snapshot.textContent=`Aquila connection test complete · ${h.shots_returned} shots returned · 25 October 2026. Submission to completion: 18 hours 52 minutes.`;
  const result=document.createElement('details'),heading=document.createElement('summary'),body=document.createElement('p'),limit=document.createElement('p');
  heading.textContent='Recorded Aquila measurements';body.textContent=`${h.fully_occupied_shots} shots began with both atoms present; ${h.initially_empty_shots} had a missing atom. Among the fully occupied shots: `+Object.entries(h.counts_fully_occupied).map(([k,v])=>`${k}: ${v}`).join(' · ')+'.';limit.textContent=h.interpretation;result.append(heading,body,limit);snapshot.after(result);
 }else snapshot.textContent=`Aquila: last verified ${h.last_verified_status.toLowerCase()} · ${h.last_checked_utc}. Archived task status.`;
 if(current)render();
}).catch(e=>{$('#hardware-snapshot').textContent='Archived hardware status unavailable. '+e.message;$('#comparison-note').textContent='Recorded GPU data unavailable; reload to retry.';});
simulate(BASE,{saveRun:false});
