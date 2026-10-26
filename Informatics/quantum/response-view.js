const ns='http://www.w3.org/2000/svg',choices=new Map();
const labels={ordinary:'Ordinary input',fibonacci:'Native Fibonacci',phi:'Phi Ratio',alpha:'Alpha · M9 P24',beta:'Beta · M9 P24',gamma:'Gamma · M9 P24',epsilon:'Epsilon · M9 P24',shuffled:'Shuffled control'};
const meanings={ordinary:'The starting comparison: DNA pattern matches, the gaps between them and local DNA features, without adding Fibonacci-specific weights.',fibonacci:'Uses gaps near the Fibonacci numbers 1, 2, 3, 5, 8, 13, 21, 34 and 55 when preparing the laser program. DNA pattern matches and local sequence features remain included.',phi:'Compares consecutive gaps between DNA pattern matches with the Phi Ratio, then uses that comparison to prepare the laser program.',alpha:'Uses all 24 values of Fibonacci M9 P24 to change how strongly the DNA measurements contribute at each of the 24 laser-program steps.',beta:'Repeats Beta’s 4, 8, 7, 5, 1, 2 sequence across the 24 laser-program steps, changing how strongly the DNA measurements contribute.',gamma:'Uses Gamma’s two triangle paths through six M9 P24 positions to set the weights across the 24 laser-program steps.',epsilon:'Uses Epsilon’s seven-visit path through the same six positions, including the return to its starting point, to set the weights across the 24 laser-program steps.',shuffled:'Mixes up the order of the same ordinary input values. This checks whether changing the order alone can alter the atom readings.'};
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const svgNode=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
const average=a=>a.reduce((s,v)=>s+v,0)/a.length,pp=v=>(v>=0?'+':'')+v.toFixed(1)+' percentage points';
export async function renderResponse(slot,report){
 if(report.response_data!=='oxtr-qrc-qualification.json')return;
 const box=el('div',undefined,'response-view');box.append(el('p','Loading recorded responses…'));slot.append(box);
 try{
  const response=await fetch('data/oxtr-qrc-qualification.json',{cache:'no-store'});if(!response.ok)throw Error();const data=await response.json();
  if(data.input_sha256!==report.receipt.input_sha256||!Array.isArray(data.records)||data.records.length!==16)throw Error();
  if(!box.isConnected)return;
  const quantum=report.kind==='quantum',field=quantum?'hardware_observables':'gpu_observables',source=quantum?'Aquila':'GPU simulation';
  const records=data.records.filter(r=>Array.isArray(r[field])&&r[field].length===21&&r[field].every(p=>Number.isFinite(p)&&p>=0&&p<=1));if(records.length!==16)throw Error();
  box.replaceChildren();
  const intro=el('section',undefined,'reading-guide');intro.append(el('h4','What went into the machine?'),el('p','We used two short DNA sequences near OXTR. CTCF is one transcription factor, a protein that binds DNA. For each sequence, we calculated matches to its known DNA pattern, the gaps between those matches and local DNA features. We prepared these measurements eight ways: ordinary, Fibonacci numbers, Phi Ratio, Alpha, Beta, Gamma, Epsilon and shuffled. Each preparation sets a laser-pulse program. The DNA itself stayed unchanged.'),el('p','The pulses act on six atoms. We count how often each atom is read in its excited state, a higher-energy state. The six circles below represent those atoms. Their percentages describe the machine’s readings, not the chance of a protein binding to DNA.'));
  const controls=el('div',undefined,'response-controls'),windowLabel=el('label','Choose a DNA sample'),windowSelect=el('select'),variantLabel=el('label','Choose a calculation method'),variantSelect=el('select');
  const windows=[...new Set(records.map(r=>r.window))];for(const [i,w]of windows.entries()){const o=el('option',`DNA sample ${i+1} · 256 bases`);o.value=w;windowSelect.append(o);}for(const [id,name]of Object.entries(labels)){const o=el('option',name);o.value=id;variantSelect.append(o);}
  const saved=choices.get(slot.id);if(saved&&windows.includes(saved.window)){windowSelect.value=saved.window;variantSelect.value=saved.variant;}
  windowLabel.append(windowSelect);variantLabel.append(variantSelect);controls.append(windowLabel);
  const sampleNote=el('p',undefined,'sample-definition');
  const comparison=el('section',undefined,'representation-comparison');
  const atoms=el('section',undefined,'atom-detail'),variantMeaning=el('p',undefined,'representation-definition'),selectedMeaning=el('p',undefined,'selected-response-meaning');
  atoms.append(el('h4','Read the six atoms'),el('p',`Each percentage is the frequency of an excited-state reading across repeated ${quantum?'hardware runs':'simulated runs'}. One repetition is called a shot. Select an atom to read its percentage in words.`),variantLabel,variantMeaning,selectedMeaning);
  const visual=el('div',undefined,'response-visual'),svg=svgNode('svg',{viewBox:'0 0 380 310',role:'group','aria-label':'Six atom positions. Select an atom for its explanation.'}),table=el('table'),atomMeaning=el('p',undefined,'atom-explanation');atomMeaning.setAttribute('aria-live','polite');
  visual.append(svg,table);atoms.append(visual,atomMeaning,el('p','Calculated % predicts the atom reading when hardware noise is left out. The 95% sampling range shows uncertainty from the limited number of shots. It excludes other hardware errors and is not a biological confidence score.','measurement-note'));
  if(quantum)atoms.append(el('p','Aquila infers excitation from an atom missing at final readout; atom loss can mimic excitation. Shots with an atom missing before the experiment are excluded.','measurement-note'));
  box.append(intro,controls,sampleNote,comparison,atoms);let selectedAtom=saved?.atom??1;
  function draw(){
   const r=records.find(x=>x.window===windowSelect.value&&x.variant===variantSelect.value);if(!r)return;
   choices.set(slot.id,{window:r.window,variant:r.variant,atom:selectedAtom});svg.replaceChildren();table.replaceChildren();comparison.replaceChildren();
   const index=windows.indexOf(r.window),rows=records.filter(x=>x.window===r.window),base=rows.find(x=>x.variant==='ordinary'),windowRecord=data.windows.find(x=>x.id===r.window);
   sampleNote.textContent=`A “window” is a short stretch of DNA. Sample ${index+1} is 256 bases near OXTR on chromosome 3 (GRCh38: ${(windowRecord.start+1).toLocaleString('en-US')}–${windowRecord.end.toLocaleString('en-US')}, one-based positions). Each sample is one DNA sequence, not a person. We chose these sequences because they contain many matches to the CTCF pattern. Laboratory binding results were not used to select them.`;
   comparison.append(el('h4','What changed inside the six-atom experiment?'),el('p',`The bars show the change in excited-state readings, averaged across the six atoms, compared with ordinary input for this DNA sample. Its ${source} average is ${(100*average(base[field].slice(0,6))).toFixed(1)}%.`));
   const guide=el('div',undefined,'chart-guide');
   for(const [cls,title,text] of [['model-key','Blue · calculated prediction','A separate computer calculation predicts the atom readings with hardware noise left out. This is the reference for checking the program, not a desired biological outcome.'],['measured-key',quantum?'Mint · Aquila measurements':'Mint · CUDA-Q sample readings',quantum?'The readings returned by the real quantum machine.':'Readings from 100 repetitions of the CUDA-Q program on an NVIDIA GPU through qBraid.']]){const row=el('p');row.append(el('span','',cls),el('strong',title),el('span',text));guide.append(row);}
   guide.append(el('p','Left = fewer excited-state readings than ordinary input. Right = more. Neither side is the “good” side. Success means more accurate binding predictions in the next test. A change from 20% to 23% is +3 percentage points.','chart-direction'));comparison.append(guide);
   const axis=el('div',undefined,'response-axis');axis.append(el('span','−15 pp · fewer'),el('span','0 · same'),el('span','+15 pp · more'));comparison.append(axis);
   for(const item of rows){
    const measured=100*(average(item[field].slice(0,6))-average(base[field].slice(0,6))),reference=100*(average(item.ideal_observables.slice(0,6))-average(base.ideal_observables.slice(0,6)));
    const row=el('div',undefined,'representation-row');row.dataset.variant=item.variant;const label=el('div');label.append(el('strong',labels[item.variant]),el('small',`${source}: ${pp(measured)}`),el('small',`Calculated: ${pp(reference)}`));
    const bar=svgNode('svg',{viewBox:'0 0 500 38',preserveAspectRatio:'none',role:'img','aria-label':`${labels[item.variant]}: ${source} change ${pp(measured)}; calculated change ${pp(reference)}. Neither direction indicates prediction quality.`});const x=v=>250+v*15;
    bar.append(svgNode('line',{x1:25,x2:475,y1:19,y2:19,stroke:'#334c56'}),svgNode('line',{x1:250,x2:250,y1:0,y2:38,stroke:'#7a919a'}),svgNode('line',{x1:250,x2:x(reference),y1:11,y2:11,stroke:'#a7c8e9','stroke-width':4}),svgNode('line',{x1:250,x2:x(measured),y1:27,y2:27,stroke:'#7fe3c6','stroke-width':6}),svgNode('circle',{cx:x(reference),cy:11,r:3,fill:'#a7c8e9'}),svgNode('circle',{cx:x(measured),cy:27,r:4,fill:'#7fe3c6'}));row.append(label,bar);comparison.append(row);
   }
   comparison.append(el('p','Each colour uses its own ordinary-input baseline. The gap between the bars shows how closely the measured change follows the calculated change. Small gaps may come from limited sampling. Aquila ran each setting once, in a fixed order; repeat runs are needed to check consistency.','measurement-note'));
   variantMeaning.textContent=meanings[r.variant];const change=100*(average(r[field].slice(0,6))-average(base[field].slice(0,6)));
   selectedMeaning.textContent=r.variant==='ordinary'?'Ordinary input contains the DNA measurements without the added Fibonacci weights. It provides the starting point for comparing all eight ways of preparing the measurements.':`${labels[r.variant]} changes the ${source} average by ${pp(change)} in this DNA sample. This tells us how the machine reacted to this way of preparing the measurements. Binding-site accuracy is measured in the next test against laboratory data.`;
   const n=quantum?r.hardware_valid_shots:r.shots;
   svg.append(svgNode('polygon',{points:Array.from({length:6},(_,i)=>{const t=i*Math.PI/3;return`${190+96*Math.sin(t)},${155-96*Math.cos(t)}`;}).join(' '),fill:'none',stroke:'#415b65','stroke-width':1}));
   table.append(el('caption',`${source} · ${n} usable shots`));const head=el('tr');for(const text of ['Atom','Calculated %',quantum?'Aquila %':'GPU %','Sampling range'])head.append(el('th',text));const thead=el('thead'),tbody=el('tbody');thead.append(head);table.append(thead,tbody);
   for(let i=0;i<6;i++){
    const p=r[field][i],k=Math.round(p*n),t=i*Math.PI/3,x=190+96*Math.sin(t),y=155-96*Math.cos(t),z=1.96,d=1+z*z/n,c=(p+z*z/(2*n))/d,e=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;
    const group=svgNode('g',{role:'button',tabindex:0,'aria-pressed':String(selectedAtom===i),'aria-label':`Atom ${i+1}: ${k} excited-state readings out of ${n} shots. Select for explanation.`});
    group.append(svgNode('circle',{cx:x,cy:y,r:24,fill:'#143c3b',stroke:selectedAtom===i?'#e9d86b':'#a4e4d0','stroke-width':selectedAtom===i?3:1.5}),svgNode('text',{x,y:y+4,fill:'#eef6f7','text-anchor':'middle','font-size':13},String(i+1)),svgNode('text',{x:190+139*Math.sin(t),y:159-139*Math.cos(t),fill:'#e9d86b','text-anchor':'middle','font-size':13},(p*100).toFixed(0)+'%'));
    const select=()=>{selectedAtom=i;draw();svg.querySelectorAll('[role=button]')[i].focus({preventScroll:true});};group.addEventListener('click',select);group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});svg.append(group);
    const tr=el('tr');if(selectedAtom===i)tr.className='selected-atom-row';for(const v of [String(i+1),(r.ideal_observables[i]*100).toFixed(1)+'%',(p*100).toFixed(0)+'%',`${(Math.max(0,c-e)*100).toFixed(0)}–${(Math.min(1,c+e)*100).toFixed(0)}%`])tr.append(el('td',v));tbody.append(tr);
   }
   const k=Math.round(r[field][selectedAtom]*n),pct=(100*r[field][selectedAtom]).toFixed(0);
   atomMeaning.textContent=`Atom ${selectedAtom+1}: ${k} excited-state readings out of ${n} usable shots (${pct}%, rounded). The noise-free calculation predicts ${(100*r.ideal_observables[selectedAtom]).toFixed(1)}%. The percentage describes this atom’s quantum-state readout. It is not a ${pct}% chance that a protein binds to OXTR.`;
  }
  windowSelect.onchange=variantSelect.onchange=draw;draw();
 }catch{box.replaceChildren(el('p','The recorded response data did not load. Refresh to retry.'));}
}
