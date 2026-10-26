'use strict';
(async()=>{
 const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
 if(window.parent===window){
  const film=document.createElement('video');film.className='molecular-film';film.muted=true;film.defaultMuted=true;film.loop=true;film.playsInline=true;film.preload='metadata';film.src='https://rmu.universalmind.tv/rihanna-global-citizen-loop.mp4';film.setAttribute('aria-hidden','true');
  const shade=document.createElement('div');shade.className='molecular-film-shade';shade.setAttribute('aria-hidden','true');document.body.prepend(film,shade);
  function syncFilm(){const playing=!film.paused;$('#background-play').textContent=playing?'Ⅱ':'▷';$('#background-play').setAttribute('aria-label',playing?'Pause background video':'Play background video');$('#background-play').title=playing?'Pause background video':'Play background video';$('#background-sound').textContent=film.muted?'♪̸':'♫';$('#background-sound').setAttribute('aria-label',film.muted?'Unmute background video':'Mute background video');$('#background-sound').title=film.muted?'Unmute background video':'Mute background video';}
  $('#background-play').onclick=()=>{if(film.paused)film.play().catch(syncFilm);else film.pause();};$('#background-sound').onclick=()=>{film.muted=!film.muted;if(film.paused)film.play().catch(syncFilm);syncFilm();};film.addEventListener('play',syncFilm);film.addEventListener('pause',syncFilm);
  let resumeFilm=false;document.addEventListener('visibilitychange',()=>{if(document.hidden){resumeFilm=!film.paused;film.pause();}else if(resumeFilm)film.play().catch(syncFilm);});window.addEventListener('pagehide',()=>film.pause(),{once:true});
  if(!matchMedia('(prefers-reduced-motion: reduce)').matches)film.play().catch(syncFilm);syncFilm();
 }

 const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 let records=[],record,selectedTarget,engine,request=0,queue=Promise.resolve(),picked=null,activeView='structure',activeSequence='',spinning=false,geometryTimer,geneTimer,level=['gene','protein','docking','quantum'].includes(new URLSearchParams(location.search).get('level'))?new URLSearchParams(location.search).get('level'):'gene',gene,geneRepresentation='cartoon',geneWindow=0,geneIndex=new Map(),geneCache=new Map();
 const isInteraction=()=>['docking','quantum'].includes(level);
 const interactionUi=window.createInteractionUI({enqueue,geometryChanged:()=>enqueue(updateGeometry),showEvidence:()=>changeView('evidence',true),download});
 document.body.dataset.level=level;
 const geometryState=()=>({scale:Number($('#geometry-scale').value),rotation:Number($('#geometry-rotation').value),offset:Number($('#geometry-offset').value),opacity:Number($('#geometry-opacity').value)});
 function busy(message){document.body.dataset.busy='true';$('#loading').hidden=false;$('#loading').dataset.error='false';$('#loading-message').textContent=message;$('#retry').hidden=true;$('#structure-status').textContent=message;}
 function fail(error){document.body.dataset.busy='false';$('#loading').hidden=false;$('#loading').dataset.error='true';$('#loading-message').textContent='View unavailable. '+error.message;$('#retry').hidden=false;$('#structure-status').textContent='View unavailable';console.error(error);}
 function enqueue(fn){queue=queue.then(fn).catch(fail);return queue;}
 function changeView(view,focus=false){if(view!=='structure')interactionUi.pause();activeView=view;document.body.dataset.view=view;$$('.primary [data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===view)));$('#collection-panel').hidden=view!=='collection';$('#detail-panel').hidden=!['sequence','evidence','geometry'].includes(view);if(view==='sequence')sequencePanel();if(view==='evidence')evidencePanel();if(view==='geometry')geometryPanel();if(focus&&!$('#detail-panel').hidden)$('#close-detail').focus();requestAnimationFrame(()=>engine?.plugin.canvas3d.requestResize());}
 function catalogue(){
  const available=isInteraction()?records.filter(r=>r.id==='OXTR'):records;
  $('#target-selector').innerHTML=available.map(r=>`<button class="target-card" data-target="${r.id}" aria-pressed="false" aria-label="Explore ${escape(r.name)}"><img src="${level==='gene'?'gene-thumb.svg':'thumbnails/'+r.id.toLowerCase()+'.png'}" alt="" width="150" height="77"><span>${r.id}</span></button>`).join('');
  $('#collection-grid').innerHTML=available.map(r=>`<button class="collection-item" data-target="${r.id}"><img src="${level==='gene'?'gene-thumb.svg':'thumbnails/'+r.id.toLowerCase()+'.png'}" alt="" width="240" height="105"><strong>${r.id}</strong><span>${escape(r.name)}</span><span>${level==='gene'?r.genes.join(' · ')+' · DNA':level==='quantum'?'Teaching & calculated orbitals':level==='docking'?'OXT–OXTR contact · '+r.pdb:'Protein · '+r.pdb+' · '+r.resolution.toFixed(2)+' Å'}</span></button>`).join('');
  if(isInteraction())$('#target-selector').insertAdjacentHTML('beforeend','<div class="interaction-footer"><strong>FIRST MOLECULAR STUDY</strong>Oxytocin · human oxytocin receptor<br>Two independent views of the same molecular encounter.</div>');
  $$('button[data-target]').forEach(button=>button.addEventListener('click',()=>{changeView('structure');select(button.dataset.target);}));
 }
 function genePosition(offset){return gene.strand===1?gene.start+offset:gene.end-offset;}
 function geneOffset(position){return gene.strand===1?position-gene.start:gene.end-position;}
 const number=n=>Number(n).toLocaleString('en-US');
 const windowSize=48;
 async function fetchGene(symbol){
  if(geneCache.has(symbol))return geneCache.get(symbol);
  const response=await fetch('data/genes/'+symbol.toLowerCase()+'.json?v=1');if(!response.ok)throw Error('The DNA reference did not load.');
  const g=await response.json(),expected=geneIndex.get(symbol);if(g.symbol!==symbol||g.sequence.length!==g.length||g.sha256!==expected.sha256)throw Error('DNA reference metadata mismatch.');
  const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(g.sequence));
  const hash=[...new Uint8Array(digest)].map(n=>n.toString(16).padStart(2,'0')).join('');if(hash!==expected.sha256)throw Error('DNA reference checksum mismatch.');
  geneCache.set(symbol,g);return g;
 }
 function updateLevel(){
  document.body.dataset.level=level;$$('.biology-toggle [data-level]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.level===level)));
  $('.primary [data-view=sequence]').textContent=level==='docking'?'Contacts':level==='quantum'?'Orbitals':'Sequence';
  if(isInteraction()){
   $('#level-description').textContent=level==='docking'?'OXT–OXTR · molecular encounter':'OXT–OXTR · electronic structure';
   $('#pick-hint').textContent='EXPLORE THE MODEL';$('#source-heading').textContent=level==='docking'?'STRUCTURE SOURCE':'MODEL SOURCE';
   $('.scene').setAttribute('aria-label',level==='docking'?'Interactive OXT–OXTR docking model':'Interactive quantum fragment model');
   $('#source-open').setAttribute('aria-label','Inspect model evidence');$('#save-image').setAttribute('aria-label','Download model image');
   $('#footer-context').textContent=level==='docking'?'ILLUSTRATIVE MOTION · EXPERIMENTAL BOUND POSE':'CALCULATED ORBITALS · STATIONARY MODEL';
   $('#reset-view').setAttribute('aria-label','Reset model view');$('#rotate-view').setAttribute('aria-label','Rotate model automatically');catalogue();return;
  }
  $('#level-description').textContent=level==='gene'?'Gene · modelled B-DNA':'Protein · experimental structure';
  $('#pick-hint').textContent=level==='gene'?'CLICK A BASE PAIR':'CLICK A RESIDUE';
  $('#source-heading').textContent=level==='gene'?'GENE SOURCE':'PROTEIN SOURCE';
  $('.scene').setAttribute('aria-label',level==='gene'?'Modelled gene molecular structure':'Experimental protein structure');$('#source-open').setAttribute('aria-label',level==='gene'?'Inspect gene evidence':'Inspect protein evidence');$('#save-image').setAttribute('aria-label',level==='gene'?'Download gene molecular image':'Download protein image');
  $('#footer-context').innerHTML=(level==='gene'?'REFERENCE SEQUENCE · MODELLED B-DNA':'EXPERIMENTAL PROTEIN')+' <span class="footer-divider">·</span> EXPLORATORY GEOMETRY';
  $('#reset-view').setAttribute('aria-label',level==='gene'?'Reset DNA view':'Reset protein view');
  $('#rotate-view').setAttribute('aria-label',level==='gene'?'Rotate DNA automatically':'Rotate protein automatically');
  $('#clear-selection').setAttribute('aria-label',level==='gene'?'Clear selected base':'Clear selected residue');
  representationLabels();catalogue();
 }
 function representationLabels(){
  if(isInteraction())return;
  const type=level==='gene'?geneRepresentation:engine?.snapshot().representation||'cartoon';
  $('#cartoon-label').textContent=level==='gene'?'Helix':'Ribbon';
  $('#representation-note').textContent=level==='gene'?({'cartoon':'Backbone and base rings from the atomic model.','ball-and-stick':'Atoms and bonds · carbon by strand; nitrogen blue, oxygen red, phosphorus orange.','molecular-surface':'Molecular surface calculated from the same atoms.'}[type]||''):'';
  $('#representation-note').hidden=level!=='gene';
  $$('button[data-representation]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.representation===type)));document.body.dataset.representation=type;
 }
 function setLevel(next){if(next===level||!['gene','protein','docking','quantum'].includes(next))return;interactionUi.pause();changeView('structure');level=next;updateLevel();select(selectedTarget||record?.id||new URLSearchParams(location.search).get('target')||'OXTR');}
 async function select(id,symbol){
  const next=records.find(r=>r.id===(isInteraction()?'OXTR':id))||records[0],token=++request,requestedLevel=level;
  selectedTarget=next.id;
  const symbolToLoad=next.genes.includes(symbol)?symbol:next.genes.includes(gene?.symbol)?gene.symbol:next.genes[0];
  busy(isInteraction()?'Preparing '+(level==='docking'?'molecular docking':'quantum view')+'…':requestedLevel==='gene'?'Loading '+symbolToLoad+' reference DNA…':'Loading '+next.name.toLowerCase()+' protein…');$('#selection').hidden=true;$('#geometry-badge').hidden=true;
  $('#target-title').textContent=requestedLevel==='gene'?symbolToLoad:next.id;$('#target-name').textContent=requestedLevel==='gene'?(next.id==='NMDA'?'NMDA receptor subunit '+(symbolToLoad==='GRIN1'?'1':'2B')+' gene':next.id==='CD38'?'CD38 gene':next.name+' gene'):next.name;$('#target-context').textContent=requestedLevel==='gene'?'GENE · MODELLED B-DNA':'PROTEIN · '+next.context;$('#target-number').textContent=String(records.indexOf(next)+1).padStart(2,'0')+' / 08';
  if(isInteraction()){$('#target-title').textContent=level==='docking'?'OXT · OXTR':'Quantum';$('#target-name').textContent=level==='docking'?'The aromatic encounter':'The aromatic electron system';$('#target-context').textContent=level==='docking'?'Tyr2 · Phe291 · receptor pocket':'Tyr2–Phe291 · calculated fragment pair';$('#target-number').textContent='01 / 01';}
  $$('button[data-target]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.target===next.id)));
  await enqueue(async()=>{
   if(token!==request)return;
   let loaded;interactionUi.dispose();
   if(['docking','quantum'].includes(requestedLevel)){loaded=await interactionUi.load(requestedLevel,engine,next);}
   else if(requestedLevel==='gene'){
    const nextGene=await fetchGene(symbolToLoad);if(token!==request)return;
    gene=nextGene;geneWindow=Math.max(0,Math.min(gene.strand===1?gene.tss-gene.start:gene.end-gene.tss,gene.length-windowSize));
    loaded=await engine.loadGene(gene,geneWindow,false,geneRepresentation);
   }else loaded=await engine.load(next);
   if(token!==request)return;
   record=next;picked=null;activeSequence=record.chains[0];spinning=false;engine.spin(false);$('#rotate-view').setAttribute('aria-pressed','false');
   if(requestedLevel==='gene'){
    renderGeneControls();geneStatus();$('#pdb-link').href=gene.geneSource;$('#pdb-link').innerHTML=`${gene.symbol} · Ensembl <span>↗</span>`;$('#method-summary').textContent='GRCh38 sequence · ideal B-DNA model';
   }else if(requestedLevel==='protein'){
    $('#ligand-toggle').disabled=!loaded.ligand;$('#ligand-toggle').checked=loaded.ligand;$('#complex-toggle').disabled=!loaded.context;$('#complex-toggle').checked=false;
    $('#ligand-label').textContent=record.id==='OXTR'?'Oxytocin':'Bound ligand';
    $$('button[data-representation]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.representation==='cartoon')));
    $('#pdb-link').href=record.source;$('#pdb-link').innerHTML=`PDB ${record.pdb} <span>↗</span>`;$('#method-summary').textContent=`${record.method==='ELECTRON MICROSCOPY'?'Cryo-EM':'X-ray diffraction'} · ${record.resolution.toFixed(2)} Å`;
    $('#structure-status').textContent=`PROTEIN · ${record.pdb} · ${record.resolution.toFixed(2)} Å · ${loaded.atoms.toLocaleString()} modelled atoms`;
   }
   representationLabels();await updateGeometry();$('#loading').hidden=true;document.body.dataset.busy='false';document.body.dataset.target=record.id;document.body.dataset.ready='true';
   const url=new URL(location.href);url.searchParams.set('target',record.id);url.searchParams.set('level',level);history.replaceState(null,'',url);
   if(['sequence','evidence'].includes(activeView))changeView(activeView);
  });
 }
 function geneStatus(){
  $('#structure-status').textContent=`DNA · chr${gene.chromosome}:${number(genePosition(geneWindow))}–${number(genePosition(geneWindow+windowSize-1))} · 48 bp · ${number(engine.snapshot().atomCount)} atoms`;
  document.body.dataset.gene=gene.symbol;document.body.dataset.window=String(geneWindow);
 }
 function renderGeneControls(){
  $('#gene-symbol').innerHTML=record.genes.map(g=>`<option value="${g}"${g===gene.symbol?' selected':''}>${g}${record.id==='NMDA'?' · NMDA subunit gene':''}</option>`).join('');
  $('#gene-symbol').disabled=record.genes.length===1;
  $('#gene-locus').textContent=`GRCh38 · chr${gene.chromosome}:${number(gene.start)}–${number(gene.end)} · ${gene.strand===1?'+':'−'} strand · ${number(gene.length)} bp`;
  const scale=196/gene.length;
  $('#gene-track').innerHTML=`<svg viewBox="0 0 200 34" role="img" aria-label="${gene.symbol}: ${gene.exons.length} exons on the canonical transcript, in transcription order"><path d="M2 17H198" stroke="#61737e"/>${gene.exons.map(e=>`<rect x="${2+e.offset*scale}" y="10" width="${Math.max(1,e.length*scale)}" height="14" fill="#d9b27a"><title>Exon ${e.number} · ${number(e.length)} bp</title></rect>`).join('')}<path d="M${2+geneWindow*scale} 4v26" stroke="#82d4ef" stroke-width="2"/></svg>`;
  $('#gene-exon').innerHTML='<option value="">Choose a position</option><option value="tss">Transcription start</option>'+gene.exons.map(e=>`<option value="${e.offset}">Exon ${e.number} · ${number(e.length)} bp</option>`).join('');
  $('#gene-offset').max=String(gene.length-windowSize);$('#gene-offset').value=String(geneWindow);$('#gene-offset-label').textContent=number(geneWindow+1)+'–'+number(geneWindow+windowSize);
  $('#gene-position').max=String(gene.length-windowSize+1);$('#gene-position').value=String(geneWindow+1);$('#gene-prev').disabled=geneWindow===0;$('#gene-next').disabled=geneWindow===gene.length-windowSize;
 }
 function changeWindow(offset){
  const token=request,symbol=gene?.symbol;if(level!=='gene'||!gene)return Promise.resolve();
  const next=Math.max(0,Math.min(Math.round(Number(offset)||0),gene.length-windowSize));
  return enqueue(async()=>{if(token!==request||level!=='gene'||gene.symbol!==symbol)return;geneWindow=next;await engine.loadGene(gene,geneWindow,true,geneRepresentation);renderGeneControls();geneStatus();await updateGeometry();if(activeView==='sequence')sequencePanel();});
 }
 function geneSequencePanel(){
  $('#detail-title').textContent='DNA sequence · '+gene.symbol;
  const first=Math.floor(geneWindow/192)*192,last=Math.min(first+192,gene.length);
  $('#detail-content').innerHTML=`<span class="evidence-badge">GENE · REFERENCE DNA</span><h3>${gene.symbol} · ${number(gene.length)} bases</h3><p>These are the reference DNA letters, read in the gene’s 5′ → 3′ direction. The gold letters mark the 48-base window in the helix. Select a letter to explore that part of the gene.</p><p class="dna-coordinates">GRCh38 · chr${gene.chromosome}:${number(genePosition(first))}–${number(genePosition(last-1))} · ${gene.strand===1?'+':'−'} strand</p><div class="dna-page-controls"><button id="dna-page-prev" ${first===0?'disabled':''} aria-label="Previous DNA sequence page">←</button><span>Bases ${number(first+1)}–${number(last)}</span><button id="dna-page-next" ${last===gene.length?'disabled':''} aria-label="Next DNA sequence page">→</button></div><div class="sequence-grid">${[...gene.sequence.slice(first,last)].map((base,i)=>`<button class="residue dna-base${first+i>=geneWindow&&first+i<geneWindow+windowSize?' in-window':''}${picked?.kind==='gene'&&picked.offset===first+i?' selected':''}" data-base="${first+i}" title="${base} · chr${gene.chromosome}:${genePosition(first+i)}" aria-label="${base}, gene base ${first+i+1}, chromosome position ${genePosition(first+i)}">${base}</button>`).join('')}</div><p class="evidence-note">The DNA letters are sourced reference data. All three views use the same constructed atomic model of this segment; the model does not reproduce its folding inside a cell.</p><button id="dna-download" class="text-button dna-download">Download full DNA · FASTA ↓</button><button id="dna-annotation" class="text-button">Download gene annotation · JSON ↓</button>`;
  $('#dna-page-prev').onclick=()=>changeWindow(first-192);$('#dna-page-next').onclick=()=>changeWindow(first+192);
  $$('[data-base]').forEach(b=>b.onclick=()=>{picked={kind:'gene',offset:Number(b.dataset.base)};changeWindow(Math.max(0,picked.offset-24));});
  $('#dna-download').onclick=()=>download(gene.symbol+'-GRCh38-DNA.fasta',`>${gene.symbol} ${gene.ensembl}.${gene.version} GRCh38 chr${gene.chromosome}:${gene.start}-${gene.end} strand=${gene.strand} gene-oriented-5prime-to-3prime\n`+gene.sequence.match(/.{1,80}/g).join('\n')+'\n','text/plain');
  $('#dna-annotation').onclick=()=>{const {sequence,...annotation}=gene;download(gene.symbol+'-GRCh38-annotation.json',JSON.stringify(annotation,null,2),'application/json');};
 }
 function geneEvidencePanel(){
  $('#detail-title').textContent='Gene evidence · '+gene.symbol;
  $('#detail-content').innerHTML=`<span class="evidence-badge">GENE · REFERENCE DNA</span><h3>${gene.symbol}</h3><dl><dt>Assembly</dt><dd>Human GRCh38</dd><dt>Gene identifier</dt><dd>${gene.ensembl}.${gene.version}</dd><dt>Chromosome</dt><dd>${gene.chromosome}</dd><dt>Genomic span</dt><dd>${number(gene.start)}–${number(gene.end)}</dd><dt>Gene length</dt><dd>${number(gene.length)} base pairs</dd><dt>Strand</dt><dd>${gene.strand===1?'Forward (+)':'Reverse (−)'}</dd><dt>Transcript</dt><dd>${gene.canonicalTranscript}</dd><dt>Transcript exons</dt><dd>${gene.exons.length}</dd><dt>Transcription start</dt><dd>${number(gene.tss)}</dd><dt>Retrieved</dt><dd>${escape(gene.retrieved.slice(0,10))} UTC</dd></dl><h4>SEQUENCE AND ANNOTATION</h4><p><a href="${gene.geneSource}" target="_blank" rel="noopener">Ensembl gene record ↗</a><br><a href="${escape(gene.annotationSource)}" target="_blank" rel="noopener">Ensembl transcript and exon annotation ↗</a><br><a href="${escape(gene.sequenceSource)}" target="_blank" rel="noopener">UCSC GRCh38 genomic DNA ↗</a><br><a href="${escape(gene.crossCheckSource)}" target="_blank" rel="noopener">Ensembl sequence cross-check ↗</a></p><p>The UCSC and Ensembl reference sequences match exactly. Genomic coordinates are 1-based and inclusive. Reverse-strand genes are displayed as the reverse complement, in the gene’s 5′ → 3′ direction.</p><h4>MOLECULAR MODEL AND VIEWS</h4><p class="evidence-note">The selected 48-base segment is built as an atomic B-DNA duplex, using NAB Langridge B-DNA templates from BioExcel. Helix traces its backbone and base rings; Atoms shows atoms and bonds; Surface calculates the molecular envelope from those same atoms. The complementary strand runs in the opposite direction. Hydrogen atoms are omitted.</p><p>This is an ideal molecular model with a uniform 3.38 Å rise and 36° twist per base pair. It is not an experimentally measured structure of this gene, a relaxed simulation, or its folded arrangement inside a cell. DNA sequence alone does not establish that cellular shape.</p><p><a href="${MolecularEngine.dnaModelSource.referenceSource}" target="_blank" rel="noopener">BioExcel · reference atomic coordinates ↗</a><br><a href="${MolecularEngine.dnaModelSource.builderSource}" target="_blank" rel="noopener">NAB builder · construction method ↗</a><br><a href="vendor/dna-model-notice.txt" target="_blank" rel="noopener">Template attribution and license ↗</a><br><button id="gene-model-download" class="text-button">Download modelled segment · PDB ↓</button></p><p>The full gene includes introns. Exon markers and the transcription start refer to the named canonical transcript; other transcripts may differ. The reference is not a personal DNA sequence.</p>${record.id==='NMDA'?'<p>NMDA is a receptor assembled from multiple protein subunits. This Atlas pairs its GluN1/GluN2B protein model with the separate GRIN1 and GRIN2B genes.</p>':''}${record.id==='OXTR'?'<p><a href="../?area=genomics" data-dna-context>Explore OXTR transcription and regulatory evidence ↗</a></p>':''}<h4>GEOMETRY</h4><p>Mathematical overlays remain independent, exploratory constructions. Their overlap with this model does not establish a biological mechanism.</p>`;
  $('#gene-model-download').onclick=()=>download(gene.symbol+'-modelled-B-DNA-'+(geneWindow+1)+'-'+(geneWindow+48)+'.pdb',engine.geneCoordinates(),'chemical/x-pdb');
  wireDNAContext();
 }
 function wireDNAContext(){ $('[data-dna-context]')?.addEventListener('click',e=>{if(window.parent!==window){e.preventDefault();window.parent.postMessage({type:'molecular:gene-context'},location.origin);}}); }
 async function updateGeometry(){
  if(level==='quantum'){$('#geometry-badge').hidden=true;document.body.dataset.geometry='none';await engine.geometryLayer(null);return;}
  const enabled=$('#geometry-toggle').checked,name=$('#geometry-kind').value;
  $('#geometry-sliders').hidden=!enabled;$('#geometry-badge').hidden=!enabled;
  $('#scale-value').textContent=Number($('#geometry-scale').value).toFixed(2)+'×';$('#rotation-value').textContent=$('#geometry-rotation').value+'°';$('#offset-value').textContent=Number($('#geometry-offset').value).toFixed(2);$('#opacity-value').textContent=Math.round(Number($('#geometry-opacity').value)*100)+'%';
  const g=MolecularEngine.geometry(name);$('#geometry-label').textContent=g.name;$('#geometry-note').textContent=enabled?g.note:'Choose a construction and enable the geometry layer.';
  if(engine&&record)await engine.geometryLayer(enabled?name:null,{...geometryState(),...(level==='docking'?{sphere:interactionUi.model?.geometrySphere()}:{})});
  document.body.dataset.geometry=enabled?name:'none';if(activeView==='geometry')geometryPanel();
 }
 function evidencePanel(){if(!record)return;if(isInteraction()){$('#detail-title').textContent=level==='docking'?'Docking · evidence & method':'Quantum · evidence & method';$('#detail-content').innerHTML=interactionUi.detail('evidence');return;}if(level==='gene')return geneEvidencePanel();$('#detail-title').textContent='Protein evidence · '+record.id;const c=record.citation;
  $('#detail-content').innerHTML=`<span class="evidence-badge">PROTEIN · EXPERIMENTALLY DETERMINED</span><h3>${escape(record.title)}</h3><dl><dt>Structure</dt><dd><a href="${record.source}" target="_blank" rel="noopener">PDB ${record.pdb} ↗</a></dd><dt>Method</dt><dd>${escape(record.method)}</dd><dt>Resolution</dt><dd>${record.resolution.toFixed(2)} Å</dd><dt>Target genes</dt><dd>${record.genes.join(' · ')}</dd><dt>Displayed chains</dt><dd>${record.sequences.map(s=>escape(s.authorChain)).join(', ')} (author identifiers)</dd><dt>Source retrieved</dt><dd>21 October 2026</dd></dl><p class="evidence-note">${escape(record.note)} Coordinates show modelled atoms only; unresolved residues are absent.</p><h4>PRIMARY PUBLICATION</h4><p>${escape(c.title||record.title)}</p><p>${escape(c.journal_abbrev||'')} ${escape(c.year||'')} ${c.pdbx_database_id_PubMed?`· <a target="_blank" rel="noopener" href="https://pubmed.ncbi.nlm.nih.gov/${c.pdbx_database_id_PubMed}/">PubMed ↗</a>`:''} ${c.pdbx_database_id_DOI?`· <a target="_blank" rel="noopener" href="https://doi.org/${encodeURI(c.pdbx_database_id_DOI)}">Paper ↗</a>`:''}</p><h4>GENE CONTEXT</h4><p>${record.genes.map((g,i)=>`<a target="_blank" rel="noopener" href="https://www.ncbi.nlm.nih.gov/gene/${record.geneIds[i]}">${g} · NCBI Gene ↗</a>`).join('<br>')}</p>${record.id==='OXTR'?'<p><a href="../?area=genomics" data-dna-context>Explore OXTR DNA and regulatory evidence ↗</a></p>':''}<h4>GEOMETRY &amp; INTERPRETATION</h4><p>Geometry layers use their own defined mathematical coordinates. Their placement, scale and rotation are exploratory. A visual overlap does not establish molecular binding, a biological mechanism or predictive accuracy.</p><h4>DATA &amp; RENDERING</h4><p><a href="data/${record.pdb.toLowerCase()}.bcif" download>Download deposited coordinates · BinaryCIF ↓</a><br><a href="https://molstar.org/" target="_blank" rel="noopener">Molecular rendering: Mol* ↗</a></p>`;
  $('[data-dna-context]')?.addEventListener('click',e=>{if(window.parent!==window){e.preventDefault();window.parent.postMessage({type:'molecular:gene-context'},location.origin);}});
 }
 function sequencePanel(){if(!record)return;if(isInteraction()){$('#detail-title').textContent=level==='docking'?'Molecular contacts':'Electronic orbitals';$('#detail-content').innerHTML=interactionUi.detail('sequence');return;}if(level==='gene')return geneSequencePanel();$('#detail-title').textContent='Protein sequence · '+record.id;
  const sequence=record.sequences.find(s=>s.chain===activeSequence)||record.sequences[0];activeSequence=sequence.chain;
  const observed=new Set(engine.observed(sequence.chain));
  $('#detail-content').innerHTML=`<span class="evidence-badge">DEPOSITED PROTEIN CONSTRUCT</span><h3>${escape(record.name)}</h3><p>Select a modelled amino acid to locate it in 3D. Numbering below follows the deposited construct; the selected residue readout uses the author’s numbering.</p><div class="sequence-toolbar"><label for="sequence-chain">Chain</label><select id="sequence-chain">${record.sequences.map(s=>`<option value="${s.chain}"${s.chain===activeSequence?' selected':''}>${escape(s.authorChain)} · ${s.length} residues</option>`).join('')}</select></div><p class="sequence-legend">${observed.size} modelled residues / ${sequence.length} deposited residues · dimmed letters have no coordinates.</p><div class="sequence-grid">${[...sequence.sequence].map((letter,i)=>`<button class="residue${picked?.labelChain===sequence.chain&&picked.seqId===i+1?' selected':''}" data-seq="${i+1}" data-missing="${!observed.has(i+1)}" ${!observed.has(i+1)?'disabled':''} aria-label="${letter}, construct position ${i+1}${observed.has(i+1)?'':', no coordinates'}" title="${letter} · ${i+1}">${letter}</button>`).join('')}</div><p class="evidence-note">${escape(record.note)}</p><p>Protein coordinates and DNA regulatory coordinates describe different biological levels. This sequence view does not project transcription-factor binding sites onto the protein.</p>`;
  $('#sequence-chain').onchange=e=>{activeSequence=e.target.value;sequencePanel();};
  $$('.residue:not(:disabled)').forEach(b=>b.onclick=()=>{engine.focusResidue(activeSequence,Number(b.dataset.seq));$$('.residue.selected').forEach(x=>x.classList.remove('selected'));b.classList.add('selected');$('#structure-status').textContent=`${record.pdb} · chain ${sequence.authorChain} · construct residue ${b.dataset.seq}`;});
 }
 function geometryPanel(){const key=$('#geometry-kind').value,g=MolecularEngine.geometry(key);$('#detail-title').textContent='Geometry · '+g.name;
  $('#detail-content').innerHTML=`<span class="evidence-badge">MATHEMATICAL CONSTRUCTION</span><h3>${g.name}</h3><p>${escape(g.note)}</p>${g.vertices.length?`<dl><dt>Vertices</dt><dd>${g.vertices.length}</dd><dt>Edges</dt><dd>${g.edges.length}</dd></dl>`:''}${key==='odom'?`<h4>ODOM’S CONSTRUCTION</h4><p>On each equilateral triangle, join the midpoints of two sides and extend that line to the circumcircle. The midpoint segment divided by either adjacent extension is the Phi Ratio.</p><p class="evidence-note">φ = (1 + √5) / 2 = ${MolecularEngine.PHI.toFixed(9)}<br>Computed segment ratio: ${g.ratio.toFixed(9)}</p><p>The original construction is planar. Here, four exact planar copies lie on the faces of a regular tetrahedron. This is an exploratory spatial arrangement, not a claim that Odom proposed a new solid.</p><p><a href="https://www.journal-1.eu/2021/Tran%20Quang%20Hung%2C%20Floor%20van%20Lamoen.%20Odom%27s%20Triangle%2C%20pp%2068-77.pdf" target="_blank" rel="noopener">Odom’s Triangle · Hung &amp; van Lamoen, 2021 ↗</a></p>`:''}<h4>PLACEMENT</h4><p>The construction is centred on the selected structure’s bounding sphere. Scale is relative to that sphere’s radius; horizontal offset is measured in the same radius units. Rotation is around the model’s vertical coordinate axis. These are display choices, not fitted biological findings.</p><h4>REPRODUCIBILITY</h4><p>Save this view to download the source identifier, coordinate checksum, camera, representation, layer settings and geometric parameters.</p><h4>KEYBOARD</h4><p class="keyboard-help">Focus the molecular canvas. Arrow keys rotate; + and − zoom; R resets. Escape closes this panel.</p>`;
 }
 function download(name,data,type){const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 function pick(item){if(isInteraction()){$('#selection').hidden=false;$('#selection-label').textContent=`${item.name} ${item.residue} · chain ${item.chain}`;$('#selection-sequence').hidden=true;return;}if(level==='gene'){picked=item;$('#selection').hidden=false;$('#selection-label').textContent=`${item.base}–${item.complement} · chr${gene.chromosome}:${number(item.position)}${item.atom?' · '+item.atom+' ('+item.element+')':''}`;if(activeView==='sequence')sequencePanel();return;}picked=item;$('#selection').hidden=false;$('#selection-label').textContent=`${item.name} ${item.residue} · chain ${item.chain}`;activeSequence=item.labelChain;if(activeView==='sequence')sequencePanel();}
 $$('.biology-toggle [data-level]').forEach(b=>b.onclick=()=>setLevel(b.dataset.level));
 $('#gene-symbol').onchange=e=>select(record.id,e.target.value);$('#gene-exon').onchange=e=>{if(e.target.value!=='')changeWindow(e.target.value==='tss'?geneOffset(gene.tss):Number(e.target.value));};
 $('#gene-offset').oninput=e=>{clearTimeout(geneTimer);const offset=Number(e.target.value);geneTimer=setTimeout(()=>changeWindow(offset),80);};$('#gene-position').onchange=e=>changeWindow(Number(e.target.value)-1);$('#gene-prev').onclick=()=>changeWindow(geneWindow-windowSize);$('#gene-next').onclick=()=>changeWindow(geneWindow+windowSize);
 $$('.primary [data-view]').forEach(b=>b.onclick=()=>changeView(b.dataset.view,true));$('#close-detail').onclick=()=>{changeView('structure');$('.primary [data-view="structure"]').focus();};$('#source-open').onclick=()=>changeView('evidence',true);$('#geometry-info').onclick=()=>changeView('geometry',true);
 $$('button[data-representation]').forEach(b=>b.onclick=()=>{const token=request,chosenLevel=level;return enqueue(async()=>{if(token!==request||chosenLevel!==level)return;await engine.render(b.dataset.representation);if(level==='gene')geneRepresentation=b.dataset.representation;representationLabels();});});
 $('#ligand-toggle').onchange=e=>enqueue(()=>engine.toggle('ligand',e.target.checked));$('#complex-toggle').onchange=e=>enqueue(()=>engine.toggle('context',e.target.checked));
 $('#geometry-toggle').onchange=()=>enqueue(updateGeometry);$('#geometry-kind').onchange=()=>{$('#geometry-toggle').checked=true;enqueue(updateGeometry);};
 $$('.geometry-sliders input').forEach(input=>input.oninput=()=>{clearTimeout(geometryTimer);geometryTimer=setTimeout(()=>enqueue(updateGeometry),70);});
 $('#reset-view').onclick=()=>isInteraction()?interactionUi.reset():engine?.reset();$('#zoom-in').onclick=()=>engine?.zoom(.8);$('#zoom-out').onclick=()=>engine?.zoom(1.25);
 $('#rotate-view').onclick=()=>{spinning=!spinning;engine.spin(spinning);$('#rotate-view').setAttribute('aria-pressed',String(spinning));};
 $('#selection-sequence').onclick=()=>changeView('sequence',true);$('#clear-selection').onclick=()=>{picked=null;$('#selection').hidden=true;if(!isInteraction())engine?.reset();};
 $('#save-image').onclick=()=>enqueue(async()=>{const uri=await engine.screenshot(),a=document.createElement('a');a.href=uri;a.download=`${record.id}-${isInteraction()?level:level==='gene'?'modelled-DNA':record.pdb}-molecular-atlas.png`;a.click();});
 $('#save-state').onclick=()=>{if(!record)return;if(isInteraction()){download(`OXTR-${level}-view.json`,JSON.stringify({schema:'universal-informatics/interaction-view/v1',created:new Date().toISOString(),level,camera:engine.snapshot().camera,...interactionUi.snapshot(),...(level==='docking'?{geometry:{enabled:$('#geometry-toggle').checked,kind:$('#geometry-kind').value,...geometryState()}}:{})},null,2),'application/json');return;}const source=level==='gene'?{source:gene.geneSource,gene:gene.symbol,assembly:gene.assembly,sequenceSHA256:gene.sha256,window:{offset:geneWindow,bases:windowSize,genomicStart:genePosition(geneWindow),genomicEnd:genePosition(geneWindow+windowSize-1)},interpretation:'Reference sequence in a constructed ideal B-DNA atomic model; no measured cellular fold or biological fit is inferred.'}:{source:record.source,coordinateSHA256:record.sha256,interpretation:'Experimental protein coordinates with exploratory geometric placement; no biological fit is inferred.'};download(`${record.id}-${level}-view.json`,JSON.stringify({schema:'universal-informatics/molecular-view/v2',created:new Date().toISOString(),target:record.id,level,...source,...engine.snapshot(),...(level==='protein'?{layers:{ligand:$('#ligand-toggle').checked,complex:$('#complex-toggle').checked}}:{}),geometry:{enabled:$('#geometry-toggle').checked,kind:$('#geometry-kind').value,...geometryState()}},null,2),'application/json');};
 $('#retry').onclick=()=>{if(engine)select(new URLSearchParams(location.search).get('target')||'OXTR');else location.reload();};
 if(window.parent!==window){$('#return').innerHTML='↗';$('#return').setAttribute('aria-label','Return to Geometry Atlas');$('#return').onclick=e=>{e.preventDefault();window.parent.postMessage({type:'molecular:close'},location.origin);};$('.brand').onclick=e=>{e.preventDefault();window.parent.postMessage({type:'molecular:close'},location.origin);};}
 document.addEventListener('keydown',e=>{if(e.key!=='Escape')return;if(activeView!=='structure'){changeView('structure');$('.primary [data-view="structure"]').focus();}else if(window.parent!==window)window.parent.postMessage({type:'molecular:close'},location.origin);});
 window.addEventListener('message',event=>{if(event.origin!==location.origin||event.source!==window.parent||event.data?.type!=='molecular:visibility')return;if(!event.data.active&&engine){interactionUi.pause();spinning=false;engine.spin(false);$('#rotate-view').setAttribute('aria-pressed','false');}else engine?.plugin.canvas3d.requestResize();});
 document.addEventListener('visibilitychange',()=>{if(document.hidden&&engine){interactionUi.pause();spinning=false;engine.spin(false);$('#rotate-view').setAttribute('aria-pressed','false');}});
 window.addEventListener('pagehide',()=>{interactionUi.dispose();engine?.dispose();},{once:true});
 try{const [response,genesResponse]=await Promise.all([fetch('data/catalogue.json?v=1'),fetch('data/genes/index.json?v=1')]);if(!response.ok||!genesResponse.ok)throw Error('The source catalogue did not load.');records=(await response.json()).records;geneIndex=new Map((await genesResponse.json()).genes.map(g=>[g.symbol,g]));updateLevel();engine=await MolecularEngine.create($('#viewport'),pick);window.MolecularAtlas={engine,interactionUi,get record(){return record;},get gene(){return gene;},get level(){return level;},select,setLevel,changeWindow,whenIdle:()=>queue};await select(new URLSearchParams(location.search).get('target')||'OXTR');}
 catch(error){fail(error);}
})();
