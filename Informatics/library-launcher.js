'use strict';
(()=>{
const root=new URL('.',document.currentScript.src),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// The two sixfold groups preserve the Phase Zulu research hierarchy.
// Display labels and library filters differ for NMDAR and telomeres.
const outerTargets = [
 {id:'DRD2',aim:'dopamine',type:'Gene',description:'Dopamine D2 receptor',gene:'1813',x:903.1,y:235},
 {id:'SLC6A4',aim:'serotonin',type:'Gene',description:'Serotonin transporter',gene:'6532',x:903.1,y:585},
 {id:'NMDAR',library:'NMDA',aim:'learning + memory',type:'Receptor family',description:'NMDA receptors; GRIN-family subunits',x:600,y:760},
 {id:'IL6',aim:'immunity + inflammation',type:'Gene',description:'Interleukin-6',gene:'3569',x:296.9,y:585},
 {id:'DOCK2',aim:'PTSD',type:'Gene',description:'Dedicator of cytokinesis 2',gene:'1794',x:296.9,y:235},
 {id:'NR3C1',aim:'cortisol',type:'Gene',description:'Glucocorticoid receptor',gene:'2908',x:600,y:60}
];
const innerTargets = [
 {id:'NOS3',aim:'gasotransmitter',type:'Gene',description:'Endothelial nitric oxide synthase',gene:'4846',x:500,y:236.8},
 {id:'NAD+',aim:'electron transfer',type:'Metabolite',description:'Redox cofactor',x:700,y:236.8},
 {id:'Telomeres',library:'Telomerase',aim:'biological clock',type:'Chromosome structures',description:'Telomere maintenance; the library includes telomerase, TERT and TERC',x:800,y:410},
 {id:'CD38',aim:'oxytocin release',type:'Gene',description:'NAD metabolism; oxytocin release. Additional research interest: breast cancer outlier',gene:'952',x:700,y:583.2},
 {id:'COL1A1',aim:'collagen',type:'Gene',description:'Type I collagen alpha 1 chain',gene:'1277',x:500,y:583.2},
 {id:'TERRA',aim:'chromosome stability',type:'RNA',description:'Telomeric repeat-containing RNA',x:400,y:410}
];
const oxtr={id:'OXTR',aim:'oxytocin',type:'Gene',description:'Oxytocin receptor; initial investigation',gene:'5021'};
const nos1={id:'NOS1',aim:'neurotransmitter',type:'Gene',description:'Neuronal nitric oxide synthase; nested minor target',gene:'4842',x:600,y:492};
const libraryURL=t=>{const url=new URL('library.html',root);url.searchParams.set('target',t.library||t.id);return url.href;};
const directory=(title,targets)=>`<section><h4>${title}</h4>${targets.map(t=>`<article><a data-library="all" data-library-gene="${esc(t.library||t.id)}" href="${esc(libraryURL(t))}"><strong>${esc(t.id)}</strong><span>${esc(t.aim)}</span></a><p>${esc(t.type)} · ${esc(t.description)}${t.gene?` · <a href="https://www.ncbi.nlm.nih.gov/gene/${t.gene}" target="_blank" rel="noopener noreferrer">Gene reference ↗</a>`:''}</p></article>`).join('')}</section>`;
let libraryOpener,networkOpener;
const library=document.createElement('dialog');
library.className='library-overlay';library.setAttribute('aria-labelledby','library-overlay-title');
library.innerHTML='<div class="library-overlay-bar"><h2 id="library-overlay-title">UNIVERSAL RESEARCH</h2><button aria-label="Close Universal Research">×</button></div><iframe title="Universal Research"></iframe>';
document.body.append(library);
const network=document.createElement('dialog');
network.className='network-overlay';network.setAttribute('aria-labelledby','network-title');
network.innerHTML=`
 <div class="library-overlay-bar"><span>UNIVERSAL INFORMATICS / RESEARCH ROADMAP</span><div class="network-media-controls"><button data-network-play aria-label="Pause background video" title="Pause background video">Ⅱ</button><button data-network-close aria-label="Close candidate network">×</button></div></div>
 <div class="network-body">
  <div class="network-heading"><div><h2 id="network-title">Phase Zulu</h2><p>Gene Network Expansion</p></div><div class="network-system-heading"><h3 id="network-sequence-title">Transcript Element Listening System</h3><p>Pathway of transcription of neurotransmitter-based genes</p></div></div>
  <figure class="network-figure">
   <div class="network-map-scroll" tabindex="0" aria-label="Phase Zulu map; scroll to explore when enlarged"><div id="network-diagram"></div></div>
   <figcaption><span>Select a target to explore its research library.</span><button class="network-zoom" aria-pressed="false" aria-controls="network-diagram">Enlarge map +</button></figcaption>
  </figure>
  <section class="network-sequence" aria-labelledby="network-sequence-title">
   <div class="network-sequence-key"><h3>Proposed research sequence</h3><p>6 outer · 6 inner · NOS1 + OXTR</p></div>
   <ol aria-label="Proposed research sequence">${[oxtr,...outerTargets].map(t=>`<li><a data-library="all" data-library-gene="${esc(t.library||t.id)}" href="${esc(libraryURL(t))}">${esc(t.id)}</a></li>`).join('')}</ol>
  </section>
  <p class="network-lead">We begin with OXTR. If independent research verifies a biologically meaningful relationship between Fibonacci-derived features and OXTR transcription, we plan to expand across this candidate network.</p>
  <p class="network-status">Research priorities and proposed relationships · validation pending</p>
  <p class="network-legend">The nested geometry preserves two groups of six around OXTR, with NOS1 as a separate minor target. Labels identify research aims; the sequence is a proposed investigation, with molecular relationships still to be tested.</p>
  <details class="network-directory"><summary>Research targets &amp; biological identities <span>14 targets</span></summary><div class="network-directory-groups">${directory('Outer sequence',outerTargets)}${directory('Inner biological targets',innerTargets)}${directory('Central focus &amp; nested target',[oxtr,nos1])}</div><p class="network-legend">OXTR encodes the oxytocin receptor. The programme spans genes, receptors, RNA, metabolites and telomere biology. Telomeres link to the telomerase research collection. CD38 retains the additional breast cancer outlier research interest from Phase Zulu.</p></details>
  <section class="network-references"><h3>Pathway context</h3><div>
   <article><a href="https://www.nature.com/articles/s41467-019-08503-8" target="_blank" rel="noopener noreferrer">Nature · OXTR parent pathways ↗</a><p>Oxytocin pathway gene networks in the human brain · 2019. Brain expression maps and co-expression analyses provide pathway context. Co-expression does not establish a causal pathway.</p></article>
   <article><a href="https://www.frontiersin.org/journals/molecular-neuroscience/articles/10.3389/fnmol.2022.1055344/full" target="_blank" rel="noopener noreferrer">Frontiers · OXTR–DRD2 receptor interactions ↗</a><p>The oxytocin receptor as a hub in the GPCR heteroreceptor network · 2022. This review discusses receptor complexes, including OXTR–D2 findings largely from cellular and animal research. It does not demonstrate human DRD2 transcriptional regulation.</p></article>
  </div><p>Neither paper tests the OXTR–Fibonacci hypothesis or verifies the complete candidate network.</p></section>
 </div>`;
document.body.append(network);
const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
svg.setAttribute('viewBox','0 0 1200 820');svg.setAttribute('aria-labelledby','network-svg-title network-svg-desc');
const petals=(radius,rotation,cls)=>Array.from({length:6},(_,i)=>{const angle=(rotation+i*60)*Math.PI/180;return `<circle class="${cls}" cx="${600+radius*Math.cos(angle)}" cy="${410+radius*Math.sin(angle)}" r="${radius}"/>`;}).join('');
const node=(t,group,index)=>`<a class="network-node network-node--${group}" data-network-target="${esc(t.id)}" data-ring="${group}" data-library="all" data-library-gene="${esc(t.library||t.id)}" href="${esc(libraryURL(t))}" aria-label="${esc(t.id)} — ${esc(t.aim)}; open research library" transform="translate(${t.x} ${t.y})"><title>${esc(t.id)} · ${esc(t.aim)}. ${esc(t.type)}: ${esc(t.description)}.</title><ellipse class="network-node-hit" cy="9" rx="${group==='outer'?130:group==='inner'?94:68}" ry="${group==='outer'?61:group==='inner'?42:30}"/>${group==='outer'?`<text class="network-node-order" y="-37">${String(index+1).padStart(2,'0')}</text>`:''}<text class="network-node-symbol" y="0">${esc(t.id)}</text><text class="network-node-aim" y="${group==='outer'?35:group==='inner'?27:23}">${esc(t.aim)}</text>${t.id==='CD38'?'<text class="network-node-note" y="46">breast cancer outlier*</text>':''}</a>`;
svg.innerHTML=`
 <title id="network-svg-title">Phase Zulu: two sixfold groups around OXTR</title>
 <desc id="network-svg-desc">OXTR is the central initial target. The outer research sequence runs clockwise: DRD2, dopamine; SLC6A4, serotonin; NMDAR, learning and memory; IL6, immunity and inflammation; DOCK2, PTSD; NR3C1, cortisol. Six inner biological targets: NOS3, gasotransmitter; NAD+, electron transfer; telomeres, biological clock; CD38, oxytocin release; COL1A1, collagen; TERRA, chromosome stability. NOS1, neurotransmitter, is a smaller separate target nested beneath OXTR. Fourteen targets in total. The geometry represents research priorities, with biological relationships pending validation.</desc>
 <g class="network-geometry" aria-hidden="true">
  <circle class="network-orbit network-orbit--outer" cx="600" cy="410" r="350"/>
  ${petals(175,-90,'network-petal network-petal--outer')}
  <circle class="network-orbit network-orbit--inner" cx="600" cy="410" r="200"/>
  ${petals(100,-60,'network-petal network-petal--inner')}
  <circle class="network-core" cx="600" cy="410" r="132"/>
  <circle class="network-core-track" cx="600" cy="410" r="140"/>
  <path class="network-dna" d="M593 307C619 319 581 332 607 344M607 307C581 319 619 332 593 344M595 312H605M595 320H605M595 330H605M595 339H605"/>
  <circle class="network-nos1-casing" cx="600" cy="492" r="41"/>
 </g>
 <g class="network-outer-targets">${outerTargets.map((t,i)=>node(t,'outer',i)).join('')}</g>
 <g class="network-inner-targets">${innerTargets.map(t=>node(t,'inner')).join('')}</g>
 <a class="network-node network-node--core" data-network-target="OXTR" data-ring="core" data-library="all" data-library-gene="OXTR" href="${esc(libraryURL(oxtr))}" aria-label="OXTR — oxytocin; initial research target; open research library" transform="translate(600 386)"><title>OXTR · oxytocin. Initial research target.</title><ellipse class="network-node-hit" rx="120" ry="58"/><text class="network-node-symbol"><tspan fill="#5ca8ff">O</tspan><tspan fill="#ffda60">X</tspan><tspan fill="#ff84c5">T</tspan><tspan fill="#62d998">R</tspan></text><text class="network-node-aim" y="43">oxytocin</text></a>
 ${node(nos1,'nested')}`;
network.querySelector('#network-diagram').append(svg);
network.querySelector('.network-zoom').onclick=e=>{const expanded=e.currentTarget.getAttribute('aria-pressed')!=='true';e.currentTarget.setAttribute('aria-pressed',String(expanded));e.currentTarget.textContent=expanded?'Fit map −':'Enlarge map +';network.querySelector('figcaption span').textContent=expanded?'Scroll the map to explore each target.':'Select a target to explore its research library.';const scroll=network.querySelector('.network-map-scroll');scroll.classList.toggle('is-enlarged',expanded);scroll.scrollLeft=expanded?(scroll.scrollWidth-scroll.clientWidth)/2:0;scroll.scrollTop=expanded?(scroll.scrollHeight-scroll.clientHeight)/2:0;};
// Reuse the active experience's film. A muted fallback serves the landing-page roadmap.
let networkFilm,networkFallback,networkResume=false;
const networkPlay=network.querySelector('[data-network-play]');
function syncNetworkFilm(){const paused=!networkFilm||networkFilm.paused;networkPlay.textContent=paused?'▷':'Ⅱ';networkPlay.setAttribute('aria-label',paused?'Play background video':'Pause background video');networkPlay.title=networkPlay.getAttribute('aria-label');}
function startNetworkFilm(){
 const film=document.getElementById('film'),experience=document.getElementById('experience');
 if(!experience.hidden&&film.currentSrc){networkFilm=film;network.dataset.video='shared';}
 else{
  if(!networkFallback){networkFallback=document.createElement('video');networkFallback.className='network-film';networkFallback.muted=true;networkFallback.defaultMuted=true;networkFallback.loop=true;networkFallback.playsInline=true;networkFallback.setAttribute('aria-hidden','true');network.prepend(networkFallback);}
  networkFilm=networkFallback;networkFallback.src=film.currentSrc||film.dataset.src;network.dataset.video='fallback';
  if(!matchMedia('(prefers-reduced-motion: reduce)').matches)networkFallback.play().catch(syncNetworkFilm);
 }
 networkFilm.addEventListener('play',syncNetworkFilm);networkFilm.addEventListener('pause',syncNetworkFilm);syncNetworkFilm();
}
function stopNetworkFilm(){
 if(networkFilm){networkFilm.removeEventListener('play',syncNetworkFilm);networkFilm.removeEventListener('pause',syncNetworkFilm);}
 if(networkFallback){networkFallback.pause();networkFallback.removeAttribute('src');networkFallback.load();}
 networkFilm=null;networkResume=false;
}
networkPlay.onclick=()=>{if(!networkFilm)return;if(networkFilm.paused)networkFilm.play().catch(syncNetworkFilm);else networkFilm.pause();};
document.addEventListener('visibilitychange',()=>{if(networkFilm!==networkFallback||!networkFilm)return;if(document.hidden){networkResume=!networkFilm.paused;networkFilm.pause();}else if(networkResume&&network.open)networkFilm.play().catch(syncNetworkFilm);});
window.addEventListener('pagehide',()=>networkFallback?.pause());
function container(dialog){const parent=document.fullscreenElement||document.body;if(dialog.parentElement!==parent)parent.append(dialog);}
function closeLibrary(){library.querySelector('iframe').removeAttribute('src');library.close();}
function openLibrary(collection='all',gene='',opener,page='library.html'){libraryOpener=opener?.closest('[role=menu]')?document.querySelector('#atlas-toggle'):opener||document.activeElement;container(library);const url=new URL(page,root);url.searchParams.set('embedded','1');if(collection!=='all')url.searchParams.set('collection',collection);if(gene)url.searchParams.set('target',gene);library.querySelector('iframe').src=url.href;if(!library.open)library.showModal();}
document.addEventListener('click',e=>{const connect=e.target.closest('[data-connect]');if(connect){e.preventDefault();openLibrary('all','',connect,'connect.html');}const open=e.target.closest('[data-library]');if(open){e.preventDefault();openLibrary(open.dataset.library,open.dataset.libraryGene,open);}const n=e.target.closest('[data-network]');if(n){e.preventDefault();networkOpener=n;container(network);network.showModal();startNetworkFilm();}});
library.querySelector('button').onclick=closeLibrary;network.querySelector('[data-network-close]').onclick=()=>network.close();function restoreFocus(dialog,opener){const active=document.activeElement;if(opener&&(active===document.body||active===opener||active===dialog||dialog.contains(active)))opener.focus({preventScroll:true});}library.addEventListener('close',()=>{if(library.open)return;library.querySelector('iframe').removeAttribute('src');restoreFocus(library,libraryOpener);});network.addEventListener('close',()=>{if(!network.open){stopNetworkFilm();restoreFocus(network,networkOpener);}});window.addEventListener('message',e=>{if(e.origin!==location.origin||e.source!==library.querySelector('iframe').contentWindow)return;if(e.data?.type==='informatics:close-library')closeLibrary();if(e.data?.type==='informatics:connect')openLibrary('all','',libraryOpener,'connect.html');if(e.data?.type==='informatics:library-navigate'&&['geometry','transcription','molecular','about','sound','vision','language','partners'].includes(e.data.area)){closeLibrary();if(network.open)network.close();document.dispatchEvent(new CustomEvent('atlas:navigate',{detail:e.data.area}));}if(e.data?.type==='informatics:library-background'&&['video','black','white'].includes(e.data.mode))library.dataset.readingBackground=e.data.mode;});
library.querySelector('iframe').addEventListener('load',()=>{const f=library.querySelector('iframe');const title=f.contentWindow.location.pathname.endsWith('/connect.html')?'CONNECT YOUR AI':'UNIVERSAL RESEARCH';library.querySelector('h2').textContent=title;library.querySelector('button').setAttribute('aria-label',title==='CONNECT YOUR AI'?'Close Connect your AI':'Close Universal Research');f.title=title;});
library.addEventListener('cancel',e=>{e.preventDefault();closeLibrary();});
for(const d of [library,network])d.addEventListener('click',e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom){if(d===library)closeLibrary();else d.close();}}});
})();
