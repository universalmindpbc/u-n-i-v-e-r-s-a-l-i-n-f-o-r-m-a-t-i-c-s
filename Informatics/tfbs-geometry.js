import {withBindingEvidence} from './tfbs-evidence.js?v=2';
// A browser view of versioned Python/Biopython results, with independent Wolfram checks.
// Coordinate choices are fixed in experiment-v1/manifest.json and included in every export.
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = n => n.toLocaleString('en-AU');
const colors = {alpha:'#76baff', beta:'#e9d86b', gamma:'#e99ac6', epsilon:'#e3e9f0'};
const names = {studies:'Published studies',encode:'ENCODE',jaspar:'JASPAR'};
const densityThreshold = 2;
const views = {line:'Original DNA line',circle:'Full-position circle',m9:'Modulus 9 circle'};
const point = (turn,r=125,cx=210,cy=190) => [cx+r*Math.sin(turn*2*Math.PI),cy-r*Math.cos(turn*2*Math.PI)];
const xy = a => a.map(x=>x.toFixed(3)).join(',');
const text = (x,y,s,attrs='') => `<text x="${x}" y="${y}" ${attrs}>${esc(s)}</text>`;
const circle = (x,y,r,attrs='') => `<circle cx="${x}" cy="${y}" r="${r}" ${attrs}/>`;
const diamond = (x,y,r=5,attrs='') => `<path d="M${x},${y-r} L${x+r},${y} L${x},${y+r} L${x-r},${y} Z" ${attrs}/>`;
const arc = (a,b,r,cx=210,cy=190) => {
  const p=point(a,r,cx,cy),q=point(b,r,cx,cy);
  if(b-a>=.999999)return `M${xy(p)} A${r},${r} 0 1 1 ${xy(point(a+.5,r,cx,cy))} A${r},${r} 0 1 1 ${xy(p)}`;
  return `M${xy(p)} A${r},${r} 0 ${b-a>.5?1:0} 1 ${xy(q)}`;
};
const svg = (body,label) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 380" role="img" aria-label="${esc(label)}" class="geometry-svg"><style>text{font-family:Arial,sans-serif;font-size:var(--geometry-text,11px);fill:#a8bfcd;stroke:none;text-anchor:middle} .axis{stroke:#3c515f;fill:none} .observation{fill:#78cabb;stroke:#78cabb;cursor:pointer} .observation:hover{stroke:#fff;fill:#fff} .individual{fill:#ffb574;stroke:#0b141b;stroke-width:1} .selected{stroke:#fff;stroke-width:2} .count-grid{stroke:#344b5a;stroke-width:.6;fill:none} .callout{fill:#ffb574;font-size:var(--geometry-callout,10px);paint-order:stroke;stroke:#0b141b;stroke-width:3px} .ref{fill:none;stroke-width:1.7} .grid-label{fill:#e5edf3;font-size:15px} .small{font-size:var(--geometry-small,9px)} .count-label{paint-order:stroke;stroke:#0b141b;stroke-width:3px;stroke-linejoin:round} .large{font-size:23px;fill:#e5edf3} .faint{opacity:.3}</style>${body}</svg>`;

// Exported for the independent Node/Python parity test; display uses precomputed mappings.
export function projectInterval(start,end,lo=8740380,hi=8779628) {
  const a=Math.max(start,lo),b=Math.min(end,hi);
  if(!Number.isInteger(start)||!Number.isInteger(end)||start>=end||a>=b)throw Error('Invalid interval');
  const d0=hi-b,d1=hi-a,d=Math.floor((d0+d1-1)/2),r=d%9;
  return {offset_start:d0,offset_end_exclusive:d1,anchor_offset_bp:d,anchor_genomic_0based:hi-1-d,m9_residue:r,m9_display:r||9,full_circle_turns:d/(hi-lo)};
}

// The Atlas guide keeps its native grid; only its position in the shared view changes.
export function atlasGuide(atlas,pattern) {
  const grid=pattern==='beta'?9:24;
  const indices=pattern==='alpha'?Array.from({length:24},(_,i)=>i):pattern==='beta'?atlas.beta.interval_residues:atlas.gamma.alpha_indices;
  const paths=pattern==='gamma'?atlas.gamma.triangle_paths:pattern==='epsilon'?[atlas.epsilon.alpha_indices]:pattern==='beta'?[[...atlas.beta.interval_residues,atlas.beta.interval_residues[0]]]:[];
  return {grid,indices,turns:indices.map(i=>i/grid),values:indices.map(i=>pattern==='beta'?(i||9):atlas.alpha.display_values[i]),paths};
}
// Only observations with the identical plotted base share a count.
export function positionCounts(records,lo=0,hi=Infinity) {
  const counts=new Map();
  for(const r of records){const d=r.mapping.anchor_offset_bp;if(d>=lo&&d<hi)counts.set(d,(counts.get(d)||0)+1);}
  return [...counts].sort((a,b)=>a[0]-b[0]).map(([offset,count])=>({offset,count}));
}

export function countRadius(count,maxCount,maxRadius=19) {
  return count>0&&maxCount>0?maxRadius*Math.sqrt(count/maxCount):0;
}

export async function mountTfbsGeometry(host, catalogue, dataRoot) {
  const root=new URL('experiment-v1/',dataRoot),cache=new Map();
  let manifest,checks,overlay,rows=[],filtered=[],inspected=[],active=null,source='studies',view='line',pattern='beta',blend=false,group=null,page=0,request=0,ready=false,lineFocus=null;
  const pageSize=12;
  const get=s=>host.querySelector(s);
  host.innerHTML=`<p class="science-label">COMPARE POSITIONS · OXTR</p><h3>Binding-site geometry</h3><p class="geometry-intro">Explore whether the four M9 P24 geometries help predict OXTR binding-site locations. Compare DNA observations with the original Atlas geometry side by side. Blend adds the selected geometry to the DNA panel.</p><div class="geometry-controls"><label>Data source<select id="geometry-source"><option value="studies">Published studies · 128 mapped observations</option><option value="encode">ENCODE · 255 experimental regions</option><option value="jaspar">JASPAR · 9,164 predicted matches</option></select></label><label class="geometry-studies-kind">Study observations<select id="geometry-studies-kind"><option value="all">All 128 observations</option><option value="sequence">Sequence elements · 2</option><option value="occupancy">BrainTF occupancy regions · 126</option></select></label><label>Find a factor, cell type or observation<input id="geometry-search" type="search" placeholder="e.g. US-1, CTCF or an ID" autocomplete="off"></label><label class="geometry-score" hidden>JASPAR score<select id="geometry-score"><option value="400">400 and above</option><option value="500">500 and above</option><option value="600">600 and above</option><option value="800">800 and above</option></select></label></div><section class="geometry-source-context" aria-labelledby="geometry-evidence-title"><h4 id="geometry-evidence-title"></h4><div class="geometry-context-columns"><div><h5>Evidence selected</h5><p class="geometry-source-note"></p></div><div><h5>What the graph plots</h5><p class="geometry-mark-meaning"></p></div></div></section><div class="geometry-switches"><div><span class="geometry-control-label">View the DNA</span><div class="geometry-buttons" role="group" aria-label="DNA coordinate view">${Object.entries(views).map(([id,name],i)=>`<button type="button" data-geometry-view="${id}" aria-pressed="${id===view}"><span>${i+1}</span> ${name}</button>`).join('')}</div></div><div><span class="geometry-control-label">Compare with M9 P24</span><div class="geometry-buttons" role="group" aria-label="Atlas reference sequence">${Object.keys(colors).map(id=>`<button type="button" data-geometry-pattern="${id}" style="--pattern:${colors[id]}" aria-pressed="${id===pattern}">${id[0].toUpperCase()+id.slice(1)}</button>`).join('')}</div></div></div><p class="geometry-status" role="status" aria-live="polite">Loading the verified coordinate data…</p><div class="geometry-blend-row"><button type="button" data-geometry-blend aria-pressed="false">Blend · off</button><span>Overlay the selected Atlas geometry on the DNA panel</span></div><p class="geometry-window">Same 39,248-base window for every source · OXTR gene + 10,000 bases on each side</p><div class="geometry-outcome-summary" role="status"></div><div class="geometry-panels"><figure><figcaption class="geometry-data-title"></figcaption><div class="geometry-legend"><span><i class="geometry-data-key"></i><span class="geometry-data-legend"></span></span><span class="geometry-blend-key" hidden><i class="geometry-atlas-key"></i><span>Atlas reference positions</span></span></div><div class="geometry-data-plot"></div><div class="geometry-zoom"><div class="geometry-buttons"><button type="button" data-geometry-range="full">Full DNA window</button><button type="button" data-geometry-range="busy">Zoom to busiest region</button></div><label><input id="geometry-zoom" type="checkbox"> Zoom to selected centre · ±50 bases</label></div><p class="geometry-data-caption"></p></figure><figure><figcaption class="geometry-reference-title"></figcaption><div class="geometry-legend"><span><i class="geometry-atlas-key"></i><span>Original Atlas geometry · fixed alignment</span></span></div><div class="geometry-reference-plot"></div><p class="geometry-reference-caption"></p></figure></div><div class="geometry-node-counts" role="group" aria-label="Observations at each M9 node" hidden></div><p class="geometry-group"></p><section class="geometry-inspector" aria-label="Selected DNA observation"><div class="geometry-inspector-body"></div><div class="geometry-buttons"><button type="button" data-geometry-step="-1">← Previous observation</button><button type="button" data-geometry-step="1">Next observation →</button></div></section><details class="geometry-records"><summary>Inspect the observations</summary><div class="geometry-table-wrap" tabindex="0" role="region" aria-label="Mapped observation selection"><table><thead><tr><th>Observation</th><th>Original interval · GRCh38</th><th>Offset / M9</th></tr></thead><tbody></tbody></table></div><div class="geometry-pagination"><button type="button" data-geometry-page="-1">← Previous page</button><span></span><button type="button" data-geometry-page="1">Next page →</button></div></details><nav class="geometry-downloads" aria-label="Experiment downloads"><button type="button" data-geometry-export="json">Selection · JSON ↓</button><button type="button" data-geometry-export="csv">Selection · CSV ↓</button><button type="button" data-geometry-export="svg">Figure · SVG ↓</button><button type="button" data-geometry-export="png">Figure · PNG ↓</button><a href="${new URL('methods.md?v=position-counts-v1',root)}" download>Methods ↓</a></nav><details class="geometry-methods"><summary>Mapping &amp; verification</summary><p>Fixed human OXTR window: GRCh38 chr3:8,740,380–8,779,628 (0-based, end-exclusive). The direction follows OXTR’s minus strand, from high to low genomic coordinates. The origin is the window edge, not the transcription start site.</p><p>The line and full circle retain interval widths. For M9, one central base represents each interval within this window. An even-width interval uses the first central base in the OXTR direction. Regions extending beyond the window are clipped for display; their original coordinates remain in every export.</p><p><code>d = floor((offset_start + offset_end − 1) / 2)</code><br><code>Full circle = d / 39,248 turns · M9 = d mod 9</code><br>Zero is displayed as 9. All circles start at north and run clockwise. Compression merges positions; node counts expose those overlaps.</p><p>The Atlas reference keeps its original grid and phase: Alpha, Gamma and Epsilon use 24 period positions; Beta uses nine residue positions. The shared circle aligns both layers at north, clockwise, with no fitted rotation. On the full-position circle, one DNA window fills one turn. On the line, Atlas turn × window length gives the guide position; only node positions are shown, without connecting curves. In M9, the DNA has nine residue positions while each Atlas guide retains its native grid. A shared angle does not equate a period index with a residue. The Atlas paths show geometry, not measured movement or binding order.</p><div class="geometry-checks"></div><p>The browser filters and draws precomputed results. Python/Biopython and Wolfram are used in the reproducible preparation and verification, rather than called as live services for each selection.</p><nav class="geometry-downloads">${[['manifest.json','Experiment manifest'],['overlay-v1.json','Comparison & blend rules'],['../binding-outcomes.json','Binding outcomes & priorities'],['../mapped-binding-outcomes.csv','Mapped binding outcomes · CSV'],['display-audit.json','Source-density audit'],['display-audit-fields.csv','Source-density audit · CSV'],['python-validation.json','Python / Biopython checks'],['wolfram-validation.json','Wolfram checks'],['build_experiment.py','Python script'],['verify.wl','Wolfram code'],['requirements.txt','Python environment']].map(([f,l])=>`<a href="${new URL(f+'?v=position-counts-v1',root)}" download>${l} ↓</a>`).join('')}</nav></details><p class="geometry-boundary">Initial visual experiment · statistical association has not yet been tested. No negative binding controls have been curated for this map; an unrecorded position is not a failed binding test.</p>`;


  // Keep the comparison controls and both complete figures in one desktop viewport.
  const board=document.createElement('section');
  board.className='geometry-board';board.setAttribute('aria-label','DNA and Atlas comparison');board.tabIndex=-1;
  const switches=get('.geometry-switches'),blendButton=get('[data-geometry-blend]');
  switches.before(board);
  const blendControl=document.createElement('div');blendControl.className='geometry-blend-control';
  blendControl.innerHTML='<span class="geometry-control-label">Overlay on DNA</span>';
  blendControl.append(blendButton);switches.append(blendControl);
  get('.geometry-blend-row').remove();
  board.before(get('.geometry-status'),get('.geometry-window'));
  board.append(switches,get('.geometry-outcome-summary'),get('.geometry-panels'));
  const notes=document.createElement('div');notes.className='geometry-plot-notes';
  const dnaNotes=document.createElement('div'),atlasNotes=document.createElement('div');
  dnaNotes.append(get('.geometry-zoom'),get('.geometry-data-caption'));
  atlasNotes.append(get('.geometry-reference-caption'));
  notes.append(dnaNotes,atlasNotes);board.after(notes);
  get('.geometry-intro').insertAdjacentHTML('afterend','<button type="button" class="geometry-jump" data-geometry-jump>View comparison ↓</button>');
  const dialog=host.closest('dialog')||host.closest('#transcription-panel'),readerBar=host.closest('.science-tfbs')?.querySelector('.science-reader-bar');
  function sizeBoard(){
    const barHeight=readerBar?.getBoundingClientRect().height||64;
    host.style.setProperty('--reader-bar-height',barHeight+'px');
    const height=dialog?.clientHeight||window.innerHeight;
    if(height>0)board.style.setProperty('--comparison-height',Math.max(400,height-barHeight-24)+'px');
  }
  function scrollBoard(){
    if(!dialog)return;
    const top=dialog.scrollTop+board.getBoundingClientRect().top-dialog.getBoundingClientRect().top-(readerBar?.getBoundingClientRect().height||0)-12;
    dialog.scrollTo({top:Math.max(0,top),behavior:'instant'});
  }
  function alignBoard(){
    sizeBoard();
    if(dialog&&readerBar&&window.matchMedia('(min-width:960px) and (min-height:620px)').matches){
      const rect=board.getBoundingClientRect(),bounds=dialog.getBoundingClientRect(),bar=readerBar.getBoundingClientRect();
      if(rect.top<bar.bottom+8||rect.bottom>bounds.bottom-8)scrollBoard();
    }
  }
  get('[data-geometry-jump]').onclick=()=>{sizeBoard();scrollBoard();board.focus({preventScroll:true});};
  const boardObserver=new ResizeObserver(sizeBoard);if(dialog)boardObserver.observe(dialog);
  sizeBoard();
  host.style.scrollMarginTop='calc(var(--reader-bar-height,64px) + 12px)';
  // Keep small labels readable when the figure scales to a shorter screen.
  function sizePlotLabels(){
    host.querySelectorAll('.geometry-svg').forEach(plot=>{
      const matrix=plot.getScreenCTM(),scale=matrix?Math.hypot(matrix.a,matrix.b):0;
      if(scale<=0)return;
      plot.style.setProperty('--geometry-small',Math.max(9,11/scale)+'px');
      plot.style.setProperty('--geometry-text',Math.max(11,11/scale)+'px');
      plot.style.setProperty('--geometry-callout',Math.max(10,12/scale)+'px');
    });
  }
  const plotObserver=new ResizeObserver(sizePlotLabels);
  plotObserver.observe(get('.geometry-data-plot'));plotObserver.observe(get('.geometry-reference-plot'));

  async function read(file) {
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),30000);
    try {
      const response=await fetch(new URL(file+'?v=position-counts-v1',root),{signal:controller.signal});
      if(!response.ok)throw Error('Experiment data unavailable');
      let bytes=new Uint8Array(await response.arrayBuffer());
      if(bytes[0]===31&&bytes[1]===139)bytes=new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      return JSON.parse(new TextDecoder().decode(bytes));
    } finally {clearTimeout(timer);}
  }
  function controls() {
    get('.geometry-score').hidden=source!=='jaspar';
    get('.geometry-studies-kind').hidden=source!=='studies';
    get('#geometry-evidence-title').textContent={studies:'Published studies · DNA elements and brain occupancy',encode:'ENCODE · Laboratory measurements',jaspar:'JASPAR · Computer predictions'}[source];
    get('[data-geometry-blend]').setAttribute('aria-pressed',String(blend));
    get('[data-geometry-blend]').textContent='Blend · '+(blend?'on':'off');
    get('.geometry-blend-key').hidden=!blend;
    host.querySelectorAll('[data-geometry-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.geometryView===view)));
    host.querySelectorAll('[data-geometry-pattern]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.geometryPattern===pattern)));
    get('.geometry-zoom').hidden=view!=='line';
  }
  async function load(id) {
    const token=++request;source=id;ready=false;rows=[];filtered=[];inspected=[];active=null;group=null;page=0;lineFocus=null;
    controls();get('.geometry-status').textContent=`Loading ${names[id]}…`;
    get('.geometry-mark-meaning').textContent='The plotted observations will appear after this source has loaded.';
    for(const cls of ['geometry-data-plot','geometry-reference-plot','geometry-inspector-body','geometry-group','geometry-outcome-summary'])get('.'+cls).innerHTML='';
    get('tbody').innerHTML='';get('.geometry-node-counts').hidden=true;
    host.querySelectorAll('[data-geometry-export],[data-geometry-step],[data-geometry-page]').forEach(b=>b.disabled=true);
    get('.geometry-source-note').textContent={studies:'128 mapped human observations: two sequence elements and 126 screened BrainTF occupancy regions. Other literature findings remain below, including reports whose individual positions are unresolved.',encode:'Experimental regions of protein occupancy, with source experiments and cell types retained. A region’s central base is a plotting representative.',jaspar:'Predicted motif matches at UCSC score 400 and above. The full archive, including lower scores, remains in the source-data browser below.'}[id];
    try {
      if(!manifest){
        const [m,p,w,o]=await Promise.all([read('manifest.json'),read('python-validation.json'),read('wolfram-validation.json'),read('overlay-v1.json')]);
        if(m.version!==p.version||o.coordinate_experiment!==m.version||p.status!=='passed'||w.status!=='passed'||p.wolfram_coordinate_table_sha256!==w.coordinate_table_sha256)throw Error('Verification mismatch');
        manifest=m;checks={python:p,wolfram:w};overlay=o;
        get('.geometry-checks').innerHTML=`<p><strong>Calculations checked</strong> · Python ${esc(p.python)} · Biopython ${esc(p.biopython)} · Wolfram Language</p><p>${num(p.records_projected)} observations mapped. Python and Wolfram agree across all ${num(w.offsets_checked)} possible base offsets, including circle coordinates to nine decimal places. Biopython rechecked the two published sequence placements and ${num(p.sequence_checks.jaspar)} JASPAR sequences; one match extends beyond the reference window.</p>`;
      }
      if(!cache.has(id)){
        const data=await read(manifest.files[id].file);
        if(!Array.isArray(data)||data.length!==manifest.summaries[id].count||data.some(r=>!r.id||!Number.isFinite(r.mapping?.anchor_offset_bp)))throw Error('Coordinate format unavailable');
        cache.set(id,data);
      }
      if(token!==request)return;
      rows=cache.get(id).map(r=>withBindingEvidence(catalogue,id,r));ready=true;draw();
    }catch{
      if(token!==request)return;
      get('.geometry-status').innerHTML='The experiment data have not loaded. <button type="button" data-geometry-retry>Retry experiment</button>';
    }
  }

  function lineRange() {
    const sel=active?.mapping,n=manifest.reference.length_bp;
    if(get('#geometry-zoom').checked&&sel)return {lo:Math.max(0,sel.anchor_offset_bp-50),hi:Math.min(n,sel.anchor_offset_bp+51),zoom:true};
    return lineFocus?{...lineFocus,zoom:true}:{lo:0,hi:n,zoom:false};
  }

  function guideSVG(markers=false, mode=view, reference=false) {
    const a=atlasGuide(manifest.atlas,pattern),color=colors[pattern],n=manifest.reference.length_bp;
    let body='';
    if(mode==='line') {
      const {lo,hi}=lineRange(),x=i=>32+356*((i/a.grid*n)-lo)/(hi-lo);
      if(markers)for(let k=0;k<a.indices.length;k++){
        const i=a.indices[k],px=x(i);if(px<32||px>388)continue;
        body+=`<line x1="${px}" x2="${px}" y1="190" y2="240" stroke="${color}" stroke-dasharray="2 3" opacity=".65"/>`+circle(px,240,3,`fill="${color}"`);
        body+=text(px,258,a.values[k],`style="fill:${color}" class="small"`);
      }

    }else {
      if(markers)for(let k=0;k<a.indices.length;k++){
        const p=point(a.indices[k]/a.grid),q=point(a.indices[k]/a.grid,109);body+=circle(...p,3.4,`fill="${color}" stroke="#0b141b" stroke-width="1"`);
        if(mode!=='m9'||reference)body+=text(q[0],q[1]+3,a.values[k],`class="small" style="fill:${color}"`);
      }
      else for(const path of a.paths){
        body+=`<polyline points="${path.map(i=>xy(point(i/a.grid))).join(' ')}" class="ref" stroke="${color}"/>`;
        if(pattern==='beta'||pattern==='epsilon')for(let j=0;j<path.length-1;j++){
          const p=point(path[j]/a.grid),q=point(path[j+1]/a.grid),x=p[0]+.64*(q[0]-p[0]),y=p[1]+.64*(q[1]-p[1]),angle=Math.atan2(q[1]-p[1],q[0]-p[0])*180/Math.PI;
          body+=`<path d="M-5,-3 L0,0 L-5,3" fill="none" stroke="${color}" transform="translate(${x},${y}) rotate(${angle})"/>`;
        }
      }
    }
    return `<g class="atlas-overlay ${markers?'atlas-markers':'atlas-paths'}" data-atlas-pattern="${pattern}" data-atlas-grid="${a.grid}" data-atlas-turns="${a.turns.join(',')}" style="pointer-events:none" aria-label="${pattern} geometry, ${a.grid} native positions, fixed alignment">${body}</g>`;
  }
  function referenceSVG() {
    const a=atlasGuide(manifest.atlas,pattern);
    const sequence=pattern==='beta'?'4 → 8 → 7 → 5 → 1 → 2':pattern==='epsilon'?'9 → 3 → 6 → 9 → 3 → 6 → 9':pattern==='gamma'?'Two equilateral triangles':'24 Fibonacci residues, in sequence order';
    return svg(circle(210,190,125,'class="axis"')+guideSVG(false,'circle',true)+guideSVG(true,'circle',true)+text(210,28,'9 / 0 at north · clockwise','class="small"')+text(210,345,sequence,'class="small"')+text(210,366,`${a.grid} native Atlas positions`,'class="small"'),`${pattern} · original M9 P24 Atlas geometry`);
  }
  function binsFor(records,lo=0,hi=manifest.reference.length_bp) {
    const bins=Array(72).fill(0);
    records.forEach(r=>{const d=r.mapping.anchor_offset_bp;if(d>=lo&&d<hi)bins[Math.min(71,Math.floor((d-lo)/(hi-lo)*72))]++;});
    return bins;
  }
  function dataSVG() {
    const n=manifest.reference.length_bp,sel=active?.mapping,unit=manifest.summaries[source].unit;
    let s=blend?guideSVG():'';
    if(view==='m9'){
      const counts=Array.from({length:9},(_,i)=>filtered.filter(r=>r.mapping.m9_residue===i).length),max=Math.max(1,...counts);
      s=circle(210,190,125,'class="axis"')+s;
      for(let i=0;i<9;i++){
        const p=point(i/9),q=point(i/9,163),c=point(i/9,88),radius=countRadius(counts[i],max),percent=filtered.length?100*counts[i]/filtered.length:0;
        const disc=counts[i]?circle(...p,radius,`class="count-disc" data-count="${counts[i]}" fill="#78cabb" fill-opacity=".72" stroke="none"`):circle(...p,2.5,'fill="#344b5a" stroke="none"');
        s+=`<g data-geometry-residue="${i}" class="observation"><title>Node ${i||9}: ${counts[i]} ${unit} (${percent.toFixed(1)}%). Select to inspect.</title>${disc}${sel?.m9_residue===i?circle(...p,radius+3,'fill="none" stroke="#fff" stroke-width="1.5"'):''}${text(q[0],q[1]+4,i||9,'class="grid-label"')}${text(c[0],c[1]+2,num(counts[i]),'class="small count-label"')}${text(c[0],c[1]+18,percent.toFixed(1)+'%','class="small count-label"')}</g>`;
      }
      s+=text(210,10,`${num(filtered.length)} ${unit} · grouped into 9 nodes`,'class="small"')+text(210,366,'Circle area = count · percentages = share of this selection','class="small"');
    }else if(view==='circle'){
      s=circle(210,190,125,'class="axis"')+s;
      for(let i=0;i<4;i++){const p=point(i/4,174);s+=text(p[0],p[1]+4,num(i*n/4)+' bp','class="small"');}
      if(filtered.length>densityThreshold){
        const positions=positionCounts(filtered),max=Math.max(1,...positions.map(p=>p.count));
        s+=circle(210,190,143,'class="count-grid" stroke-dasharray="2 3"')+circle(210,190,161,'class="count-grid" stroke-dasharray="2 3"');
        for(const {offset,count} of positions){
          const a=point(offset/n),b=point(offset/n,125+36*count/max);
          s+=`<g class="observation position-count" data-geometry-position="${offset}" data-count="${count}"><title>${num(offset)} bp from origin: ${count} observations at this plotted centre. Select to inspect.</title><line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke-width="1"/>${circle(...b,1.3)}</g>`;
        }
        s+=blend?text(210,350,`Count scale: 0–${num(max)} observations at one centre`,'class="small"'):text(210,154,`${num(filtered.length)} observations`,'class="large count-label"')+text(210,180,`${num(positions.length)} distinct plotted centres`,'class="small count-label"')+text(210,207,`Longest mark: ${num(max)} observations at one centre`,'class="small count-label"')+text(210,234,'Each mark keeps its exact DNA position','class="small count-label"');
      }else{
        filtered.forEach((r,i)=>{const m=r.mapping,p=point(m.full_circle_turns);s+=`<g class="observation" data-geometry-record="${esc(r.id)}"><title>${esc(r.id+' · '+r.factor+' · '+num(m.anchor_offset_bp)+' bp from origin')}</title><path d="${arc(m.offset_start/n,m.offset_end_exclusive/n,119)}" fill="none" stroke="#ffb574" stroke-width="3"/>${diamond(...p,5,`class="individual ${active?.id===r.id?'selected':''}"`)}${source==='studies'&&filtered.length<=2?`<path d="M${xy(p)} L${378+i*12},${p[1]} L${378+i*12},${40+i*24} L305,${40+i*24}" stroke="#ffb574" stroke-width=".7" fill="none"/>`+text(300,43+i*24,r.label.split(' ')[0],'class="callout" style="text-anchor:end"'):''}</g>`;});
      }
      if(sel){const p=point(sel.full_circle_turns);s+=`<path class="selected-extent" d="${arc(sel.offset_start/n,sel.offset_end_exclusive/n,119)}" fill="none" stroke="#ffb574" stroke-width="4"/>`+diamond(...p,7,'class="individual selected"')+text(210,282,`Selected: ${active.id}`,'class="small count-label"');}
      // Window length and direction remain in the description outside the SVG.
    }else{
      const {lo,hi,zoom}=lineRange(),span=hi-lo,x=d=>32+356*(d-lo)/span;
      s=`<line x1="32" y1="190" x2="388" y2="190" class="axis" stroke-width="2"/>`+s;
      for(let i=0;i<=4;i++){const d=lo+span*i/4;s+=`<line x1="${x(d)}" y1="190" x2="${x(d)}" y2="200" class="axis"/>`+text(x(d),218,num(Math.round(d)),`class="small" style="text-anchor:${i===0?'start':i===4?'end':'middle'}"`);}
      const visible=filtered.filter(r=>r.mapping.offset_start<hi&&r.mapping.offset_end_exclusive>lo);
      if(filtered.length>densityThreshold){
        const positions=positionCounts(filtered,lo,hi),max=Math.max(1,...positions.map(p=>p.count));
        for(const count of [...new Set([0,Math.ceil(max/2),max])]){const y=185-110*count/max;s+=`<line x1="32" x2="388" y1="${y}" y2="${y}" class="count-grid"/>`+text(27,y+3,num(count),'class="small" style="text-anchor:end"');}
        s+=text(210,26,'Observations at each plotted centre','class="small"')+text(210,47,`${num(positions.reduce((sum,p)=>sum+p.count,0))} observations · ${num(positions.length)} distinct centres`,'class="small"');
        for(const {offset,count} of positions){const px=x(offset),y=185-110*count/max;
          s+=`<g class="observation position-count" data-geometry-position="${offset}" data-count="${count}"><title>${num(offset)} bp from origin: ${count} observations at this plotted centre. Select to inspect.</title><line x1="${px}" x2="${px}" y1="185" y2="${y}" stroke-width="1"/>${circle(px,y,1.7)}</g>`;
        }
        s+=text(210,245,zoom?`${num(lo)}–${num(hi-1)} bp from the fixed origin`:'Nearby marks overlap at this scale; zoom to separate them.','class="small"');
      }else{
        visible.forEach((r,i)=>{const m=r.mapping,a=x(Math.max(lo,m.offset_start)),b=x(Math.min(hi,m.offset_end_exclusive)),p=x(m.anchor_offset_bp);s+=`<g class="observation" data-geometry-record="${esc(r.id)}"><title>${esc(r.id+' · '+r.factor+' · strand '+(r.strand||'unassigned'))}</title><line x1="${a}" y1="183" x2="${b}" y2="183" stroke="#ffb574" stroke-width="3"/>${p>=32&&p<=388?diamond(p,190,5,`class="individual ${active?.id===r.id?'selected':''}"`):''}${source==='studies'&&filtered.length<=2?`<path d="M${p},183 L${p},${110+i*30} L230,${110+i*30}" stroke="#ffb574" stroke-width=".7" fill="none"/>`+text(240,103+i*30,`${r.label.split(' ')[0]} · ${num(m.anchor_offset_bp)} bp`,'class="callout" style="text-anchor:start"'):''}</g>`;});
      }
      if(sel&&sel.anchor_offset_bp>=lo&&sel.anchor_offset_bp<hi){const p=x(sel.anchor_offset_bp);s+=`<line class="selected-extent" x1="${x(Math.max(lo,sel.offset_start))}" x2="${x(Math.min(hi,sel.offset_end_exclusive))}" y1="183" y2="183" stroke="#ffb574" stroke-width="4"/><line class="selection-guide" x1="${p}" y1="70" x2="${p}" y2="198" stroke="#fff" stroke-dasharray="3 4"/>`+diamond(p,190,6,'class="individual selected"')+text(210,294,`Selected: ${active.id} · ${num(sel.anchor_offset_bp)} bp`,'class="small"');}
      if(blend)s+=text(210,280,'Atlas node positions','class="small"');
      if(!zoom){
        const a=x(10000),b=x(29248);
        s+=`<line x1="32" x2="388" y1="329" y2="329" class="axis" stroke-width="3"/><rect x="${a}" y="324" width="${b-a}" height="10" fill="#8094a2"/>`+text(210,318,'OXTR gene body →','class="small"')+text((32+a)/2,349,'10 kb upstream','class="small"')+text((b+388)/2,349,'10 kb downstream','class="small"');
      }
      s+=text(210,371,zoom?'Zoomed line · mapping origin remains fixed':'DNA bases from window edge · OXTR direction →','class="small"');
    }
    if(blend)s+=guideSVG(true);
    return svg(s,`${views[view]}: ${filtered.length} ${unit} from ${names[source]}${blend?', blended with '+pattern:''}`);
  }

  function draw() {
    if(!ready)return;
    controls();
    const terms=get('#geometry-search').value.toLowerCase().trim().split(/\s+/).filter(Boolean),score=Number(get('#geometry-score').value);
    const studyKind=get('#geometry-studies-kind').value;
    filtered=rows.filter(r=>(source!=='studies'||studyKind==='all'||(studyKind==='occupancy'?r.study_id==='braintf':r.study_id!=='braintf'))&&(source!=='jaspar'||r.score>=score)&&terms.every(t=>[r.id,r.factor,r.label,r.matrix_id,r.sequence_oriented,...(r.cell_types||[])].join(' ').toLowerCase().includes(t)));
    inspected=group?filtered.filter(r=>group.kind==='residue'?r.mapping.m9_residue===group.value:r.mapping.anchor_offset_bp>=group.lo&&r.mapping.anchor_offset_bp<group.hi):filtered;
    if(!inspected.some(r=>r.id===active?.id))active=null;
    if(!active)get('#geometry-zoom').checked=false;
    const unit=manifest.summaries[source].unit;
    get('.geometry-status').textContent=`${num(filtered.length)} ${unit} · ${names[source]} · fixed origin and original Atlas alignment`;
    get('.geometry-data-title').textContent=views[view];
    get('.geometry-reference-title').textContent=`${pattern[0].toUpperCase()+pattern.slice(1)} geometry · M9 P24`;
    const dense=view!=='m9'&&filtered.length>densityThreshold;
    get('.geometry-data-legend').textContent=view==='m9'?`${num(filtered.length)} ${unit} · counts per node`:dense?`${names[source]} · count at each plotted centre`:`${names[source]} · ${source==='studies'?'DNA elements and occupancy regions':source==='jaspar'?'predicted matches':'experimental regions'}`;
    get('.geometry-data-key').className='geometry-data-key '+(view==='m9'?'is-count':dense?'is-bar':'is-diamond');
    const outcomes=new Map();for(const r of filtered){const e=r.binding_evidence;outcomes.set(e.label,(outcomes.get(e.label)||0)+1);}
    get('.geometry-outcome-summary').innerHTML=!filtered.length?'<strong>No observations in this selection</strong>':[...outcomes].map(([label,count])=>`<strong>${esc(label)} · ${num(count)} ${unit}</strong>`).join('')+(source==='encode'?'<span>Positive regional signals · exact binding sites unresolved</span>':source==='jaspar'?'<span>No location-specific binding result established by these predictions</span>':'<span>Sequence binding and regional occupancy remain distinct · inspect each observation below</span>');
    const models=new Set(filtered.map(r=>r.matrix_id)).size, factors=new Set(filtered.map(r=>r.factor)).size, cells=new Set(filtered.flatMap(r=>r.cell_types||[])).size;
    get('.geometry-source-note').textContent=source==='studies'?`${num(filtered.filter(r=>r.study_id!=='braintf').length)} sequence elements and ${num(filtered.filter(r=>r.study_id==='braintf').length)} BrainTF occupancy regions in this selection. Select a position mark or table row to inspect the original position, assay and source. BrainTF contexts share four donors; repeated regions are separate observations, not distinct binding sites.`:source==='jaspar'?`${num(filtered.length)} predicted matches from ${num(models)} motif models in this selection. Several models can match the same DNA location. These are sequence predictions, not measured activity.`:`${num(filtered.length)} experimental regions spanning ${num(factors)} protein targets and ${num(cells)} cell types in this selection. The combined view brings together separate experiments; it is not one cell’s binding pattern.`;
    const anchorMeaning=source==='encode'?'The centre of a region is a plotting choice, not an experimentally identified binding site.':source==='jaspar'?'These locations are computer predictions, not measurements of binding in cells.':'US-1 and US-2 are sequence-mapped elements. BrainTF region centres are plotting choices, not identified contact bases.';
    get('.geometry-mark-meaning').textContent=(view==='m9'?'Each large circle counts observations whose centre maps to that M9 node. ':dense?'Each vertical or radial mark counts observations at one identical plotted centre. Nearby positions stay separate. ':source==='encode'?'An orange diamond marks a region’s centre; its extent is also shown. ':source==='jaspar'?'An orange diamond marks the centre of a predicted DNA match. ':'Orange diamonds mark observation centres; the interval extents are also shown. ')+anchorMeaning;
    get('#geometry-zoom').disabled=!active;
    get('.geometry-reference-plot').innerHTML=referenceSVG();
    host.style.setProperty('--atlas-color',colors[pattern]);
    host.querySelectorAll('.geometry-atlas-key').forEach(key=>key.classList.toggle('is-dot',pattern==='alpha'||(key.closest('.geometry-blend-key')&&view==='line')));
    get('.geometry-data-plot').innerHTML=dataSVG();
    get('.geometry-data-plot svg').setAttribute('role','group');
    get('.geometry-data-plot').querySelectorAll('.observation').forEach(mark=>{
      mark.setAttribute('tabindex','0');mark.setAttribute('role','button');
      mark.setAttribute('aria-label',mark.querySelector('title')?.textContent||'Inspect observation');
    });
    get('[data-geometry-range="busy"]').disabled=!filtered.length;
    get('[data-geometry-range="full"]').setAttribute('aria-pressed',String(!lineRange().zoom));
    get('[data-geometry-range="busy"]').setAttribute('aria-pressed',String(!!lineFocus&&!get('#geometry-zoom').checked));
    get('.geometry-data-plot').querySelectorAll('[data-geometry-record] title').forEach(t=>{const r=filtered.find(r=>r.id===t.parentElement.dataset.geometryRecord);if(r)t.textContent+=' · '+r.binding_evidence.label;});
    get('.geometry-data-caption').textContent=(view==='circle'?'Full 39,248-base window · clockwise in OXTR direction. ':'')+(view==='m9'?'Each node groups positions with the same remainder after division by nine. It combines different DNA locations. Counts describe this arithmetic grouping; no predictive value for binding has been established.':dense?'Mark height counts observations at exactly the same plotted centre. Close marks can overlap on screen; use the DNA line zoom to separate them. Repeated observations are not a count of distinct binding sites.':'Orange diamonds mark individual DNA observations; short orange marks show their interval widths. Select a diamond or a table row for details.');
    get('.geometry-reference-caption').textContent=(pattern==='beta'?'Beta retains its nine residue positions.':'Alpha, Gamma and Epsilon retain their 24-position period grid.')+(blend?' Blend is on. The left panel uses the same fixed alignment; the DNA line shows node positions only.':' Blend is off. The DNA data and Atlas reference are displayed independently.');
    const nodeBox=get('.geometry-node-counts');nodeBox.hidden=view!=='m9';
    if(view==='m9')nodeBox.innerHTML=Array.from({length:9},(_,i)=>{const count=filtered.filter(r=>r.mapping.m9_residue===i).length;return `<button type="button" data-geometry-residue="${i}" aria-pressed="${group?.kind==='residue'&&group.value===i}"><span>Node ${i||9}</span><strong>${num(count)}</strong><small>${(filtered.length?100*count/filtered.length:0).toFixed(1)}%</small></button>`;}).join('');
    get('.geometry-group').innerHTML=group?`${num(inspected.length)} ${unit} in ${group.kind==='residue'?'node '+(group.value||9):num(group.lo)+' bp from origin'} · <button type="button" data-geometry-clear-group>Show all observations</button>`:'';
    const body=get('.geometry-inspector-body');
    if(active){
      const r=active,m=r.mapping,index=inspected.indexOf(r),sourceLink=`<a href="${esc(r.source_url)}" target="_blank" rel="noopener noreferrer">Original source ↗</a>`;
      body.innerHTML=`<p class="science-label">SELECTED OBSERVATION · ${num(index+1)} OF ${num(inspected.length)}</p><h4>${esc(r.study_id==='braintf'?r.factor+' · '+r.label:r.label||r.factor)} <span>${esc(r.id)}</span></h4><section class="geometry-binding-result"><h5>${esc(r.binding_evidence.label)}</h5><p>${esc(r.binding_evidence.result)}</p><dl><dt>How binding was assessed</dt><dd>${esc(r.binding_evidence.assay)}</dd><dt>Biological context</dt><dd>${esc(r.binding_evidence.context)}</dd><dt>Effect on gene activity</dt><dd>${esc(r.binding_evidence.expression_effect)}</dd><dt>Source review</dt><dd>${esc(r.binding_evidence.review_depth)}</dd></dl><a href="${esc(r.binding_evidence.source_url)}" target="_blank" rel="noopener noreferrer">Evidence for this interpretation ↗</a></section><p>chr3:${num(r.start_0based)}–${num(r.end_exclusive)} · ${num(r.end_exclusive-r.start_0based)} bp · strand ${esc(r.strand||'not assigned')}</p><p><strong>${num(m.anchor_offset_bp)} bp</strong> from origin → <strong>${m.full_circle_degrees.toFixed(3)}°</strong> on the full circle → <strong>M9 node ${m.m9_display}</strong></p>${m.clipped_to_window?'<p class="geometry-note">This region extends beyond the saved window. The plotted centre uses its visible intersection.</p>':''}${r.sequence_oriented?`<p class="geometry-sequence"><code>${esc(r.sequence_oriented)}</code></p><p class="geometry-note">${esc(r.sequence_validation)}</p>`:''}${r.cell_types?.length?`<p class="geometry-note">${esc(r.cell_types.join(' · '))}</p>`:''}${r.study_id==='braintf'?`<p class="geometry-note">Replication QC: ${esc(r.replication_qc)} · minimum FRiP ${(100*r.minimum_FRiP).toFixed(2)}% · author HOT status: ${esc(r.author_HOT_status)}<br>Source row ${num(r.source_row_1based)} · ${esc(r.source_file)}</p>`:''}<p>${sourceLink}${r.experiments?.length?` · ${num(r.experiments.length)} source experiments in the download`:''}</p>`;
    }else body.innerHTML=inspected.length?'<p>Select a DNA position mark or table row to inspect its original position and source. Records are listed in DNA order.</p>':'<p>No observations match this selection. Clear the search or choose another node or data source.</p>';
    host.querySelectorAll('[data-geometry-step]').forEach(b=>b.disabled=!inspected.length||(!!active&&inspected.length<2));
    page=Math.min(page,Math.max(0,Math.ceil(inspected.length/pageSize)-1));
    get('tbody').innerHTML=inspected.slice(page*pageSize,(page+1)*pageSize).map(r=>`<tr><td><button type="button" data-geometry-record="${esc(r.id)}" aria-pressed="${r.id===active?.id}">${esc(r.study_id==='braintf'?r.factor+' · '+r.label:r.label||r.factor)} · ${esc(r.id)}</button><span class="geometry-row-outcome">${esc(r.binding_evidence.label)}</span></td><td>${num(r.start_0based)}–${num(r.end_exclusive)}<br>strand ${esc(r.strand||'unassigned')}</td><td>${num(r.mapping.anchor_offset_bp)} bp / ${r.mapping.m9_display}</td></tr>`).join('')||'<tr><td colspan="3">No matching observations.</td></tr>';
    get('.geometry-pagination span').textContent=inspected.length?`Page ${page+1} of ${Math.ceil(inspected.length/pageSize)}`:'No results';
    get('[data-geometry-page="-1"]').disabled=page===0;
    get('[data-geometry-page="1"]').disabled=(page+1)*pageSize>=inspected.length;
    host.querySelectorAll('[data-geometry-export]').forEach(b=>b.disabled=!inspected.length);
    sizePlotLabels();
  }

  function saveBlob(body,type,name) {
    const url=URL.createObjectURL(new Blob([body],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
  }
  function combinedFigure() {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="740" viewBox="0 0 1200 740"><rect width="1200" height="740" fill="#0b141b"/><style>text{font-family:Arial,sans-serif;fill:#d9e7ef;text-anchor:middle}</style>${text(600,35,'OXTR binding-site geometry · '+names[source],'style="font-size:23px;fill:#d9e7ef"')}${text(600,63,`${num(filtered.length)} observations · ${views[view]} · ${pattern} · Blend ${blend?'on':'off'}`,'style="font-size:16px;fill:#d9e7ef"')}<svg x="20" y="85" width="560" height="507" viewBox="0 0 420 380">${get('.geometry-data-plot svg').innerHTML}</svg><svg x="620" y="85" width="560" height="507" viewBox="0 0 420 380">${get('.geometry-reference-plot svg').innerHTML}</svg>${text(600,618,view==='m9'?'DNA observations: counts per M9 node · Atlas: coloured circles and paths':'DNA observations: orange diamonds / counts at exact centres · Atlas: coloured circles and paths','style="font-size:14px;fill:#d9e7ef"')}${text(600,645,'GRCh38 chr3:8740380–8779628 · fixed origin · OXTR minus strand','style="font-size:14px;fill:#d9e7ef"')}${text(600,672,source==='studies'?'Sequence elements and regional occupancy · geometric association untested':source==='encode'?'Regional association detected · exact binding sites unresolved':'Prediction only · binding at these locations not established','style="font-size:14px;fill:#d9e7ef"')}<metadata>${esc(JSON.stringify({version:manifest.version,overlay,source,view,pattern,blend,search:get('#geometry-search').value,study_observations:source==='studies'?get('#geometry-studies-kind').value:null,minimum_score:source==='jaspar'?Number(get('#geometry-score').value):null,group,selected_observation:active?.id,line_zoom:lineRange().zoom,line_range:lineRange(),position_count_rule:'Identical anchor_offset_bp only; no neighbouring-base aggregation',mapping:manifest.mapping,reference:manifest.reference}))}</metadata></svg>`;
  }

  async function save(format) {
    if(!ready)return;draw();if(!inspected.length)return;
    const parameters={source,view,blend,overlay_version:overlay.version,atlas_sequence:pattern,search:get('#geometry-search').value,study_observations:source==='studies'?get('#geometry-studies-kind').value:null,minimum_score:source==='jaspar'?Number(get('#geometry-score').value):null,inspection_group:group,selected_observation:active?.id,line_zoom:lineRange().zoom,line_range:lineRange(),position_count_rule:'Identical anchor_offset_bp only; no neighbouring-base aggregation'};
    const name=`oxtr-${source}-${view}-${pattern}`;
    if(format==='json')saveBlob(JSON.stringify({version:manifest.version,count:inspected.length,parameters,overlay,reference:manifest.reference,mapping:manifest.mapping,atlas:manifest.atlas,verification:checks,sources:manifest.source_sha256,records:inspected},null,2),'application/json',name+'.json');
    if(format==='csv'){
      const flat=inspected.map(r=>({...r,...r.mapping,experiment_version:manifest.version,view,atlas_sequence:pattern,window_start_0based:manifest.reference.start_0based,window_end_exclusive:manifest.reference.end_exclusive,origin_genomic_base_0based:manifest.reference.origin_genomic_base_0based,coordinate_convention:manifest.reference.coordinate_convention,mapping_rules:manifest.mapping,overlay_rules:overlay,selection_parameters:parameters,atlas_reference:manifest.atlas,source_sha256:manifest.source_sha256}));
      const fields=[...new Set(flat.flatMap(Object.keys))],cell=v=>{const s=v==null?'':typeof v==='object'?JSON.stringify(v):String(v);return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
      saveBlob('\uFEFF'+[fields,...flat.map(r=>fields.map(k=>r[k]))].map(r=>r.map(cell).join(',')).join('\r\n')+'\r\n','text/csv;charset=utf-8',name+'.csv');
    }
    if(format==='svg')saveBlob(combinedFigure(),'image/svg+xml',name+'.svg');
    if(format==='png'){
      const image=new Image(),url=URL.createObjectURL(new Blob([combinedFigure()],{type:'image/svg+xml'}));
      try{image.src=url;await image.decode();const canvas=document.createElement('canvas');canvas.width=2400;canvas.height=1480;canvas.getContext('2d').drawImage(image,0,0,2400,1480);const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));if(!blob)throw Error('PNG unavailable');saveBlob(blob,'image/png',name+'.png');}
      catch{get('.geometry-status').textContent='The PNG export did not complete. The SVG export retains the full vector figure.';}
      finally{URL.revokeObjectURL(url);}
    }
  }
  host.addEventListener('click',event=>{
    const target=event.target.closest('[data-geometry-blend],[data-geometry-view],[data-geometry-pattern],[data-geometry-record],[data-geometry-residue],[data-geometry-position],[data-geometry-range],button[data-geometry-step],button[data-geometry-page],button[data-geometry-export],button[data-geometry-clear-group],button[data-geometry-retry]');
    if(!target||!host.contains(target))return;
    if(target.hasAttribute('data-geometry-retry')){load(source);return;}if(!ready)return;
    const d=target.dataset;
    if(target.hasAttribute('data-geometry-blend'))blend=!blend;
    if(d.geometryView){view=d.geometryView;group=null;page=0;}
    if(d.geometryPattern)pattern=d.geometryPattern;
    if(d.geometryRecord){active=filtered.find(r=>r.id===d.geometryRecord);if(!inspected.includes(active))group=null;}
    if(d.geometryResidue!==undefined){get('.geometry-records').open=true;group={kind:'residue',value:Number(d.geometryResidue)};page=0;}
    if(d.geometryPosition!==undefined){
      const offset=Number(d.geometryPosition);group={kind:'position',lo:offset,hi:offset+1};page=0;
      active=filtered.find(r=>r.mapping.anchor_offset_bp===offset);get('.geometry-records').open=true;
    }
    if(d.geometryRange){
      get('#geometry-zoom').checked=false;
      if(d.geometryRange==='full')lineFocus=null;
      else {const bins=binsFor(filtered),i=bins.indexOf(Math.max(...bins)),n=manifest.reference.length_bp;lineFocus={lo:Math.floor(i*n/72),hi:Math.ceil((i+1)*n/72)};}
    }
    if(target.hasAttribute('data-geometry-clear-group')){group=null;page=0;}
    if(d.geometryStep){const i=inspected.indexOf(active);active=inspected[i<0?0:(i+Number(d.geometryStep)+inspected.length)%inspected.length];}
    if(d.geometryPage)page+=Number(d.geometryPage);
    if(d.geometryExport){save(d.geometryExport);return;}
    const restore=target===document.activeElement;
    draw();
    if(d.geometryView||d.geometryPattern||d.geometryRange||target.hasAttribute('data-geometry-blend'))alignBoard();
    if(restore&&!target.isConnected){
      const key=Object.keys(d).find(k=>k.startsWith('geometry'));
      const attr=key?.replace(/[A-Z]/g,c=>'-'+c.toLowerCase());
      if(attr)host.querySelector(`[data-${attr}="${CSS.escape(d[key])}"]`)?.focus({preventScroll:true});
    }
  });
  host.addEventListener('keydown',event=>{if((event.key==='Enter'||event.key===' ')&&event.target.matches('svg [role="button"]')){event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
  get('#geometry-source').addEventListener('change',()=>{get('#geometry-search').value='';get('#geometry-studies-kind').value='all';get('#geometry-zoom').checked=false;load(get('#geometry-source').value);});
  let timer;get('#geometry-search').addEventListener('input',()=>{group=null;lineFocus=null;page=0;clearTimeout(timer);timer=setTimeout(draw,120);});
  get('#geometry-studies-kind').addEventListener('change',()=>{group=null;lineFocus=null;page=0;draw();});
  host.addEventListener('tfbs-select-source',async event=>{
    get('#geometry-source').value=event.detail.source;
    get('#geometry-studies-kind').value=event.detail.studyKind||'all';
    get('#geometry-search').value=event.detail.search||'';
    get('#geometry-zoom').checked=false;
    await load(event.detail.source);
    host.scrollIntoView({block:'start',behavior:'instant'});get('#geometry-source').focus({preventScroll:true});
  });
  get('#geometry-score').addEventListener('change',()=>{group=null;lineFocus=null;page=0;draw();});
  get('#geometry-zoom').addEventListener('change',draw);
  await load(source);
}
