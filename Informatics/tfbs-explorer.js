import {withBindingEvidence} from './tfbs-evidence.js?v=2';
// Browse the archived OXTR observations without merging different evidence units.
const number = n => n.toLocaleString('en-AU');
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const link = (url, label) => `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(label)} ↗</a>`;
const csvCell = value => {
  const text = value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"','""')}"` : text;
};

export function mountTfbsExplorer(host, catalogue, root) {
  const summary = catalogue.summary;
  const mapped = catalogue.evidence.filter(r => r.species === 'Homo sapiens' && r.reported_sequence && r.normalized_location?.start_0based != null);
  const collections = {
    published: {label:'Studies', unit:'observations', count:mapped.length+summary.braintf_screened_observations, note:'Two published DNA elements and 126 screened BrainTF occupancy records. The brain records span 43 protein targets and 13 contexts; overlapping peaks and shared donors are retained. A regional peak does not identify an exact binding site. The enhancer paper appears in the literature evidence below.'},
    encode: {label:'ENCODE', unit:'regions', count:summary.encode_clusters, note:'Laboratory measurements. ENCODE brings together experiments in cells. Each record is a stretch of DNA with evidence of protein association, with the protein, cell types and original experiments attached. A region can contain several binding sites.'},
    jaspar: {label:'JASPAR', unit:'matches', count:summary.jaspar_default_threshold_predictions, note:'Computer predictions. JASPAR provides DNA-binding patterns learned from experiments; a computer searches OXTR DNA for matching sequences. Several models can match the same location. A prediction does not establish binding in a cell.'},

  };
  const cache = new Map();
  let selected = 'encode', rows = [], filtered = [], page = 0, request = 0, ready = false;
  const pageSize = 50;
  host.innerHTML = `<h3 class="tfbs-explorer-heading" id="tfbs-explorer-heading">Human OXTR data to explore</h3><p class="science-small">Choose a collection to browse its positions and download the matching data. Counts describe the units named below; the collections can overlap.</p><div class="tfbs-datasets" role="group" aria-label="Choose binding-site data">${Object.entries(collections).map(([id,c]) => `<button type="button" data-tfbs-dataset="${id}" aria-pressed="${id===selected}"><strong>${number(c.count)}</strong><span>${c.label}</span><small>${id==='published'?'Reported findings':id==='jaspar'?'Computer predictions · score ≥400':id==='encode'?'Laboratory measurements':'Curated annotations · ORegAnno'}</small></button>`).join('')}</div><h4 class="tfbs-data-heading"></h4><p class="tfbs-data-description"></p><div class="tfbs-data-filters"><label>Find a factor, cell type, sequence or ID<input type="search" id="tfbs-data-search" placeholder="e.g. CTCF, AP1 or an accession" autocomplete="off"></label><label>DNA position · GRCh38<input type="text" id="tfbs-data-position" placeholder="e.g. chr3:8770000-8771000" aria-describedby="tfbs-coordinate-help" autocomplete="off" spellcheck="false"></label><label class="tfbs-score-filter" hidden>Prediction score<select id="tfbs-data-score"><option value="400">400 and above · ${number(summary.jaspar_default_threshold_predictions)} matches</option><option value="0">All archived scores · ${number(summary.jaspar_all_predictions)} matches</option></select></label></div><p id="tfbs-coordinate-help" class="science-small">Human OXTR and 10,000 DNA bases on each side. Positions use GRCh38, 0-based starts and end-exclusive ranges. A position search finds overlapping intervals.</p><p class="tfbs-data-status" role="status" aria-live="polite"></p><div class="tfbs-data-actions"><nav class="tfbs-downloads" aria-label="Selected binding-site data downloads"><button type="button" class="science-register-link" data-tfbs-data-export="json" disabled>Matching data · JSON ↓</button><button type="button" class="science-register-link" data-tfbs-data-export="csv" disabled>Matching data · CSV ↓</button></nav><button type="button" class="science-register-link" data-tfbs-data-clear>Clear search</button></div><p class="science-table-hint">Scroll the table horizontally to read positions and sources.</p><div class="science-table-scroll tfbs-data-table" tabindex="0" role="region" aria-label="Human OXTR binding-site data"><table><caption class="science-sr-only">Selected OXTR observations with genomic positions and original sources.</caption><thead><tr><th scope="col">Factor / element</th><th scope="col">Position &amp; sequence</th><th scope="col">Evidence &amp; source</th></tr></thead><tbody></tbody></table></div><nav class="tfbs-data-pages" aria-label="Binding-site results pages"><button type="button" class="science-register-link" data-tfbs-data-page="previous" disabled>← Previous</button><span></span><button type="button" class="science-register-link" data-tfbs-data-page="next" disabled>Next →</button></nav><p class="science-small tfbs-unresolved">2006 human study: <strong>15 reported sites</strong> — individual sequences and positions await recovery. These remain in the literature collection and are not yet plotted as 15 coordinates.</p>`;
  const get = selector => host.querySelector(selector);
  const status = get('.tfbs-data-status'), tbody = get('tbody');

  async function read(file, signal) {
    const response = await fetch(new URL(file + '?v=brain-studies-v1', root), {signal});
    if (!response.ok) throw Error('Data unavailable');
    if (!file.endsWith('.gz')) return response.json();
    const bytes = new Uint8Array(await response.arrayBuffer());
    // Some servers decode gzip in transit; handle either response explicitly.
    if (bytes[0] !== 31 || bytes[1] !== 139) return JSON.parse(new TextDecoder().decode(bytes));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).json();
  }

  async function recordsFor(id) {
    if (cache.has(id)) return cache.get(id);
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 45000);
    try {
      let records;
      if (id === 'published') records = [...mapped.map(r => ({...r,...r.normalized_location,sequence:r.reported_sequence})), ...await read('study-additions-v1/braintf-peaks.json',controller.signal)];
      if (id === 'encode') {
        const [clusters, provenance] = await Promise.all([read('encode-clusters.json',controller.signal),read('encode-provenance.json',controller.signal)]);
        const byId = new Map();
        for (const entry of provenance) { if (!byId.has(entry.cluster_id)) byId.set(entry.cluster_id,[]); byId.get(entry.cluster_id).push(entry); }
        records = clusters.map(r => ({...r,experiments:byId.get(r.id)||[],cell_types:[...new Set((byId.get(r.id)||[]).map(p=>p.cell_type))]}));
      }
      if (id === 'jaspar') records = await read('jaspar-predictions.json.gz',controller.signal);
      if (id === 'oreganno') records = await read('oreganno-regions.json',controller.signal);
      if (!Array.isArray(records) || records.some(r => !r.id || !Number.isFinite(r.start_0based) || !Number.isFinite(r.end_exclusive))) throw Error('Unexpected coordinate data');
      records=records.map(r=>withBindingEvidence(catalogue,id,r));
      records.sort((a,b) => a.start_0based-b.start_0based || a.end_exclusive-b.end_exclusive || a.id.localeCompare(b.id));
      const indexed = records.map(record => ({record,search:[record.id,record.factor,record.target,record.label,record.matrix_id,record.sequence,record.sequence_oriented,...(record.cell_types||[]),...(record.experiments||[]).map(p=>p.experiment)].filter(Boolean).join(' ').toLowerCase()}));
      cache.set(id,indexed);
      return indexed;
    } finally { clearTimeout(timeout); }
  }

  function intervalQuery() {
    const value = get('#tfbs-data-position').value.trim().replaceAll(',','').replace(/[–—]/g,'-');
    if (!value) return null;
    const match = /^(?:(chr\d+):)?(\d+)(?:\s*-\s*(\d+))?$/i.exec(value);
    if (!match) return false;
    const start = Number(match[2]), end = match[3] ? Number(match[3]) : start+1;
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || end<=start) return false;
    return {chromosome:(match[1]||'chr3').toLowerCase(),start,end};
  }

  function rowMarkup(r) {
    const factor = r.factor || r.target || r.label || 'Factor not recorded';
    const sequence = r.sequence || r.sequence_oriented;
    const strand = r.strand ?? r.annotation_strand;
    const position = `${r.chromosome}:${r.start_0based}–${r.end_exclusive}`;
    const browser = `https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position=${encodeURIComponent(`${r.chromosome}:${r.start_0based+1}-${r.end_exclusive}`)}`;
    let evidence;
    if (selected === 'published') {
      const paper = catalogue.publications.find(p=>p.pmid===r.pmid);
      evidence = `${link(paper.doi_url,paper.title)}<p>${escape(r.finding)}</p><p><strong>Effect on gene activity</strong><br>${escape(r.binding_evidence.expression_effect)}</p><details><summary>Methods &amp; mapping</summary><p>${escape(r.methods)}</p>${r.study_id==='braintf'?`<p>Brain context: ${escape(r.context)} · replication: ${escape(r.replication_qc)} · minimum FRiP: ${(100*r.minimum_FRiP).toFixed(2)}% · author HOT class: ${escape(r.author_HOT_status)}</p><p>${link(r.source_url,'Original peak file')} · source row ${r.source_row_1based}</p>`:''}<p>${escape(r.normalized_location.mapping_status)}</p><p><strong>Interpretation:</strong> ${escape(r.limitation)}</p></details>`;
    } else if (selected === 'encode') {
      const sources = r.experiments;
      evidence = `<p>Experimental occupancy · ${number(r.source_count)} source observations</p><p>${escape(r.cell_types.join(' · ')||'Cell type unavailable')}</p><details><summary>Source experiments</summary>${sources.map(p=>`<p>${link(p.experiment_url,p.experiment||p.source_name)} · ${escape(p.cell_type)}<br>${escape(p.lab)} · ${escape(p.file_accession)}</p>`).join('')}</details>`;
    } else if (selected === 'jaspar') {
      evidence = `${link(`https://jaspar.elixir.no/matrix/${encodeURIComponent(r.matrix_id)}/`,r.matrix_id)}<p>Predicted motif match · UCSC score ${r.score}</p>`;
    } else evidence = `${link(`https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position=${encodeURIComponent(`${r.chromosome}:${r.start_0based+1}-${r.end_exclusive}`)}&oreganno=pack`,'ORegAnno track')}<p>Regional annotation · factor identity ${r.factor?'recorded':'unresolved'}</p>`;
    return `<tr data-observation-id="${escape(r.id)}"><td><span class="science-study-id">${escape(r.id)}</span><strong>${escape(factor)}</strong>${r.label && r.factor ? `<p>${escape(r.label)}</p>`:''}</td><td><p class="tfbs-position">${escape(position)}</p><p>${number(r.end_exclusive-r.start_0based)} bp · ${strand?`strand ${escape(strand)}`:'strand not assigned'}</p>${sequence?`<p><code>${escape(sequence)}</code></p>`:''}${link(browser,'Genome position')}</td><td><p class="tfbs-binding-outcome">${escape(r.binding_evidence.label)}</p>${evidence}</td></tr>`;
  }

  function draw() {
    if (!ready) return;
    const interval = intervalQuery(), terms = get('#tfbs-data-search').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const minimum = Number(get('#tfbs-data-score').value);
    filtered = interval === false ? [] : rows.filter(({record:r,search}) => (selected!=='jaspar'||r.score>=minimum) && terms.every(term=>search.includes(term)) && (!interval || (r.chromosome.toLowerCase()===interval.chromosome && r.start_0based<interval.end && r.end_exclusive>interval.start))).map(x=>x.record);
    get('#tfbs-data-position').setAttribute('aria-invalid',String(interval===false));
    page = Math.min(page,Math.max(0,Math.ceil(filtered.length/pageSize)-1));
    const start = page*pageSize, shown = filtered.slice(start,start+pageSize);
    status.textContent = interval===false ? 'Enter one position or a range, for example chr3:8770000-8771000. The end must be after the start.' : `${number(filtered.length)} ${collections[selected].unit} match your selection${filtered.length?` · showing ${number(start+1)}–${number(start+shown.length)}`:''}. Downloads include all matching rows.`;
    tbody.innerHTML = shown.length ? shown.map(rowMarkup).join('') : `<tr><td colspan="3">${interval===false?'Check the position format above.':'No data match this search. Clear the search or choose another collection.'}</td></tr>`;
    host.querySelectorAll('[data-tfbs-data-export]').forEach(b=>b.disabled=!filtered.length);
    get('[data-tfbs-data-page="previous"]').disabled=page===0;
    get('[data-tfbs-data-page="next"]').disabled=start+pageSize>=filtered.length;
    get('.tfbs-data-pages span').textContent=filtered.length?`Page ${page+1} of ${Math.ceil(filtered.length/pageSize)}`:'No results';
  }

  async function choose(id) {
    const version = ++request;
    selected=id;page=0;ready=false;rows=[];filtered=[];
    host.querySelectorAll('[data-tfbs-dataset]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tfbsDataset===id)));
    get('.tfbs-data-heading').textContent=collections[id].label;
    get('.tfbs-data-description').textContent=collections[id].note;
    get('.tfbs-score-filter').hidden=id!=='jaspar';
    status.textContent=`Loading ${collections[id].label.toLowerCase()}…`;
    tbody.innerHTML='';get('.tfbs-data-pages span').textContent='';
    host.querySelectorAll('[data-tfbs-data-export],[data-tfbs-data-page]').forEach(b=>b.disabled=true);
    try {
      const loaded = await recordsFor(id);
      if (version!==request) return;
      rows=loaded;ready=true;draw();
    } catch {
      if (version!==request) return;
      status.innerHTML='This collection has not loaded. <button type="button" class="science-register-link" data-tfbs-data-retry>Retry collection</button>. The complete archive downloads remain available below.';
    }
  }

  function save(format) {
    if (!ready) return;
    draw(); // Export the current inputs even if their search debounce is pending.
    if (!filtered.length) return;
    const reference = {species:'Homo sapiens',gene:'OXTR',assembly:'GRCh38',chromosome:'chr3',window_start_0based:8740380,window_end_exclusive:8779628,coordinate_convention:'0-based, end-exclusive',selection:'OXTR gene plus 10,000 bases on each side; overlapping records may extend beyond the window'};
    const filters = {search:get('#tfbs-data-search').value.trim(),position:get('#tfbs-data-position').value.trim(),minimum_score:selected==='jaspar'?Number(get('#tfbs-data-score').value):null};
    const sourceFiles = {published:['catalogue.json','sequence-mapping.json','study-additions-v1/braintf-peaks.json','study-additions-v1/review.json'],encode:['encode-clusters.json','encode-provenance.json'],jaspar:['jaspar-predictions.json.gz'],oreganno:['oreganno-regions.json']};
    let body;
    if (format==='json') body=JSON.stringify({version:catalogue.harmonisation.version,collection:selected,count:filtered.length,unit:collections[selected].unit,scope:collections[selected].note,reference,filters,sources:sourceFiles[selected].map(file=>new URL(file,root).href),records:filtered,...(selected==='published'?{publications:catalogue.publications.filter(p=>filtered.some(r=>r.pmid===p.pmid))}:{})},null,2);
    else {
      const flat = filtered.map(r=>({...r,species:'Homo sapiens',gene:'OXTR',collection:selected,count_unit:collections[selected].unit,coordinate_convention:reference.coordinate_convention,collection_note:collections[selected].note,source_files:sourceFiles[selected].map(file=>new URL(file,root).href)}));
      const fields=[...new Set(flat.flatMap(r=>Object.keys(r)))];
      body='\uFEFF'+[fields,...flat.map(r=>fields.map(field=>r[field]))].map(row=>row.map(csvCell).join(',')).join('\r\n')+'\r\n';
    }
    const url=URL.createObjectURL(new Blob([body],{type:format==='json'?'application/json':'text/csv;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download=`oxtr-${selected}-selection.${format}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  host.addEventListener('click',e=>{
    const button=e.target.closest('button');if(!button||!host.contains(button))return;
    if(button.dataset.tfbsDataset)choose(button.dataset.tfbsDataset);
    if(button.hasAttribute('data-tfbs-data-retry'))choose(selected);
    if(button.hasAttribute('data-tfbs-data-clear')){get('#tfbs-data-search').value='';get('#tfbs-data-position').value='';page=0;draw();}
    if(button.dataset.tfbsDataPage){page+=button.dataset.tfbsDataPage==='next'?1:-1;draw();get('.tfbs-data-table').scrollIntoView({block:'nearest'});}
    if(button.dataset.tfbsDataExport)save(button.dataset.tfbsDataExport);
  });
  let debounce;
  host.addEventListener('input',()=>{clearTimeout(debounce);debounce=setTimeout(()=>{page=0;draw();},120);});
  get('#tfbs-data-score').addEventListener('change',()=>{page=0;draw();});
  choose(selected);
}
