'use strict';
(() => {
  const $ = selector => document.querySelector(selector);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const safeURL = value => { try { const url = new URL(value, location.href); return ['https:', 'http:'].includes(url.protocol) ? url.href : '#'; } catch { return '#'; } };
  const normalize = value => String(value ?? '').normalize('NFKC').toLowerCase().replace(/interleukin[\s-]*6|il[\s-]+6/g, 'il6').replace(/oxytocin receptor/g, 'oxtr').replace(/serotonin transporter/g, 'slc6a4').replace(/glucocorticoid receptor/g, 'nr3c1').replace(/[^\p{L}\p{N}+]+/gu, ' ').trim();
  const model = globalThis.UniversalResearchModel;
  let publicationAccess = {}, clinicalConditions = {}, fullTextAccess = {};
  const multi = globalThis.ResearchMultiSelect;
  const params = new URLSearchParams(location.search);
  const collections = ['sound', 'vision', 'language', 'general', 'all'];
  const fields = ['query', 'topic', 'journal', 'category', 'physiology', 'target', 'condition', 'kind', 'period', 'trust', 'jadad', 'sort'];
  let collection = collections.includes(params.get('collection')) ? params.get('collection') : ['target','gene','record','query','condition'].some(key => params.has(key)) ? 'all' : 'sound';
  let view = params.get('view') === 'search' || !['sound','vision','language','general'].includes(collection) || ['target','gene','query','condition'].some(key => params.has(key)) ? 'search' : 'chapters';
  let chapterNumber = Number(params.get('chapter')) || 0;
  let edition, editions = {}, records = [], filtered = [], papers = [], pageIndex = 0, articleOpener, loading = true;
  const PAGE_SIZE = 15;
  if (params.get('embedded') === '1') document.body.classList.add('embedded');
  const targets = ['OXTR','OXT','DRD2','SLC6A4','NR3C1','DOCK2','IL6','NMDA','TERRA','NOS3','NAD+','Telomerase','CD38','COL1A1','NOS1','FKBP5','BDNF','CRHR2','TNF','SNCA','FOXP2'];
  $('#target').innerHTML += targets.map(t => `<option>${esc(t)}</option>`).join('');
  const multiFields = fields.filter(key=>!['query','sort'].includes(key));
  multi.init(multiFields);
  function restoreFilters(p) { for (const key of fields) { if (multiFields.includes(key)) multi.set(key,p.getAll(key)); else $('#'+key).value=p.get(key) || (key==='sort' ? 'genomics' : ''); } if(p.has('gene'))multi.set('target',[...multi.values('target'),...p.getAll('gene')]); }
  restoreFilters(params);

  function crosses(score, labelled = false, scope = 'Topic') {
    if (!score) return '';
    const icon = '<svg viewBox="0 0 18 18" aria-hidden="true"><circle cx="9" cy="9" r="8" fill="#1468a8" stroke="currentColor" stroke-width=".5"/><path d="M7 3.5h4V7h3.5v4H11v3.5H7V11H3.5V7H7z" fill="#fff"/></svg>';
    return `<span class="trust-score" role="img" aria-label="${scope} trust score: ${score} of 3" title="${scope} trust score: ${score} of 3">${labelled ? '<span class="trust-label">Trust score</span>' : ''}${icon.repeat(score)}</span>`;
  }
  const topicScoreStatus = topic => topic.trustAssessment?.status === 'clinical-guidance' ? '<span class="topic-score-status">Clinical guidance</span>' : topic.trustAssessment?.status === 'below-minimum' ? '<span class="topic-score-status">Not yet graded · minimum 3 papers</span>' : '';
  function topicCounts(topic) {
    return `<span class="contents-topic-meta"><span class="paper-count">${paperCount(topic.papers)}</span>${crosses(topic.trustScore)}${topicScoreStatus(topic)}</span>`;
  }
  $('#trust-key').innerHTML = [[3,'At least 3 double-blind randomized controlled trials'],[2,'At least 3 single-blind randomized controlled trials'],[1,'At least 3 peer-reviewed journal articles']].map(([score,label]) => `<div>${crosses(score)}<span>${label}</span></div>`).join('');
  function route(number = 0, topic = '', destination = collection) {
    const url = new URL(location.href);
    for (const key of [...fields, 'record', 'paper', 'gene', 'view', 'chapter']) url.searchParams.delete(key);
    url.searchParams.set('collection', destination);
    if (number) url.searchParams.set('chapter', number);
    url.hash = topic ? 'topic-' + topic : '';
    return url.href;
  }
  function syncURL(push = false) {
    const url = new URL(location.href);
    url.searchParams.set('collection', collection);
    url.searchParams.delete('gene');
    if (view === 'chapters') {
      url.searchParams.delete('view');
      if (chapterNumber) url.searchParams.set('chapter', chapterNumber); else url.searchParams.delete('chapter');
      for (const key of fields) url.searchParams.delete(key);
    } else {
      url.searchParams.set('view', 'search'); url.searchParams.delete('chapter'); url.hash = '';
      for (const key of fields) {
        url.searchParams.delete(key);
        const selected = multiFields.includes(key) ? multi.values(key) : [$('#'+key).value];
        for (const value of selected) if(value && !(key==='sort' && value==='genomics'))url.searchParams.append(key,value);
      }
    }
    history[push ? 'pushState' : 'replaceState'](null, '', url);
  }
  const collectionTitle = value => value.charAt(0).toUpperCase()+value.slice(1);
  const quoteText = quote => quote.map(span => span.text).join('');
  const quoteHTML = quote => quote.map(span => span.emphasis ? `<strong>${esc(span.text)}</strong>` : esc(span.text)).join('');
  function evidenceMarkup(evidence) {
    if (!evidence) return '';
    const values = [['Evidence',evidence.evidenceType],['Intervention',evidence.intervention],['Comparison',evidence.comparator],['Participants',evidence.sample],['Measurement',evidence.measurement],['Timing',evidence.timing],['Genes measured',evidence.measuredGenes?.join(', ')],['Pathways / gene sets',evidence.pathways?.join(', ')],['Trial',evidence.trialId]];
    return `<details class="study-methods"><summary>Study methods and measurements</summary><dl>${values.filter(([,value])=>value).map(([label,value])=>`<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>${evidence.relationshipNote ? `<p>${esc(evidence.relationshipNote)}</p>` : ''}</details>`;
  }
  function bibliographyMarkup(paper) {
    if (!paper.bibliography) return '';
    return `<details class="research-bibliography"><summary>Research bibliography</summary><p class="publication">${esc(paper.bibliographyAttribution)}</p>${paper.bibliographyNote ? `<p>${esc(paper.bibliographyNote)}</p>` : ''}${paper.bibliography.map(block => block.kind==='heading' ? `<h5>${esc(block.text)}</h5>` : `<p class="bibliography-reference">${esc(block.text)}</p>`).join('')}</details>`;
  }
  function journalDisplay(publication) {
    const journal = publication.journalLabel || publication.journal || '';
    const id = publication.recordId || publication.id;
    const family = publication.publisherFamily || publicationAccess[id]?.publisherFamily || records.find(record => record.id === id || record.aliases.includes(id))?.publisherFamily;
    return journal && family === 'nature' && !/\bnature\b/i.test(journal) ? `${journal} (Nature Portfolio)` : journal;
  }
  function paperMarkup(paper, heading = 'h4') {
    const quotes = paper.quotes || [];
    const review = paper.quotationReview;
    const publication = [journalDisplay(paper), paper.year].filter(Boolean).join(' · ');
    const details = [paper.design, paper.citationCount !== null && paper.citationCount !== undefined ? `${paper.citationCount.toLocaleString()} ${paper.citationCount===1 ? 'citation' : 'citations'}` : ''].filter(Boolean).join(' · ');
    const url = safeURL(paper.urls?.[0] || '#');
    return `<article class="paper" id="paper-${esc(paper.id)}" data-paper="${esc(paper.recordId)}">
      <${heading}><a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(paper.title)}</a></${heading}>
      ${publication ? `<p class="publication">${esc(publication)}</p>` : ''}
      ${details ? `<p class="study-details">${esc(details)}</p>` : ''}
      ${quotes.filter((q,index) => index !== review?.index).length ? `<blockquote cite="${esc(url)}">${quotes.map((q,index) => index === review?.index ? '' : `<p>${quoteHTML(q)}</p>`).join('')}</blockquote>` : ''}
      ${paper.summary ? `<div class="study-summary"><p class="eyebrow">Study summary</p><p>${esc(paper.summary)}</p>${paper.limitation ? `<p class="summary-limitation">${esc(paper.limitation)}</p>` : ''}</div>` : ''}

      ${!paper.summary && paper.limitation ? `<p class="summary-limitation">${esc(paper.limitation)}</p>` : ''}${(paper.figures || []).map(figure => `<figure class="research-figure"><img src="${esc(safeURL(figure.src))}" alt="${esc(figure.caption)}" loading="lazy"><figcaption>${esc(figure.caption)}</figcaption></figure>`).join('')}${evidenceMarkup(paper.evidence)}${paper.studyDesignAssessment ? `<details class="study-methods design-assessment"><summary>Study design and blinding</summary><p>${esc(paper.studyDesignAssessment.basis)}</p><p>${paper.studyDesignAssessment.sources.map((source,index)=>`<a href="${esc(safeURL(source))}" target="_blank" rel="noopener noreferrer">${source.includes('clinicaltrials.gov') || source.includes('isrctn.com') ? 'Trial register' : `Study source ${index+1}`} ↗</a>`).join(' · ')}</p></details>` : ''}${bibliographyMarkup(paper)}
      <div class="paper-links"><a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${url.includes('web.archive.org/web/') ? 'Read archived publication' : 'Read publication'} ↗</a><button class="text-button" data-paper-detail="${esc(paper.id)}">Citation details</button></div>
    </article>`;
  }
  const paperCount = items => { const count = model.uniqueCount(items); return `${count.toLocaleString()} ${count === 1 ? 'paper' : 'papers'}`; };
  function renderStatistics() {
    const stats = model.statistics(records,collection);
    $('#collection-statistics').hidden = false;
    $('#collection-statistics').innerHTML = `<dl><div><dt>Papers &amp; references</dt><dd data-stat="papers">${stats.papers.toLocaleString()}</dd></div><div><dt>Journals</dt><dd data-stat="journals">${stats.journals.toLocaleString()}</dd></div><div><dt>Recorded citations</dt><dd data-stat="citations">${stats.citations.toLocaleString()}</dd></div></dl><p>Each publication and identified journal is counted once. Citation counts are available for ${stats.citationCoverage.toLocaleString()} of ${stats.papers.toLocaleString()} publications; these are historical counts, not live totals.</p>`;
  }
  function populateFilters() {
    const relevant = records.filter(r => collection === 'all' || r.collections.includes(collection));
    for (const [id,label,values] of [
      ['journal','All journals',[...relevant.filter(r=>r.publisherFamily).map(r=>['family:'+r.publisherFamily,r.publisherFamilyLabel+' — all journals']),...relevant.filter(r=>r.journalKey).map(r=>[r.journalKey,journalDisplay(r)])]],
      ['topic','All topics',relevant.flatMap(r=>r.topics.filter(t=>collection==='all'||t.collection===collection).map(t=>[t.key,t.title]))]
    ]) {
      const selected = loading ? params.getAll(id) : multi.values(id);
      const options = [...new Map(values).entries()].sort((a,b)=>a[1].localeCompare(b[1]));
      $('#'+id).innerHTML = `<option value="">${label}</option>`+options.map(([value,title])=>`<option value="${esc(value)}">${esc(title)}</option>`).join('');
      multi.set(id,selected);
    }
  }
  function renderContents() {
    return `${edition.introduction ? `<section class="collection-introduction">${edition.introduction.title ? `<h2>${esc(edition.introduction.title)}</h2>` : ''}${edition.introduction.paragraphs.map(p=>`<p>${esc(p)}</p>`).join('')}</section>` : ''}<div class="edition-intro"><p class="eyebrow">${esc(collectionTitle(collection))} RESEARCH</p><h2>Contents</h2></div><div class="contents-grid">${edition.chapters.map(chapter => `<section class="contents-chapter"><h3><span class="chapter-number">${String(chapter.number).padStart(2,'0')}</span><a data-chapter="${chapter.number}" href="${esc(route(chapter.number))}">${esc(chapter.title)}</a></h3><p class="paper-count">${paperCount(chapter.topics.flatMap(t=>t.papers))}</p><ol>${chapter.topics.map(topic => `<li><a data-chapter="${chapter.number}" data-topic="${topic.id}" href="${esc(route(chapter.number,topic.id))}">${esc(topic.title)}${topicCounts(topic)}</a></li>`).join('')}</ol></section>`).join('')}</div>`;
  }
  function renderChapter() {
    const chapter = edition.chapters.find(c => c.number === chapterNumber);
    if (!chapter) { chapterNumber = 0; return renderContents(); }
    const rail = `<aside class="contents-rail" aria-label="${esc(collectionTitle(collection))} contents"><a class="contents-home" data-chapter="0" href="${esc(route())}">← ${esc(collectionTitle(collection))} contents</a><ol>${edition.chapters.map(c => `<li><a data-chapter="${c.number}" href="${esc(route(c.number))}" ${c.number === chapterNumber ? 'aria-current="page"' : ''}>${String(c.number).padStart(2,'0')} &nbsp; ${esc(c.title)}<span class="paper-count">${paperCount(c.topics.flatMap(t=>t.papers))}</span></a>${c.number === chapterNumber ? `<ol>${c.topics.map(t => `<li><a data-chapter="${c.number}" data-topic="${t.id}" href="${esc(route(c.number,t.id))}">${esc(t.title)}</a></li>`).join('')}</ol>` : ''}</li>`).join('')}</ol></aside>`;
    const chapterIndex = edition.chapters.indexOf(chapter);
    const previous = edition.chapters[chapterIndex - 1];
    const next = edition.chapters[chapterIndex + 1];
    return `<div class="narrative-layout">${rail}<div class="chapter-reader"><div class="chapter-opening"><p class="eyebrow">${esc(collectionTitle(collection))} · CHAPTER ${String(chapter.number).padStart(2,'0')}</p><h2>${esc(chapter.title)}</h2><p class="paper-count">${paperCount(chapter.topics.flatMap(t=>t.papers))}</p></div>${chapter.topics.map(topic => `<section class="topic" id="topic-${topic.id}"><div class="topic-heading"><h3>${esc(topic.title)}</h3>${crosses(topic.trustScore,true)}</div><p class="paper-count">${paperCount(topic.papers)}${topicScoreStatus(topic)}</p>${topic.description ? `<p class="topic-description">${esc(topic.description)}</p>` : ''}${topic.papers.map(p => paperMarkup(p)).join('')}</section>`).join('')}<nav class="chapter-pagination" aria-label="Read another chapter">${previous ? `<a data-chapter="${previous.number}" href="${esc(route(previous.number))}">← ${esc(previous.title)}</a>` : `<a data-chapter="0" href="${esc(route())}">← Contents</a>`}${next ? `<a data-chapter="${next.number}" href="${esc(route(next.number))}">${esc(next.title)} →</a>` : `<a data-chapter="0" href="${esc(route())}">Back to contents ↑</a>`}</nav></div></div>`;
  }
  function filter() {
    const q = normalize($('#query').value).split(' ').filter(Boolean);
    const chosen = Object.fromEntries(multiFields.map(key=>[key,multi.values(key)]));
    const any = (key,test) => !chosen[key].length || chosen[key].some(test);
    filtered = records.filter(r => (collection==='all' || r.collections.includes(collection)) &&
      any('topic',v=>r.topics.some(t=>t.key===v)) && any('journal',v=>v.startsWith('family:') ? r.publisherFamily===v.slice(7) : r.journalKey===v) &&
      any('physiology',v=>r.categories.includes(v)) && any('category',v=>r.categories.includes(v)) &&
      any('target',v=>r.targets.includes(v)) && any('condition',v=>r.conditions.includes(v)) &&
      any('jadad',v=>model.matchesJadad(r,v)) && any('kind',v=>model.matchesStudy(r,v)) &&
      any('period',v=>v==='recent' && r.year>=2022 && r.year<=2026 || v==='older' && r.year && r.year<2022) &&
      any('trust',v=>chosen.topic.length ? chosen.topic.some(t=>model.matchesTopicTrust(r,Number(v),collection,t)) : model.matchesTopicTrust(r,Number(v),collection,'')) &&
      q.every(word=>r._search.includes(word)));
    const sort = $('#sort').value;
    filtered.sort((a,b) => sort === 'title' ? a.title.localeCompare(b.title) : sort === 'newest' ? (b.year || 0)-(a.year || 0) : a.priority-b.priority || (b.year || 0)-(a.year || 0) || a.title.localeCompare(b.title));
    const active = Object.values(chosen).reduce((sum,values)=>sum+values.length,0);
    $('#active-filter-count').textContent = active ? `· ${active} selected` : '';
    for(const key of multiFields)multi.refresh(key);
    multi.chips($('#selected-filters'),search);
    renderResults();
  }
  function selectedEntry(record) {
    const topics=multi.values('topic'), trust=multi.values('trust').map(Number);
    const eligible = record.chapterEntries.filter(p => (collection === 'all' || p.collection === collection) && (!topics.length || topics.includes(p.collection+':'+p.topicId)));
    return eligible.find(p => !trust.length || trust.includes(p.trustScore)) || eligible[0];
  }
  function renderResults() {
    const totalPages = Math.max(1, Math.ceil(filtered.length/PAGE_SIZE));
    pageIndex = Math.min(pageIndex,totalPages-1);
    $('#result-count').textContent = `${filtered.length.toLocaleString()} ${filtered.length === 1 ? 'paper' : 'papers'}${collection === 'all' ? '' : ' in '+collection[0].toUpperCase()+collection.slice(1)}`;
    $('#page-status').textContent = `${pageIndex+1} / ${totalPages}`;
    $('#previous').disabled = pageIndex === 0; $('#next').disabled = pageIndex >= totalPages-1;
    $('#export-json').disabled = !filtered.length;
    $('#export-csv').disabled = !filtered.length;
    $('#results').innerHTML = filtered.slice(pageIndex*PAGE_SIZE,(pageIndex+1)*PAGE_SIZE).map(record => {
      const entry = selectedEntry(record);
      const excerpt = record.quotes[0] || record.summary;
      const trimmed = excerpt.length > 340 ? excerpt.slice(0,337).trim()+'…' : excerpt;
      return `<article class="search-paper" data-id="${esc(record.id)}"><h3><button data-record="${esc(record.id)}">${esc(record.title)}</button></h3><p class="publication">${esc([journalDisplay(record),record.year,record.design].filter(Boolean).join(' · '))}</p>${trimmed ? record.quotes.length ? `<blockquote>${esc(trimmed)}</blockquote>` : `<p class="finding">${esc(trimmed)}</p>` : ''}${entry ? `<p class="search-context">${esc(collectionTitle(entry.collection))} · ${esc(entry.topicTitle)} ${crosses(entry.trustScore)}</p>` : ''}<div class="search-tags">${record.categories.slice(0,2).map(c=>`<span>${esc(c)}</span>`).join('')}${record.targets.slice(0,5).map(t=>`<span class="gene">${esc(t)}</span>`).join('')}</div><button class="text-button" data-record="${esc(record.id)}">Read paper →</button></article>`;
    }).join('') || '<div class="empty-state"><h2>No matching papers.</h2><p>Try a different term or clear the filters.</p><button data-reset>Clear filters</button></div>';
  }
  function render() {
    if (loading) return;
    edition = editions[collection];
    renderStatistics();
    $('#sound-assessment').hidden = collection !== 'sound';
    $('#language-assessment').hidden = collection !== 'language';
    $('#vision-assessment').hidden = collection !== 'vision';
    populateFilters();
    const reading = !!edition && view === 'chapters';
    $('#view-options').hidden = !edition;
    $('#view-options').setAttribute('aria-label',collectionTitle(collection)+' research view');
    document.querySelectorAll('[data-collection]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.collection === collection)));
    document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.view === view)));
    $('#reading-content').hidden = !reading; $('#search-view').hidden = reading;
    if (reading) $('#reading-content').innerHTML = chapterNumber ? renderChapter() : renderContents(); else filter();
  }
  function goChapter(number, topic = '', push = true, destination = collection) {
    collection = destination; view = 'chapters'; chapterNumber = number; pageIndex = 0;
    restoreFilters(new URLSearchParams());
    const url = route(number,topic);
    history[push ? 'pushState' : 'replaceState'](null,'',url);
    render();
    requestAnimationFrame(() => {
      const destination = topic ? document.getElementById('topic-'+topic) : $('#reading-content');
      destination?.scrollIntoView({block:'start'});
      if (destination) { destination.tabIndex = -1; destination.focus({preventScroll:true}); }
    });
  }
  function showPaper(paper, opener) {
    articleOpener = opener || document.activeElement;
    const topic = papers.find(p => p.id === paper.id);
    const record = records.find(r=>r.id===paper.recordId || r.aliases.includes(paper.recordId));
    const jadad = model.jadadStatus(record || {});
    const citations = [paper.citationSnapshot ? `${paper.citationSnapshot.count.toLocaleString()} citations · ${paper.citationSnapshot.source} · checked ${paper.citationSnapshot.retrievedAt}` : '',paper.authors, paper.doi ? 'DOI: '+paper.doi : '', paper.pmid ? 'PMID: '+paper.pmid : ''].filter(Boolean);
    $('#article-body').innerHTML = `${topic ? `<p class="article-chapter-link"><a data-chapter="${topic.chapterNumber}" data-topic="${topic.topicId}" data-chapter-collection="${topic.collection}" href="${esc(route(topic.chapterNumber,topic.topicId,topic.collection))}">${esc(collectionTitle(topic.collection))} · ${esc(topic.topicTitle)} →</a></p>` : ''}${paperMarkup(paper,'h2').replace('<h2>', '<h2 id="article-title">')}<div class="citation-details"><p>Jadad: ${jadad.status==='assessed' ? jadad.score+'/5' : jadad.status==='not-applicable' ? 'Not applicable' : 'Not yet assessed'}</p>${citations.map(line => `<p>${esc(line)}</p>`).join('')}${paper.urls?.slice(1).map(url => `<p><a href="${esc(safeURL(url))}" target="_blank" rel="noopener noreferrer">${esc(url)}</a></p>`).join('') || ''}</div>`;
    $('#article-body').querySelector('[data-paper-detail]')?.remove();
    if (!$('#article').open) $('#article').showModal();
    $('#article').scrollTop = 0;
    const url = new URL(location.href); url.searchParams.set('record',paper.recordId); url.searchParams.set('paper',paper.id); history.replaceState(null,'',url);
  }
  function openRecord(id, opener) {
    const record = records.find(r => r.id === id || r.aliases.includes(id));
    if (!record) return;
    const entry = selectedEntry(record);
    if (entry) return showPaper(entry,opener);
    const paper = {id:record.id,recordId:record.id,title:record.title,journal:record.journal,year:record.year,authors:record.authors,design:record.design,urls:record.urls,quotes:record.quotes.map(text => [{text,emphasis:false}]),summary:record.summary,limitation:record.limitation,citationCount:record.citationSnapshot?.count ?? record.citationCount,citationSnapshot:record.citationSnapshot,studyDesignAssessment:record.studyDesignAssessment,sourcePages:[],doi:record.doi,pmid:record.pmid,evidence:record.evidence};
    showPaper(paper,opener);
  }
  function buildRecords(data) {
    const aliasToId = new Map();
    for (const paper of papers) for (const alias of paper.aliases || []) aliasToId.set(alias,paper.recordId);
    return model.deduplicate(data.records.filter(r => !aliasToId.has(r.id)).map(r => {
      const chapterEntries = papers.filter(p => p.recordId === r.id);
      const soundEntries = chapterEntries.filter(p=>p.collection==='sound');
      const otherEntries = (r.archiveEntries || []).filter(a => !editions[a.collection]);
      const collections = [...new Set([...(r.collections || []).filter(c => c !== 'sound' || soundEntries.length || ['update','portal'].includes(r.origin)),...chapterEntries.map(p=>p.collection)])];
      if (!collections.length) return null;
      const first = chapterEntries[0];
      const access = publicationAccess[r.id] || {};
      const citationsByCollection = Object.fromEntries(collections.map(c=>[c,model.maximumCount([...chapterEntries.filter(p=>p.collection===c).map(p=>p.citationCount),...otherEntries.filter(a=>a.collection===c).map(a=>a.historicalCitationCount)])]));
      const topics = [...chapterEntries.map(p=>({key:p.collection+':'+p.topicId,title:p.topicTitle,collection:p.collection,trustScore:p.trustScore})),...otherEntries.map(a=>({key:a.collection+':'+a.sectionPage,title:a.section.replace(/^.*?\b[ivxlcdm]+\.\s*/i,''),collection:a.collection,trustScore:a.sectionGrade}))];
      const quotes = first ? first.quotes.filter((q,index) => index !== first.quotationReview?.index).map(quoteText) : otherEntries.flatMap(a => (a.originalNote || '').split('\n\n').filter(Boolean));
      const result = {id:r.id,publicationKey:access.publicationKey,aliases:first?.aliases || [],title:first?.title || r.title,journal:access.journalLabel || first?.journal || r.journal || '',publisherFamily:access.publisherFamily || r.publisherFamily,publisherFamilyLabel:access.publisherFamilyLabel || r.publisherFamilyLabel,neurotransmitters:r.neurotransmitters || [],year:first?.year || r.year,authors:first?.authors || r.authors || '',design:r.studyDesignAssessment ? r.design : first?.design || r.sourceDesign || r.design || '',kind:r.kind,doi:access.doi || first?.doi || r.doi,pmid:access.pmid || first?.pmid || r.pmid,urls:access.urls || first?.urls || r.urls,collections,categories:r.categories || [],targets:r.targets || [],priority:r.priority ?? 1,quotes,summary:first?.summary || r.finding || r.portalNotes?.[0]?.finding || '',limitation:first?.limitation || r.limitation || r.portalNotes?.[0]?.limitation || '',citationCount:model.maximumCount(Object.values(citationsByCollection)),citationSnapshot:r.citationSnapshot,studyDesignAssessment:r.studyDesignAssessment,citationsByCollection,topics,journalKey:access.journalKey ?? model.journalKey(first?.journal || r.journal),journalLabel:access.journalLabel || first?.journal || r.journal || '',topicScores:[...new Set([...chapterEntries.map(p=>p.trustScore),...otherEntries.map(a=>a.sectionGrade)].filter(Boolean))],soundEntries,chapterEntries,evidence:r.evidence,jadad:model.jadadStatus(r)};
      result._search = normalize([result.title,result.journal,result.publisherFamilyLabel,...result.neurotransmitters,result.authors,result.design,result.doi,result.pmid,...result.quotes,result.summary,result.limitation,...result.targets,...result.categories,...chapterEntries.map(p=>p.topicTitle+' '+p.quotes.map(quoteText).join(' ')+' '+(p.bibliography || []).map(b=>b.text).join(' ')),...otherEntries.map(a=>a.originalNote),JSON.stringify(r.evidence || {})].join(' '));
      return result;
    }).filter(Boolean));
  }
  function search() { if (loading) return; view = 'search'; pageIndex = 0; syncURL(); render(); }
  function reset() { restoreFilters(new URLSearchParams()); search(); }
  document.addEventListener('click', event => {
    const chapter = event.target.closest('[data-chapter]');
    if (chapter) { if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return; event.preventDefault(); if ($('#article').open) closePaper(); goChapter(Number(chapter.dataset.chapter),chapter.dataset.topic || '',true,chapter.dataset.chapterCollection || collection); return; }
    const button = event.target.closest('button');
    if (!button) return;
    if (button.dataset.collection) { $('#topic').value = ''; $('#journal').value = ''; collection = button.dataset.collection; view = editions[collection] && !$('#query').value ? 'chapters' : 'search'; chapterNumber = 0; pageIndex = 0; syncURL(true); render(); }
    if (button.dataset.view) { if (button.dataset.view === 'chapters') goChapter(chapterNumber); else search(); }
    if (button.dataset.record) openRecord(button.dataset.record,button);
    if (button.dataset.paperDetail) { const paper = papers.find(p => p.id === button.dataset.paperDetail); if (paper) showPaper(paper,button); }
    if (button.dataset.close) closePaper();
    if (button.hasAttribute('data-reset')) reset();
  });
  $('#search-form').onsubmit = event => { event.preventDefault(); search(); };
  let searchTimer;
  $('#query').oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(search,180); };
  for (const key of fields.filter(key=>key!=='query')) $('#'+key).onchange = search;
  $('#reset').onclick = reset;
  $('#previous').onclick = () => { pageIndex--; renderResults(); $('#results').scrollIntoView(); };
  $('#next').onclick = () => { pageIndex++; renderResults(); $('#results').scrollIntoView(); };
  const exportRecords = () => filtered.map(({_search,soundEntries,topicScores,...record}) => ({...record,topicTrustScores:topicScores,fullText:fullTextAccess[record.id] || {status:'not-indexed',retrievalAvailable:false}}));
  const download = (text,type,name) => {const url=URL.createObjectURL(new Blob([text],{type}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  $('#export-json').onclick = () => download(JSON.stringify({title:'Universal Research',schemaVersion:'2026-10-24',records:exportRecords()},null,2)+'\n','application/json','universal-research.json');
  $('#export-csv').onclick = () => {
    const rows=exportRecords();
    const columns=[...new Set(['id','title','journal','year','authors','design','doi','pmid',...rows.flatMap(Object.keys)])];
    const cell=value=>{let text=value==null?'':typeof value==='object'?JSON.stringify(value):String(value);if(/^[=+@\-\t\r]/.test(text))text="'"+text;return '"'+text.replaceAll('"','""')+'"';};
    download('\ufeff'+[columns,...rows.map(r=>columns.map(c=>r[c]))].map(row=>row.map(cell).join(',')).join('\r\n')+'\r\n','text/csv;charset=utf-8','universal-research.csv');
  };
  window.UniversalResearchContext=()=>({collection,chapter:chapterNumber,query:$('#query').value,filters:Object.fromEntries(multiFields.map(key=>[key,multi.values(key)]))});
  function closePaper() {
    const url = new URL(location.href); url.searchParams.delete('record'); url.searchParams.delete('paper'); history.replaceState(null,'',url);
    $('#article').close();
  }
  $('#article').addEventListener('close', () => {
    const url = new URL(location.href); url.searchParams.delete('record'); url.searchParams.delete('paper'); history.replaceState(null,'',url);
    if (articleOpener?.isConnected) articleOpener.focus({preventScroll:true});
  });
  $('#article').addEventListener('click', event => {
    if (event.target !== $('#article')) return;
    const rect = $('#article').getBoundingClientRect();
    if (event.clientX<rect.left || event.clientX>rect.right || event.clientY<rect.top || event.clientY>rect.bottom) closePaper();
  });
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || event.defaultPrevented) return;
    if ($('#article').open) { event.preventDefault(); closePaper(); }
    else if (window.parent !== window) { event.preventDefault(); parent.postMessage({type:'informatics:close-library'},location.origin); }
  });
  window.addEventListener('popstate', () => {
    const p = new URLSearchParams(location.search);
    collection = collections.includes(p.get('collection')) ? p.get('collection') : 'sound';
    view = p.get('view') === 'search' || !['sound','vision','language','general'].includes(collection) ? 'search' : 'chapters';
    chapterNumber = Number(p.get('chapter')) || 0;
    populateFilters();
    restoreFilters(p);
    render();
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  });
  async function load() {
    $('#load-error').hidden = true;
    try {
      const responses = await Promise.all(['sound-edition','library','publication-access','language-edition','general-edition','mindfulness-expansion','vision-edition','vision-expansion','clinical-conditions','nature-expansion','fulltext-availability','fibonacci-expansion'].map(name=>fetch('library-data/'+name+'.json?v=daytab-20261026')));
      if (responses.some(response => !response.ok)) throw new Error('Research unavailable');
      const [sound,data,access,language,general,expansion,vision,visionExpansion,conditions,natureExpansion,fulltexts,fibonacciExpansion] = await Promise.all(responses.map(response=>response.json()));
      fullTextAccess=fulltexts.records;
      if (!Array.isArray(sound.chapters) || !Array.isArray(data.records)) throw new Error('Invalid research data');
      publicationAccess = {...access.records,...visionExpansion.publicationAccess};
      for (const [id,verified] of Object.entries(natureExpansion.publicationAccess)) publicationAccess[id]={...publicationAccess[id],...verified};
      clinicalConditions=conditions.records;
      $('#condition').innerHTML='<option value="">All conditions and symptoms</option>'+conditions.taxonomy.map(t=>`<option value="${esc(t.id)}">${esc(t.label)}</option>`).join('');
      editions = {sound,language,general,vision};
      const byId = new Map(data.records.map(r=>[r.id,r]));
      for (const additional of [...expansion.records,...visionExpansion.records,...natureExpansion.records,...fibonacciExpansion.records]) {
        const existing = byId.get(additional.id);
        byId.set(additional.id,{...existing,...additional,collections:[...new Set([...(existing?.collections || []),...additional.collections])],archiveEntries:existing?.archiveEntries || []});
      }
      data.records = [...byId.values()];
      for (const [name,book] of Object.entries(editions)) for (const chapter of book.chapters) for (const topic of chapter.topics) for (const paper of topic.papers) {
        const resolved = publicationAccess[paper.recordId];
        if (resolved?.journalLabel) paper.journal = resolved.journalLabel;
        if (resolved?.urls) paper.urls = resolved.urls;
        if (resolved?.doi) paper.doi = resolved.doi;
        if (resolved?.pmid) paper.pmid = resolved.pmid;
        if (resolved?.publicationKey) paper.publicationKey = resolved.publicationKey;
      }
      papers = Object.entries(editions).flatMap(([collection,book])=>book.chapters.flatMap(chapter=>chapter.topics.flatMap(topic=>topic.papers.map(p=>({...p,collection,chapterNumber:chapter.number,topicId:topic.id,topicTitle:topic.title,trustScore:topic.trustScore,evidence:collection==='sound' ? p.evidence : byId.get(p.recordId)?.evidence || p.evidence})))));
      const allTargets=[...new Set([...targets,...data.records.flatMap(r=>r.targets || [])])].sort();
      $('#target').innerHTML='<option value="">All targets</option>'+allTargets.map(t=>`<option>${esc(t)}</option>`).join('');
      restoreFilters(params);
      records = buildRecords(data);
      for(const record of records){record.conditionEvidence=[record.id,...record.aliases].flatMap(id=>clinicalConditions[id] || []);record.conditions=[...new Set(record.conditionEvidence.map(c=>c.id))];}
      populateFilters(); loading = false; render();
      if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
      if (params.get('paper') && papers.some(p=>p.id===params.get('paper'))) showPaper(papers.find(p=>p.id===params.get('paper')));
      else if (params.get('record')) openRecord(params.get('record'));
    } catch (error) {
      $('#load-error').hidden = false; $('#reading-content').innerHTML = ''; $('#search-view').hidden = true;
    }
  }
  $('#retry').onclick = load;
  load();
})();
