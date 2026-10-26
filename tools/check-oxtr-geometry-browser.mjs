import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {gunzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const candidate=process.env.INFORMATICS_CANDIDATE==='1',base='https://staging.universalmind.tv/informatics/';
const out='/private/tmp/oxtr-geometry-'+(candidate?'candidate':'staging');await fs.mkdir(out,{recursive:true});
const dataRoot=path.join(root,'Informatics/pilot-v1/oxtr-tfbs-v1/experiment-v1');
const manifest=JSON.parse(await fs.readFile(path.join(dataRoot,'manifest.json'),'utf8'));
const data={};for(const [k,{file}] of Object.entries(manifest.files)){const b=await fs.readFile(path.join(dataRoot,file));data[k]=JSON.parse(file.endsWith('.gz')?gunzipSync(b):b);}
const server=await chromium.launchServer({headless:true,executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),report=[];
try{
 for(const viewport of [{width:1440,height:1000},{width:1366,height:768},{width:390,height:844}]){
  const context=await browser.newContext({viewport,acceptDownloads:true,reducedMotion:'reduce'});
  await context.addInitScript(()=>{let full=null;Object.defineProperty(document,'fullscreenElement',{get:()=>full});Element.prototype.requestFullscreen=async function(){full=this;document.dispatchEvent(new Event('fullscreenchange'));};document.exitFullscreen=async()=>{full=null;document.dispatchEvent(new Event('fullscreenchange'));};});
  if(candidate)await context.route(base+'**',async route=>{
   const rel=decodeURIComponent(new URL(route.request().url()).pathname.slice('/informatics/'.length))||'index.html';
   const file=path.resolve(root,'Informatics',rel);if(!file.startsWith(path.join(root,'Informatics')+'/'))return route.abort();
   try{await route.fulfill({status:200,body:await fs.readFile(file),contentType:{'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png'}[path.extname(file)]||'application/octet-stream'});}catch{return route.fulfill({status:404,body:'Missing candidate asset'});}
  });
  let fail=true;
  await context.route(base+'pilot-v1/oxtr-tfbs-v1/experiment-v1/studies.json*',async route=>{if(fail){fail=false;return route.fulfill({status:503,body:'Deliberate loading failure'});}return route.fallback();});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  try{
   await page.goto(base+'?atlas=1&record=fibonacci-m9-p24',{waitUntil:'domcontentloaded'});
   await page.waitForFunction(()=>!document.querySelector('#launch').disabled);await page.locator('#launch').click();
   const frame=page.frameLocator('#atlas-frame');await frame.locator('.intro h1').filter({hasText:'M9 / P24'}).waitFor();
   const entry=frame.locator('#atlas-science-summary [data-science-tfbs]');await entry.click();
   const dialog=frame.locator('.science-tfbs'),geo=dialog.locator('.tfbs-geometry');
   await geo.getByRole('button',{name:'Retry experiment',exact:true}).click();
   const status=geo.locator('.geometry-status');
   const expectCount=async n=>status.filter({hasText:new RegExp('^'+n.toLocaleString('en-AU')+' ')}).waitFor({timeout:60000});
   await expectCount(128);assert.equal(await geo.locator('[data-geometry-view="line"]').getAttribute('aria-pressed'),'true');
   const startUrl=page.url();
   const before=await page.evaluate(()=>({full:document.fullscreenElement?.id,srcs:[...document.querySelectorAll('audio,video')].map(e=>e.currentSrc),times:[...document.querySelectorAll('audio,video')].map(e=>e.currentTime)}));
   for(const source of ['studies','encode','jaspar']){
    await geo.locator('#geometry-source').selectOption(source);await expectCount(data[source].length);
    for(const view of ['line','circle','m9']){
     await geo.locator(`[data-geometry-view="${view}"]`).click();
     for(const pattern of ['alpha','beta','gamma','epsilon']){
      await geo.locator(`[data-geometry-pattern="${pattern}"]`).click();
      assert.equal(await geo.locator(`[data-geometry-pattern="${pattern}"]`).getAttribute('aria-pressed'),'true');
      assert.equal(await geo.locator('.geometry-panels svg').count(),2);
      const guide=geo.locator('.geometry-reference-plot .atlas-markers');
      assert.equal(await guide.getAttribute('data-atlas-pattern'),pattern);
      assert.equal(await guide.getAttribute('data-atlas-grid'),String(pattern==='beta'?9:24));
      const indices=pattern==='alpha'?Array.from({length:24},(_,i)=>i):pattern==='beta'?[4,8,7,5,1,2]:[0,4,8,12,16,20];
      assert.equal(await guide.getAttribute('data-atlas-turns'),indices.map(i=>i/(pattern==='beta'?9:24)).join(','));
      const originalReference=await geo.locator('.geometry-reference-plot').innerHTML();
      assert.equal(await geo.locator('[data-geometry-blend]').getAttribute('aria-pressed'),'false');
      assert.equal(await geo.locator('.geometry-data-plot .atlas-overlay').count(),0);
      await geo.locator('[data-geometry-blend]').click();
      assert.equal(await geo.locator('.geometry-data-plot .atlas-markers').getAttribute('data-atlas-turns'),indices.map(i=>i/(pattern==='beta'?9:24)).join(','));
      assert.equal(await geo.locator('.geometry-reference-plot').innerHTML(),originalReference);
      if(view==='line')assert.equal(await geo.locator('.geometry-data-plot .atlas-paths path').count(),0);
      assert.equal(await geo.locator('.geometry-reference-plot').locator('..').locator('.geometry-atlas-key').evaluate(e=>e.classList.contains('is-dot')),pattern==='alpha');
      assert.equal(await geo.locator('.geometry-blend-key .geometry-atlas-key').evaluate(e=>e.classList.contains('is-dot')),pattern==='alpha'||view==='line');
      if(viewport.width>=960){
        const fit=await geo.locator('.geometry-board').evaluate(board=>{
          const dialog=board.closest('dialog'),d=dialog.getBoundingClientRect(),bar=dialog.querySelector('.science-reader-bar').getBoundingClientRect();
          const b=board.getBoundingClientRect();
          const marks=[...board.querySelectorAll('svg,[data-geometry-view],[data-geometry-pattern],[data-geometry-blend]')].map(e=>{const r=e.getBoundingClientRect();return {top:r.top,bottom:r.bottom,left:r.left,right:r.right};});
          return {fits:marks.every(r=>r.top>=bar.bottom&&r.bottom<=d.bottom&&r.left>=d.left&&r.right<=d.right),boardHeight:b.height,plotHeights:[...board.querySelectorAll('svg')].map(e=>e.getBoundingClientRect().height)};
        });
        assert.equal(fit.fits,true,JSON.stringify({viewport,source,view,pattern,fit}));
        assert.ok(Math.min(...fit.plotHeights)>=250,JSON.stringify(fit));
        const labelSizes=await geo.locator('svg text.small').evaluateAll(labels=>labels.map(e=>parseFloat(getComputedStyle(e).fontSize)*Math.hypot(e.getScreenCTM().a,e.getScreenCTM().b)));
        assert.ok(labelSizes.every(size=>size>=10.9),JSON.stringify({viewport,labelSizes}));
      }

      if(view==='circle'&&source==='encode'&&pattern==='alpha')await page.screenshot({path:path.join(out,`comparison-board-${viewport.width}.png`)});
      await geo.locator('[data-geometry-blend]').click();
      if(view==='circle'&&source==='encode')await geo.locator('.geometry-panels').screenshot({path:path.join(out,`${source}-${view}-${pattern}-${viewport.width}.png`)});
     }
     if(view==='m9'){
      const counts=await geo.locator('.geometry-node-counts strong').allTextContents();
      assert.deepEqual(counts.map(s=>Number(s.replaceAll(',',''))),manifest.summaries[source].residue_counts_0_to_8);
      const discs=await geo.locator('.count-disc').evaluateAll(es=>es.map(e=>({count:Number(e.dataset.count),radius:Number(e.getAttribute('r'))})));
      const scale=discs[0].radius**2/discs[0].count;
      discs.forEach(d=>assert.ok(Math.abs(d.radius**2/d.count-scale)<1e-10));
      assert.equal(await geo.locator('.geometry-node-counts small').count(),9);
     }
     if(view==='circle'||view==='line'){
      const positions=await geo.locator('.position-count').evaluateAll(es=>es.map(e=>({offset:Number(e.dataset.geometryPosition),count:Number(e.dataset.count)})));
      const expected=new Map();for(const r of data[source]){const d=r.mapping.anchor_offset_bp;expected.set(d,(expected.get(d)||0)+1);}
      assert.deepEqual(positions,[...expected].sort((a,b)=>a[0]-b[0]).map(([offset,count])=>({offset,count})));
      const counts=positions.map(p=>p.count);
      assert.equal(counts.reduce((a,b)=>a+b,0),data[source].length);
      assert.equal(counts.length,new Set(data[source].map(r=>r.mapping.anchor_offset_bp)).size);
      await geo.locator('.geometry-panels').screenshot({path:path.join(out,`${source}-circle-${viewport.width}.png`)});
     }
    }
    await geo.locator('[data-geometry-pattern="beta"]').click();
    await geo.locator('.geometry-panels').screenshot({path:path.join(out,`${source}-m9-${viewport.width}.png`)});
    // Node drill-down retains all rows in an export, not just the visible table page.
    const residue=source==='studies'?7:0;
    const node=geo.locator(`.geometry-node-counts [data-geometry-residue="${residue}"]`);await node.focus();await page.keyboard.press('Enter');
    const expected=data[source].filter(r=>r.mapping.m9_residue===residue);
    const promise=page.waitForEvent('download');await geo.locator('[data-geometry-export="json"]').click();
    const file=path.join(out,`${source}-${viewport.width}.json`);await(await promise).saveAs(file);
    const result=JSON.parse(await fs.readFile(file,'utf8'));assert.equal(result.count,expected.length);assert.deepEqual(result.records.map(r=>r.id),expected.map(r=>r.id));
    const expectedOutcome={studies:'binding_detected',encode:'regional_association_detected',jaspar:'prediction_only'}[source];
    assert.ok(result.records.every(r=>r.binding_evidence.status===(r.study_id==='braintf'?'regional_association_detected':expectedOutcome)));
    assert.ok(result.records.every(r=>r.binding_evidence.source_url&&r.binding_evidence.expression_effect&&r.binding_evidence.review_depth));
    assert.equal(result.mapping.modulo9,manifest.mapping.modulo9);assert.equal(result.parameters.inspection_group.value,residue);
    assert.equal(result.parameters.overlay_version,result.overlay.version);assert.equal(result.overlay.atlas_registration.rotation_turns,0);
    assert.equal(result.verification.python.wolfram_coordinate_table_sha256,result.verification.wolfram.coordinate_table_sha256);
    for(const r of result.records)assert.deepEqual(r.mapping,expected.find(x=>x.id===r.id).mapping);
    await geo.locator('[data-geometry-clear-group]').click();
   }
   await geo.locator('#geometry-search').fill('not-a-factor');await expectCount(0);assert.equal(await geo.locator('[data-geometry-export="json"]').isDisabled(),true);assert.match(await geo.locator('.geometry-outcome-summary').textContent(),/No observations/);
   await geo.locator('#geometry-search').fill('');await expectCount(9164);
   await geo.locator('#geometry-score').selectOption('800');await expectCount(data.jaspar.filter(r=>r.score>=800).length);
   await geo.locator('#geometry-source').selectOption('encode');await expectCount(255);
   await geo.locator('#geometry-search').fill('CTCF');await expectCount(data.encode.filter(r=>r.factor.includes('CTCF')).length);
   await geo.locator('#geometry-source').selectOption('studies');await expectCount(128);
   await geo.locator('[data-geometry-view="line"]').click();
   // Close neighbouring centres stay separate; a zoom never changes their coordinates.
   await geo.locator('[data-geometry-view="line"]').click();
   assert.equal(await geo.locator('.position-count').count(),81);
   assert.equal(await geo.locator('.density-bar').count(),0);
   await geo.locator('[data-geometry-range="busy"]').click();
   const zoomed=await geo.locator('.position-count').evaluateAll(es=>es.map(e=>({offset:Number(e.dataset.geometryPosition),count:Number(e.dataset.count)})));
   assert.equal(zoomed.length,66);assert.equal(zoomed.reduce((s,p)=>s+p.count,0),113);assert.equal(Math.max(...zoomed.map(p=>p.count)),6);
   await geo.locator(viewport.width<960?'.geometry-data-plot':'.geometry-panels').screenshot({path:path.join(out,`exact-centres-zoom-${viewport.width}.png`)});
   await geo.locator('[data-geometry-position="10092"]').focus();await page.keyboard.press('Enter');
   assert.match(await geo.locator('.geometry-group').textContent(),/6 observations in 10,092 bp/);
   const positionDownload=page.waitForEvent('download');await geo.locator('[data-geometry-export="json"]').click();
   const positionPath=path.join(out,`exact-position-${viewport.width}.json`);await(await positionDownload).saveAs(positionPath);
   const positionRecords=JSON.parse(await fs.readFile(positionPath,'utf8'));assert.equal(positionRecords.count,6);
   assert.ok(positionRecords.records.every(r=>r.mapping.anchor_offset_bp===10092&&r.mapping.m9_residue===3));
   await geo.locator('[data-geometry-clear-group]').click();await geo.locator('[data-geometry-range="full"]').click();
   await geo.locator('#geometry-source').selectOption('studies');await expectCount(128);
   assert.equal(await geo.locator('.selection-guide').count(),0);
   assert.equal(await geo.locator('#geometry-zoom').isDisabled(),true);
   await geo.locator('#geometry-studies-kind').selectOption('sequence');await expectCount(2);
   await geo.locator('.geometry-data-plot [data-geometry-record="TFBS012"] .callout').click();
   await geo.locator('#geometry-zoom').check();
   await geo.locator('.geometry-panels').screenshot({path:path.join(out,`studies-line-zoom-${viewport.width}.png`)});
   assert.match(await geo.locator('.geometry-binding-result').textContent(),/Binding detected/);
   assert.match(await geo.locator('.geometry-binding-result').textContent(),/reporter construct/);
   await geo.locator('.geometry-binding-result').screenshot({path:path.join(out,`binding-outcome-${viewport.width}.png`)});
   const first=await geo.locator('.geometry-inspector-body h4').textContent();
   await geo.locator('[data-geometry-step="1"]').click();assert.notEqual(await geo.locator('.geometry-inspector-body h4').textContent(),first);
   // Blend and record selection survive view/sequence changes; the source data stay fixed.
   await geo.locator('[data-geometry-blend]').click();
   await geo.locator('[data-geometry-view="circle"]').click();
   await geo.locator('[data-geometry-pattern="alpha"]').click();
   assert.equal(await geo.locator('[data-geometry-blend]').getAttribute('aria-pressed'),'true');
   assert.equal(await geo.locator('.geometry-data-plot .atlas-markers').getAttribute('data-atlas-pattern'),'alpha');
   assert.equal(await geo.locator('.geometry-data-plot .individual').count(),3);
   await geo.locator('.geometry-panels').screenshot({path:path.join(out,`studies-alpha-blend-${viewport.width}.png`)});
   await geo.locator('[data-geometry-blend]').click();
   // Verify CSV and vector/raster exports are actual downloadable data and images.
   for(const format of ['csv','svg','png']){
    const download=page.waitForEvent('download');await geo.locator(`[data-geometry-export="${format}"]`).click();
    const file=path.join(out,`studies-${viewport.width}.${format}`);await(await download).saveAs(file);const bytes=await fs.readFile(file);
    if(format==='csv'){const s=bytes.toString();assert.ok(s.includes('TFBS012')&&s.includes('TFBS013')&&s.includes('anchor_offset_bp')&&s.includes('mapping_rules')&&s.includes('overlay_rules'));}
    if(format==='svg')assert.ok(bytes.toString().includes('<metadata>')&&bytes.toString().includes('OXTR binding-site geometry')&&bytes.toString().includes('atlas-overlay'));
    if(format==='png')assert.equal(bytes.subarray(1,4).toString(),'PNG');
   }
   await dialog.locator('[data-source-definition="studies"] [data-tfbs-compare]').click();await expectCount(128);
   assert.equal(await geo.locator('#geometry-studies-kind').inputValue(),'all');
   assert.match(await geo.locator('#geometry-source option:checked').textContent(),/128 mapped observations/);
   assert.match(await geo.locator('.geometry-outcome-summary').textContent(),/Binding detected · 2 observations/);
   assert.match(await geo.locator('.geometry-outcome-summary').textContent(),/Regional association detected · 126 observations/);
   const allWait=page.waitForEvent('download');await geo.locator('[data-geometry-export="json"]').click();
   const allPath=path.join(out,`all-studies-${viewport.width}.json`);await(await allWait).saveAs(allPath);
   const all=JSON.parse(await fs.readFile(allPath,'utf8'));assert.equal(all.count,128);
   assert.deepEqual(all.records.map(r=>r.id),data.studies.map(r=>r.id));
   assert.equal(all.records.filter(r=>r.study_id==='braintf').length,126);
   const allCsvWait=page.waitForEvent('download');await geo.locator('[data-geometry-export="csv"]').click();
   const allCsvPath=path.join(out,`all-studies-${viewport.width}.csv`);await(await allCsvWait).saveAs(allCsvPath);
   const allCsv=await fs.readFile(allCsvPath,'utf8');assert.ok(data.studies.every(r=>allCsv.includes(r.id)));
   assert.ok(allCsv.includes('source_sha256')&&allCsv.includes('binding_evidence')&&allCsv.includes('anchor_offset_bp'));
   await geo.locator('#geometry-studies-kind').selectOption('occupancy');await expectCount(126);
   await geo.locator('#geometry-search').fill('REST');
   const rest=data.studies.filter(r=>r.factor==='REST');await expectCount(rest.length);assert.ok(rest.length>0);
   if(!await geo.locator('.geometry-records').evaluate(e=>e.open))await geo.locator('.geometry-records summary').click();
   const firstPeak=geo.locator('.geometry-records [data-geometry-record]').first();await firstPeak.click();
   assert.match(await geo.locator('.geometry-inspector-body').textContent(),/REST/);
   assert.match(await geo.locator('.geometry-binding-result').textContent(),/Regional association detected/);
   assert.match(await geo.locator('.geometry-inspector-body').textContent(),/Replication QC: pass/);
   for(const view of ['line','circle']){
    await geo.locator(`[data-geometry-view="${view}"]`).click();assert.equal(await geo.locator('.selected-extent').count(),1);
   }
   const regionalWait=page.waitForEvent('download');await geo.locator('[data-geometry-export="json"]').click();
   const regionalPath=path.join(out,`REST-studies-${viewport.width}.json`);await(await regionalWait).saveAs(regionalPath);
   const regional=JSON.parse(await fs.readFile(regionalPath,'utf8'));assert.equal(regional.count,rest.length);
   assert.ok(regional.records.every(r=>r.binding_evidence.status==='regional_association_detected'&&r.source_sha256&&r.replication_qc==='pass'));
   await geo.locator('[data-geometry-view="m9"]').click();
   const restCounts=await geo.locator('.geometry-node-counts strong').allTextContents();assert.equal(restCounts.reduce((s,n)=>s+Number(n),0),rest.length);
   await geo.locator('.geometry-inspector').screenshot({path:path.join(out,`brain-region-inspector-${viewport.width}.png`)});
   await dialog.locator('[data-source-definition="studies"] [data-tfbs-compare]').click();await expectCount(128);
   await geo.locator('[data-geometry-view="line"]').click();
   await geo.locator('.geometry-source-context').screenshot({path:path.join(out,`all-studies-context-${viewport.width}.png`)});
   assert.equal(await dialog.locator('[data-source-definition]').count(),3);
   assert.match(await dialog.locator('.tfbs-evidence-priority').textContent(),/positive binding result/);
   assert.match(await geo.locator('.geometry-outcome-summary').textContent(),/Binding detected/);
   const hierarchy=await dialog.evaluate(e=>{
     const guide=e.querySelector('.tfbs-source-guide'),experiment=e.querySelector('.tfbs-geometry');
     return {before:!!(guide.compareDocumentPosition(experiment)&Node.DOCUMENT_POSITION_FOLLOWING),font:parseFloat(getComputedStyle(guide.querySelector('.tfbs-source-definition')).fontSize)};
   });
   assert.ok(hierarchy.before&&hierarchy.font>=16);
   await dialog.locator('.tfbs-source-guide').screenshot({path:path.join(out,`source-guide-${viewport.width}.png`)});
   await geo.locator('.geometry-source-context').screenshot({path:path.join(out,`graph-context-${viewport.width}.png`)});
   assert.equal(await dialog.evaluate(e=>e.scrollWidth<=e.clientWidth+1),true);
   await dialog.evaluate(e=>e.scrollTop=0);await page.screenshot({path:path.join(out,`experiment-${viewport.width}.png`)});
   // Preserve the source-data browser, learning lightbox, and Atlas/music context.
   assert.equal(await dialog.locator('.tfbs-datasets button').count(),3);
   await dialog.locator('[data-science-tfbs-learning]').click();await frame.locator('.science-tfbs-learning').waitFor({state:'visible'});await page.keyboard.press('Escape');
   assert.equal(await dialog.evaluate(e=>e.open),true);
   await page.keyboard.press('Escape');assert.equal(await dialog.evaluate(e=>e.open),false);assert.equal(await entry.evaluate(e=>e===document.activeElement),true);
   const after=await page.evaluate(()=>({full:document.fullscreenElement?.id,srcs:[...document.querySelectorAll('audio,video')].map(e=>e.currentSrc),times:[...document.querySelectorAll('audio,video')].map(e=>e.currentTime)}));
   assert.equal(page.url(),startUrl);assert.equal(after.full,before.full);assert.deepEqual(after.srcs,before.srcs);
   assert.deepEqual(errors,[]);
   report.push({viewport,combinations:36,blend_states_per_combination:2,source_definitions_prominent:true,binding_outcomes_and_exports:true,side_by_side:true,controls_and_complete_graphs_fit:viewport.width>=960,legend_matches_marks:true,blend_default_off:true,selection_deliberate:true,exact_position_counts:true,studies_distinct_centres:81,studies_maximum_same_centre:6,zoom_region_centres:66,zoom_region_observations:113,source_counts:{studies:128,encode:255,jaspar:9164},retry:true,exports:['json','csv','svg','png'],keyboard:true,overflow:false,media_context_preserved:true,errors});
  } finally {await page.evaluate(()=>document.querySelectorAll('audio,video').forEach(e=>e.pause())).catch(()=>{});await context.close();}
 }
 await fs.writeFile(path.join(out,'report.json'),JSON.stringify({candidate,results:report},null,2));console.log(JSON.stringify({out,results:report},null,2));
}finally{await browser.close();await server.close();}
