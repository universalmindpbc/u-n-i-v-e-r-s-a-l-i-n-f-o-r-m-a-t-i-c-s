import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {checkTfbsExplorer} from './check-oxtr-data-explorer.mjs';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const base='https://staging.universalmind.tv/informatics/';
const candidate=process.env.INFORMATICS_CANDIDATE==='1';
const out='/private/tmp/oxtr-browser-'+(candidate?'candidate':'staging');
await fs.mkdir(out,{recursive:true});
const data=JSON.parse(await fs.readFile(path.join(root,'Informatics/pilot-v1/oxtr-tfbs-v1/catalogue.json'),'utf8'));
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const server=await chromium.launchServer({downloadsPath:out,executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'),'Process-level silence required before navigation');
const browser=await chromium.connect(server.wsEndpoint()),results=[];
try{
 for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
  const context=await browser.newContext({viewport,acceptDownloads:true,reducedMotion:'reduce'});
  await context.addInitScript(()=>{let full=null;Object.defineProperty(document,'fullscreenElement',{get:()=>full});Element.prototype.requestFullscreen=async function(){full=this;document.dispatchEvent(new Event('fullscreenchange'));};document.exitFullscreen=async()=>{full=null;document.dispatchEvent(new Event('fullscreenchange'));};});
  if(candidate)await context.route(base+'**',async route=>{
   const relative=decodeURIComponent(new URL(route.request().url()).pathname.slice('/informatics/'.length))||'index.html';
   const file=path.resolve(root,'Informatics',relative);if(!file.startsWith(path.join(root,'Informatics')+'/'))return route.abort();
   try{await route.fulfill({status:200,body:await fs.readFile(file),contentType:{'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png','.csv':'text/csv','.zip':'application/zip'}[path.extname(file)]||'application/octet-stream'});}catch{await route.fulfill({status:404,body:'Missing candidate asset'});}
  });
  let failFirstCatalogue=true;
  await context.route(base+'pilot-v1/oxtr-tfbs-v1/catalogue.json*',async route=>{
   if(failFirstCatalogue){failFirstCatalogue=false;return route.fulfill({status:503,body:'Deliberate retry check'});}
   return route.fallback();
  });
  let failFirstRegions=true;
  await context.route(base+'pilot-v1/oxtr-tfbs-v1/encode-clusters.json*',async route=>{
   if(failFirstRegions){failFirstRegions=false;return route.fulfill({status:503,body:'Deliberate collection retry check'});}
   return route.fallback();
  });
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  try{
   await page.goto(base+'?atlas=1&record=fibonacci-m9-p24',{waitUntil:'domcontentloaded'});
   await page.waitForFunction(()=>!document.querySelector('#launch').disabled);await page.locator('#launch').click();
   const frame=page.frameLocator('#atlas-frame');
   await frame.locator('.intro h1').filter({hasText:'M9 / P24'}).waitFor();
   let originalUrl=page.url();
   const before=await page.evaluate(()=>({fullscreen:document.fullscreenElement?.id,media:[...document.querySelectorAll('video,audio')].map(m=>({src:m.currentSrc,paused:m.paused,time:m.currentTime}))}));
   const entry=frame.locator('#atlas-science-summary [data-science-tfbs]');
   await entry.click();
   const dialog=frame.locator('.science-tfbs');
   await dialog.getByRole('button',{name:'Try again',exact:true}).click();
   await dialog.locator('#tfbs-species').waitFor({state:'attached'});
   assert.equal(await dialog.locator('#tfbs-title').textContent(),'OXTR transcription-factor binding-site catalogue');
   const teachingLink=dialog.locator('[data-science-transcription]');
   await teachingLink.click();
   const teaching=frame.locator('.science-transcription');
   await teaching.locator('img').evaluate(img=>img.decode());
   assert.equal(await teaching.locator('img').evaluate(img=>img.naturalWidth),1536);
   assert.equal(await teaching.evaluate(e=>e.open),true);
   assert.equal(await teaching.locator('.transcription-takeaways p').count(),3);
   await page.screenshot({path:path.join(out,`transcription-${viewport.width}.png`)});
   await teaching.getByRole('button',{name:'Enlarge diagram',exact:true}).click();
   assert.equal(await teaching.locator('figure').evaluate(e=>e.scrollWidth),1536);
   await teaching.getByRole('button',{name:'Fit diagram to screen',exact:true}).click();
   await page.keyboard.press('Escape');
   assert.equal(await teaching.evaluate(e=>e.open),false);
   assert.equal(await dialog.evaluate(e=>e.open),true);
   assert.equal(await teachingLink.evaluate(e=>e===document.activeElement),true);
   const learningLink=dialog.locator('[data-science-tfbs-learning]');
   await learningLink.click();
   const learning=frame.locator('.science-tfbs-learning');
   await learning.locator('img').evaluate(img=>img.decode());
   assert.equal(await learning.locator('img').evaluate(img=>img.naturalWidth),1536);
   assert.equal(await learning.evaluate(e=>e.open),true);
   await page.screenshot({path:path.join(out,`learning-${viewport.width}.png`)});
   await learning.getByRole('button',{name:'Enlarge diagram',exact:true}).click();
   assert.equal(await learning.locator('figure').evaluate(e=>e.scrollWidth),1536);
   await learning.getByRole('button',{name:'Fit diagram to screen',exact:true}).click();
   await learning.getByText('The 15 sites, ENCODE clusters and predicted matches',{exact:true}).click();
   assert.equal(await learning.locator('details').evaluate(e=>e.open),true);
   await page.keyboard.press('Escape');
   assert.equal(await learning.evaluate(e=>e.open),false);
   assert.equal(await dialog.evaluate(e=>e.open),true);
   assert.equal(await learningLink.evaluate(e=>e===document.activeElement),true);
   await dialog.locator('.tfbs-study-additions summary').click();assert.ok((await dialog.locator('.tfbs-study-additions').innerText()).includes('126 screened'));await page.screenshot({path:path.join(out,`new-studies-${viewport.width}.png`)});await dialog.locator('.tfbs-study-additions summary').click();
   await checkTfbsExplorer({page,dialog,root,out,viewport});
   const literature=dialog.locator('.tfbs-literature');
   assert.equal(await literature.evaluate(e=>e.open),false);
   await literature.locator(':scope > summary').click();
   assert.equal(await literature.locator('tbody tr').count(),data.evidence.filter(r=>r.species==='Homo sapiens').length);
   await dialog.locator('#tfbs-species').selectOption('animal');
   assert.equal(await literature.locator('tbody tr').count(),data.evidence.filter(r=>r.species!=='Homo sapiens').length);
   await dialog.locator('#tfbs-species').selectOption('all');
   assert.equal(await literature.locator('tbody tr').count(),45);
   const kinds=data.harmonisation.categories.map(c=>c.id);
   assert.deepEqual(await dialog.locator('#tfbs-kind option').evaluateAll(options=>options.map(o=>o.value)),['all',...kinds]);
   for(const system of ['human','animal','all']){
    await dialog.locator('#tfbs-species').selectOption(system);
    const systemRows=data.evidence.filter(r=>system==='all'||(system==='human'?r.species==='Homo sapiens':r.species!=='Homo sapiens'));
    for(const kind of ['all',...kinds]){
     await dialog.locator('#tfbs-kind').selectOption(kind);
     const expected=systemRows.filter(r=>kind==='all'||r.evidence_categories.includes(kind));
     assert.deepEqual(await literature.locator('tbody .science-study-id').evaluateAll(items=>items.map(e=>e.textContent.split(' · ')[0])),expected.map(r=>r.id));
     const label=kind==='all'?'All findings':data.harmonisation.categories.find(c=>c.id===kind).label;
     assert.equal(await dialog.locator('#tfbs-kind option:checked').textContent(),`${label} (${expected.length} records)`);
    }
   }
   await dialog.locator('#tfbs-species').selectOption('human');
   await dialog.locator('#tfbs-kind').selectOption('binding_evidence');
   assert.equal(await literature.locator('tbody tr').count(),12);
   assert.ok((await literature.locator('tbody .science-study-id').allTextContents()).some(t=>t.startsWith('TFBS044')));
   await dialog.locator('#tfbs-species').selectOption('animal');
   assert.equal(await literature.locator('tbody tr').count(),7);
   await dialog.locator('#tfbs-species').selectOption('human');
   await dialog.locator('#tfbs-kind').selectOption('functional_evidence');
   assert.equal(await literature.locator('tbody tr').count(),9);
   assert.equal(await dialog.locator('.tfbs-evidence-guide').evaluate(e=>e.open),false);
   await dialog.locator('.tfbs-evidence-guide summary').click();
   assert.ok((await dialog.locator('.tfbs-evidence-guide').textContent()).includes('A record can appear in several evidence filters'));
   await dialog.locator('.tfbs-evidence-guide summary').click();
   assert.ok((await literature.locator('tbody .tfbs-kind').allTextContents()).every(t=>t.includes('Functional evidence')));
   const selectedDownload=page.waitForEvent('download');
   await dialog.locator('[data-tfbs-export="json"]').click();
   const selected=await selectedDownload,selectedPath=path.join(out,`selection-${viewport.width}.json`);await selected.saveAs(selectedPath);
   const selection=JSON.parse(await fs.readFile(selectedPath,'utf8'));
   assert.equal(selection.evidence.length,9);
   assert.deepEqual(selection.filters,{study_system:'human',evidence_category:'functional_evidence'});
   assert.ok(selection.evidence.every(r=>r.species==='Homo sapiens'&&r.evidence_categories.includes('functional_evidence')));
   const selectedCsvWait=page.waitForEvent('download');
   await dialog.locator('[data-tfbs-export="csv"]').click();
   const selectedCsv=await selectedCsvWait,selectedCsvPath=path.join(out,`selection-${viewport.width}.csv`);await selectedCsv.saveAs(selectedCsvPath);
   const selectedCsvText=await fs.readFile(selectedCsvPath,'utf8');
   assert.equal((selectedCsvText.match(/\r\nTFBS/g)||[]).length,9);
   assert.ok(selectedCsvText.includes('limitation,reference_transcript,coordinate_status'));
   await dialog.locator('.tfbs-harmonisation summary').click();
   await dialog.getByRole('link',{name:'Mapped literature · BED ↓',exact:true}).waitFor();
   await dialog.locator('.tfbs-harmonisation summary').click();
   await dialog.locator('#tfbs-kind').selectOption('all');
   await dialog.locator('#tfbs-species').selectOption('human');
   await literature.locator(':scope > summary').click();
   await dialog.evaluate(e=>e.scrollTop=0);
   assert.equal(await dialog.evaluate(e=>e.scrollWidth<=e.clientWidth+1),true);
   await page.screenshot({path:path.join(out,`catalogue-${viewport.width}.png`)});
   await dialog.locator('.tfbs-explorer').scrollIntoViewIfNeeded();
   await page.screenshot({path:path.join(out,`restored-filters-${viewport.width}.png`)});
   if(viewport.width===1440){
    for(const [label,file] of [['Complete JSON data','oxtr-tfbs-json.zip'],['Complete CSV data','oxtr-tfbs-csv.zip'],['Literature table','literature-evidence.csv']]){
     const link=dialog.getByRole('link',{name:label+' ↓',exact:true});
     if(candidate){
      const url=await link.getAttribute('href');
      const digest=await page.evaluate(async u=>{const bytes=await (await fetch(u)).arrayBuffer();return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join('');},url);
      assert.equal(digest,hash(await fs.readFile(path.join(root,'Informatics/pilot-v1/oxtr-tfbs-v1',file))));
     }else{
      const wait=page.waitForEvent('download',{timeout:120000});await link.click();const download=await wait;const saved=path.join(out,file);await download.saveAs(saved);
      assert.equal(hash(await fs.readFile(saved)),hash(await fs.readFile(path.join(root,'Informatics/pilot-v1/oxtr-tfbs-v1',file))));
     }
    }
   }
   await page.keyboard.press('Escape');
   assert.equal(await dialog.evaluate(e=>e.open),false);
   assert.equal(await entry.evaluate(e=>e===document.activeElement),true);
   await page.keyboard.press('Enter');
   assert.equal(await dialog.evaluate(e=>e.open),true);
   const bounds=await dialog.boundingBox();
   await page.mouse.click(bounds.x-3,bounds.y+20);
   assert.equal(await dialog.evaluate(e=>e.open),false);
   assert.equal(await entry.evaluate(e=>e===document.activeElement),true);
   await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();
   await frame.locator('#summary-science-slot [data-science-tfbs]').click();
   assert.equal(await dialog.evaluate(e=>e.open),true);
   await dialog.getByRole('button',{name:'Close binding-site catalogue',exact:true}).click();
   // Shared background entry remains available on a different modulus.
   await page.locator('#record-toggle').click();await page.locator('#record-search').fill('18');await page.locator('[data-record="fibonacci-m18-p24"]').click();originalUrl=page.url();
   await frame.locator('.intro h1').filter({hasText:'M18 / P24'}).waitFor();
   assert.equal(await frame.locator('#atlas-science-summary [data-science-tfbs]').count(),1);
   assert.equal(page.url().split('#')[0],originalUrl.split('#')[0]);
   const after=await page.evaluate(()=>({fullscreen:document.fullscreenElement?.id,media:[...document.querySelectorAll('video,audio')].map(m=>({src:m.currentSrc,paused:m.paused,time:m.currentTime}))}));
   assert.equal(after.fullscreen,before.fullscreen);
   assert.deepEqual(after.media.map(m=>[m.src,m.paused]),before.media.map(m=>[m.src,m.paused]));
   // About -> Genomics -> existing register -> catalogue stays in the experience.
   await page.locator('#atlas-toggle').click();await page.locator('#atlas-menu [data-about]').click();
   await page.locator('#about [data-science-genomics]').click();
   await page.locator('.science-genomics [data-science-register]').click();
   await page.locator('.science-register:not(.science-tfbs) [data-science-tfbs]').click();
   await page.locator('.science-tfbs #tfbs-species').waitFor({state:'attached'});
   assert.equal(page.url().split('#')[0],originalUrl.split('#')[0]);
   assert.equal(await page.evaluate(()=>document.fullscreenElement?.id),before.fullscreen);
   await page.keyboard.press('Escape');
   assert.equal(await page.locator('.science-register[aria-labelledby="science-register-title"]').evaluate(e=>e.open),true);
   assert.deepEqual(errors,[]);
   results.push({viewport,transcriptionTeachingAndFocus:true,teachingGraphicLoaded:true,teachingZoomAndGlossary:true,nestedTeachingEscapeAndFocus:true,dataExplorer:true,overlappingEvidenceFilters:true,all18FilterCombinations:true,selectedExportMatchesFilter:true,originalLabelsPreserved:true,filters:true,retryAfterFailure:true,keyboardAndOutsideClose:true,escapeAndFocus:true,summaryAndSidebar:true,aboutNestedLightbox:true,fullscreenStatePreserved:true,mediaStatePreserved:true,downloadsMatched:viewport.width===1440,errors});
  }finally{await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause())).catch(()=>{});await context.close();}
 }
 const result={passed:true,candidate,processMuted:true,results};await fs.writeFile(path.join(out,'results.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();await server.close();}
