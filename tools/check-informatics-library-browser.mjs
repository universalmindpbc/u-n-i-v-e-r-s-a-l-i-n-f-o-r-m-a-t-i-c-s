import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const base='https://staging.universalmind.tv/informatics/';
const candidate=process.env.INFORMATICS_CANDIDATE==='1';
const output=process.env.INFORMATICS_QA_OUTPUT||'/private/tmp/informatics-library-qa';
await fs.mkdir(output,{recursive:true});
const data=JSON.parse(await fs.readFile(path.join(root,'Informatics/library-data/library.json'),'utf8'));
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),results=[];
try{
 for(const viewport of [{width:1470,height:956},{width:1024,height:768},{width:390,height:844},{width:320,height:844}]){
  const context=await browser.newContext({viewport,reducedMotion:'reduce'});
  await context.addInitScript(()=>{
   let full=null;window.__fullscreenCalls={enter:0,exit:0};
   Object.defineProperty(document,'fullscreenElement',{get:()=>full});
   Element.prototype.requestFullscreen=async function(){full=this;window.__fullscreenCalls.enter++;document.dispatchEvent(new Event('fullscreenchange'));};
   document.exitFullscreen=async()=>{full=null;window.__fullscreenCalls.exit++;};
  });
  if(candidate)await context.route(/https:\/\/staging\.universalmind\.tv\/(?:I|i)nformatics\//,async route=>{
   const relative=decodeURIComponent(new URL(route.request().url()).pathname).replace(/^\/(?:I|i)nformatics\//,'')||'index.html',file=path.resolve(root,'Informatics',relative);
   if(!file.startsWith(path.join(root,'Informatics')+path.sep))return route.abort();
   try{await route.fulfill({status:200,contentType:{'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml'}[path.extname(file)]||'application/octet-stream',body:await fs.readFile(file)});}catch{await route.fulfill({status:404,body:'Missing candidate file'});}
  });
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'library.html',{waitUntil:'domcontentloaded'});
  await page.locator('.research-card').first().waitFor();
  assert.equal(await page.locator('#result-count').innerText(),`${data.summary.records} records`);
  assert.equal(await page.locator('.research-card').count(),20);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.screenshot({path:path.join(output,`library-${viewport.width}.png`)});
  for(const domain of ['sound','vision','language']){
   await page.locator(`[data-collection="${domain}"]`).click();
   assert.match(await page.locator('#result-count').innerText(),new RegExp('^'+data.summary.collections[domain]+' records'));
   await page.locator('#origin').selectOption('update');
   const expected=data.records.filter(r=>r.collections.includes(domain)&&r.origin==='update');
   assert.match(await page.locator('#result-count').innerText(),new RegExp('^'+expected.length+' records'));
   await page.locator('#reset').click();
  }
  await page.locator('[data-collection="all"]').click();
  await page.locator('#query').fill('IL-6');const il6=await page.locator('#result-count').innerText();
  await page.locator('#query').fill('interleukin 6');assert.equal(await page.locator('#result-count').innerText(),il6);
  await page.locator('#reset').click();await page.locator('#target').selectOption('IL6');await page.locator('#origin').selectOption('update');
  assert.ok(await page.locator('[data-record="pmid-41974356"]').count()>0);
  await page.locator('[data-record="pmid-41974356"]').first().click();
  assert.match(await page.locator('#article-body').innerText(),/protein|salivary/i);
  await page.screenshot({path:path.join(output,`article-${viewport.width}.png`)});
  const downloadEvent=page.waitForEvent('download');await page.locator('#record-export').click();const download=await downloadEvent;
  const exportedPath=path.join(output,'download-'+viewport.width+'.json');await download.saveAs(exportedPath);const exported=JSON.parse(await fs.readFile(exportedPath,'utf8'));assert.deepEqual(exported.records[0],data.records.find(r=>r.pmid==='41974356'));assert.ok(!('_search' in exported.records[0]));
  await page.keyboard.press('Escape');assert.equal(await page.locator('#article').evaluate(e=>e.open),false);
  assert.equal(await page.locator('[data-record="pmid-41974356"]').first().evaluate(e=>e===document.activeElement),true);
  await page.locator('#reset').click();await page.locator('#stars').selectOption('3');
  assert.match(await page.locator('#result-count').innerText(),new RegExp('^'+data.records.filter(r=>r.stars===3).length+' records'));
  await page.locator('#reset').click();await page.locator('#query').fill('zzzz_no_possible_match_8844');assert.equal(await page.locator('.research-card').count(),0);assert.equal(await page.locator('#export-json').isDisabled(),true);await page.locator('#empty-reset').click();
  await page.locator('#next').click();assert.match(await page.locator('#page-status').innerText(),/^Page 2 /);await page.locator('#previous').click();
  await page.locator('#notebook-open').click();await page.locator('#notebook-page-body h3').waitFor();
  for(const [domain,count] of [['sound',144],['vision',19],['language',73]]){
   await page.locator('#notebook-collection').selectOption(domain);await page.waitForFunction(n=>document.querySelector('#notebook-page').options.length===n,count);
   await page.locator('#notebook-page').selectOption(String(count));assert.equal(await page.locator('#notebook-page-body h3').innerText(),'Source page '+count);
  }
  await page.locator('#notebook-query').fill('gene');await page.waitForFunction(()=>document.querySelector('#notebook-status').textContent.includes('source-text search'));
  assert.ok(await page.locator('#notebook-page option').count()>0);await page.keyboard.press('Escape');
  assert.equal(await page.locator('#notebook-open').evaluate(e=>e===document.activeElement),true);
  await page.goto(base+'library.html?record=pmid-40749778',{waitUntil:'domcontentloaded'});await page.locator('#article[open]').waitFor();assert.match(await page.locator('#article-body').innerText(),/Oxtr|OXTR/);await page.keyboard.press('Escape');
  await page.goto(base+'?about=1',{waitUntil:'domcontentloaded'});await page.locator('#about[open]').waitFor();
  await page.screenshot({path:path.join(output,`about-${viewport.width}.png`)});
  await page.locator('[data-network]').click();await page.locator('.network-overlay[open]').waitFor();
  assert.equal(await page.locator('[data-network-target]').count(),14);
  assert.match(await page.locator('.network-body').innerText(),/validation pending/);
  await page.screenshot({path:path.join(output,`network-${viewport.width}.png`)});
  assert.equal(await page.locator('.network-body').evaluate(e=>e.scrollWidth<=e.clientWidth+1),true);
  await page.locator('[data-network-target="IL6"]').click();
  const frame=page.frameLocator('.library-overlay iframe');await frame.locator('.research-card').first().waitFor();assert.equal(await frame.locator('#target').inputValue(),'IL6');
  await frame.locator('#query').focus();await page.keyboard.press('Escape');await page.locator('.library-overlay').waitFor({state:'hidden'});
  assert.equal(await page.locator('[data-network-target="IL6"]').evaluate(e=>e===document.activeElement),true);
  await page.keyboard.press('Escape');assert.equal(await page.locator('[data-network]').evaluate(e=>e===document.activeElement),true);
  await page.locator('#about-close').click();await page.waitForFunction(()=>!document.querySelector('#launch').disabled);await page.locator('#launch').click();
  await page.waitForFunction(()=>{const v=document.querySelector('#film');return !v.paused&&v.currentTime>.2;},null,{timeout:60000});
  const initialURL=page.url();
  await page.evaluate(()=>{window.__media=document.querySelector('#film');window.__mediaTime=__media.currentTime;window.__mediaEvents=[];for(const event of ['pause','emptied','loadstart'])__media.addEventListener(event,()=>__mediaEvents.push(event));});
  await page.locator('#experience .brand[data-about]').click();
  await page.locator('[data-network]').click();
  assert.equal(await page.locator('.network-overlay').evaluate(e=>e.parentElement===document.fullscreenElement),true);
  await page.locator('[data-network-target="OXTR"]').click();await frame.locator('#target').waitFor();assert.equal(await frame.locator('#target').inputValue(),'OXTR');
  await page.locator('.library-overlay [aria-label="Close research library"]').click();
  await page.locator('.network-overlay [aria-label="Close candidate network"]').click();await page.locator('#about-close').click();
  assert.equal(page.url(),initialURL);
  for(const domain of ['sound','vision','language']){
   await page.locator('#atlas-toggle').click();await page.locator(`#atlas-menu [data-area="${domain}"]`).click();await page.locator(`#area-${domain}`).waitFor();const areaURL=page.url();
   await page.locator(`#area-${domain} [data-library="${domain}"]`).click();await frame.locator('.research-card').first().waitFor();
   assert.equal(await frame.locator(`[data-collection="${domain}"]`).getAttribute('aria-pressed'),'true');
   assert.equal(await page.locator('.library-overlay').evaluate(e=>e.parentElement===document.fullscreenElement),true);
   await page.locator('.library-overlay [aria-label="Close research library"]').click();assert.equal(page.url(),areaURL);
   assert.equal(await page.locator(`#area-${domain} [data-library="${domain}"]`).evaluate(e=>e===document.activeElement),true);
  }
  await page.locator('#atlas-toggle').focus();await page.keyboard.press('ArrowDown');
  assert.deepEqual(await page.locator('#atlas-menu button').allTextContents(),['Geometry Atlas','Molecular Atlas','About','Sound','Vision','Language','Universal Research','Partners']);
  await page.locator('#atlas-menu [data-library]').click();await frame.locator('.research-card').first().waitFor();
  await page.locator('.library-overlay [aria-label="Close research library"]').click();
  await page.waitForFunction(()=>document.activeElement===document.querySelector('#atlas-toggle'));
  // Existing record navigation remains available after the new collection views.
  await page.locator('#record-toggle').focus();await page.keyboard.press('Enter');
  await page.locator('.record-pinned [data-mode="research"]').click();
  const atlas=page.frameLocator('#atlas-frame');await atlas.locator('.intro h1').filter({hasText:'M9 / P24'}).waitFor();
  await atlas.getByRole('tab',{name:'Beta',exact:true}).click();
  assert.equal(await atlas.getByRole('tab',{name:'Beta',exact:true}).getAttribute('aria-selected'),'true');
  const media=await page.evaluate(()=>({same:__media===document.querySelector('#film'),advanced:__media.currentTime>__mediaTime,paused:__media.paused,events:__mediaEvents}));
  assert.deepEqual(media,{same:true,advanced:true,paused:false,events:[]});assert.deepEqual(await page.evaluate(()=>__fullscreenCalls),{enter:1,exit:0});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);assert.deepEqual(errors,[]);
  results.push({viewport,searchAndFilters:true,threeCollections:true,sourceReaderPages:236,recordPermalink:true,jsonExport:true,focusRestored:true,networkTargets:14,activeExperiencePreserved:true,networkInActiveExperience:true,media,nativeFullscreen:false,errors});
  await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause()));
  if(viewport.width===1470){
   let attempts=0;const fail=/\/library-data\/library\.json\?v=1$/;
   await context.route(fail,async route=>++attempts===1?route.fulfill({status:503,body:'Test unavailable response'}):route.fallback());
   await page.goto(base+'library.html');await page.locator('#load-error').waitFor({state:'visible'});
   assert.equal(await page.locator('.research-card').count(),0);await page.locator('#retry').click();await page.locator('.research-card').first().waitFor();assert.equal(attempts,2);
   results.push({unavailableStateAndRetry:true,noFallbackData:true});
  }
  await context.close();
 }
 const result={passed:true,candidate,checkedAt:new Date().toISOString(),base,processMuted:true,records:data.summary.records,results};
 await fs.writeFile(path.join(output,'verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
}finally{for(const context of browser.contexts()){for(const page of context.pages())await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause())).catch(()=>{});await context.close();}await browser.close().catch(()=>{});await server.close().catch(()=>{});}
