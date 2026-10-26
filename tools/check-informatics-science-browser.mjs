import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const base='https://staging.universalmind.tv/informatics/';
const candidate=process.env.INFORMATICS_CANDIDATE==='1';
const output=process.env.INFORMATICS_QA_OUTPUT||'/private/tmp/first-principles-qa';
await fs.mkdir(output,{recursive:true});
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint());
const results=[];
try{
 for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
  const context=await browser.newContext({viewport,reducedMotion:'reduce'});
  await context.addInitScript(()=>{
   // Model the fullscreen container without entering native fullscreen.
   let fullscreenElement=null;window.__fullscreenCalls={enter:0,exit:0};
   Object.defineProperty(document,'fullscreenElement',{get:()=>fullscreenElement});
   Element.prototype.requestFullscreen=async function(){fullscreenElement=this;window.__fullscreenCalls.enter++;document.dispatchEvent(new Event('fullscreenchange'));};
   document.exitFullscreen=async()=>{fullscreenElement=null;window.__fullscreenCalls.exit++;};
  });
  if(candidate)await context.route(/https:\/\/staging\.universalmind\.tv\/(?:I|i)nformatics\//,async route=>{
   const relative=decodeURIComponent(new URL(route.request().url()).pathname).replace(/^\/(?:I|i)nformatics\//,'')||'index.html';
   const file=path.resolve(root,'Informatics',relative);
   if(!file.startsWith(path.join(root,'Informatics')+path.sep))return route.abort();
   try{await route.fulfill({status:200,contentType:{'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.png':'image/png','.svg':'image/svg+xml'}[path.extname(file)]||'application/octet-stream',body:await fs.readFile(file)});}
   catch{await route.fulfill({status:404,body:'Missing candidate file'});}
  });
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'?about=1',{waitUntil:'domcontentloaded'});
  assert.equal(await page.locator('#about').evaluate(e=>e.open),true);
  assert.match(await page.locator('#about .about-lead').innerText(),/Investigating relationships.*Fibonacci.*OXTR/);
  assert.equal(await page.locator('#about .science-teachables').count(),0);
  assert.equal(await page.locator('#about .about-resources a').count(),4);
  await page.locator('#about img').evaluateAll(imgs=>Promise.all(imgs.map(img=>img.decode())));
  await page.locator('#about-close').click();
  await page.waitForFunction(()=>!document.querySelector('#launch').disabled);
  await page.locator('#launch').click();
  await page.waitForFunction(()=>{const v=document.querySelector('#film');return !v.paused&&v.currentTime>0.2;},null,{timeout:45000});
  const initialURL=page.url();
  await page.evaluate(()=>{window.__music=document.querySelector('#film');window.__musicStart=__music.currentTime;window.__musicEvents=[];for(const event of ['pause','emptied','loadstart'])__music.addEventListener(event,()=>__musicEvents.push(event));});
  await page.locator('#experience [data-about]').click();
  await page.screenshot({path:path.join(output,`about-${viewport.width}.png`)});
  await page.getByRole('link',{name:'Science and Open Questions'}).click();
  const genomics=page.locator('.science-genomics');
  await genomics.locator('h1').waitFor();
  assert.equal(page.url(),initialURL);
  assert.equal(await genomics.evaluate(e=>e.parentElement===document.fullscreenElement),true);
  assert.equal(await genomics.locator('h1').innerText(),'Explore First Principles Genomics');
  assert.match(await page.locator('.genomics-lead').innerText(),/Our hypothesis.*in the context of OXTR/);
  assert.match(await genomics.innerText(),/initial research is specific to OXTR/);
  assert.doesNotMatch(await genomics.innerText(),/broader programme/);
  assert.equal(await page.locator('.genomics-targets li').count(),5);
  assert.equal(await page.locator('.science-teachables button').count(),4);
  await page.locator('.science-teachables img').evaluateAll(imgs=>Promise.all(imgs.map(img=>img.decode())));
  assert.equal(await page.locator('.science-register').evaluate(e=>e.open),false);
  assert.doesNotMatch(await genomics.innerText(),/golden[ -]ratio|Explore the science/i);
  await genomics.screenshot({path:path.join(output,`genomics-${viewport.width}.png`)});
  await page.locator('.science-teachables [data-science-card="gamma"]').click();
  const reader=page.locator('.science-reader');
  assert.equal(await reader.locator('.science-visual').evaluate(e=>e.open),true);
  await reader.locator('.science-image-button img').scrollIntoViewIfNeeded();
  await reader.locator('.science-image-button img').evaluate(img=>img.decode());
  await reader.locator('.science-image-button').click();
  assert.equal(await page.locator('.science-zoom').evaluate(e=>e.open),true);
  await page.keyboard.press('Escape');await page.keyboard.press('Escape');
  await page.locator('.genomics-main [data-science-register]').click();
  const register=page.locator('.science-register');
  assert.equal(await register.evaluate(e=>e.open),true);
  assert.equal(await register.locator('tbody tr').count(),14);
  assert.equal(await register.locator('.science-hero').count(),4);
  assert.deepEqual(await register.locator('tr:has(.science-hero)').evaluateAll(rows=>rows.map(r=>r.dataset.studyId)),['E03','E06','E08','E11']);
  assert.equal(new Set(await register.locator('.science-study-title').evaluateAll(links=>links.map(a=>a.href))).size,14);
  const data=await page.evaluate(()=>window.InformaticsStudies);
  assert.equal(data.length,14);
  assert.ok(data.every(s=>s.score===Object.values(s.scoreParts).reduce((a,b)=>a+b,0)&&s.score>=0&&s.score<=10&&s.jadad==='Not applicable'));
  for(const study of data){
   await register.locator(`[data-study-score="${study.id}"]`).click();
   const appraisal=page.locator('.science-appraisal');
   assert.equal(await appraisal.evaluate(e=>e.open),true);
   assert.match(await appraisal.innerText(),/Jadad: not applicable/);
   assert.match(await appraisal.locator('h2').innerText(),new RegExp('^'+study.score+' / 10'));
   await page.keyboard.press('Escape');
   assert.equal(await register.evaluate(e=>e.open),true);
   assert.equal(await register.locator(`[data-study-score="${study.id}"]`).evaluate(e=>e===document.activeElement),true);
  }
  await register.evaluate(e=>{e.scrollTop=0;e.querySelector('.science-table-scroll').scrollLeft=0;});
  await register.screenshot({path:path.join(output,`register-${viewport.width}.png`)});
  await register.locator('[data-science-card="gamma"]').click();
  assert.equal(await reader.locator('.science-visual').evaluate(e=>e.open),false);
  assert.match(await reader.innerText(),/Cryo-electron microscopy/);
  await reader.locator('.science-atlas-link').click();
  assert.equal(page.url(),initialURL);
  assert.equal(await genomics.evaluate(e=>e.open),false);
  const frame=page.frameLocator('#atlas-frame');
  await frame.locator('#atlas-science-summary').waitFor({state:'visible'});
  assert.equal(await frame.locator('#research-record').inputValue(),'fibonacci-m9-p24');
  assert.equal(await frame.locator('#atlas-science-summary img, #atlas-science-summary article').count(),0);
  assert.match(await frame.locator('.science-module-intro').innerText(),/background for our OXTR research programme/);
  await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();
  assert.equal(await frame.locator('#summary-science-slot #atlas-science-summary').count(),1);
  await frame.locator('#atlas-science-summary').screenshot({path:path.join(output,`atlas-context-${viewport.width}.png`)});
  await frame.locator('#atlas-science-summary [data-science-genomics]').click();
  const atlasGenomics=frame.locator('.science-genomics');
  await atlasGenomics.locator('h1').waitFor();
  assert.equal(page.url(),initialURL);
  assert.equal(await atlasGenomics.locator('.science-teachables button').count(),4);
  await atlasGenomics.screenshot({path:path.join(output,`atlas-genomics-${viewport.width}.png`)});
  await page.keyboard.press('Escape');
  assert.equal(await atlasGenomics.evaluate(e=>e.open),false);
  assert.equal(await frame.locator('[data-science-genomics]').evaluate(e=>e===document.activeElement),true);
  await frame.locator('[data-science-genomics]').click();
  await atlasGenomics.locator('[data-science-atlas]').click();
  assert.equal(await atlasGenomics.evaluate(e=>e.open),false);
  assert.equal(await frame.locator('[role=tab][aria-selected=true]').innerText(),'Geometry summary');
  await frame.locator('#atlas-science-summary [data-science-register]').click();
  const atlasRegister=frame.locator('.science-register'),atlasReader=frame.locator('.science-reader');
  assert.equal(await atlasRegister.locator('tbody tr').count(),14);
  for(const id of ['alpha','beta','gamma','epsilon']){
   await atlasRegister.locator(`[data-science-card="${id}"]`).click();
   assert.equal(await atlasReader.locator('.science-visual').evaluate(e=>e.open),false);
   assert.match(await atlasReader.locator('.science-evidence').innerText(),/OPEN ATLAS QUESTION/);
   assert.match(await atlasReader.locator('.science-question-text').innerText(),/OXTR/);
   if(id==='epsilon'){
    await atlasReader.locator('.science-visual>summary').click();
    await atlasReader.locator('.science-image-button img').scrollIntoViewIfNeeded();
    await atlasReader.locator('.science-image-button img').evaluate(img=>img.decode());
    await atlasReader.locator('.science-atlas-link').click();
   }else await atlasReader.locator('button[aria-label="Close research card"]').click();
  }
  await frame.locator('[role=tab][aria-selected=true]').filter({hasText:/^Epsilon$/}).waitFor();
  assert.equal(await atlasRegister.evaluate(e=>e.open),false);
  for(const id of ['fibonacci-m18-p24','fibonacci-m1-p1','fibonacci-m48','m-summary']){
   const value=id==='fibonacci-m48'?await frame.locator('#research-record option').evaluateAll(opts=>opts.find(o=>o.textContent.startsWith('Fibonacci · M48 /')).value):id;
   await frame.locator('#research-record').selectOption(value);
   await frame.locator('#atlas-science-summary').waitFor({state:value==='m-summary'?'hidden':'visible'});
   if(value==='m-summary')await frame.locator('#family-view').waitFor({state:'visible'});
   else await frame.locator('.intro h1').filter({hasText:new RegExp('^M'+value.match(/-m(\d+)-/)[1]+' /')}).waitFor();
   if(value==='fibonacci-m18-p24'){
    await frame.getByRole('tab',{name:'Epsilon',exact:true}).click();await frame.locator('#epsilon-reduced').click();
    assert.equal(await frame.locator('#epsilon-reduced').getAttribute('aria-pressed'),'true');
   }
  }
  // Every native record offers the same shared collection, independently of geometry.
  const options=await frame.locator('#research-record option').evaluateAll(opts=>opts.filter(o=>o.value!=='m-summary').map(o=>({value:o.value,modulus:Number(o.value.match(/-m(\d+)-/)[1])})));
  assert.equal(options.length,48);
  for(const option of options){
   await frame.locator('#research-record').selectOption(option.value);
   await frame.locator('.intro h1').filter({hasText:new RegExp('^M'+option.modulus+' /')}).waitFor();
   const viewNames=await frame.locator('[role=tab]').allTextContents();
   for(const name of viewNames.filter(name=>!name.includes('Odom'))){
    await frame.getByRole('tab',{name,exact:true}).click();
    await frame.locator('#atlas-science-summary').waitFor({state:'visible'});
    assert.equal(await frame.locator('#atlas-science-summary [data-science-register]').count(),1);
    assert.equal(await frame.locator('#atlas-science-summary [data-science-genomics]').count(),1);
    assert.match(await frame.locator('#atlas-science-summary summary').innerText(),/How scientific significance is framed/);
   }
   await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();
   await frame.locator('#summary-science-slot #atlas-science-summary').waitFor({state:'visible'});
   assert.match(await frame.locator('.science-module-intro').innerText(),/14 peer-reviewed studies.*four HERO references/);
   await frame.locator('#atlas-science-summary [data-science-register]').click();
   assert.equal(await atlasRegister.locator('tbody tr').count(),14);
   assert.equal(await atlasRegister.locator('.science-hero').count(),4);
   assert.match(await atlasRegister.locator('.science-register-intro').innerText(),/Shared background literature.*M1–M48/);
   await atlasRegister.locator('button[aria-label="Close research register"]').click();
   assert.equal(await frame.locator('#research-record').inputValue(),option.value);
   if([1,18,27,36,45,48].includes(option.modulus)&&viewport.width===1440)await frame.locator('#atlas-science-summary').screenshot({path:path.join(output,`atlas-m${option.modulus}.png`)});
  }
  await frame.locator('#research-record').selectOption('m-summary');
  await frame.locator('#family-view').waitFor({state:'visible'});
  // Return from About into an existing non-M9 Atlas without reloading the shell.
  await page.locator('#experience [data-about]').click();
  await page.locator('#about [data-science-genomics]').click();
  await genomics.locator('[data-science-atlas]').click();
  await frame.locator('#atlas-science-summary').waitFor({state:'visible'});
  assert.equal(await frame.locator('#research-record').inputValue(),'fibonacci-m9-p24');
  assert.equal(await frame.locator('[role=tab][aria-selected=true]').innerText(),'Geometry summary');
  assert.equal(page.url(),initialURL);
  assert.deepEqual(await page.evaluate(()=>__fullscreenCalls),{enter:1,exit:0});
  assert.equal(await page.evaluate(()=>document.fullscreenElement===document.querySelector('#experience')),true);
  const media=await page.evaluate(()=>({same:__music===document.querySelector('#film'),advanced:__music.currentTime>__musicStart+1,paused:__music.paused,events:__musicEvents}));
  assert.deepEqual(media,{same:true,advanced:true,paused:false,events:[]});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.equal(await frame.locator('html').evaluate(e=>e.scrollWidth<=innerWidth),true);
  assert.deepEqual(errors,[]);
  results.push({viewport,compactAbout:true,genomicsLightboxBothEntries:true,musicContinuous:true,fullscreenContainerPreserved:true,studies:14,heroes:4,scoreBreakdowns:14,atlasLiteratureFirst:true,all48RegistersVerified:true,allAvailableSequenceViewsVerified:true,errors});
  await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause()));
  if(viewport.width===1440){
   let attempts=0,release;
   const gate=new Promise(resolve=>{release=resolve;});
   const failureRoute=/\/genomics\.html\?v=lightbox1$/;
   await context.route(failureRoute,async route=>{attempts++;if(attempts===1)return route.fulfill({status:503,body:'Temporarily unavailable'});await gate;return route.fallback();});
   await page.goto(base+'?about=1',{waitUntil:'domcontentloaded'});
   await page.locator('#about [data-science-genomics]').click();
   await page.locator('[data-genomics-retry]').waitFor();
   assert.equal(page.url(),base+'?about=1');
   await page.locator('[data-genomics-retry]').click();
   await page.getByText('Loading First Principles Genomics…',{exact:true}).waitFor();
   await page.getByRole('button',{name:'Close First Principles Genomics',exact:true}).click();
   await page.locator('#about [data-science-genomics]').click();
   release();
   await page.locator('.science-genomics h1').waitFor();
   assert.equal(attempts,2);
   assert.equal(await page.locator('.science-genomics .science-teachables button').count(),4);
   await page.keyboard.press('Escape');
   assert.equal(await page.locator('#about [data-science-genomics]').evaluate(e=>e===document.activeElement),true);
   await context.unroute(failureRoute);
   results.push({failedFetchRetry:true,closeReopenDuringLoading:true,noPageNavigation:true});
  }
  await context.close();
 }
 console.log(JSON.stringify({passed:true,candidate,processMuted:true,nativeFullscreen:false,results,output},null,2));
}finally{await browser.close().catch(()=>{});await server.close().catch(()=>{});}
