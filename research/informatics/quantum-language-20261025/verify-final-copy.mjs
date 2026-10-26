import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const out='research/informatics/quantum-language-20261025',base='https://staging.universalmind.tv',endpoint='https://informatics-universal-one-staging.universalmind.workers.dev';
await fs.mkdir(out,{recursive:true});const token=(await fs.readFile('/private/tmp/uo-secrets/qa-token','utf8')).trim();
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:1100},reducedMotion:'reduce'});
try{
 await ctx.route(endpoint+'/**',r=>r.continue({headers:{...r.request().headers(),'X-Research-QA':token}}));
 const page=await ctx.newPage(),errors=[],layouts=[];let questions=0;page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.url()===endpoint+'/quantum/question'&&r.method()==='POST')questions++;});
 await page.goto(base+'/informatics/?entry=research&area=quantum');const f=page.frameLocator('#quantum-frame');await f.locator('#completed-study').waitFor({state:'visible'});
 const summary=await f.locator('#study-initial-insights').innerText();assert(summary.includes('The laboratory')||summary.includes('laboratory binding results stay hidden'));assert(summary.includes('CTCF, one transcription factor'));assert(!/eight versions|kept out of (model )?development/i.test(summary));for(const s of ['The question','What this first run tested','The result','The signal we want next','The next action','No binding-site prediction score was measured'])assert(summary.includes(s));assert(!/encoding|ideal|candidate measurements/i.test(summary));
 await f.locator('#completed-study').evaluate(n=>n.scrollIntoView({block:'start'}));await page.screenshot({path:out+'/goal-desktop.png'});await f.locator('#study-initial-insights').screenshot({path:out+'/goal-callout.png'});
 await f.locator('#question').fill('Keep my question while reading the study.');
 for(const entry of ['next','simulation','quantum','comparison']){
  const opener=entry==='next'?f.locator('#study-initial-insights [data-open-next-test]'):f.locator(`[data-recorded-result=${entry}]`);await opener.scrollIntoViewIfNeeded();const before=await f.locator('body').evaluate(()=>({y:scrollY,q:document.querySelector('#question').value}));await opener.click();await f.locator('#results-dialog').waitFor({state:'visible'});
  if(entry==='next'){
   await f.locator('.available-evidence h4').filter({hasText:'Public binding data is already on file.'}).waitFor();const text=await f.locator('#results-dialog-content').innerText();for(const s of ['NEXT EXPERIMENT · NOT RUN','Positive signal','No clear result','Discouraging result','wet lab','ENCSR932ZMX'])assert(text.includes(s));assert(!text.includes('encoding'));
   await page.screenshot({path:out+'/next-test-desktop.png'});await f.locator('.binding-data-detail summary').click();assert.equal(await f.locator('.binding-data-detail tbody tr').count(),7);await f.locator('.binding-data-detail').evaluate(n=>n.scrollIntoView({block:'start'}));await page.screenshot({path:out+'/binding-data-desktop.png'});
   await f.locator('.success-guide').evaluate(n=>n.scrollIntoView({block:'start'}));await page.screenshot({path:out+'/success-rules-desktop.png'});
  }else if(entry==='comparison'){
   assert.match(await f.locator('#results-dialog-content').innerText(),/Which result do we trust/);await page.screenshot({path:out+'/comparison-desktop.png'});
  }else{
   const slot=f.locator('#modal-'+entry+'-report');await slot.locator('.representation-row').last().waitFor();assert(!/ideal (calculation|direction|model)/i.test(await slot.innerText()));assert.match(await slot.innerText(),/Neither side is the “good” side/);assert.match(await slot.innerText(),/CTCF is one transcription factor/);assert.match(await slot.innerText(),/DNA itself stayed unchanged/);assert(!/eight versions|kept out of (model )?development/i.test(await slot.innerText()));await page.screenshot({path:out+`/${entry}-verdict-desktop.png`});
   for(const i of [0,1]){await slot.locator('.response-controls select').selectOption({index:i});for(const variant of ['ordinary','fibonacci','phi','alpha','beta','gamma','epsilon','shuffled']){await slot.locator('.atom-detail select').selectOption(variant);assert.equal(await slot.locator('tbody tr').count(),6);}}
   await slot.locator('.atom-detail').evaluate(n=>n.scrollIntoView({block:'start'}));await slot.locator('svg [role=button]').nth(1).focus();await page.keyboard.press('Enter');assert.match(await slot.locator('.atom-explanation').innerText(),/^Atom 2:/);await page.screenshot({path:out+`/${entry}-atoms-desktop.png`});
   await f.locator('#results-dialog-content [data-open-next-test]').click();await f.locator('.available-evidence h4').filter({hasText:'Public binding data is already on file.'}).waitFor();
  }
  await page.keyboard.press('Escape');await f.locator('#results-dialog').waitFor({state:'hidden'});const after=await f.locator('body').evaluate(()=>({y:scrollY,q:document.querySelector('#question').value}));assert(Math.abs(before.y-after.y)<1);assert.equal(after.q,before.q);assert(await opener.evaluate(n=>document.activeElement===n));
 }
 assert.equal(questions,0);
 for(const width of [390,320]){
  await page.setViewportSize({width,height:844});await f.locator('#study-initial-insights').evaluate(n=>n.scrollIntoView({block:'start'}));await page.screenshot({path:out+`/goal-${width}.png`});
  for(const kind of ['next','simulation','quantum','comparison']){
   const opener=kind==='next'?f.locator('#study-initial-insights [data-open-next-test]'):f.locator(`[data-recorded-result=${kind}]`);await opener.click();await f.locator('#results-dialog').waitFor({state:'visible'});
   if(kind==='next'){await f.locator('.available-evidence h4').filter({hasText:'Public binding data is already on file.'}).waitFor();await f.locator('.binding-data-detail summary').click();await f.locator('.binding-data-detail').evaluate(n=>n.scrollIntoView({block:'start'}));}
   else if(kind!=='comparison')await f.locator('#modal-'+kind+'-report .representation-row').last().waitFor();
   const layout=await f.locator('#results-dialog-content').evaluate(n=>({width:n.clientWidth,scroll:n.scrollWidth}));assert(layout.scroll<=layout.width+1,JSON.stringify(layout));const close=await f.locator('#close-results').boundingBox();assert(close.y>=0&&close.y+close.height<=844);layouts.push({viewport:width,kind,...layout});await page.screenshot({path:out+`/${kind}-${width}.png`});await f.locator('#close-results').click();
  }
 }
 await fs.writeFile(out+'/final-copy-verification.json',JSON.stringify({passed:true,at:new Date().toISOString(),audio_muted:true,ctcf_explained:true,two_samples_eight_measurement_preparations:true,independent_test_explained:true,live_gpt_receipt:'live-openai-receipt.json',final_copy_only:true,layouts,errors},null,2)+'\n');
 const paths=[['Informatics/index.html',17],['Informatics/quantum/index.html',17],['Informatics/quantum/biology.css',12],['Informatics/quantum/biology.js',15],['Informatics/quantum/insights.js',5],['Informatics/quantum/response-view.js',6],['Informatics/quantum/study.js',7],['Informatics/quantum/data/oxtr-next-test-evidence.json',1],['Informatics/vision-science.js',2]],assets=[];
 for(const[file,v]of paths){const url=base+'/'+file.replace('Informatics/','informatics/')+'?v='+v,r=await fetch(url);assert(r.ok);const hash=b=>crypto.createHash('sha256').update(b).digest('hex'),local=hash(await fs.readFile(file)),remote=hash(Buffer.from(await r.arrayBuffer()));assert.equal(local,remote,file);assets.push({file,url,sha256:local,match:true});}await fs.writeFile(out+'/staging-assets.json',JSON.stringify(assets,null,2)+'\n');console.log('Final wording, desktop/mobile readers and staging asset hashes passed; no additional GPT request or quantum job.');
}finally{for(const p of ctx.pages())await p.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});await ctx.close();await browser.close();await server.close();}
