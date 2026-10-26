import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const live=process.env.QUANTUM_LIVE==='1',out=process.env.QUANTUM_QA_OUT||'/private/tmp/quantum-lab-candidate';
await fs.mkdir(out,{recursive:true});
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),records=[];
async function candidate(ctx){
 if(live)return;
 for(const prefix of ['/informatics/','/Informatics/'])await ctx.route('https://staging.universalmind.tv'+prefix+'**',async route=>{
  const url=new URL(route.request().url());let rel=decodeURIComponent(url.pathname.slice(prefix.length));if(!rel||rel.endsWith('/'))rel+='index.html';
  const full=path.resolve(root,'Informatics',rel);if(!full.startsWith(root+'/Informatics/'))return route.abort();
  try{const body=await fs.readFile(full);await route.fulfill({body,contentType:{'.html':'text/html','.css':'text/css','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.png':'image/png','.svg':'image/svg+xml','.py':'text/plain'}[path.extname(rel)]||'application/octet-stream'});}catch{await route.fulfill({status:404,body:'Missing candidate asset'});}
 });
}
try{
 for(const viewport of [{width:1470,height:956},{width:1024,height:768},{width:390,height:844},{width:320,height:844}]){
  const ctx=await browser.newContext({viewport,reducedMotion:'reduce',acceptDownloads:true});await candidate(ctx);
  const page=await ctx.newPage(),errors=[],badRequests=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()!=='GET'&&new URL(r.url()).pathname!=='/cdn-cgi/rum')badRequests.push(r.url());});
  await page.goto('https://staging.universalmind.tv/informatics/quantum/');
  await page.locator('#run-status').filter({hasText:'Reference ready'}).waitFor();await page.locator('#papers article').nth(4).waitFor();
  assert.equal(await page.locator('#export').isEnabled(),true);
  assert.ok(await page.locator('#atom-a-ring').evaluate(n=>parseFloat(getComputedStyle(n).strokeDasharray))>.48);
  assert.match(await page.locator('#outcomes').innerText(),/9,812/);assert.match(await page.locator('#outcomes').innerText(),/9,507/);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.screenshot({path:out+'/workspace-'+viewport.width+'.png',fullPage:false});
  await page.locator('#time').fill('0');assert.match(await page.locator('#atom-a-prob').textContent(),/0.00%/);assert.equal(await page.locator('#gpu-key').isVisible(),false);
  const max=await page.locator('#time').getAttribute('max');await page.locator('#time').fill(max);assert.equal(await page.locator('#gpu-key').isVisible(),true);
  await page.locator('#distance_um').fill('7');assert.equal(await page.locator('#export').isEnabled(),false);await page.locator('#run').click();await page.locator('#run-status').filter({hasText:/Run .+ complete/}).waitFor();
  assert.equal(await page.locator('#history-count').textContent(),'1');assert.equal(await page.locator('#gpu-key').isVisible(),false);assert.match(await page.locator('#distance-label').textContent(),/7.00/);
  const [dl]=await Promise.all([page.waitForEvent('download'),page.locator('#export').click()]);const file=out+'/run-'+viewport.width+'.json';await dl.saveAs(file);const run=JSON.parse(await fs.readFile(file,'utf8'));assert.equal(run.spec.distance_um,7);assert.equal(run.execution,'browser-cpu');assert.ok(run.result.convergence<1e-6);
  const [figure]=await Promise.all([page.waitForEvent('download'),page.locator('#export-figure').click()]);await figure.saveAs(out+'/figure-'+viewport.width+'.svg');assert.match(await fs.readFile(out+'/figure-'+viewport.width+'.svg','utf8'),/No matching GPU or QPU/);
  await page.locator('#rerun').click();await page.locator('#history-count').filter({hasText:'2'}).waitFor();
  const [repeatDL]=await Promise.all([page.waitForEvent('download'),page.locator('#export').click()]);await repeatDL.saveAs(out+'/repeat-'+viewport.width+'.json');const repeat=JSON.parse(await fs.readFile(out+'/repeat-'+viewport.width+'.json','utf8'));assert.notEqual(run.run_id,repeat.run_id);assert.equal(repeat.parent_run_id,run.run_id);assert.deepEqual(repeat.result.probabilities,run.result.probabilities);
  await page.reload();await page.locator('#run-status').filter({hasText:'Reference ready'}).waitFor();assert.equal(await page.locator('#history-count').textContent(),'2');
  await page.locator('#import').setInputFiles(file);await page.locator('#draft-status').filter({hasText:'Imported parameters only'}).waitFor();assert.equal(await page.locator('#export').isEnabled(),false);await page.locator('#run').click();await page.locator('#history-count').filter({hasText:'3'}).waitFor();
  await page.locator('[data-stage="qbraid"]').click();assert.match(await page.locator('#stage-detail').innerText(),/exported circuit/);
  await page.locator('#workflow').scrollIntoViewIfNeeded();await page.screenshot({path:out+'/workflow-'+viewport.width+'.png'});
  await page.locator('#shots').fill('100');const [hw]=await Promise.all([page.waitForEvent('download'),page.locator('#prepare-request').click()]);await hw.saveAs(out+'/review-'+viewport.width+'.json');const request=JSON.parse(await fs.readFile(out+'/review-'+viewport.width+'.json','utf8'));assert.equal(request.status,'LOCAL_DRAFT_NOT_SUBMITTED');assert.equal(request.requested_shots,100);
  assert.match(await page.locator('#request-status').innerText(),/No job was submitted/);
  await page.locator('#import').setInputFiles({name:'bad.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({model_version:'invalid',spec:run.spec}))});await page.locator('#draft-status').filter({hasText:'Import failed'}).waitFor();
  await page.locator('#history-panel summary').click();await page.locator('#clear-history').click();assert.equal(await page.locator('#history-count').textContent(),'0');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);assert.deepEqual(errors,[]);assert.deepEqual(badRequests,[]);
  records.push({viewport,freeSimulation:true,distinctReruns:true,reproducibleImport:true,persistentLocalHistory:true,reviewDownloadOnly:true,noOverflow:true,errors});await ctx.close();
 }
 const ctx=await browser.newContext({viewport:{width:1470,height:956},reducedMotion:'reduce'});await candidate(ctx);const page=await ctx.newPage();
 await page.goto('https://staging.universalmind.tv/informatics/?entry=research&area=quantum');
 await page.locator('#quantum-panel').waitFor({state:'visible'});const frame=page.frameLocator('#quantum-frame');await frame.locator('#run-status').filter({hasText:'Reference ready'}).waitFor();
 const labels=await page.locator('#atlas-menu [role=menuitem]').allTextContents();assert.equal(labels[labels.indexOf('Transcription Atlas')+1],'Quantum Lab');assert.equal(labels[labels.indexOf('Quantum Lab')+1],'Molecular Atlas');
 assert.equal(await page.evaluate(()=>document.fullscreenElement===null),true);assert.equal(await page.locator('#record-navigation').isVisible(),false);
 await page.screenshot({path:out+'/embedded-1470.png'});
 await page.locator('#atlas-toggle').click();await page.locator('#atlas-menu [data-area=vision]').click();assert.equal(await page.locator('#quantum-panel').isVisible(),false);assert.equal(await page.locator('#area-vision').isVisible(),true);
 await page.locator('#atlas-toggle').click();await page.locator('#atlas-menu [data-area=quantum]').click();assert.equal(await frame.locator('#lab-title').isVisible(),true);
 // Keyboard navigation uses the shared menu and retains focus after Escape.
 await page.locator('#atlas-menu-toggle').focus();await page.keyboard.press('ArrowDown');await page.keyboard.press('Escape');assert.equal(await page.locator('#atlas-menu').isVisible(),false);
 assert.equal(await page.evaluate(()=>document.activeElement.id),'atlas-menu-toggle');
 await page.evaluate(()=>document.querySelectorAll('audio,video').forEach(x=>x.pause()));await ctx.close();
 await fs.writeFile(out+'/verification.json',JSON.stringify({live,processMuted:true,embeddedNavigation:true,keyboardMenu:true,records},null,2));console.log(JSON.stringify({passed:true,live,viewports:4,out}));
}finally{for(const ctx of browser.contexts()){for(const p of ctx.pages())await p.evaluate(()=>document.querySelectorAll('audio,video').forEach(x=>x.pause())).catch(()=>{});await ctx.close();}await browser.close();await server.close();}
