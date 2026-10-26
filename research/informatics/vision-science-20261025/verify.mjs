import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const audit=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(audit,'../../..'),live=process.argv.includes('--live'),out=path.join(audit,live?'live':'local');await fs.mkdir(out,{recursive:true});
const expected=JSON.parse(await fs.readFile(path.join(audit,'source-paragraphs.json'),'utf8'));
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));const browser=await chromium.connect(server.wsEndpoint()),results=[];
try{
 for(const viewport of [{width:1470,height:956},{width:1024,height:768},{width:390,height:844},{width:320,height:844}]){
  const context=await browser.newContext({viewport,reducedMotion:'reduce'});
  await context.addInitScript(()=>{window.__fullscreenAttempts=0;Element.prototype.requestFullscreen=async function(){window.__fullscreenAttempts++;throw Error('Native fullscreen disabled in QA');};});
  if(!live)await context.route(/https:\/\/staging\.universalmind\.tv\/(informatics|Informatics)\//,async route=>{let rel=decodeURIComponent(new URL(route.request().url()).pathname.replace(/^\/(informatics|Informatics)\//,''))||'index.html';if(rel.endsWith('/'))rel+='index.html';const file=path.resolve(root,'Informatics',rel);if(!file.startsWith(path.join(root,'Informatics')+'/'))return route.abort();try{await route.fulfill({status:200,body:await fs.readFile(file),contentType:{'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml'}[path.extname(file)]||'application/octet-stream'});}catch{return route.continue();}});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  try{
   await page.goto('https://staging.universalmind.tv/informatics/?entry=research&area=vision',{waitUntil:'domcontentloaded'});
   await page.locator('#area-vision:not([hidden]) .vision-science').waitFor();
   assert.equal(await page.locator('.vision-science').count(),1);
   assert.equal(await page.locator('#area-vision').evaluate(e=>e.lastElementChild.id),'vision-science');
   const parity=await page.evaluate(expected=>{const clean=s=>s.replace(/\s+/g,' ').trim();return Object.entries(expected).map(([i,html])=>{const tmp=document.createElement('div');tmp.innerHTML=html;return{id:i,equal:clean(tmp.textContent)===clean(document.querySelector(`[data-source-paragraph="${i}"]`).textContent)};});},expected);assert.ok(parity.every(p=>p.equal));
   assert.deepEqual(await page.locator('.vision-comparison-panel figcaption').allTextContents(),['Saccadic · Hexagonal','Bilateral']);
   const allVisible=await page.locator('.vision-science').evaluate(e=>[e,...e.querySelectorAll('*')].filter(n=>n.getBoundingClientRect().width>0&&!n.closest('svg')).every(n=>{const r=n.getBoundingClientRect();return r.left>=-1&&r.right<=innerWidth+1}));assert.ok(allVisible,'Explainer stays within viewport');
   await page.locator('#vision-science-title').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,`intro-${viewport.width}.png`)});
   await page.locator('.vs-chapter-nav a[href="#vs-retina"]').click();await page.locator('#vs-retina-title:focus').waitFor();
   await page.locator('[data-vs-layer="triangles"]').click();assert.equal(await page.locator('[data-vs-triangles]').getAttribute('visibility'),'visible');
   await page.locator('[data-vs-layer="hexagons"]').focus();await page.keyboard.press('Enter');assert.equal(await page.locator('[data-vs-hexagons]').getAttribute('visibility'),'visible');assert.equal(await page.locator('[data-vs-triangles]').getAttribute('visibility'),'hidden');
   await page.screenshot({path:path.join(out,`retina-${viewport.width}.png`)});
   await page.locator('[data-vs-layer="cells"]').click();
   await page.locator('#vs-grid-rotation').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#vs-rotation-value').textContent(),'5.5°');assert.equal(await page.locator('#vs-cortex svg g[transform]').getAttribute('transform'),'rotate(5.5 220 200)');
   await page.locator('#vs-cortex').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,`cortex-${viewport.width}.png`)});
   await page.locator('#vs-space').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,`space-${viewport.width}.png`)});
   await page.locator('.vs-summary').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,`summary-${viewport.width}.png`)});
   const references=await page.locator('.vs-publication').evaluateAll(es=>es.map(e=>({url:e.href,target:e.target,rel:e.rel})));assert.equal(references.length,3);assert.ok(references.every(r=>r.url.startsWith('https://www.nature.com/articles/')&&r.target==='_blank'&&r.rel.includes('noopener')));
   await page.locator('.vs-colophon a').click();await page.locator('#vision-science-title:focus').waitFor();
   for(const area of ['sound','vision']){await page.locator('#atlas-toggle').click();await page.locator(`#atlas-menu [data-area="${area}"]`).click();await page.waitForFunction(a=>document.querySelector('#experience').dataset.area===a,area);}
   assert.equal(await page.locator('.vision-science').count(),1);assert.equal(await page.evaluate(()=>window.__fullscreenAttempts),0);assert.deepEqual(errors,[]);
   results.push({viewport,sourceParagraphsPreserved:parity.length,chapters:3,retinaControls:true,moireKeyboardControl:true,navigation:true,comparisonPreserved:true,scopeAndOverflow:true,references,errors});
  }finally{for(const frame of page.frames())await frame.evaluate(()=>document.querySelectorAll('video,audio').forEach(e=>e.pause())).catch(()=>{});await context.close();}
 }
 const report={passed:true,live,processAudioDisabled:true,nativeFullscreen:false,results};await fs.writeFile(path.join(out,'verification.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}finally{await browser.close().catch(()=>{});await server.close().catch(()=>{});}
