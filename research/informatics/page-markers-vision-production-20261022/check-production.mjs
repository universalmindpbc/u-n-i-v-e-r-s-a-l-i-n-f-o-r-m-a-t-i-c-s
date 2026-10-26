import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../Informatics');
const live=process.env.ATLAS_LIVE==='1',out=process.env.ATLAS_QA_OUT||'/private/tmp/atlas-page-markers-qa';
const base='https://universalmind.tv/informatics/';
await fs.mkdir(out,{recursive:true});
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),results=[];
async function choose(page,area){await page.locator('#atlas-toggle').click();await page.locator(`#atlas-menu [data-area="${area}"]`).click();await page.waitForFunction(a=>document.querySelector('#experience').dataset.area===a,area);}
async function marker(locator,text){await locator.waitFor();assert.equal((await locator.innerText()).trim().toLowerCase(),text.toLowerCase());assert.equal(await locator.evaluate(e=>getComputedStyle(e).textTransform),'uppercase');}
try{
 for(const viewport of [{width:1470,height:956},{width:1024,height:768},{width:390,height:844},{width:320,height:844}]){
  const context=await browser.newContext({viewport,reducedMotion:'reduce'});
  await context.addInitScript(()=>{let full=null;Object.defineProperty(document,'fullscreenElement',{get:()=>full});Element.prototype.requestFullscreen=async function(){full=this;document.dispatchEvent(new Event('fullscreenchange'));};document.exitFullscreen=async()=>{full=null;document.dispatchEvent(new Event('fullscreenchange'));};});
  if(!live)await context.route(/https:\/\/staging\.universalmind\.tv\/(informatics|Informatics)\//,async route=>{const rel=decodeURIComponent(new URL(route.request().url()).pathname.replace(/^\/(informatics|Informatics)\//,''))||'index.html',file=path.resolve(root,rel.endsWith('/')?rel+'index.html':rel);if(!file.startsWith(root+'/'))return route.abort();try{await route.fulfill({status:200,body:await fs.readFile(file),contentType:{'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png'}[path.extname(file)]||'application/octet-stream'});}catch{await route.fulfill({status:404,body:'Missing'});}});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  try{
   await page.goto(base+'?area=geometry');await page.waitForFunction(()=>!document.querySelector('#launch').disabled);await page.locator('#launch').click();
   const geometry=page.frameLocator('#atlas-frame');await marker(geometry.locator('main>.atlas-page-marker'),'Geometry Atlas');
   const geoMarker=await geometry.locator('main>.atlas-page-marker').boundingBox(),geoEyebrow=await geometry.locator('.intro .eyebrow').boundingBox();assert.ok(geoMarker.y+geoMarker.height<geoEyebrow.y);
   if(viewport.width===1470)await page.screenshot({path:path.join(out,'geometry-1470.png')});
   for(const [area,title] of [['sound','Sound Atlas'],['language','Language Atlas'],['partners','Partners']]){await choose(page,area);await marker(page.locator(`#area-${area}>.atlas-page-marker`),title);if(viewport.width===1470)await page.screenshot({path:path.join(out,`${area}-1470.png`)});}
   await choose(page,'vision');await marker(page.locator('#area-vision>.atlas-page-marker'),'Vision Atlas');
   assert.deepEqual(await page.locator('.vision-comparison-panel figcaption').allTextContents(),['Saccadic · Hexagonal','Bilateral']);
   const boxes=await page.locator('.vision-comparison-panel').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return{x:r.x,y:r.y,w:r.width,h:r.height};}));
   assert.ok(boxes[0].x+boxes[0].w<=boxes[1].x&&Math.abs(boxes[0].y-boxes[1].y)<1,'Comparison panels side by side');
   const bottom=await page.locator('.vision-comparison-controls').evaluate(e=>e.getBoundingClientRect().bottom);assert.ok(bottom<=viewport.height,`Comparison above fold: ${bottom}/${viewport.height}`);
   assert.equal(await page.locator('#area-vision').getAttribute('data-motion'),'paused');
   const dot=page.locator('.vision-bilateral-dot');assert.equal(await dot.evaluate(e=>e.getAnimations().length),0);
   await page.locator('#vision-motion').click();await page.waitForFunction(()=>document.querySelector('.vision-bilateral-dot').getAnimations().some(a=>a.playState==='running'));
   const first=await dot.evaluate(e=>getComputedStyle(e).transform);await page.waitForFunction(t=>getComputedStyle(document.querySelector('.vision-bilateral-dot')).transform!==t,first);
   assert.equal(await dot.getAttribute('cy'),'330');assert.equal(await dot.getAttribute('fill'),'#62d998');
   const endpoints=await dot.evaluate(e=>{const a=e.getAnimations()[0];a.pause();return[0,1400,2800].map(t=>{a.currentTime=t;return new DOMMatrixReadOnly(getComputedStyle(e).transform).m41;});});assert.deepEqual(endpoints,[0,480,0]);
   await page.locator('#vision-motion').click();assert.equal(await page.locator('#area-vision').getAttribute('data-motion'),'paused');
   const paused=await dot.evaluate(e=>getComputedStyle(e).transform);await page.waitForTimeout(180);assert.equal(await dot.evaluate(e=>getComputedStyle(e).transform),paused);
   await page.screenshot({path:path.join(out,`vision-${viewport.width}.png`)});
   await page.emulateMedia({reducedMotion:'no-preference'});await page.waitForFunction(()=>document.querySelector('#area-vision').dataset.motion==='playing');
   await choose(page,'sound');assert.equal(await dot.evaluate(e=>e.getAnimations().length),0);
   await choose(page,'molecular');const molecular=page.frameLocator('#molecular-frame');await molecular.locator('body[data-ready=true]').waitFor({timeout:60000});await marker(molecular.locator('.product-name'),'Molecular Atlas');
   await page.locator('#atlas-toggle').click();await page.locator('#atlas-menu [data-science-tfbs]').click();await page.locator('.science-tfbs[open] .tfbs-source-card').first().waitFor({timeout:60000});await marker(page.locator('.science-tfbs[open] .atlas-page-marker'),'Transcription Atlas');
   const tfbsMarker=await page.locator('.science-tfbs .atlas-page-marker').boundingBox(),regulation=await page.locator('.science-reader-location>span').boundingBox();assert.ok(tfbsMarker.y+tfbsMarker.height<=regulation.y);
   if(viewport.width===1470)await page.screenshot({path:path.join(out,'transcription-1470.png')});
   await page.locator('.science-tfbs .science-close').click();await page.locator('#atlas-toggle:focus').waitFor();
   await page.locator('#atlas-toggle').click();await page.locator('#atlas-menu [data-about]').click();await marker(page.locator('#about>.atlas-page-marker'),'About');await page.locator('#about-close').click();
   await page.locator('#atlas-toggle').click();await page.locator('#atlas-menu [data-library]').click();await marker(page.frameLocator('.library-overlay iframe').locator('.library-intro>.atlas-page-marker'),'Universal Research');
   await page.locator('.library-overlay [aria-label="Close Universal Research"]').click();
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);assert.deepEqual(errors,[]);
   results.push({viewport,allNineGlobalDestinationsLabelled:true,geometryMarkerAboveSequence:true,transcriptionMarkerAboveRegulation:true,comparisonSideBySide:true,comparisonBottom:bottom,aboveFold:true,bilateralGreenDot:true,animatedHorizontalMovement:true,pauseAndReducedMotion:true,noAnimationInHiddenSection:true,noPageErrors:true});
  }finally{for(const frame of page.frames())await frame.evaluate(()=>document.querySelectorAll('video,audio').forEach(v=>v.pause())).catch(()=>{});await context.close();}
 }
 const report={passed:true,live,processMuted:true,nativeFullscreen:false,results};await fs.writeFile(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}finally{await browser.close().catch(()=>{});await server.close().catch(()=>{});}
