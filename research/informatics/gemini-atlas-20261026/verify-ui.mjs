import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const origin='https://staging.universalmind.tv',out=new URL('./',import.meta.url).pathname;
const proof={origin,errors:[],assets:[],muted:true};
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:1000},reducedMotion:'reduce'}),page=await ctx.newPage();
page.on('pageerror',e=>proof.errors.push(e.message));
page.on('response',r=>{if(/scientific-review\.(js|css)/.test(r.url()))proof.assets.push({url:r.url(),status:r.status()});});
async function openPage(p){await p.goto(origin+'/informatics/?area=quantum&entry=research');const f=p.frameLocator('#quantum-frame');await f.locator('#completed-study:not([hidden])').waitFor();return f;}
try{
 const f=await openPage(page);
 await f.locator('#final-review-status').filter({hasText:'Review passed'}).waitFor();
 assert.match(await f.locator('.live-flow').innerText(),/GPT-6 Astra/);
 assert.match(await f.locator('#recorded-review-stamp').innerText(),/26 Oct 2026.*11:31 am.*AEDT/);
 await f.locator('#scientific-review-rail').scrollIntoViewIfNeeded();
 await page.screenshot({path:out+'diagram-desktop.png'});
 for(const phase of ['design','final']){
  await f.locator(`#scientific-review-rail [data-scientific-review="${phase}"]`).click();
  await f.locator('#scientific-review-dialog[open]').waitFor();
  const text=await f.locator('#scientific-review-content').innerText();
  assert(text.includes(phase==='design'?'after the original calculations':'no clear improvement'));
  assert(text.includes('Review passed.'));
  if(phase==='final')await page.screenshot({path:out+'review-desktop.png'});
  await page.keyboard.press('Escape');
  assert.equal(await f.locator('#scientific-review-dialog').evaluate(n=>n.open),false);
  assert.equal(await f.locator(`[data-scientific-review="${phase}"]`).first().evaluate(n=>n===document.activeElement),true);
 }
 await f.locator('#recorded-review-stamp button').click();
 await f.locator('.result-reader-nav button').filter({hasText:'Study design'}).click();
 await f.locator('.result-reader-nav button').filter({hasText:'Findings'}).click();
 await f.locator('#close-scientific-review').click();
 for(const type of ['simulation','quantum','genomic','comparison']){
  await f.locator('[data-recorded-result='+type+']').click();
  await f.locator('#results-dialog[open]').waitFor();
  assert((await f.locator('#results-dialog-content').innerText()).length>100);
  await f.locator('#close-results').click();
 }
 for(const width of [390,320]){
  await page.setViewportSize({width,height:844});
  await f.locator('#scientific-review-rail').scrollIntoViewIfNeeded();
  assert(!await f.locator('body').evaluate(n=>n.scrollWidth>innerWidth+1));
  await page.screenshot({path:out+`diagram-${width}.png`});
  await f.locator('#scientific-review-rail [data-scientific-review="final"]').click();
  assert(!await f.locator('#scientific-review-content').evaluate(n=>n.scrollWidth>n.clientWidth+1));
  await page.screenshot({path:out+`review-${width}.png`});
  await f.locator('#close-scientific-review').click();
 }
 await f.locator('body').evaluate(()=>window.dispatchEvent(new CustomEvent('quantum:agent',{detail:{id:'new-study',status:'running'}})));
 assert.equal(await f.locator('#final-review-status').innerText(),'Not yet reviewed');
 assert(await f.locator('#scientific-review-rail [data-scientific-review="final"]').isDisabled());
 assert.match(await f.locator('#recorded-review-stamp').innerText(),/Reviewed by Gemini/);
 const fail=await ctx.newPage();
 await fail.route('**/reviews/oxtr-study-01/r1/review.json',async route=>{
  const r=await route.fetch(),data=await r.json();data.study_files[0].sha256='0'.repeat(64);await route.fulfill({response:r,json:data});
 });
 const broken=await openPage(fail);
 await broken.locator('#final-review-status').filter({hasText:'Review unavailable'}).waitFor();
 assert(await broken.locator('#scientific-review-rail [data-scientific-review="final"]').isDisabled());
 await fail.close();
 for(const format of ['json','csv']){
  const path=`Informatics/quantum/reviews/oxtr-study-01/r1/review.${format}`;
  const r=await ctx.request.get(origin+'/'+path.replace('Informatics','informatics'));
  assert(r.ok());assert.deepEqual(await r.body(),await fs.readFile(new URL('../../../'+path,import.meta.url)));
 }
 assert(proof.assets.every(x=>x.status===200));assert.deepEqual(proof.errors,[]);
 proof.passed=true;proof.checks=['desktop and mobile layout','design and findings review','correct timestamp','close Escape and focus return','four existing results dialogs','new agent task cannot inherit saved approval','changed evidence fails closed','JSON and CSV download equality'];
}finally{
 for(const p of ctx.pages())for(const frame of p.frames())await frame.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});
 await ctx.close();await browser.close();await server.close();await fs.writeFile(out+'ui-verification.json',JSON.stringify(proof,null,2)+'\n');
}
console.log(JSON.stringify(proof));
