import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const origin=process.env.QA_ORIGIN||'https://staging.universalmind.tv',out=process.env.QA_OUT||new URL('./',import.meta.url).pathname;
const proof={origin,errors:[],assets:[],muted:true};
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:1000},reducedMotion:'reduce'}),page=await ctx.newPage();
page.on('pageerror',e=>proof.errors.push(e.message));
const assetChecks=[];
page.on('response',r=>{const name=new URL(r.url()).pathname.split('/').at(-1);if(['biology.js','biology.css','scientific-review.js','scientific-review.css','study.js','study-progress.js','insights.js','agent.js'].includes(name))assetChecks.push((async()=>{const body=await r.body(),local=await fs.readFile('Informatics/quantum/'+name);const record={url:r.url(),status:r.status(),sha256:createHash('sha256').update(body).digest('hex'),matches_source:body.equals(local)};proof.assets.push(record);})());});
try{
 await page.goto(origin+'/informatics/?area=quantum&entry=research');const f=page.frameLocator('#quantum-frame');
 await f.locator('#completed-study:not([hidden])').waitFor();await f.locator('#recorded-review-stamp button').waitFor();
 assert((await f.locator('.stage-state').allTextContents()).every(x=>!x));assert.equal(await f.locator('#workflow-state').innerText(),'');
 assert.equal(await f.locator('#classical-step-status').innerText(),'');assert.equal(await f.locator('#quantum-step-status').innerText(),'');
 assert(await f.locator('#scientific-review-rail [data-scientific-review=final]').isDisabled());
 assert(!/152|16|passed|retrospective|ready/i.test(await f.locator('#workflow').innerText()));
 await f.locator('#scientific-review-rail').scrollIntoViewIfNeeded();await page.screenshot({path:out+'fresh-desktop.png'});
 const paths=await f.locator('#review-connectors>path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d')));assert(paths.every(Boolean));proof.connectorPaths=paths;
 for(const type of ['simulation','quantum','genomic','comparison']){await f.locator('[data-recorded-result='+type+']').click();await f.locator('#results-dialog[open]').waitFor();assert((await f.locator('#results-dialog-content').innerText()).length>100);await f.locator('#close-results').click();assert.equal(await f.locator('#workflow-state').innerText(),'');}
 await f.locator('#recorded-review-stamp button').click();await f.locator('#scientific-review-dialog[open]').waitFor();assert.match(await f.locator('#scientific-review-content').innerText(),/saved OXTR study/);await page.keyboard.press('Escape');
 for(const width of [390,320]){await page.setViewportSize({width,height:844});await f.locator('#scientific-review-rail').scrollIntoViewIfNeeded();assert(!await f.locator('body').evaluate(n=>n.scrollWidth>innerWidth+1));await page.screenshot({path:out+`fresh-${width}.png`});}
 // Explicit UI fixture: an active investigation never inherits the public review.
 await f.locator('body').evaluate(()=>window.dispatchEvent(new CustomEvent('quantum:agent',{detail:{id:'fixture-only',status:'running',stage:'design_review',events:[{stage:'model'}],reviews:[]}})));
 assert.equal(await f.locator('#design-review-status').innerText(),'Reviewing…');assert.equal(await f.locator('#final-review-status').innerText(),'');
 assert(await f.locator('#scientific-review-rail [data-scientific-review=final]').isDisabled());
 await f.locator('body').evaluate(()=>{window.dispatchEvent(new CustomEvent('quantum:agent',{detail:null}));window.dispatchEvent(new CustomEvent('quantum:question',{detail:{state:'ready'}}));});
 assert((await f.locator('.stage-state').allTextContents()).every(x=>!x));assert.equal(await f.locator('#workflow-state').innerText(),'');
 await Promise.all(assetChecks);for(const asset of proof.assets){assert.equal(asset.matches_source,true,asset.url);assert.equal(asset.status,200,asset.url);}assert.equal(proof.assets.filter(r=>r.url.includes('/agent.js?')).length,1);assert.deepEqual(proof.errors,[]);proof.passed=true;await fs.writeFile(out+'ui-proof.json',JSON.stringify(proof,null,2)+'\n');console.log(JSON.stringify(proof));
}finally{for(const p of ctx.pages())for(const fr of p.frames())await fr.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause())).catch(()=>{});await browser.close();await server.close();}
