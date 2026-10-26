import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const origin=process.argv[2]||'https://staging.universalmind.tv',out=process.argv[3],proof={origin,errors:[],muted:true,assets:[]};
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:1000},reducedMotion:'reduce'}),page=await ctx.newPage();page.on('pageerror',e=>proof.errors.push(e.message));page.on('response',r=>{if(/quantum\/(biology|study)\.js/.test(r.url()))proof.assets.push({url:r.url(),status:r.status()});});
try{
 await page.goto(origin+'/informatics/?area=quantum&entry=research');const f=page.frameLocator('#quantum-frame');await f.locator('#completed-study:not([hidden])').waitFor();await f.locator('.study-verdict').waitFor();
 const body=await f.locator('body').innerText();for(const text of ['New calculations: classical timing is an estimate','Your question is sent to OpenAI to select the calculation','STUDY 01 · 16 GPU simulations'])assert(!body.includes(text));assert.equal(await f.locator('#study-state-caption').count(),0);
 await page.screenshot({path:out+'/quantum-desktop.png'});
 for(const type of ['simulation','quantum','genomic','comparison']){await f.locator('[data-recorded-result='+type+']').click();await f.locator('#results-dialog[open]').waitFor();assert((await f.locator('#results-dialog-content').innerText()).length>100);await f.locator('#close-results').click();}
 await f.locator('[data-step="study"]').click();await f.locator('[data-step="ask"]').click();
 await page.setViewportSize({width:390,height:844});assert(!await f.locator('body').evaluate(n=>n.scrollWidth>innerWidth+1));await page.screenshot({path:out+'/quantum-mobile.png'});
 assert(proof.assets.some(r=>r.url.includes('biology.js?v=clean-3')&&r.status===200));assert(proof.assets.some(r=>r.url.includes('study.js?v=clean-3')&&r.status===200));assert.deepEqual(proof.errors,[]);proof.passed=true;proof.dialogs=4;
}finally{for(const frame of page.frames())await frame.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});await ctx.close();await browser.close();await server.close();await fs.writeFile(out+'/quantum-cleanup.json',JSON.stringify(proof,null,2)+'\n');}
console.log(JSON.stringify(proof));
