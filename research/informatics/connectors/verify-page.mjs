import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {installQAAuth,endpoint,origin} from '../universal-one/qa-auth.mjs';
const {chromium}=await import('/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const out=new URL(process.env.UO_PROOF_DIR||'./verification/',import.meta.url);
await fs.mkdir(out,{recursive:true});
const report={checkedAt:new Date().toISOString(),origin,endpoint,processAudioDisabled:true,layouts:[],consoleErrors:[],researchResponses:[]};
const browser=await chromium.launch({headless:true,args:['--mute-audio']});
const page=await browser.newPage({viewport:{width:1470,height:1000}});await installQAAuth(page);
page.on('pageerror',e=>report.consoleErrors.push(e.message));
page.on('response',r=>{if(r.url()===endpoint+'/research')report.researchResponses.push({status:r.status()});});
try{
 await page.goto(origin+'/informatics/connect.html',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>document.querySelector('#inventory')?.textContent.includes('811'));
 assert.equal(await page.locator('.provider-card').count(),9);assert.equal(await page.locator('#personal-agents article').count(),4);
 await page.locator('#test-mcp').click();await page.waitForFunction(()=>document.querySelector('#mcp-result').textContent.includes('Endpoint verified'),null,{timeout:30000});report.mcpTest=await page.locator('#mcp-result').innerText();
 await page.locator('summary').filter({hasText:'Try a research query'}).click();await page.locator('#api-query-input').fill('music gene expression');await page.locator('#api-query button').click();await page.waitForFunction(()=>document.querySelectorAll('#api-results article').length>0,null,{timeout:30000});report.query=await page.locator('#api-result-status').innerText();
 await page.locator('.provider-card button').first().click();assert.ok(await page.locator('#connection-guide').evaluate(d=>d.open));await page.keyboard.press('Escape');assert.ok(!await page.locator('#connection-guide').evaluate(d=>d.open));report.dialogKeyboard=true;
 for(const [width,height] of [[1470,1000],[1024,900],[390,844],[320,740]]){
  await page.setViewportSize({width,height});await page.evaluate(()=>scrollTo(0,0));await page.waitForTimeout(300);
  const dims=await page.evaluate(()=>({viewport:innerWidth,scroll:document.documentElement.scrollWidth}));assert.ok(dims.scroll<=width+1,JSON.stringify(dims));
  report.layouts.push({width,height,...dims});await page.screenshot({path:new URL(`connect-${width}.png`,out).pathname});
 }
 await page.setViewportSize({width:1470,height:1000});await page.locator('[data-reading-background=white]').click();await page.screenshot({path:new URL('connect-white.png',out).pathname});assert.equal(await page.locator('html').getAttribute('data-reading-background'),'white');await page.locator('[data-reading-background=video]').click();
 const downloadPromise=page.waitForEvent('download');await page.locator('[data-download="publications.json"]').click();const download=await downloadPromise;const file='/private/tmp/informatics-browser-publications.json';await download.saveAs(file);const payload=JSON.parse(await fs.readFile(file));assert.equal(payload.records.length,811);report.jsonDownload=payload.records.length;
 await page.locator('.uo .iam-realtime-voice__summon').click();await page.locator('#uo-question').fill('Which indexed studies link music making to gene expression? Name one study, its measured finding and a limitation in three sentences.');await page.locator('[data-send]').click();await page.waitForSelector('.uo-message[data-role=assistant] .uo-sources',{timeout:240000});report.answer=await page.locator('.uo-message[data-role=assistant]').last().innerText();assert.ok(report.researchResponses.some(x=>x.status===200));await page.screenshot({path:new URL('connect-research-answer.png',out).pathname});await page.evaluate(()=>UniversalOneResearch.close());
 // Exercise the real main navigation and its existing lightbox.
 await page.goto(origin+'/informatics/?area=geometry&entry=research',{waitUntil:'domcontentloaded'});
 await page.locator('#atlas-toggle').click();await page.locator('[data-connect]').click();
 const frame=page.frameLocator('.library-overlay iframe');await frame.locator('#inventory').filter({hasText:'811'}).waitFor({timeout:30000});assert.equal(await page.locator('#library-overlay-title').innerText(),'CONNECT YOUR AI');report.embeddedNavigation=true;
 await page.screenshot({path:new URL('connect-lightbox.png',out).pathname});
 assert.deepEqual(report.consoleErrors,[]);report.passed=true;
}catch(e){report.error=e.stack;throw e;}finally{for(const p of browser.contexts().flatMap(c=>c.pages()))await p.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause())).catch(()=>{});await browser.close();report.browserClosed=true;await fs.writeFile(new URL('page.json',out),JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({passed:report.passed,layouts:report.layouts,download:report.jsonDownload,embeddedNavigation:report.embeddedNavigation,researchResponses:report.researchResponses}));
