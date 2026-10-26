import fs from 'node:fs';
import assert from 'node:assert/strict';
import {installQAAuth,endpoint} from './qa-auth.mjs';
const {chromium}=await import('/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const output='/private/tmp/uo-followup-qa';
const report={syntheticHistoryStressTest:true,processAudioDisabled:true,requests:[]};
const browser=await chromium.launch({headless:true,args:['--mute-audio']});
const page=await browser.newPage({viewport:{width:1470,height:1000}});await installQAAuth(page);
page.on('response',response=>{if(response.url()===endpoint+'/research'){const body=response.request().postData();report.requests.push({status:response.status(),bytes:Buffer.byteLength(body),history:JSON.parse(body).history.length,references:JSON.parse(body).references.length});}});
async function ask(question){const before=await page.locator('.uo-message[data-role=assistant]').count();await page.locator('#uo-question').fill(question);await page.locator('[data-send]').click();await page.waitForFunction(n=>document.querySelectorAll('.uo-message[data-role=assistant]').length>n,before,{timeout:240000});assert(await page.locator('.uo-message[data-role=assistant]').last().locator('.uo-sources').count());return page.locator('.uo-message[data-role=assistant]').last().innerText();}
try{
 await page.goto('https://staging.universalmind.tv/informatics/?area=sound',{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.UniversalOneResearch&&window.UniversalOneContext);
 await page.locator('.uo .iam-realtime-voice__summon').click();
 report.initialAnswer=await ask('Find an indexed recreational music-making study involving gene expression. Give a title and one measured finding briefly.');
 report.fixture=await page.evaluate(()=>{const pair=structuredClone(UniversalOneConversation);for(let i=1;i<100;i++)UniversalOneConversation.push(...structuredClone(pair));return {simulatedExchanges:100,rawBytes:new TextEncoder().encode(JSON.stringify(UniversalOneConversation)).length,sourceIDs:pair.flatMap(m=>(m.sources||[]).map(s=>s.id))};});
 assert(report.fixture.rawBytes>50000);
 report.followupAnswer=await ask('Which genes were measured in that study? Keep the answer brief.');
 assert(report.requests.length===2&&report.requests.every(r=>r.status===200&&r.bytes<=44000));
 assert(report.requests[1].references>0);assert.equal(await page.locator('.uo-privacy').count(),0);
 report.retainedHistory=await page.evaluate(()=>UniversalOneConversation.length);assert(report.retainedHistory>=202);
 await page.waitForTimeout(700);await page.screenshot({path:output+'/history-stress-transcript.png'});
 await page.setViewportSize({width:390,height:844});await page.locator('.uo-message[data-role=assistant]').last().locator('.uo-sources').scrollIntoViewIfNeeded();await page.waitForTimeout(700);await page.screenshot({path:output+'/history-stress-mobile.png'});
 report.success=true;
}catch(error){report.failure=error.stack;throw error;}finally{await page.evaluate(()=>{UniversalOneResearch?.close();document.querySelectorAll('video,audio').forEach(m=>m.pause());}).catch(()=>{});await browser.close();report.browserClosed=true;fs.writeFileSync(output+'/history-stress.json',JSON.stringify(report,null,2));}
console.log(JSON.stringify({passed:true,rawBytes:report.fixture.rawBytes,sentBytes:report.requests.at(-1).bytes,historyRetained:report.retainedHistory}));
