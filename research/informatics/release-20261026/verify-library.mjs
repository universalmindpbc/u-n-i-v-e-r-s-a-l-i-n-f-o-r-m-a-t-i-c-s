import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
import {installQAAuth,endpoint,origin} from '../universal-one/qa-auth.mjs';

const out='research/informatics/release-20261026';
const records=JSON.parse(await fs.readFile('research/informatics/geometry-literature-20261026/records.json','utf8')).records;
const proof={started:new Date().toISOString(),processAudioDisabled:true,errors:[],publications:[],memory:[],answers:[]};
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),context=await browser.newContext({viewport:{width:1470,height:1000},acceptDownloads:true,reducedMotion:'reduce'}),page=await context.newPage();
await installQAAuth(page);page.on('pageerror',e=>proof.errors.push(e.message));
const checkpoint=()=>fs.writeFile(out+'/live.json',JSON.stringify(proof,null,2)+'\n');
async function responseJSON(response){assert.equal(response.status(),200,await response.text());return response.json();}
async function ask(question,expectedIds,file){
 const count=await page.locator('.uo-message[data-role=assistant]').count();
 await page.locator('#uo-question').fill(question);
 const pending=page.waitForResponse(r=>r.url()===endpoint+'/research'&&r.request().method()==='POST',{timeout:240000});
 await page.locator('[data-send]').click();const data=await responseJSON(await pending);
 await page.waitForFunction(n=>document.querySelectorAll('.uo-message[data-role=assistant]').length>n,count,{timeout:30000});
 const message=page.locator('.uo-message[data-role=assistant]').last();
 assert(await message.locator('.uo-sources').count());
 const ids=data.sources.map(s=>s.id||s.publicationId||s.publication_id);
 for(const id of expectedIds)assert(ids.includes(id),JSON.stringify({expected:id,sources:data.sources}));
 const visible=await message.innerText();assert(visible.length>100);
 proof.answers.push({question,model:data.model,version:data.version,sources:data.sources,answer:visible,http:200});
 await message.scrollIntoViewIfNeeded();await page.screenshot({path:out+'/'+file+'.png'});await checkpoint();
}
try{
 const headers={Origin:origin};
 const inventory=await responseJSON(await context.request.get(endpoint+'/api/v1/inventory',{headers}));
 assert.equal(inventory.counts.publications,821);assert.equal(inventory.datasetVersion,'research-20261026-geometry-v1');proof.inventory=inventory;
 for(const record of records){const result=await responseJSON(await context.request.get(endpoint+'/api/v1/publications/'+record.id,{headers}));assert.equal(result.record.title,record.title);assert.equal(result.record.doi,record.doi);assert(result.record.summary&&result.record.limitation);proof.publications.push({id:record.id,title:result.record.title,doi:result.record.doi});}
 const mcp=await responseJSON(await context.request.post(endpoint+'/mcp',{headers:{...headers,'Content-Type':'application/json'},data:{jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'get_publication',arguments:{id:'quantum-penrose-fibonacci-code-2024'}}}}));
 assert.equal(mcp.result.structuredContent.record.id,'quantum-penrose-fibonacci-code-2024');proof.mcp=true;
 await page.goto(origin+'/informatics/library.html?collection=vision&chapter=6',{waitUntil:'domcontentloaded'});
 const f=page.frameLocator('#research-frame');await f.locator('#collection-statistics:not([hidden])').waitFor({timeout:60000});
 assert.match(await f.locator('.chapter-opening').innerText(),/Geometry of vision/);
 for(const record of records.filter(r=>r.collections.includes('vision')))assert((await f.locator('.chapter-reader').innerText()).includes(record.title));
 await f.locator('.chapter-opening').scrollIntoViewIfNeeded();await page.screenshot({path:out+'/vision-desktop.png'});
 await f.locator('[data-paper-detail]').first().click();await f.locator('#article[open]').waitFor();assert.match(await f.locator('#article-body').innerText(),/photoreceptor/i);await page.screenshot({path:out+'/vision-paper.png'});await f.locator('#article [data-close]').click();
 await f.locator('[data-collection=general]').click();await f.locator('[data-chapter="10"]').first().click();await f.locator('.chapter-opening').scrollIntoViewIfNeeded();assert.match(await f.locator('.chapter-opening').innerText(),/Quantum Computing/);
 const chapter=(await f.locator('.chapter-reader').innerText()).replace(/\s+/g,' ');for(const record of records.filter(r=>r.collections.includes('general')))assert(chapter.includes(record.title.replace(/\s+/g,' ')),record.title);assert(chapter.includes('Dynamical topological phase realized'));
 await page.screenshot({path:out+'/quantum-desktop.png'});await page.setViewportSize({width:390,height:844});await page.screenshot({path:out+'/quantum-mobile.png'});assert(!await f.locator('body').evaluate(n=>n.scrollWidth>innerWidth+1));await page.setViewportSize({width:1470,height:1000});
 await f.locator('[data-collection=all]').click();await f.locator('#export-json:not([disabled])').waitFor();
 for(const format of ['json','csv']){const pending=page.waitForEvent('download');await f.locator('#export-'+format).click();await (await pending).saveAs(out+'/library-download.'+format);}
 const downloaded=JSON.parse(await fs.readFile(out+'/library-download.json','utf8'));assert.equal(downloaded.records.length,821);for(const record of records)assert(downloaded.records.some(r=>r.id===record.id&&r.title===record.title));proof.libraryDownloads={jsonRecords:downloaded.records.length,newRecordsIncluded:records.length,csvSaved:true};
 await page.locator('.uo .iam-realtime-voice__summon').click();
 await ask('Using the papers in our repository, explain what Hexadirectional coding of visual space in human entorhinal cortex found, and what links to M9 P24 remain hypotheses. Cite the paper.',['vision-hexadirectional-2018'],'universal-one-vision');
 await page.locator('[data-clear]').click();
 await ask('In our research repository, explain The Penrose Tiling is a Quantum Error-Correcting Code and its Fibonacci tiling variants. State its publication status and compare it with Dynamically Generated Logical Qubits. Cite both.',['quantum-penrose-fibonacci-code-2024','quantum-honeycomb-hastings-haah-2021'],'universal-one-quantum');
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:out+'/universal-one-mobile.png'});
 assert.deepEqual(proof.errors,[]);proof.passed=true;
}catch(error){proof.error=error.stack;await page.screenshot({path:out+'/failure.png'}).catch(()=>{});throw error;}
finally{for(const frame of page.frames())await frame.evaluate(()=>document.querySelectorAll('audio,video').forEach(m=>m.pause())).catch(()=>{});await context.close();await browser.close();await server.close();proof.browserClosed=true;await checkpoint();}
console.log(JSON.stringify({passed:proof.passed,records:proof.publications.length,answers:proof.answers.length,mcp:proof.mcp}));
