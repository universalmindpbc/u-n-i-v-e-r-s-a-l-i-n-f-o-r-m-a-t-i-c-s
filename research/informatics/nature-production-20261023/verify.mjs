import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=new URL('.',import.meta.url).pathname;
const base='https://universalmind.tv/informatics/library.html';
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()), checks=[], errors=[];
async function exported(page,name){const pending=page.waitForEvent('download');await page.locator('#export-json').click();const download=await pending;const file='/private/tmp/'+name+'.json';await download.saveAs(file);return JSON.parse(await fs.readFile(file,'utf8')).records;}
async function choose(page,id,value){const toggle=page.locator('#'+id+'-toggle');if(!await toggle.evaluate(e=>e.parentElement.open))await toggle.click();await page.locator('#'+id+'-options input[value='+JSON.stringify(value)+']').check();}
async function clear(page){for(const control of await page.locator('.multi-select[open]>summary').all())await control.click();await page.locator('#reset').click();}
const ids=rs=>rs.map(r=>r.id).sort();
try{
 for(const width of [1470,1024,390,320]){
  const context=await browser.newContext({viewport:{width,height:956},acceptDownloads:true}),page=await context.newPage();
  page.on('pageerror',e=>errors.push(e.message));
  try{
   await page.goto(base+'?collection=all&view=search',{waitUntil:'domcontentloaded'});await page.locator('#export-json:not([disabled])').waitFor();
   assert.match(await page.locator('#result-count').innerText(),/^781 papers$/);
   let all;
   if(width===1470){
    all=await exported(page,'research-all-nature');assert.equal(all.length,781);
    assert.equal(new Set(all.map(r=>r.doi ? r.doi.toLowerCase() : r.id)).size,781);
    for(const [family,count] of [['nature',49],['frontiers',42]]){
     await choose(page,'journal','family:'+family);const actual=await exported(page,'research-'+family);
     assert.equal(actual.length,count);assert.deepEqual(ids(actual),ids(all.filter(r=>r.publisherFamily===family)));
     assert.ok(actual.every(r=>r.journal===r.journalLabel));
     await page.reload();await page.locator('#export-json:not([disabled])').waitFor();assert.equal(await page.locator('#journal-options input:checked').count(),1);
     await clear(page);
    }
    await choose(page,'journal','family:nature');await choose(page,'journal','family:frontiers');
    assert.equal((await exported(page,'research-publisher-union')).length,91);
    await choose(page,'journal','nature');assert.equal((await exported(page,'research-publisher-overlap')).length,91);
    await choose(page,'target','OXTR');await choose(page,'target','DRD2');
    let expected=all.filter(r=>['nature','frontiers'].includes(r.publisherFamily)&&r.targets.some(t=>['OXTR','DRD2'].includes(t)));
    assert.deepEqual(ids(await exported(page,'research-publishers-genes')),ids(expected));
    await clear(page);
    await choose(page,'journal','nature');assert.equal((await exported(page,'research-nature-journal')).length,6);
    await clear(page);await choose(page,'journal','family:nature');await page.locator('#query').fill('serotonin');await page.waitForTimeout(350);
    assert.ok((await exported(page,'research-nature-serotonin')).some(r=>r.doi==='10.1038/nature12518'));
    await clear(page);
    const music=all.find(r=>r.doi==='10.1038/srep09506'),ot=all.find(r=>r.doi==='10.1038/s41386-024-01996-y');
    assert.equal(await page.evaluate(r=>UniversalResearchModel.matchesStudy(r,'trial'),music),false);
    assert.equal(await page.evaluate(r=>UniversalResearchModel.matchesStudy(r,'double-blind'),ot),true);
    assert.match(ot.summary,/impaired/);assert.ok(!ot.categories.includes('Gene expression'));
    assert.ok(all.find(r=>r.doi==='10.1038/nature12518').neurotransmitters.includes('Serotonin'));
   }
   for(const mode of ['video','black','white']){
    await page.evaluate(mode=>{document.documentElement.dataset.readingBackground=mode;},mode);
    await choose(page,'journal','family:nature');await choose(page,'journal','family:frontiers');
    await page.locator('#journal-toggle').scrollIntoViewIfNeeded();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
    await page.screenshot({path:root+`publishers-${width}-${mode}.png`});
    await clear(page);
   }
   await page.goto(base+'?collection=general&chapter=8',{waitUntil:'domcontentloaded'});await page.locator('#topic-neurotransmitter-signalling').waitFor();
   assert.equal(await page.locator('.chapter-reader .paper').count(),14);
   assert.equal(await page.locator('.chapter-reader .topic').count(),3);
   assert.equal(await page.locator('.chapter-reader .topic-heading .trust-score').count(),3);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
   await page.screenshot({path:root+`general-${width}.png`});
   await page.goto(base+'?collection=vision&chapter=5',{waitUntil:'domcontentloaded'});await page.locator('#topic-bilateral-mechanisms').waitFor();
   assert.equal(await page.locator('.chapter-reader .paper').count(),2);
   assert.equal(await page.locator('.chapter-reader .trust-score').count(),0);
   await page.goto(base+'?collection=language&chapter=4#topic-mindfulness-epigenetics',{waitUntil:'domcontentloaded'});await page.locator('#topic-mindfulness-epigenetics').waitFor();
   assert.equal(await page.locator('#topic-mindfulness-epigenetics .paper').count(),3);
   assert.equal(await page.locator('#topic-mindfulness-epigenetics .trust-score svg').count(),1);
   checks.push({width,publisherGroups:true,nature:49,frontiers:42,uniqueRecords:781,multipleSelections:true,themes:3,newChaptersAndTrustScores:true});
  }finally{for(const p of context.pages())await p.evaluate(()=>document.querySelectorAll('video,audio').forEach(e=>e.pause())).catch(()=>{});await context.close();}
 }
 assert.deepEqual(errors,[]);await fs.writeFile(root+'verification.json',JSON.stringify({base,checks,errors,audioMuted:true},null,2));console.log(JSON.stringify({checks,errors}));
}finally{await browser.close();await server.close();}
