import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=new URL('.',import.meta.url).pathname;
const base='https://universalmind.tv/informatics/';
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()), checks=[], errors=[];
async function exported(page,name){const pending=page.waitForEvent('download');await page.locator('#export-json').click();const download=await pending;const file='/private/tmp/'+name+'.json';await download.saveAs(file);return JSON.parse(await fs.readFile(file,'utf8')).records;}
async function choose(page,id,value){const details=page.locator('#'+id+'-toggle');if(!await details.evaluate(e=>e.parentElement.open))await details.click();await page.locator('#'+id+'-options input[value='+JSON.stringify(value)+']').check();}
async function clear(page){for(const control of await page.locator('.multi-select[open]>summary').all())await control.click();await page.locator('#reset').click();assert.match(await page.locator('#result-count').innerText(),/^758 papers$/);}
try{
 for(const width of [1470,1024,390,320]){
  const context=await browser.newContext({viewport:{width,height:956},acceptDownloads:true}),page=await context.newPage();
  page.on('pageerror',e=>errors.push(e.message));
  try{
   // Block the native fullscreen method before exercising the actual launch handler.
   await page.addInitScript(()=>{Element.prototype.requestFullscreen=function(){return Promise.resolve();};});
   await page.goto(base,{waitUntil:'domcontentloaded'});await page.locator('#launch:not([disabled])').waitFor();await page.locator('#launch').click();
   await page.waitForFunction(()=>document.querySelector('#film').readyState>=2,{},{timeout:45000});
   await page.locator('#experience button.brand[data-about]').click();
   const meta=await page.locator('meta[name=description]').getAttribute('content');assert.equal(meta,'Bioinformatics platform for mental health. Investigating potential relationships between the Fibonacci sequence and OXTR gene expression.');
   for(const selector of ['meta[property="og:description"]','meta[name="twitter:description"]'])assert.equal(await page.locator(selector).getAttribute('content'),meta);
   const style=await page.locator('#about').evaluate(e=>({bg:getComputedStyle(e,'::backdrop').backgroundColor,blur:getComputedStyle(e,'::backdrop').backdropFilter,overflow:e.scrollWidth>e.clientWidth+1}));
   assert.equal(style.bg,'rgba(0, 0, 0, 0.18)');assert.equal(style.blur,'none');assert.equal(style.overflow,false);
   const before=await page.locator('#film').evaluate(e=>e.currentTime);await page.waitForTimeout(500);assert.ok(await page.locator('#film').evaluate(e=>e.currentTime)>before);
   await page.screenshot({path:root+`about-${width}.png`});
   await page.locator('#initial-focus-open').focus();await page.keyboard.press('Enter');await page.locator('#initial-focus').waitFor({state:'visible'});await page.keyboard.press('Escape');assert.equal(await page.locator('#initial-focus-open').evaluate(e=>e===document.activeElement),true);
   assert.equal(await page.evaluate(()=>document.fullscreenElement===null),true);
   await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(e=>e.pause()));
   await page.goto(base+'library.html?collection=all&view=search',{waitUntil:'domcontentloaded'});await page.locator('#export-json:not([disabled])').waitFor();
   assert.match(await page.locator('#result-count').innerText(),/^758 papers$/);
   if(width===1470){
    const all=await exported(page,'research-all-multiselect');
    await choose(page,'condition','ptsd');await choose(page,'condition','anxiety');
    let actual=await exported(page,'research-conditions');let expected=all.filter(r=>r.conditions.some(c=>['ptsd','anxiety'].includes(c)));
    assert.deepEqual(actual.map(r=>r.id).sort(),expected.map(r=>r.id).sort());
    await choose(page,'target','OXTR');await choose(page,'target','NR3C1');
    actual=await exported(page,'research-condition-genes');expected=expected.filter(r=>r.targets.some(c=>['OXTR','NR3C1'].includes(c)));
    assert.deepEqual(actual.map(r=>r.id).sort(),expected.map(r=>r.id).sort());
    assert.equal(new URL(page.url()).searchParams.getAll('condition').length,2);assert.equal(new URL(page.url()).searchParams.getAll('target').length,2);
    await page.reload();await page.locator('#export-json:not([disabled])').waitFor();assert.equal(await page.locator('#condition-options input:checked').count(),2);assert.equal(await page.locator('#target-options input:checked').count(),2);
    await clear(page);
    const journals=[...new Set(all.filter(r=>r.journalKey).map(r=>r.journalKey))].slice(0,2);
    for(const journal of journals)await choose(page,'journal',journal);
    actual=await exported(page,'research-journals');assert.deepEqual(actual.map(r=>r.id).sort(),all.filter(r=>journals.includes(r.journalKey)).map(r=>r.id).sort());
    await clear(page);
    // Repeated changes after rebuilding dynamic topic/journal options retain working checkbox bindings.
    for(const journal of journals)await choose(page,'journal',journal);
    assert.equal(await page.locator('#journal-options input:checked').count(),2);
    await page.locator('#journal-options input[type=checkbox]').evaluateAll((inputs,v)=>inputs.find(i=>i.value===v).click(),journals[0]);
    assert.equal(await page.locator('#journal-options input:checked').count(),1);
    await clear(page);
    await choose(page,'kind','meta-analysis');await choose(page,'kind','trial');assert.equal(await page.locator('#kind-options input:checked').count(),2);
    await clear(page);
    const topics=[...new Set(all.flatMap(r=>r.topics.map(t=>t.key)))].slice(0,2);
    for(const topic of topics)await choose(page,'topic',topic);
    actual=await exported(page,'research-topics');assert.deepEqual(actual.map(r=>r.id).sort(),all.filter(r=>r.topics.some(t=>topics.includes(t.key))).map(r=>r.id).sort());
    await clear(page);
    await choose(page,'condition','anxiety');await page.locator('#condition-toggle').focus();await page.keyboard.press('Escape');assert.equal(await page.locator('#condition-toggle').evaluate(e=>e.parentElement.open),false);
    await page.getByRole('button',{name:'Remove Condition or symptom: Anxiety',exact:true}).click();assert.equal(await page.locator('#selected-filters button').count(),0);
    await choose(page,'condition','mania');assert.match(await page.locator('#result-count').innerText(),/^0 papers$/);assert.equal(await page.locator('#export-json').isDisabled(),true);await clear(page);
   }
   for(const mode of ['video','black','white']){
    await page.evaluate(mode=>{document.documentElement.dataset.readingBackground=mode;},mode);
    await choose(page,'condition','ptsd');await choose(page,'condition','anxiety');
    await page.locator('#condition-toggle').scrollIntoViewIfNeeded();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
    await page.screenshot({path:root+`filters-${width}-${mode}.png`});
    await clear(page);
   }
   checks.push({width,metadata:true,aboutBackdrop:style,videoContinues:true,nestedDialogAndFocus:true,multipleFilters:true,themes:3,noNativeFullscreen:true});
  }finally{for(const p of context.pages())await p.evaluate(()=>document.querySelectorAll('video,audio').forEach(e=>e.pause())).catch(()=>{});await context.close();}
 }
 assert.deepEqual(errors,[]);await fs.writeFile(root+'verification.json',JSON.stringify({base,checks,errors,audioMuted:true},null,2));console.log(JSON.stringify({checks,errors}));
}finally{await browser.close();await server.close();}
