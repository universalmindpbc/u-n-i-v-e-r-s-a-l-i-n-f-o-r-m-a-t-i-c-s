import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const base='https://staging.universalmind.tv/informatics/';
const candidate=process.env.INFORMATICS_CANDIDATE==='1';
const out=process.env.INFORMATICS_QA_OUTPUT||'/private/tmp/atlas-m1001-browser';
await fs.mkdir(out,{recursive:true});
const index=JSON.parse(await fs.readFile(path.join(root,'Informatics/pilot-v1/index.json'),'utf8'));
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),results=[];
try {
 for(const viewport of [{width:1440,height:1000},{width:390,height:844}]) {
  const context=await browser.newContext({viewport,reducedMotion:'reduce'});
  await context.addInitScript(()=>{let full=null;Object.defineProperty(document,'fullscreenElement',{get:()=>full});Element.prototype.requestFullscreen=async function(){full=this;document.dispatchEvent(new Event('fullscreenchange'));};document.exitFullscreen=async()=>{full=null;document.dispatchEvent(new Event('fullscreenchange'));};});
  if(candidate)await context.route(base+'**',async route=>{
   const relative=decodeURIComponent(new URL(route.request().url()).pathname.slice('/informatics/'.length))||'index.html';
   const file=path.resolve(root,'Informatics',relative);if(!file.startsWith(path.join(root,'Informatics')+'/'))return route.abort();
   try{await route.fulfill({status:200,body:await fs.readFile(file),contentType:{'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png'}[path.extname(file)]||'application/octet-stream'});}catch{await route.fulfill({status:404,body:'Missing candidate asset'});}
  });
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  try{
   await page.goto(base+'?atlas=1&record=m-summary',{waitUntil:'domcontentloaded'});
   await page.waitForFunction(()=>!document.querySelector('#launch').disabled);
   await page.locator('#launch').click();
   const frame=page.frameLocator('#atlas-frame');
   await frame.locator('#family-status').filter({hasText:'111 M9-family records available'}).waitFor();
   const options=await frame.locator('#research-record option').evaluateAll(os=>os.map(o=>({value:o.value,text:o.textContent})));
   assert.equal(options.length,1002);assert.deepEqual(options[0],{value:'m-summary',text:'M family'});
   assert.equal(options[1].text,'M1 / P1');assert.equal(options[2].text,'M2 / P3');
   assert.equal(options.at(-1).text,'M1001 / P560');
   assert.equal(options.filter(o=>o.text.includes('M9 family')).length,111);
   assert.equal(await frame.locator('#family-group option').count(),28);
   await frame.locator('#family-group').selectOption('27');
   await frame.locator('.family-card h3').filter({hasText:'M999 / P1368'}).waitFor();
   assert.equal(await frame.locator('.family-card').count(),3);
   await frame.locator('#family-reduced').click();
   await frame.locator('.family-projection').first().waitFor();
   assert.match(await frame.locator('#family-cards').innerText(),/57 cycles/);
   await page.screenshot({path:path.join(out,`family-last-${viewport.width}.png`)});
   await frame.getByRole('button',{name:'Explore M999 P1368',exact:true}).click();
   await frame.locator('.intro h1').filter({hasText:'M999 / P1368'}).waitFor();
   for(const m of [1,49,54,99,750,999,1001]){
    console.log('Checking viewport',viewport.width,'M'+m);
    const item=index.records.find(r=>r.modulus===m),d=JSON.parse(await fs.readFile(path.join(root,'Informatics',item.file),'utf8'));
    await frame.locator('#research-record').selectOption(item.id);
    await frame.locator('.intro h1').filter({hasText:`M${m} / P${item.pisano_period}`}).waitFor();
    for(const kind of ['alpha','beta','gamma','epsilon']){
     await frame.getByRole('tab',{name:kind[0].toUpperCase()+kind.slice(1),exact:true}).click();
     await frame.locator('#diagram').evaluate(img=>img.decode());
     assert.equal(await frame.locator('#'+kind+'-display').isVisible(),m%9===0);
     if(m%9===0){
      await frame.locator('#'+kind+'-reduced').click();
      await frame.locator('#diagram').evaluate(img=>img.decode());
      assert.equal(await frame.locator('#sequence').innerText(),d.m9_sequence_views[kind].reduced_values.join(' → '));
      assert.match(await frame.locator('#svg-download').getAttribute('href'),new RegExp(kind+'-m9.svg'));
      await frame.locator('#'+kind+'-native').click();
     }
     assert.equal(await frame.locator('#atlas-science-summary [data-science-register]').count(),1);
    }
    await frame.getByRole('tab',{name:'Other geometry',exact:true}).click();
    await frame.locator('#diagram').evaluate(img=>img.decode());
    await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();
    assert.match(await frame.locator('#verification-note').innerText(),/Wolfram core checks passed/);
    assert.equal(await frame.locator('#formation-catalogue tbody tr').count(),d.formation_catalogue.length);
    await frame.locator('#atlas-science-summary [data-science-register]').click();
    assert.equal(await frame.locator('.science-register tbody tr').count(),14);
    assert.equal(await frame.locator('.science-register .science-hero').count(),4);
    await frame.locator('.science-register [aria-label="Close research register"]').click();
    await frame.locator('#atlas-science-summary [data-science-genomics]').click();
    await frame.locator('.science-genomics h1').waitFor();
    assert.equal(await frame.locator('.science-genomics').evaluate(e=>e.open),true);
    await page.keyboard.press('Escape');
    assert.equal(await frame.locator('#research-record').inputValue(),item.id);
    assert.equal(await frame.locator('html').evaluate(e=>e.scrollWidth<=innerWidth+1),true);
   }
   await page.screenshot({path:path.join(out,`m1001-summary-${viewport.width}.png`)});
   await frame.locator('#atlas-downloads summary').click();
   assert.match(await frame.locator('#atlas-downloads').getByRole('link',{name:'Full data bundle',exact:true}).getAttribute('href'),/m1-m1001-data.zip/);
   assert.deepEqual(errors,[]);
   results.push({viewport,recordsAvailable:1001,familyMembers:111,familyFirst:true,numericOrder:true,lastFamilyGroup:true,sampledRecords:[1,49,54,99,750,999,1001],allFourViews:true,literatureLightboxes:true,noHorizontalOverflow:true,errors});
  } finally {await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause())).catch(()=>{});await context.close();}
 }
 await fs.writeFile(path.join(out,'results.json'),JSON.stringify({candidate,results},null,2));console.log(JSON.stringify({candidate,results},null,2));
} finally {await browser.close();await server.close();}
