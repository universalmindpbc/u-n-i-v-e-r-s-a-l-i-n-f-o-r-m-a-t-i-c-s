import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const base='https://staging.universalmind.tv/informatics/';
const candidate=process.env.INFORMATICS_CANDIDATE==='1';
const out=process.env.INFORMATICS_QA_OUTPUT||'/private/tmp/atlas-complete-csv-browser';
await fs.mkdir(out,{recursive:true});
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const server=await chromium.launchServer({downloadsPath:out,executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),results=[];
try{
 for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
  const context=await browser.newContext({viewport,acceptDownloads:true,reducedMotion:'reduce'});
  await context.addInitScript(()=>{let full=null;Object.defineProperty(document,'fullscreenElement',{get:()=>full});Element.prototype.requestFullscreen=async function(){full=this;document.dispatchEvent(new Event('fullscreenchange'));};document.exitFullscreen=async()=>{full=null;document.dispatchEvent(new Event('fullscreenchange'));};});
  if(candidate)await context.route(base+'**',async route=>{
   const relative=decodeURIComponent(new URL(route.request().url()).pathname.slice('/informatics/'.length))||'index.html';
   const file=path.resolve(root,'Informatics',relative);if(!file.startsWith(path.join(root,'Informatics')+'/'))return route.abort();
   try{await route.fulfill({status:200,body:await fs.readFile(file),contentType:{'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png','.csv':'text/csv'}[path.extname(file)]||'application/octet-stream'});}catch{await route.fulfill({status:404,body:'Missing candidate asset'});}
  });
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  try{
   await page.goto(base+'?atlas=1&record=fibonacci-m999-p1368',{waitUntil:'domcontentloaded'});
   await page.waitForFunction(()=>!document.querySelector('#launch').disabled);await page.locator('#launch').click();
   const frame=page.frameLocator('#atlas-frame');
   await frame.locator('.intro h1').filter({hasText:'M999 / P1368'}).waitFor();
   for(const [m,p] of [[999,1368],[1001,560]]){
    await frame.locator('#research-record').selectOption(`fibonacci-m${m}-p${p}`);
    await frame.locator('.intro h1').filter({hasText:`M${m} / P${p}`}).waitFor();
    await frame.getByRole('tab',{name:'Beta',exact:true}).click();
    const csv=frame.locator('#csv-download');assert.equal(await csv.getAttribute('href'),`pilot-v1/csv-records/m${m}.csv?v=1`);
    if(candidate){
     const digest=await page.evaluate(async url=>{const data=await (await fetch(url)).arrayBuffer();return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),b=>b.toString(16).padStart(2,'0')).join('');},base+`pilot-v1/csv-records/m${m}.csv`);
     assert.equal(digest,hash(await fs.readFile(path.join(root,`Informatics/pilot-v1/csv-records/m${m}.csv`))));
    }else{
     const wait=page.waitForEvent('download');await csv.click();const download=await wait;
     const saved=path.join(out,`m${m}-${viewport.width}.csv`);await download.saveAs(saved);assert.equal(hash(await fs.readFile(saved)),hash(await fs.readFile(path.join(root,`Informatics/pilot-v1/csv-records/m${m}.csv`))));
    }
   }
   await frame.locator('#atlas-downloads summary').click();
   const bundle=frame.locator('#atlas-downloads').getByRole('link',{name:'Complete CSV bundle',exact:true});
   assert.equal(await bundle.isVisible(),true);
   assert.equal(await frame.locator('html').evaluate(e=>e.scrollWidth<=innerWidth+1),true);
   await page.screenshot({path:path.join(out,`csv-downloads-${viewport.width}.png`)});
   if(viewport.width===1440&&!candidate){
    const wait=page.waitForEvent('download',{timeout:180000});await bundle.click();const download=await wait;
    const saved=path.join(out,'complete-csv.zip');await download.saveAs(saved);assert.equal(hash(await fs.readFile(saved)),hash(await fs.readFile(path.join(root,'Informatics/pilot-v1/m1-m1001-csv.zip'))));
   }else await frame.locator('#atlas-downloads summary').click();
   await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();
   assert.equal(await frame.locator('.pilot-downloads').getByRole('link',{name:/Complete CSV bundle/}).count(),1);
   assert.deepEqual(errors,[]);results.push({viewport,recordsChecked:[999,1001],recordContentsMatch:true,recordCheckMethod:candidate?"browser fetch":"download click",completeCsvDownloadVisible:true,fullBundleDownloadedAndMatched:viewport.width===1440&&!candidate,noHorizontalOverflow:true,errors});
  }finally{await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause())).catch(()=>{});await context.close();}
 }
 const result={passed:true,candidate,results};await fs.writeFile(path.join(out,'results.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();await server.close();}
