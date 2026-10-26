import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import crypto from 'node:crypto';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../..');
const evidence=path.join(root,'research/informatics/discovery-20261026');
const base='https://staging.universalmind.tv/informatics/';
const build=JSON.parse(await fs.readFile(path.join(evidence,'build.json')));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint());
const report={base,checkedAt:new Date().toISOString(),audioDisabledAtProcessLaunch:true,noJavaScript:true,layouts:[],assets:[],errors:[]};
try{
 const context=await browser.newContext({javaScriptEnabled:false,viewport:{width:1470,height:1000}});
 const p=await context.newPage();p.on('pageerror',e=>report.errors.push(e.message));
 let response=await p.goto(base+'read/',{waitUntil:'load'});
 assert.equal(response.status(),200);assert.match(response.headers()['x-robots-tag']||'',/noindex/);
 for(const width of [1470,390,320]){
  await p.setViewportSize({width,height:width>1000?1000:844});
  assert(await p.locator('h1').isVisible());
  assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await p.screenshot({path:path.join(evidence,'reading-'+width+'.png'),fullPage:true});report.layouts.push({width,noHorizontalOverflow:true});
 }
 await p.getByRole('link',{name:'Read the publication register',exact:true}).click();
 assert.equal(await p.locator('.record-list>li').count(),build.publications);
 const first=p.locator('.record-list a').first(),title=await first.innerText();await first.click();
 assert.equal(await p.locator('h1').innerText(),title);
 await p.getByText('Complete approved library record',{exact:true}).click();assert(await p.locator('details[open]').count());
 await p.screenshot({path:path.join(evidence,'publication-320.png'),fullPage:true});
 await p.getByRole('link',{name:'Geometry Atlas',exact:true}).click();
 assert.equal(await p.locator('.record-list>li').count(),1001);
 await p.getByRole('link',{name:'Fibonacci M9P24 · period 24',exact:true}).click();
 assert(await p.getByRole('link',{name:'Complete record JSON',exact:true}).isVisible());
 await p.getByText('research',{exact:true}).click();assert(await p.getByText('Not yet assessed',{exact:true}).count());
 await p.getByRole('link',{name:'Experiments',exact:true}).click();assert.equal(await p.locator('.record-list>li').count(),build.experiments);
 await p.locator('.record-list a').first().click();assert(await p.getByRole('link',{name:'Complete experiment JSON',exact:true}).isVisible());
 await p.getByRole('link',{name:'AI access & reuse',exact:true}).click();assert.match(await p.locator('main').innerText(),/all LLM providers/);
 await p.goto(base+'read/provenance.html',{waitUntil:'load'});
 assert.match(await p.locator('main').innerText(),/15 files were retrieved/);
 assert.match(await p.locator('main').innerText(),/33 archived files/);
 assert.match(await p.locator('main').innerText(),/combined archive manifest has been recovered/);
 assert.match(await p.locator('main').innerText(),/Bitcoin confirmation remains pending/);
 for(const width of [1470,320]){
  await p.setViewportSize({width,height:width>1000?1000:844});
  assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await p.screenshot({path:path.join(evidence,'provenance-'+width+'.png'),fullPage:true});
 }
 report.provenanceVerified=true;
 report.journeys=['overview → publication index → complete publication','geometry index → M9 → evidence status and complete data','experiment index → versioned experiment','AI access policy'];
 const samples=['index.html','connect.html','llms.txt','read/index.html','read/access.html','read/data.html','read/provenance.html','read/reading.css','read/atlas/m1.html','read/atlas/m9.html','read/atlas/m750.html','read/atlas/m1001.html','read/publications/index.html','read/experiments/index.html','exports/manifest.json'];
 const archive=JSON.parse(await fs.readFile(path.join(evidence,'archive-verification.json')));
 samples.push(...archive.files.map(x=>'read/'+x.path));
 for(const rel of samples){const r=await context.request.get(base+rel);assert.equal(r.status(),200,rel);const b=await r.body(),local=await fs.readFile(path.join(root,'Informatics',rel));assert.equal(sha(b),sha(local),rel);report.assets.push({path:rel,sha256:sha(b),matches:true,stagingNoindex:/noindex/.test(r.headers()['x-robots-tag']||'')});}
 assert(report.assets.every(x=>x.stagingNoindex));
 const robots=await context.request.get('https://staging.universalmind.tv/robots.txt');assert.match(await robots.text(),/Disallow: \/\*/);report.stagingRobotsExcluded=true;
 const sitemap=await context.request.get('https://staging.universalmind.tv/sitemap.xml');assert.equal(sha(await sitemap.body()),sha(await fs.readFile(path.join(root,'sitemap.xml'))));
 report.sitemapMatches=true;await context.close();
 const js=await browser.newContext({viewport:{width:1470,height:1000}});
 await js.addInitScript(()=>{globalThis.__fullscreenAttempts=0;Element.prototype.requestFullscreen=function(){globalThis.__fullscreenAttempts++;return Promise.reject(new Error('Native fullscreen disabled during QA'));};});
 const landing=await js.newPage();await landing.goto(base,{waitUntil:'domcontentloaded'});await landing.getByRole('link',{name:'Wiki For AI / LLMs',exact:true}).click();assert.match(landing.url(),/\/read\/$/);assert(await landing.locator('h1').isVisible());report.landingReadingLinkWorks=true;
 await landing.evaluate(()=>document.querySelectorAll('video,audio').forEach(x=>x.pause()));await js.close();
 assert.deepEqual(report.errors,[]);report.passed=true;
}catch(e){report.failure=e.stack;throw e;}finally{
 for(const page of browser.contexts().flatMap(c=>c.pages())){await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(e=>e.pause())).catch(()=>{});await page.close().catch(()=>{});}
 await browser.close();await server.close();await fs.writeFile(path.join(evidence,'staging-verification.json'),JSON.stringify(report,null,2)+'\n');
}
console.log(JSON.stringify({passed:report.passed,pages:build.pages,assets:report.assets.length,layouts:report.layouts.length}));
