import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const out=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(out,'../../../..'),base='https://staging.universalmind.tv/informatics/';
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),report={checkedAt:new Date().toISOString(),processAudioDisabled:true,layouts:[],favicons:[],assets:[]};
try{
 const context=await browser.newContext({viewport:{width:1470,height:1000}});
 await context.addInitScript(()=>{Element.prototype.requestFullscreen=()=>Promise.reject(new Error('Fullscreen disabled for QA'));});
 const p=await context.newPage();await p.goto(base,{waitUntil:'domcontentloaded'});
 assert.equal(await p.locator('.access-shell a').count(),0);
 assert.equal(await p.locator('a.ai-wiki-link').innerText(),'Wiki For AI / LLMs');
 for(const width of [1470,390,320]){
  const height=width>1000?1000:844;await p.setViewportSize({width,height});
  const link=p.locator('a.ai-wiki-link');assert(await link.isVisible());const box=await link.boundingBox();
  assert(box.x>=0&&box.x<=20);assert(box.y+box.height<=height);assert(box.y+box.height>=height-35);
  const uncovered=await link.evaluate(e=>{const b=e.getBoundingClientRect();return e.contains(document.elementFromPoint(b.x+b.width/2,b.y+b.height/2));});assert(uncovered);
  assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await p.screenshot({path:path.join(out,'landing-'+width+'.png'),fullPage:true});report.layouts.push({width,height,bottomLeft:true,uncovered:true,box});
 }
 await p.getByRole('link',{name:'Wiki For AI / LLMs',exact:true}).click();assert(p.url().endsWith('/informatics/read/'));report.wikiLinkWorks=true;
 const routes=['read/','read/provenance.html','read/publications/index.html','read/atlas/m9.html','read/experiments/index.html'];
 const favicon=await context.request.get('https://staging.universalmind.tv/vogue/iam/favicon.png');assert.equal(favicon.status(),200);const favbytes=await favicon.body();assert.equal(hash(favbytes),hash(await fs.readFile(path.join(root,'i  a m/favicon.png'))));assert(favbytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])));
 for(const route of routes){await p.goto(base+route,{waitUntil:'load'});const icons=p.locator('link[rel="icon"]');assert.equal(await icons.count(),1);assert.equal(await icons.getAttribute('href'),'/vogue/iam/favicon.png');report.favicons.push({path:route,href:'/vogue/iam/favicon.png',verifiedImageSHA256:hash(favbytes)});}
 for(const rel of ['index.html','style.css','read/index.html','read/provenance.html']){const r=await context.request.get(base+rel);assert.equal(r.status(),200);assert.equal(hash(await r.body()),hash(await fs.readFile(path.join(root,'Informatics',rel))));report.assets.push({path:rel,matches:true});}
 const html=await (await context.request.get(base)).text();assert.match(html,/<a class="ai-wiki-link" href="read\/">Wiki For AI \/ LLMs<\/a>/);report.linkPresentWithoutJavaScript=true;
 const sitemap=await (await context.request.get('https://staging.universalmind.tv/sitemap.xml')).text();assert(sitemap.includes('https://universalmind.tv/informatics/read/index.html'));report.sitemapIncludesReadingPages=true;
 const guide=await (await context.request.get(base+'llms.txt')).text();assert(guide.includes('https://universalmind.tv/informatics/read/'));report.llmsGuideLinksReadingRoom=true;
 report.passed=true;
}finally{
 for(const p of browser.contexts().flatMap(c=>c.pages())){await p.evaluate(()=>document.querySelectorAll('audio,video').forEach(e=>e.pause())).catch(()=>{});await p.close().catch(()=>{});}
 await browser.close();await server.close();await fs.writeFile(path.join(out,'landing-favicon-verification.json'),JSON.stringify(report,null,2)+'\n');
}
console.log(JSON.stringify({passed:report.passed,landingWidths:report.layouts.length,faviconPages:report.favicons.length,assetMatches:report.assets.length}));
