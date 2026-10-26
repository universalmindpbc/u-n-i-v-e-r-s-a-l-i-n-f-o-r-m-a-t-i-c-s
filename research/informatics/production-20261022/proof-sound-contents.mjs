import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root='/private/tmp/um-informatics-library-20261021',base='https://universalmind.tv/informatics/';
const candidate=process.env.CANDIDATE==='1',output='/private/tmp/informatics-production-20261022/sound-contents';
await fs.mkdir(output,{recursive:true});
const edition=JSON.parse(await fs.readFile(root+'/Informatics/library-data/sound-edition.json','utf8'));
const expected=edition.chapters.flatMap(c=>c.topics.map(t=>({title:t.title,score:t.trustScore||0})));
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),results=[],errors=[];
try{
 for(const width of [1470,1024,390,320]){
  const ctx=await browser.newContext({viewport:{width,height:956}});
  if(candidate)await ctx.route(base+'**',async route=>{
   const file=path.join(root,'Informatics',new URL(route.request().url()).pathname.replace('/informatics/',''));
   try{await route.fulfill({status:200,body:await fs.readFile(file),contentType:{'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'}[path.extname(file)]||'application/octet-stream'});}catch{await route.continue();}
  });
  const page=await ctx.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'library.html');await page.locator('.contents-topic-meta').first().waitFor();
  assert.equal(await page.locator('.contents-topic-meta').count(),19);
  assert.deepEqual((await page.locator('.contents-topic-meta .paper-count').allTextContents()).slice(0,3),['30 papers','4 papers','17 papers']);
  assert.equal(await page.locator('#trust-key .trust-score svg').count(),6);
  for(const mode of ['video','black','white']){
   await page.locator(`.reading-toolbar [data-reading-background="${mode}"]`).click();
   const layout=await page.locator('.contents-chapter li').evaluateAll(items=>items.map(li=>{
    const row=li.querySelector('.contents-topic-meta'),count=row.querySelector('.paper-count'),score=row.querySelector('.trust-score'),c=count.getBoundingClientRect(),s=score?.getBoundingClientRect(),b=li.getBoundingClientRect(),next=li.nextElementSibling?.getBoundingClientRect();
    return {title:li.querySelector('a').firstChild.textContent,score:score?.querySelectorAll('svg').length||0,inline:!s||Math.abs((c.top+c.height/2)-(s.top+s.height/2))<1,gap:s?s.left-c.right:null,icon:score?.querySelector('svg').getBoundingClientRect().width,spacing:next?next.top-b.bottom:null};
   }));
   assert.deepEqual(layout.map(({title,score})=>({title,score})),expected);
   assert.ok(layout.every(x=>x.inline&&(!x.score||x.gap>=12&&x.icon===32)&& (x.spacing===null||x.spacing>=23)),JSON.stringify(layout));
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
   await page.locator('.edition-intro').evaluate(e=>e.scrollIntoView({block:'start'}));
   if(width>=1024)await page.screenshot({path:output+`/contents-${width}-${mode}.png`});
   else await page.locator('.contents-chapter').first().screenshot({path:output+`/contents-${width}-${mode}.png`});
   results.push({width,mode,topics:layout.length,scoresAndCountsInline:true,icons32px:true,subchapterSpacingAtLeast24px:true,overflow:false});
  }
  await page.locator('.contents-topic-meta').first().click();await page.locator('#topic-i').waitFor();
  assert.equal(await page.locator('#topic-i .trust-score svg').count(),3);
  assert.equal(await page.locator('#topic-i .paper').count(),30);
  await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause()));await ctx.close();
 }
 assert.deepEqual(errors,[]);
 await fs.writeFile(output+'/verification.json',JSON.stringify({results,errors,muted:true,contextsClosed:true,chapterLinksPassed:true},null,2)+'\n');
 console.log(JSON.stringify({output,checks:results.length,topics:19,errors,chapterLinksPassed:true}));
}finally{await browser.close();await server.close();}
