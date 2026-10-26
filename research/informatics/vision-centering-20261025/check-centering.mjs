import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root='/private/tmp/um-vision-science-20261025',live=process.argv.includes('--live'),out=path.join(root,'research/informatics/vision-centering-20261025'),results=[];
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));const browser=await chromium.connect(server.wsEndpoint());
try{
 for(const viewport of (live?[{width:1470,height:956},{width:390,height:844},{width:320,height:844}]:[{width:1470,height:956}])){
  const ctx=await browser.newContext({viewport,reducedMotion:'reduce'}),page=await ctx.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  if(!live)await ctx.route(/https:\/\/staging\.universalmind\.tv\/informatics\//,async route=>{
   const rel=new URL(route.request().url()).pathname.replace('/informatics/','')||'index.html';
   if(!['index.html','vision-science.js','vision-science.css'].includes(rel))return route.continue();
   await route.fulfill({body:await fs.readFile(path.join(root,'Informatics',rel)),contentType:rel.endsWith('.js')?'text/javascript':rel.endsWith('.css')?'text/css':'text/html'});
  });
  try{
   await page.goto('https://staging.universalmind.tv/informatics/?entry=research&area=vision',{waitUntil:'domcontentloaded'});
   await page.locator('[data-vs-layer="triangles"]').click();
   const geometry=await page.locator('[data-vs-triangles]').evaluate(g=>{
    const box=g.getBBox(),cells=[...g.parentNode.querySelectorAll('[data-vs-cells] circle')].map(n=>({x:n.cx.baseVal.value,y:n.cy.baseVal.value}));
    const paths=[...g.querySelectorAll('path')],endpoints=paths.flatMap(p=>[p.getPointAtLength(0),p.getPointAtLength(p.getTotalLength())]);
    return {cx:box.x+box.width/2,cy:box.y+box.height/2,width:box.width,height:box.height,edges:paths.length,maxCellDistance:Math.max(...endpoints.map(p=>Math.min(...cells.map(c=>Math.hypot(c.x-p.x,c.y-p.y))))),maxRadius:Math.max(...endpoints.map(p=>Math.hypot(p.x-220,p.y-200))),visible:g.getAttribute('visibility')};
   });
   assert(Math.abs(geometry.cx-220)<.001);assert(Math.abs(geometry.cy-200)<.001);assert(geometry.maxCellDistance<.001);assert(geometry.maxRadius<155);assert.equal(geometry.visible,'visible');
   await page.locator('#vs-retina .vs-figure').scrollIntoViewIfNeeded();
   await page.screenshot({path:path.join(out,`${live?'staging':'local'}-triangles-${viewport.width}.png`)});
   for(const kind of ['cells','hexagons','triangles']){await page.locator(`[data-vs-layer="${kind}"]`).focus();await page.keyboard.press('Enter');assert.equal(await page.locator(`[data-vs-layer="${kind}"]`).getAttribute('aria-pressed'),'true');}
   assert.deepEqual(errors,[]);results.push({viewport,geometry,keyboardControls:true,errors});
  }finally{for(const frame of page.frames())await frame.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});await ctx.close();}
 }
 await fs.writeFile(path.join(out,`${live?'staging':'local'}-verification.json`),JSON.stringify({passed:true,live,processAudioDisabled:true,results},null,2)+'\n');console.log(JSON.stringify({passed:true,live,results}));
}finally{await browser.close();await server.close();}
