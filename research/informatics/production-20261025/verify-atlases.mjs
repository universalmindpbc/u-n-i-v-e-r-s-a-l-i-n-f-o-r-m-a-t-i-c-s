import assert from 'node:assert/strict';import fs from 'node:fs/promises';import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const base='https://universalmind.tv/informatics/',endpoint='https://informatics-universal-one.universalmind.workers.dev',out='research/informatics/production-20261025';const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});assert(server.process().spawnargs.includes('--mute-audio'));const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1024,height:768},reducedMotion:'reduce'});const page=await ctx.newPage(),errors=[],direct=[];page.on('pageerror',e=>errors.push(e.message));
try{
 for(const width of [1470,390]){await page.setViewportSize({width,height:900});
 for(const area of ['sound','vision','language','geometry']){
 await page.goto(base+'?area='+area+'&entry=research');
 await page.waitForFunction(a=>document.querySelector('#experience')?.dataset.area===a,area);
 assert(await page.locator('#atlas-navigation').isVisible());assert(await page.locator('#sound').isVisible());assert(await page.locator('#fullscreen').isVisible());
 await page.waitForTimeout(1500);
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 direct.push({area,width});}}
 assert.deepEqual(errors,[]);
 await fs.writeFile(out+'/remaining-atlases.json',JSON.stringify({passed:true,process_audio_muted:true,routes:direct,errors},null,2));console.log(JSON.stringify({passed:true,routes:direct.length}));
}finally{await page.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});await ctx.close();await browser.close();await server.close();}
