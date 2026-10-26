import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),checks=[],errors=[];
try{
 for(const width of [1470,1024,390,320]){
  const context=await browser.newContext({viewport:{width,height:956}}),page=await context.newPage();
  try{
   page.on('pageerror',e=>errors.push(e.message));await page.goto('https://universalmind.tv/informatics/?about',{waitUntil:'domcontentloaded'});
   const about=page.locator('#about');await about.waitFor({state:'visible'});
   assert.equal(await about.locator('.about-tagline').innerText(),'Bioinformatics platform for mental health');
   assert.equal(await about.locator('#initial-focus-heading').innerText(),'Initial focus · Computer vision for genomics');
   assert.equal(await about.locator('[data-library]').count(),3);
   await page.screenshot({path:`/private/tmp/informatics-production-20261023/about-${width}.png`});
   const open=page.locator('#initial-focus-open');await open.focus();await page.keyboard.press('Enter');await page.locator('#initial-focus').waitFor({state:'visible'});
   assert.equal(await page.locator('#initial-focus-title').innerText(),'A measurable first step');
   assert.match(await page.locator('#initial-focus').innerText(),/computational cost, processing speed and accuracy/);
   assert.ok(await page.locator('#initial-focus').evaluate(e=>e.scrollWidth<=e.clientWidth+1));
   assert.equal(await page.evaluate(()=>document.fullscreenElement===null),true);
   await page.screenshot({path:`/private/tmp/informatics-production-20261023/initial-focus-${width}.png`});
   await page.keyboard.press('Escape');assert.equal(await page.locator('#initial-focus').isVisible(),false);assert.equal(await about.isVisible(),true);
   assert.equal(await open.evaluate(e=>e===document.activeElement),true);
   await open.click();await page.locator('.initial-focus-back').click();assert.equal(await about.isVisible(),true);
   await page.locator('#about-close').click();assert.equal(await about.isVisible(),false);
   checks.push({width,tagline:true,nestedFocusDialog:true,keyboardAndBack:true,focusRestored:true,libraryLinks:3,fullscreen:false});
  }finally{for(const p of context.pages())await p.evaluate(()=>document.querySelectorAll('video,audio').forEach(x=>x.pause())).catch(()=>{});await context.close();}
 }
 assert.deepEqual(errors,[]);await fs.writeFile('/private/tmp/informatics-production-20261023/about.json',JSON.stringify({url:'https://universalmind.tv/informatics/?about',checks,errors,audioMuted:true},null,2));console.log(JSON.stringify({checks,errors}));
}finally{await browser.close();await server.close();}
