import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const out='research/informatics/quantum-biology-20261025';
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:956},reducedMotion:'reduce'}),page=await ctx.newPage(),checks=[];
try {
  for(const [route,id]of[['library.html','research'],['connect.html','connect']]){
    await page.goto('https://staging.universalmind.tv/informatics/'+route);
    const f=page.frameLocator('#'+id+'-frame');
    await f.locator('body.embedded').waitFor();
    assert.equal(await f.locator('.library-masthead').isVisible(),false);
    for(const mode of ['black','white','video']){
      await f.locator('button[data-reading-background="'+mode+'"]:visible').first().click();
      assert.equal(await f.locator('[data-reading-motion]:visible').count(),0);
    }
    assert.equal(await page.locator('#sound').isVisible(),true);
    assert.equal(await page.locator('#film').evaluate(v=>v.muted),true);
    checks.push({route,shared_media_controls:true,duplicate_controls_hidden:true});
  }
  await page.goto('https://staging.universalmind.tv/informatics/?entry=research&area=transcription');
  await page.locator('.tfbs-source-card').nth(2).waitFor();
  await page.waitForFunction(()=>{const v=document.querySelector('#film');return v.readyState>=2&&v.currentTime>1&&!v.paused;},null,{timeout:45000});
  await page.screenshot({path:out+'/transcription-shared-1470.png'});
  const capacity=await(await fetch('https://informatics-universal-one-staging.universalmind.workers.dev/quantum/capacity',{headers:{Origin:'https://staging.universalmind.tv'}})).json();
  assert.equal(capacity.hardware.available,false);
  await fs.writeFile(out+'/live-embedded-media-verification.json',JSON.stringify({verifiedAt:new Date().toISOString(),source:'7cfd44b0',passed:true,process_audio_muted:true,checks,public_capacity:capacity},null,2)+'\n');
  console.log(JSON.stringify({passed:true,checks:checks.length,public_capacity:true}));
}finally{
  await page.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});
  await ctx.close();await browser.close();await server.close();
}
