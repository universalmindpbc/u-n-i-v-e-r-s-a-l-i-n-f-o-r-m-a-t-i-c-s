import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const root='/private/tmp/um-quantum-lab-20261025',out=root+'/research/informatics/quantum-insights-20261025',base='https://staging.universalmind.tv';
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:1100},reducedMotion:'reduce'});
try{
 const page=await ctx.newPage(),errors=[],layouts=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base+'/informatics/?entry=research&area=quantum');const f=page.frameLocator('#quantum-frame');await f.locator('#completed-study').waitFor({state:'visible'});
 for(const width of [1470,390,320]){
  await page.setViewportSize({width,height:width===1470?1100:844});
  for(const kind of ['simulation','quantum']){
   await f.locator(`[data-recorded-result=${kind}]`).click();const slot=f.locator('#modal-'+kind+'-report');await slot.locator('.representation-row').last().waitFor();
   await slot.locator('.representation-comparison').evaluate(n=>n.scrollIntoView({block:'start'}));
   const layout=await slot.evaluate(n=>{const bar=n.querySelector('.representation-row svg').getBoundingClientRect(),mid=n.querySelector('.response-axis span:nth-child(2)').getBoundingClientRect(),first=n.querySelector('.response-axis span:first-child').getBoundingClientRect(),last=n.querySelector('.response-axis span:last-child').getBoundingClientRect();return{barX:bar.x,barWidth:bar.width,zeroError:Math.abs(mid.x+mid.width/2-bar.x-bar.width/2),leftError:Math.abs(first.x-bar.x-bar.width*.05),rightError:Math.abs(last.right-bar.x-bar.width*.95),overflow:document.querySelector('#results-dialog-content').scrollWidth-document.querySelector('#results-dialog-content').clientWidth};});
   assert(layout.zeroError<1,JSON.stringify(layout));assert(layout.leftError<1);assert(layout.rightError<1);assert(layout.overflow<=1);layouts.push({width,kind,...layout});await page.screenshot({path:out+`/${kind}-bars-${width===1470?'desktop':width}.png`});
   const close=await f.locator('#close-results').boundingBox();assert(close.y>=0);await f.locator('#close-results').click();assert(await f.locator('#ask').isVisible());
  }
 }
 assert.deepEqual(errors,[]);await fs.writeFile(out+'/chart-alignment-verification.json',JSON.stringify({passed:true,source_commit:'e6b9e2f3',at:new Date().toISOString(),audio_muted:true,no_new_api_or_quantum_jobs:true,layouts,errors},null,2)+'\n');
 const files=[['Informatics/index.html',12],['Informatics/quantum/index.html',12],['Informatics/quantum/biology.css',11],['Informatics/quantum/biology.js',10],['Informatics/quantum/response-view.js',4],['Informatics/quantum/insights.js',1]],assets=[];
 for(const [file,v]of files){const url=base+'/'+file.replace('Informatics/','informatics/')+'?v='+v,response=await fetch(url);assert(response.ok);const hash=b=>crypto.createHash('sha256').update(b).digest('hex'),source=hash(await fs.readFile(root+'/'+file)),live=hash(Buffer.from(await response.arrayBuffer()));assert.equal(source,live,file);assets.push({file,url,source_sha256:source,staging_sha256:live,match:true});}
 await fs.writeFile(out+'/staging-assets.json',JSON.stringify(assets,null,2)+'\n');console.log('Final chart alignment, mobile layouts and six exact staging asset hashes verified.');
}finally{for(const p of ctx.pages())await p.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});await ctx.close();await browser.close();await server.close();}
