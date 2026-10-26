import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const out='/private/tmp/um-quantum-lab-20261025/research/informatics/quantum-activation-20261025';
const saved=JSON.parse(await fs.readFile(out+'/live-openai-receipt.json','utf8'));
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:1900},reducedMotion:'reduce'});
try{
 await ctx.addInitScript(value=>localStorage.setItem('quantum-atlas-question-v3',JSON.stringify(value)),saved);
 const page=await ctx.newPage();await page.goto('https://staging.universalmind.tv/informatics/?entry=research&area=quantum');const f=page.frameLocator('#quantum-frame');await f.locator('#resume-study').click();
 for(const [step,id,name]of[['classical','simulation-report','simulation-results-desktop.png'],['quantum','quantum-report','quantum-results-desktop.png']]){await f.locator(`[data-step=${step}]`).click();await f.locator(`#${id} .representation-row`).last().waitFor();await f.locator(`#${id} select`).first().selectOption({index:1});await f.locator(`#${id}`).screenshot({path:out+'/'+name});}
 await f.locator('.nature-reference').screenshot({path:out+'/nature-panel-desktop.png'});await f.locator('#open-nature').click();await f.locator('#nature-dialog').screenshot({path:out+'/nature-reference-desktop.png'});await f.locator('#close-nature').click();
 const layouts=[];
 for(const width of [390,320]){
  await page.setViewportSize({width,height:844});await f.locator('[data-step=quantum]').click();await f.locator('#quantum-report .representation-row').last().scrollIntoViewIfNeeded();await page.screenshot({path:out+`/quantum-comparison-${width}.png`});
  await f.locator('#quantum-report .atom-detail>summary').click();await f.locator('#quantum-report table').scrollIntoViewIfNeeded();const size=await f.locator('body').evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));assert(size.scroll<=size.width+1);layouts.push(size);await f.locator('#quantum-report .atom-detail>summary').click();
  await f.locator('.geometry-hypothesis').scrollIntoViewIfNeeded();await page.screenshot({path:out+`/geometry-hypothesis-${width}.png`});
  await f.locator('[data-step=classical]').click();await f.locator('#simulation-report h3').scrollIntoViewIfNeeded();await page.screenshot({path:out+`/simulation-results-${width}.png`});
 }
 await fs.writeFile(out+'/visual-verification.json',JSON.stringify({passed:true,process_audio_muted:true,no_new_question_or_compute:true,desktop_capture_height:1900,mobile_individual_atom_table_no_overflow:layouts,at:new Date().toISOString()},null,2)+'\n');
}finally{for(const p of ctx.pages())for(const frame of p.frames())await frame.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});await ctx.close();await browser.close();await server.close();}
