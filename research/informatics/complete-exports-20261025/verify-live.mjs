import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const site='https://staging.universalmind.tv',api='https://informatics-universal-one-staging.universalmind.workers.dev',out='research/informatics/complete-exports-20261025';
const manifest=JSON.parse(await fs.readFile('Informatics/exports/manifest.json'));
const report={started:new Date().toISOString(),downloads:[],layouts:[],answers:[],errors:[]};
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));report.processAudioMuted=true;
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:1000},reducedMotion:'reduce',acceptDownloads:true});
const page=await ctx.newPage();page.on('pageerror',e=>report.errors.push(e.message));
try{
 const qa=(await fs.readFile('/private/tmp/uo-secrets/qa-token','utf8')).trim();
 await ctx.route(api+'/**',r=>r.continue({headers:{...r.request().headers(),'X-Research-QA':qa}}));
 await page.goto(site+'/informatics/?area=connect&entry=research');
 const frame=page.frameLocator('#connect-frame');await frame.locator('#complete-download-status').filter({hasText:'1,168'}).waitFor({timeout:60000});
 await frame.locator('.complete-dataset').scrollIntoViewIfNeeded();
 for(const width of [1470,390,320]){
  await page.setViewportSize({width,height:width<500?844:1000});
  await frame.locator('.complete-dataset').scrollIntoViewIfNeeded();
  const dims=await frame.locator('html').evaluate(el=>({viewport:innerWidth,scroll:el.scrollWidth}));assert(dims.scroll<=dims.viewport+1,JSON.stringify(dims));
  report.layouts.push({width,...dims});await page.screenshot({path:out+'/downloads-'+width+'.png'});
 }
 await page.setViewportSize({width:1470,height:1000});
 for(const format of ['json','csv']){
  const pending=page.waitForEvent('download',{timeout:300000});await frame.locator(`[data-complete-download=${format}]`).click();const download=await pending;
  assert.equal(download.suggestedFilename(),manifest.downloads[format].file);
  const dest='/private/tmp/verified-'+download.suggestedFilename();await download.saveAs(dest);
  const bytes=await fs.readFile(dest),hash=createHash('sha256').update(bytes).digest('hex');assert.equal(hash,manifest.downloads[format].sha256);
  report.downloads.push({format,file:download.suggestedFilename(),bytes:bytes.length,sha256:hash,passed:true});console.log('Verified browser download',format,bytes.length);
 }
 await frame.locator('#test-mcp').click();await frame.locator('#mcp-result').filter({hasText:'9 read-only tools'}).waitFor({timeout:30000});report.mcp=await frame.locator('#mcp-result').innerText();
 const resp=await ctx.request.get(api+'/api/v1/quantum/oxtr-qrc-01?section=sequences');assert.equal(resp.status(),200);const seq=await resp.json();assert.deepEqual(seq.data.definitions.beta.display_cycle,[4,8,7,5,1,2]);report.apiQuantum=true;
 await page.goto(site+'/informatics/?area=quantum&entry=research');
 const qframe=page.frameLocator('#quantum-frame');await qframe.locator('a[href*="experiments/"][download]').first().waitFor({timeout:30000});
 const experimentDownload=page.waitForEvent('download');await qframe.locator('a[href*="experiment.json"][download]').first().click();const ex=await experimentDownload;await ex.saveAs('/private/tmp/verified-experiment.json');const archived=JSON.parse(await fs.readFile('/private/tmp/verified-experiment.json'));assert.equal(archived.experiment_id,'oxtr-qrc-01-qualification');report.experimentDownload=true;
 await page.locator('.uo .iam-realtime-voice__summon').click();
 const questions=[
 'Explain our Quantum Atlas experiment: its goal, the eight preparations, the useful finding, what CTCF means and the next computational task. Use plain language.',
 'Give our recorded Beta, Gamma and Epsilon sequences and explain how Phi Ratio was included. Use the saved study definitions.'
 ];
 for(let i=0;i<questions.length;i++){
  await page.locator('#uo-question').fill(questions[i]);const pending=page.waitForResponse(r=>r.url()===api+'/research'&&r.request().method()==='POST',{timeout:180000});await page.locator('.uo [data-send]').click();
  const response=await pending,data=await response.json();assert.equal(response.status(),200,JSON.stringify(data));assert.equal(data.model,'gpt-6-astra');assert(data.sources.some(s=>s.kind==='quantum-experiment'));assert(data.usage.totalTokens>0);
  await page.locator('.uo-answer').last().filter({hasText:data.answer.slice(0,25).replaceAll('*','')}).waitFor({timeout:10000}).catch(async()=>{await page.waitForFunction(()=>document.querySelectorAll('.uo-answer').length>0);});
  const visible=await page.locator('.uo-answer').last().innerText();assert(visible.length>100);assert.equal(await page.locator('.uo-answer').last().isVisible(),true);
  if(i===0){assert.match(data.answer,/CTCF/);assert.match(data.answer,/two|2/i);assert.match(data.answer,/Phi Ratio/i);}else{assert.match(data.answer,/Beta/i);assert.match(data.answer,/Gamma/i);assert.match(data.answer,/Epsilon/i);}
  report.answers.push({question:questions[i],http:response.status(),...data,visibleText:visible});await page.screenshot({path:out+'/universal-one-quantum-'+i+'.png'});console.log('Verified Universal One response',i+1);
 }
 assert.deepEqual(report.errors,[]);report.passed=true;
}catch(e){report.error=e.stack;throw e;}
finally{for(const f of page.frames())await f.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});await ctx.close();await browser.close();await server.close();await fs.writeFile(out+'/live-verification.json',JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({passed:report.passed,downloads:report.downloads.length,answers:report.answers.length}));
