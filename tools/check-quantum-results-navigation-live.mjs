import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const out='research/informatics/quantum-results-navigation-20261025',base='https://staging.universalmind.tv',endpoint='https://informatics-universal-one-staging.universalmind.workers.dev';
await fs.mkdir(out,{recursive:true});
const token=(await fs.readFile('/private/tmp/uo-secrets/qa-token','utf8')).trim();
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:1100},reducedMotion:'reduce'});
try{
 await ctx.route(endpoint+'/**',r=>r.continue({headers:{...r.request().headers(),'X-Research-QA':token}}));
 const page=await ctx.newPage(),errors=[];let questionRequests=0;
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.url()===endpoint+'/quantum/question'&&r.method()==='POST')questionRequests++;});
 await page.goto(base+'/informatics/?entry=research&area=quantum');const f=page.frameLocator('#quantum-frame');
 await f.locator('#completed-study').waitFor({state:'visible'});
 assert.equal(await f.locator('#classical-step-status').innerText(),'Ready to view');assert.equal(await f.locator('#quantum-step-status').innerText(),'Ready to view');
 assert.equal(await f.locator('#quantum-step-timing').innerText(),'Recorded results available');
 const order=await f.locator('body').evaluate(()=>{const el=s=>document.querySelector(s);return {afterAsk:!!(el('#ask').compareDocumentPosition(el('#workflow'))&Node.DOCUMENT_POSITION_FOLLOWING),afterWorkflow:el('#workflow').nextElementSibling.id==='completed-study',natureAfterFoundations:el('.background-notes').nextElementSibling.classList.contains('nature-reference'),hypothesisBeforeService:el('.geometry-hypothesis').nextElementSibling.id==='access'};});assert(Object.values(order).every(Boolean));
 await f.locator('#workflow').evaluate(n=>n.scrollIntoView({block:'start'}));await page.screenshot({path:out+'/results-entry-desktop.png'});
 for(const [target,panel] of [['simulation','classical'],['quantum','quantum'],['comparison','quantum']]){
  await f.locator(`[data-recorded-result=${target}]`).click();assert.equal(await f.locator(`[data-step=${panel}]`).getAttribute('aria-selected'),'true');assert.equal(await f.locator('#'+target+'-report').getAttribute('data-state'),'complete');
  assert.match(await f.locator('#'+(panel==='classical'?'classical':'quantum')+'-results .result-question').innerText(),/^Study 01 question:/);
  assert.equal(await f.locator('#'+target+'-report').evaluate(n=>document.activeElement===n),true);
 }
 assert.equal(questionRequests,0);await page.screenshot({path:out+'/comparison-desktop.png'});
 await f.locator('[data-recorded-result=simulation]').click();await f.locator('#simulation-report .representation-row').last().waitFor();assert.equal(await f.locator('#simulation-report .representation-row').count(),8);await f.locator('#simulation-report select').first().selectOption({index:1});assert.match(await f.locator('#simulation-report').innerText(),/Phi Ratio/);
 await page.setViewportSize({width:1470,height:1900});await f.locator('#simulation-report').evaluate(n=>n.scrollIntoView({block:'start'}));await page.screenshot({path:out+'/cudaq-desktop.png'});
 await f.locator('[data-recorded-result=quantum]').click();await f.locator('#quantum-report .representation-row').last().waitFor();assert.equal(await f.locator('#quantum-report .representation-row').count(),8);await f.locator('#quantum-report select').first().selectOption({index:1});await page.screenshot({path:out+'/aquila-desktop.png'});
 await page.setViewportSize({width:1470,height:1100});
 await f.locator('.nature-reference').evaluate(n=>n.scrollIntoView({block:'start'}));await page.screenshot({path:out+'/research-context-desktop.png'});
 assert.equal(await f.locator('#open-nature').innerText(),'Read the Nature Study');await f.locator('#open-nature').click();await f.locator('#nature-dialog').waitFor({state:'visible'});await page.keyboard.press('Escape');assert.equal(await f.locator('#nature-dialog').isVisible(),false);assert(await f.locator('#open-nature').evaluate(n=>document.activeElement===n));
 await f.locator('[data-step=study]').click();await f.locator('#study-methods>summary').click();await f.locator('#study-methods details>summary').click();await f.locator('#open-verification').click();
 const old=f.frameLocator('#verification-frame');await old.locator('#hardware-snapshot').filter({hasText:'Aquila connection test complete'}).waitFor();assert.match(await old.locator('#hardware-snapshot').innerText(),/100 shots returned/);assert.match(await old.locator('#hardware-snapshot').innerText(),/18 hours 52 minutes/);await f.locator('#close-verification').click();
 await f.locator('[data-step=ask]').click();await f.locator('#question').fill('Do quantum responses generated from Fibonacci, the Phi Ratio and M9 P24 Alpha, Beta, Gamma and Epsilon improve our ability to predict transcription factor occupancy around OXTR?');
 const pending=page.waitForResponse(r=>r.url()===endpoint+'/quantum/question'&&r.request().method()==='POST',{timeout:120000});await f.locator('#ask-button').click();const response=await pending,data=await response.json();assert.equal(response.status(),200);assert.equal(data.route,'oxtr_occupancy');assert.equal(data.status,'study_linked');assert.match(data.provider_response_id,/^resp_/);await f.locator('#study-request-status').filter({hasText:data.message}).waitFor();
 await f.locator('[data-step=quantum]').click();await f.locator('#quantum-report[data-state=complete]').waitFor();assert.match(await f.locator('#quantum-results .result-question').innerText(),/^Your question:/);assert.equal(questionRequests,1);
 await fs.writeFile(out+'/live-openai-receipt.json',JSON.stringify(data,null,2)+'\n');
 await f.locator('[data-step=study]').click();await f.locator('#keep-session').click();await page.reload();await f.locator('#resume-study').click();assert.match(await f.locator('#saved-status').innerText(),/No new calculation/);assert.equal(questionRequests,1);
 const layouts=[];for(const width of [390,320]){
  await page.setViewportSize({width,height:844});await f.locator('[data-step=ask]').click();await f.locator('#completed-study').evaluate(n=>n.scrollIntoView({block:'start'}));await page.screenshot({path:out+`/results-entry-${width}.png`});
  for(const target of ['simulation','quantum','comparison']){await f.locator(`[data-recorded-result=${target}]`).click();assert.equal(await f.locator('#'+target+'-report').getAttribute('data-state'),'complete');}
  const layout=await f.locator('body').evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));assert(layout.scroll<=layout.width+1);layouts.push(layout);await page.screenshot({path:out+`/comparison-${width}.png`});
  await f.locator('[data-recorded-result=quantum]').click();await f.locator('#quantum-report .atom-detail>summary').click();const atomLayout=await f.locator('body').evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));assert(atomLayout.scroll<=atomLayout.width+1);await f.locator('#quantum-report .atom-detail>summary').click();
 }
 assert.deepEqual(errors,[]);assert.equal(questionRequests,1);
 await fs.writeFile(out+'/verification.json',JSON.stringify({passed:true,at:new Date().toISOString(),process_audio_muted:true,initial_results_without_question:true,result_buttons_submit_nothing:true,focus_on_selected_result:true,source_order:order,nature_label_and_escape:true,earlier_hardware_completed:true,real_openai_response:true,saved_study_reopened:true,layouts,errors},null,2)+'\n');
 console.log('Staging: direct recorded results, revised timing, earlier completed hardware, and live OpenAI journey passed.');
}finally{for(const p of ctx.pages())await p.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});await ctx.close();await browser.close();await server.close();}
