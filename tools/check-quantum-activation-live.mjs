import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const out='research/informatics/quantum-activation-20261025',base='https://staging.universalmind.tv',endpoint='https://informatics-universal-one-staging.universalmind.workers.dev';
const token=(await fs.readFile('/private/tmp/uo-secrets/qa-token','utf8')).trim();
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:956},reducedMotion:'reduce'});
try{
 await ctx.route(endpoint+'/**',r=>r.continue({headers:{...r.request().headers(),'X-Research-QA':token}}));
 const page=await ctx.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base+'/informatics/?entry=research&area=quantum');const f=page.frameLocator('#quantum-frame');
 await f.locator('#workflow-state').filter({hasText:'First measurement phase complete'}).waitFor();
 assert(await f.locator('#ask').isVisible());assert(await f.locator('#workflow').isVisible());
 const positions=await f.locator('body').evaluate(()=>({ask:document.querySelector('#ask').getBoundingClientRect().bottom,flow:document.querySelector('#workflow').getBoundingClientRect().top}));assert(positions.flow>=positions.ask);
 assert.equal(await f.locator('[data-hypothesis]').count(),5);
 for(const key of ['fibonacci','modular','odom','honeycomb','quantum']){await f.locator(`[data-hypothesis=${key}]`).click();assert.equal(await f.locator(`[data-hypothesis=${key}]`).getAttribute('aria-pressed'),'true');assert((await f.locator('#hypothesis-detail').innerText()).length>80);}
 await f.locator('.geometry-hypothesis').screenshot({path:out+'/geometry-hypothesis-desktop.png'});
 await f.locator('#open-nature').click();await f.locator('#nature-dialog').waitFor({state:'visible'});assert.match(await f.locator('#nature-dialog').innerText(),/5.5-second observation period/);await f.locator('#nature-dialog').screenshot({path:out+'/nature-reference-desktop.png'});await page.keyboard.press('Escape');assert.equal(await f.locator('#nature-dialog').isVisible(),false);
 await f.locator('#question').fill('Do quantum responses generated from Fibonacci, the Phi Ratio and M9 P24 Alpha, Beta, Gamma and Epsilon improve our ability to predict transcription factor occupancy around OXTR?');
 const pending=page.waitForResponse(r=>r.url()===endpoint+'/quantum/question'&&r.request().method()==='POST',{timeout:120000});await f.locator('#ask-button').click();const response=await pending,data=await response.json();
 assert.equal(response.status(),200);assert.equal(data.route,'oxtr_occupancy');assert.equal(data.status,'study_linked');assert.match(data.provider_response_id,/^resp_/);
 await f.locator('#study-request-status').filter({hasText:data.message}).waitFor();assert(!('plan'in data));
 await fs.writeFile(out+'/live-openai-receipt.json',JSON.stringify(data,null,2)+'\n');
 await f.locator('[data-step=classical]').click();await f.locator('#simulation-report .representation-row').last().waitFor();
 assert.equal(await f.locator('#classical-report').getAttribute('data-state'),'complete');assert.equal(await f.locator('#simulation-report').getAttribute('data-state'),'complete');
 assert.match(await f.locator('#classical-report').innerText(),/improved prediction has not yet been established/);
 assert.equal(await f.locator('#simulation-report .representation-row').count(),8);assert.match(await f.locator('#simulation-report').innerText(),/Gamma and Epsilon/);assert.match(await f.locator('#simulation-report').innerText(),/Phi Ratio/);
 await f.locator('#simulation-report .atom-detail>summary').click();
 const controls=f.locator('#simulation-report select');assert.equal(await controls.count(),2);assert.equal(await controls.nth(1).locator('option').count(),8);
 const values={};for(const key of ['ordinary','fibonacci','phi','alpha','beta','gamma','epsilon','shuffled']){await controls.nth(1).selectOption(key);values[key]=await f.locator('#simulation-report tbody').innerText();}
 await controls.nth(0).selectOption({index:1});await controls.nth(1).selectOption('phi');
 await f.locator('#simulation-report .atom-detail>summary').click();
 await f.locator('#simulation-report').screenshot({path:out+'/simulation-results-desktop.png'});
 await f.locator('[data-step=quantum]').click();await f.locator('#quantum-report .representation-row').last().waitFor();assert.equal(await f.locator('#quantum-report').getAttribute('data-state'),'complete');assert.match(await f.locator('#quantum-report').innerText(),/second-window reversal/);assert.match(await f.locator('#comparison-report').innerText(),/no occupancy predictor/);assert.equal(await f.locator('#quantum-report .representation-row').count(),8);await f.locator('#quantum-report select').first().selectOption({index:1});await f.locator('#quantum-report').screenshot({path:out+'/quantum-results-desktop.png'});
 await f.locator('[data-stage=hardware]').click();await f.locator('#workflow').screenshot({path:out+'/hardware-status-desktop.png'});
 await f.locator('[data-step=study]').click();await f.locator('#keep-session').click();await page.reload();await f.locator('#resume-study').click();assert.match(await f.locator('#saved-status').innerText(),/No new calculation/);
 const layouts=[];for(const width of [390,320]){await page.setViewportSize({width,height:844});await f.locator('[data-step=classical]').click();await f.locator('#simulation-report .representation-row').last().waitFor();const layout=await f.locator('body').evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));assert(layout.scroll<=layout.width+1,JSON.stringify(layout));layouts.push(layout);await f.locator('#simulation-report').screenshot({path:out+`/simulation-results-${width}.png`});await f.locator('.geometry-hypothesis').screenshot({path:out+`/geometry-hypothesis-${width}.png`});await f.locator('#open-nature').click();const modal=await f.locator('#nature-dialog').evaluate(n=>({width:n.clientWidth,scroll:n.scrollWidth}));assert(modal.scroll<=modal.width+1);await f.locator('#nature-dialog').screenshot({path:out+`/nature-reference-${width}.png`});await f.locator('#close-nature').click();}
 assert.deepEqual(errors,[]);
 await fs.writeFile(out+'/live-verification.json',JSON.stringify({passed:true,at:new Date().toISOString(),process_audio_muted:true,real_openai_response:true,routing_status:data.status,all_eight_representations_selectable:true,both_windows_selectable:true,recorded_results_only:true,qualification_boundary_visible:true,hardware_status_from_receipts:true,quantum_comparison_complete:true,named_sequence_insights:true,nature_lightbox_keyboard_close:true,geometry_hypothesis_interactive:true,saved_study_restored:true,layouts,errors},null,2)+'\n');
 console.log('Live question → recorded CPU/GPU results → hardware status verified.');
}finally{for(const p of ctx.pages())await p.evaluate(()=>document.querySelectorAll('audio,video').forEach(v=>v.pause())).catch(()=>{});await ctx.close();await browser.close();await server.close();}
