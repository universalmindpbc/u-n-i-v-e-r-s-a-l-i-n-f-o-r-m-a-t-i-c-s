import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const origin=process.env.QA_ORIGIN||'https://universalmind.tv',out=process.env.QA_OUT||new URL('./production/',import.meta.url).pathname,api=origin.includes('staging.')?'https://informatics-universal-one-staging.universalmind.workers.dev':'https://informatics-universal-one.universalmind.workers.dev';
await fs.mkdir(out,{recursive:true});
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});assert(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),ctx=await browser.newContext({viewport:{width:1470,height:1000},reducedMotion:'reduce'}),page=await ctx.newPage();let task,initial,seen='',errors=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('response',async r=>{if(r.url().includes('/agent/studies/')&&r.request().method()==='GET'){const t=await r.json().catch(()=>null);if(t?.id){task=t;await fs.writeFile(out+'live-task.json',JSON.stringify(t,null,2)+'\n');if(t.stage+':'+t.status!==seen){seen=t.stage+':'+t.status;console.log(JSON.stringify({id:t.id,stage:t.stage,status:t.status,message:t.message,reviews:t.reviews?.map(r=>({phase:r.phase,decision:r.decision}))}));}}}});
try{
 await page.goto(origin+'/informatics/?area=quantum&entry=research');const f=page.frameLocator('#quantum-frame');await f.locator('#completed-study:not([hidden])').waitFor();
 if(process.argv.includes('--resume')){
  const saved=JSON.parse(await fs.readFile('/private/tmp/astra-test-session.json','utf8'));
  initial=JSON.parse(await fs.readFile(out+'live-request.json','utf8'));
  const headers={Authorization:'Bearer '+saved.token,Origin:origin};
  let r=await ctx.request.get(api+'/agent/studies/'+saved.task_id,{headers});assert.equal(r.status(),200);task=await r.json();
  if(task.status==='needs_attention'){r=await ctx.request.post(api+'/agent/studies/'+saved.task_id+'/resume',{headers});assert.equal(r.status(),200);task=await r.json();}
  initial.agent=task;
  await fs.writeFile(out+'live-task.json',JSON.stringify(task,null,2)+'\n');
  await f.locator('body').evaluate((_,{token,receipt})=>{localStorage.setItem('astra-research-session-v1',token);localStorage.setItem('quantum-atlas-question-v3',JSON.stringify(receipt));},{token:saved.token,receipt:initial});
  await page.reload();await f.locator('#resume-study').click();console.log(JSON.stringify({resumed:task.id,status:task.status}));
 }else{
 await f.locator('#question').fill('Do quantum responses generated from Fibonacci, Phi Ratio and M9 P24 Alpha, Beta, Gamma and Epsilon representations improve our ability to predict transcription factor occupancy around OXTR? Continue the available analysis using existing data and local simulations.');
 const reply=page.waitForResponse(r=>r.url()===api+'/quantum/question'&&r.request().method()==='POST',{timeout:120000});await f.locator('#ask-button').click();const response=await reply;initial=await response.json();assert.equal(response.status(),200);assert(initial.agent?.policy.requireGeminiReview);task=initial.agent;const token=await f.locator('body').evaluate(()=>localStorage.getItem('astra-research-session-v1'));await fs.writeFile('/private/tmp/astra-test-session.json',JSON.stringify({token,task_id:task.id}),{mode:0o600});await fs.writeFile(out+'live-request.json',JSON.stringify(initial,null,2)+'\n');console.log(JSON.stringify({submitted:task.id,policy:task.policy}));
 }
 const start=Date.now();while(!['complete','failed','needs_attention','cancelled'].includes(task.status)&&Date.now()-start<1250000)await new Promise(r=>setTimeout(r,3000));
 assert.equal(task.status,'complete',task.message);assert.equal(task.report.review.status,'model_review_complete');assert.deepEqual(['design','final'].map(p=>task.reviews.filter(r=>r.phase===p).at(-1).decision),['approve','approve']);assert.equal(task.report.spend.new_Aquila_USD,0);
 await f.locator('#agent-open').click();await f.locator('#results-dialog[open]').waitFor();const insight=await f.locator('#results-dialog-content').innerText();assert.match(insight,/Reviewed by Gemini 4 Argon/);assert.match(insight,/What we found/);assert(!insight.includes('Draft for Gemini scientific review:'));await page.screenshot({path:out+'live-reviewed-insights.png'});await f.locator('#close-results').click();
 await f.locator('#scientific-review-rail [data-scientific-review=final]').click();await f.locator('#scientific-review-content summary').click();assert.match(await f.locator('#scientific-review-content').innerText(),new RegExp(task.id));await f.locator('#close-scientific-review').click();
 for(const format of ['JSON','CSV']){await f.locator('#agent-open').click();const dp=page.waitForEvent('download');await f.locator('#results-dialog-content button').filter({hasText:'Download complete study · '+format}).click();const d=await dp;await d.saveAs(out+'live-result.'+format.toLowerCase());await f.locator('#close-results').click();}
 await f.locator('#new-question').click();assert((await f.locator('.stage-state').allTextContents()).every(x=>!x));assert.equal(await f.locator('#final-review-status').innerText(),'');assert.deepEqual(errors,[]);
 await fs.writeFile(out+'live-proof.json',JSON.stringify({passed:true,task_id:task.id,model:task.report.model,reviews:task.reviews,spend:task.report.spend,errors,muted:true},null,2)+'\n');console.log('LIVE_REVIEWED_WORKFLOW_PASSED');
}finally{await ctx.storageState({path:'/private/tmp/astra-live-test-browser.json'});await fs.chmod('/private/tmp/astra-live-test-browser.json',0o600);for(const p of ctx.pages())for(const fr of p.frames())await fr.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause())).catch(()=>{});await browser.close();await server.close();}
