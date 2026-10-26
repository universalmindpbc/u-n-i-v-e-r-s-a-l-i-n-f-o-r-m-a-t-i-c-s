import fs from 'node:fs';import assert from 'node:assert/strict';
import {installQAAuth,readBudget,endpoint} from './qa-auth.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const output=process.env.UO_QA_OUTPUT||'/private/tmp/uo-specialty-qa';fs.mkdirSync(output,{recursive:true});
const report={processAudioDisabled:true,nativeFullscreenInvoked:false,checks:[],network:[],errors:[],budgetBefore:await readBudget()};
const browser=await chromium.launch({headless:true,args:['--mute-audio','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',`--use-file-for-fake-audio-capture=${output}/music-paused.wav`]});
const p=await browser.newPage({viewport:{width:1470,height:1000}});await installQAAuth(p);
await p.addInitScript(()=>{const PC=RTCPeerConnection;window.__events=[];window.__sent=[];window.__audios=[];window.RTCPeerConnection=class extends PC{createDataChannel(...args){const c=super.createDataChannel(...args);window.__channel=c;const send=c.send.bind(c);c.send=v=>{const e=JSON.parse(v);window.__sent.push(e);send(v);};c.addEventListener('message',e=>{const item=JSON.parse(e.data);window.__events.push({type:item.type,transcript:item.transcript,status:item.response?.status,error:item.error});});return c;}};const A=Audio;window.Audio=class extends A{constructor(...args){super(...args);window.__audios.push(this);}};});
p.on('pageerror',e=>report.errors.push(e.message));p.on('response',async r=>{if(r.url().startsWith(endpoint)||r.url().includes('/realtime/calls'))report.network.push({url:r.url(),status:r.status()});});
const count=()=>p.locator('.uo-message[data-role=assistant]').count();
async function answerAfter(n){await p.waitForFunction(n=>document.querySelectorAll('.uo-message[data-role=assistant]').length>n,n,{timeout:240000});return p.locator('.uo-message[data-role=assistant]').last().innerText();}
async function idle(){await p.waitForFunction(()=>UniversalOneResearch.getState().live.active&&!UniversalOneResearch.getState().live.gated&&!UniversalOneResearch.getState().live.pending,undefined,{timeout:120000});}
async function textQuestion(question){const n=await count();await p.locator('#uo-question').fill(question);await p.locator('[data-send]').click();const answer=await answerAfter(n);assert(await p.locator('.uo-message[data-role=assistant]').last().locator('.uo-sources').count());return answer;}
async function voiceText(question,expectSources=false){const n=await count();await p.evaluate(q=>{__channel.send(JSON.stringify({type:'conversation.item.create',item:{type:'message',role:'user',content:[{type:'input_text',text:q}]}}));__channel.send(JSON.stringify({type:'response.create'}));},question);await answerAfter(n);if(expectSources)await p.waitForFunction(n=>[...document.querySelectorAll('.uo-message[data-role=assistant]')].slice(n).some(e=>e.querySelector('.uo-sources')),n,{timeout:240000});await idle();return p.locator('.uo-message[data-role=assistant]').last().innerText();}
try{
 await p.goto('https://staging.universalmind.tv/informatics/?area=geometry&record=fibonacci-m9-p24',{waitUntil:'domcontentloaded'});await p.waitForFunction(()=>window.UniversalOneResearch&&window.UniversalOneNeedsResearch);await p.locator('#launch:not([disabled])').waitFor({timeout:60000});await p.evaluate(()=>document.dispatchEvent(new CustomEvent('atlas:navigate',{detail:'geometry'})));await p.locator('#experience').waitFor({state:'visible'});
 for(const width of [1470,1024,390,320]){
  await p.setViewportSize({width,height:900});await p.locator('.uo .iam-realtime-voice__summon').hover();await p.waitForTimeout(350);
  const planet=await p.locator('.uo .iam-realtime-voice__summon').boundingBox(),label=await p.locator('.uo .iam-realtime-voice__invitation').boundingBox();
  assert(Math.abs((label.y+label.height/2)-(planet.y+planet.height/2))<2);assert(label.x>=0);assert(label.x+label.width<planet.x);
  await p.screenshot({path:`${output}/rollover-${width}.png`});report.checks.push({width,rolloverAligned:true});
 }
 await p.setViewportSize({width:1470,height:1000});await p.locator('.uo .iam-realtime-voice__summon').click();
 if(!process.env.UO_QA_SKIP_TEXT){
 const text=await textQuestion('Electronic music making and gene expression.');assert.match(text,/recreational|music.making/i);assert.doesNotMatch(text,/fun mash.up/i);report.checks.push({textMusic:text});
 const follow=await textQuestion('Are there any studies on the informatics website around music making and gene expression?');assert.match(follow,/recreational|music.making/i);report.checks.push({textFollowUp:follow});
 await p.locator('[data-clear]').click();
 }
 await p.evaluate(()=>{const v=document.querySelector('video');if(v){v.volume=.8;v.muted=true;}});
 await p.locator('[data-mode=voice]').click();await p.waitForFunction(()=>UniversalOneResearch.getState().live.connected,undefined,{timeout:45000});
 await p.waitForFunction(()=>window.__events.some(e=>e.type==='conversation.item.input_audio_transcription.completed'&&/music/i.test(e.transcript)),undefined,{timeout:45000});
 await p.locator('[data-mute]').click();
 await p.waitForFunction(()=>document.querySelector('.uo-message[data-role=assistant] .uo-sources')||document.querySelector('.uo').dataset.state==='error',undefined,{timeout:240000});assert.notEqual(await p.locator('.uo').getAttribute('data-state'),'error');await idle();
 const spoken=await p.locator('.uo-message[data-role=assistant]').last().innerText();assert.match(spoken,/recreational|music.making/i);assert.doesNotMatch(spoken,/fun mash.up|daily allowance/i);
 assert(await p.evaluate(()=>__sent.some(e=>e.type==='response.create'&&e.response?.tool_choice?.name==='ask_research')));assert(await p.evaluate(()=>__audios.some(a=>a.srcObject&&a.currentTime>1)));
 console.log('Real speech, retrieval and decoded reply passed');report.checks.push({actualMicrophoneTranscript:await p.locator('.uo-message[data-role=user]').last().innerText(),forcedResearchBeforeVoice:true,spokenAnswer:spoken,remoteAudioDecoded:true});
 const voiceFollow=await voiceText('Are there any studies on the informatics website around music making and gene expression?',true);assert.match(voiceFollow,/music|gene/i);assert(await p.locator('.uo-message[data-role=assistant]').last().locator('.uo-sources').count());report.checks.push({voiceFollowUp:voiceFollow});
 const before=report.network.filter(x=>x.url.endsWith('/research')).length;const general=await voiceText('Thank you.');assert.equal(report.network.filter(x=>x.url.endsWith('/research')).length,before);assert.equal(await p.locator('.uo-message[data-role=assistant]').last().locator('.uo-sources').count(),0);report.checks.push({generalConcept:general,socialAcknowledgementWithoutLookup:true});
 for(const width of [1470,1024,390,320]){
  await p.setViewportSize({width,height:900});for(const theme of ['video','black','white']){
   await p.evaluate(t=>document.documentElement.dataset.readingBackground=t,theme);
   const details=await p.locator('.uo-dock').evaluate(e=>{const r=e.getBoundingClientRect(),s=getComputedStyle(e);return {x:r.x,right:r.right,y:r.y,bottom:r.bottom,background:s.backgroundColor,border:s.borderWidth};});assert(details.x>=0&&details.right<=width&&details.y>=0);assert(!/rgba.*0\)$|transparent/.test(details.background));assert.equal(details.border,'1px');
   for(const button of await p.locator('.uo-dock-actions button:visible').all()){const r=await button.boundingBox();assert(r.x>=details.x&&r.x+r.width<=details.right&&r.y+r.height<=details.bottom);}
  }
  await p.evaluate(()=>document.documentElement.dataset.readingBackground='video');await p.screenshot({path:`${output}/voice-casing-${width}.png`});report.checks.push({width,casingOpaque:true,allControlsContained:true,themes:3});
 }
 await p.setViewportSize({width:1470,height:1000});await p.locator('.uo .iam-realtime-voice__summon').click();await p.locator('[data-transcript]').click();await p.screenshot({path:`${output}/transcript.png`});
 await p.locator('[data-end]').click();await p.waitForTimeout(1200);assert(!await p.evaluate(()=>UniversalOneResearch.getState().microphone));
 report.budgetAfter=await readBudget();assert.deepEqual(report.budgetAfter.visitors.counts,report.budgetBefore.visitors.counts,'Verification leaves visitor allowance intact');
 assert(!report.network.some(x=>x.status>=400));assert.deepEqual(report.errors,[]);
}catch(error){report.failure=error.stack;throw error;}finally{
 report.finalState=await p.evaluate(()=>UniversalOneResearch?.getState()).catch(()=>null);report.events=await p.evaluate(()=>__events).catch(()=>[]);fs.writeFileSync(`${output}/specialty-verification.json`,JSON.stringify(report,null,2));await p.evaluate(()=>{UniversalOneResearch?.close();document.querySelectorAll('video,audio').forEach(m=>m.pause());window.__audios?.forEach(a=>a.pause());}).catch(()=>{});await browser.close();
}
console.log(JSON.stringify({passed:report.checks.length,requests:report.network.length,quotaIsolated:true}));
