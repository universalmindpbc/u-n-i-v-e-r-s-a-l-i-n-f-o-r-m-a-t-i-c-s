import {installQAAuth} from './qa-auth.mjs';
import fs from 'node:fs';import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const output=process.env.UO_QA_OUTPUT||'/private/tmp/uo-qa';fs.mkdirSync(output,{recursive:true});
// Supply synthetic question-paused.wav with silence between repetitions; never use a real microphone.
const report={processAudioDisabled:true,checks:[],network:[],errors:[]};
const b=await chromium.launch({headless:true,args:['--mute-audio','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',`--use-file-for-fake-audio-capture=${output}/question-paused.wav`]});
const p=await b.newPage({viewport:{width:1470,height:1000}});
await installQAAuth(p);
await p.addInitScript(()=>{
 const PC=window.RTCPeerConnection;window.__events=[];window.__audios=[];
 window.RTCPeerConnection=class extends PC{createDataChannel(...args){const c=super.createDataChannel(...args);window.__channel=c;c.addEventListener('message',e=>{try{const v=JSON.parse(e.data);window.__events.push({type:v.type,error:v.error,responseStatus:v.response?.status,details:v.response?.status_details,transcript:v.transcript});}catch{}});return c;}};
 const A=window.Audio;window.Audio=class extends A{constructor(...args){super(...args);window.__audios.push(this);}};
});
p.on('pageerror',e=>report.errors.push(e.message));p.on('response',r=>{if(r.url().includes('workers.dev')||r.url().includes('/realtime/calls'))report.network.push({url:r.url(),status:r.status()});});
async function until(fn,ms=180000){try{await p.waitForFunction(source=>Function('return ('+source+')()')()||document.querySelector('.uo')?.dataset.state==='error',fn.toString(),{timeout:ms});assert.notEqual(await p.locator('.uo').getAttribute('data-state'),'error');}catch(e){console.log(await p.evaluate(()=>({state:UniversalOneResearch.getState(),status:document.querySelector('.uo-status').textContent,events:window.__events?.slice(-20)})));throw e;}}
try{
 await p.goto('https://staging.universalmind.tv/informatics/library.html?collection=all&view=search',{waitUntil:'domcontentloaded'});await until(()=>window.UniversalOneResearch,30000);
 await p.locator('.uo .iam-realtime-voice__summon').hover();assert.match(await p.locator('.iam-realtime-voice__invitation').innerText(),/Ask a research question/);
 await p.locator('.uo .iam-realtime-voice__summon').click();await p.waitForTimeout(700);assert.equal(await p.locator('.uo .iam-realtime-voice__summon').isVisible(),true);assert.equal(await p.locator('.uo-character-toggle,.uo .iam-realtime-character').count(),0);assert.equal(await p.locator('.iam-realtime-voice__invitation').innerText(),'Ask a research question');assert.equal(await p.locator('.uo-header h2').innerText(),'Universal One');
 await p.locator('#uo-question').fill('What is an electron? Explain the concept in two sentences.');await p.locator('[data-send]').click();await until(()=>document.querySelector('.uo-message[data-role=assistant]'));const general=await p.locator('.uo-message[data-role=assistant]').last().innerText();assert.match(general,/electron/i);assert.equal(await p.locator('.uo-message[data-role=assistant] .uo-sources').count(),0);report.checks.push({generalText:general});await p.locator('[data-clear]').click();
 await p.locator('#uo-question').fill('How many unique publications and permitted full-text articles are indexed?');await p.locator('[data-send]').click();await until(()=>document.querySelector('.uo-message[data-role=assistant]'));
 const answer=await p.locator('.uo-message[data-role=assistant]').last().innerText();assert(answer.includes('811')&&answer.includes('149'));report.checks.push({textAstra:true,answer});console.log('Text Astra passed');
 await p.locator('.uo [data-record]').click();await until(()=>UniversalOneResearch.getState().waveform.level>.02,15000);assert(await p.locator('.uo-compose canvas').isVisible());await p.waitForTimeout(4000);await p.locator('.uo [data-record]').click();await until(()=>document.querySelector('.uo-status').textContent.includes('Review your words'),60000);const dictated=await p.locator('#uo-question').inputValue();assert(dictated.toLowerCase().includes('publication'));report.checks.push({dictation:dictated,microphoneWaveform:true});console.log('Dictation + waveform passed');
 await p.waitForTimeout(1100);await p.evaluate(()=>{const v=document.querySelector('video');v.volume=.8;v.muted=true});
 await p.locator('[data-mode=voice]').click();assert.equal(await p.locator('.uo-panel').isVisible(),false);assert.equal(await p.locator('.uo-compose').isVisible(),false);
 await until(()=>UniversalOneResearch.getState().live.connected,40000);console.log('WebRTC connected');
 await until(()=>UniversalOneResearch.getState().playing,30000);assert(await p.evaluate(()=>UniversalOneResearch.getState().inputGated));
 await until(()=>UniversalOneResearch.getState().live.inputEnabled,30000);
 await until(()=>UniversalOneResearch.getState().waveform.level>.02,15000);
 await until(()=>window.__events.some(e=>e.type==='conversation.item.input_audio_transcription.completed'),40000);
 await p.locator('[data-mute]').click();
 await until(()=>document.querySelectorAll('.uo-message[data-role=assistant]').length>=3,180000);
 await until(()=>!UniversalOneResearch.getState().live.gated&&!UniversalOneResearch.getState().live.pending,60000);
 const mix=await p.evaluate(()=>document.querySelector('video').volume);assert(Math.abs(mix-.12)<.01);assert(await p.evaluate(()=>window.__audios.some(a=>a.srcObject&&a.currentTime>1)));
 assert((await p.locator('.uo-message[data-role=assistant]').last().innerText()).includes('811'));report.checks.push({liveSpeechAutomatic:true,transcript:await p.locator('.uo-log').innerText(),realRemoteAudioDecoded:true,duckedVolume:mix});console.log('Live speech -> research -> voice passed');
 // A second question through the same Realtime connection verifies citation display.
 const before=await p.locator('.uo-message[data-role=assistant]').count();
 await p.evaluate(()=>{__channel.send(JSON.stringify({type:'conversation.item.create',item:{type:'message',role:'user',content:[{type:'input_text',text:'What is the M9 Pisano period? Cite the Atlas record.'}]}}));__channel.send(JSON.stringify({type:'response.create'}));});
 await p.waitForFunction(n=>document.querySelectorAll('.uo-message[data-role=assistant]').length>n,before,{timeout:180000});
 await until(()=>document.querySelector('.uo-message:last-child .uo-sources'),180000);
 await p.locator('.uo .iam-realtime-voice__summon').click();await p.locator('[data-transcript]').click();assert.equal(await p.locator('.uo-compose').isVisible(),false);assert(await p.locator('.uo-panel').isVisible());
 await p.waitForTimeout(100);assert.equal(await p.locator('.uo-message button:visible').count(),0);await p.waitForFunction(()=>{const e=document.querySelector('.uo-log');return Math.abs(e.scrollHeight-e.clientHeight-e.scrollTop)<3;},undefined,{timeout:5000});report.checks.push({liveCitedAnswer:await p.locator('.uo-message[data-role=assistant]').last().innerText(),optionalTranscript:true});
 // A conceptual follow-up uses the voice model directly, even after Atlas retrieval.
 await until(()=>!UniversalOneResearch.getState().live.gated&&!UniversalOneResearch.getState().live.pending,60000);
 const researchBefore=report.network.filter(r=>r.url.endsWith('/research')).length;
 const messagesBefore=await p.locator('.uo-message[data-role=assistant]').count();
 await p.evaluate(()=>{__channel.send(JSON.stringify({type:'conversation.item.create',item:{type:'message',role:'user',content:[{type:'input_text',text:'What is an electron? Explain the general concept in two sentences.'}]}}));__channel.send(JSON.stringify({type:'response.create'}));});
 await p.waitForFunction(n=>document.querySelectorAll('.uo-message[data-role=assistant]').length>n,messagesBefore,{timeout:60000});
 await until(()=>!UniversalOneResearch.getState().live.gated&&!UniversalOneResearch.getState().live.pending,60000);
 const direct=await p.locator('.uo-message[data-role=assistant]').last().innerText();assert.match(direct,/electron/i);assert.equal(report.network.filter(r=>r.url.endsWith('/research')).length,researchBefore);assert.equal(await p.locator('.uo-message[data-role=assistant]').last().locator('.uo-sources').count(),0);report.checks.push({directVoiceKnowledge:direct,noResearchRoundTrip:true,noPreviousCitationsAttached:true});
 await p.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));});assert(await p.evaluate(()=>UniversalOneResearch.getState().live.connected));assert(!await p.evaluate(()=>UniversalOneResearch.getState().live.inputEnabled));
 await p.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:false});document.dispatchEvent(new Event('visibilitychange'));});assert(await p.evaluate(()=>UniversalOneResearch.getState().live.connected));assert(await p.evaluate(()=>UniversalOneResearch.getState().live.muted));assert(!await p.locator('[data-reconnect]').isVisible());report.checks.push({visibilityReturnWithoutReconnect:true,explicitMutePreserved:true});
 await p.screenshot({path:`${output}/live-transcript.png`});await p.locator('.uo-minimize').click();assert.equal(await p.locator('.uo-panel').isVisible(),false);
 await p.screenshot({path:`${output}/live-compact.png`});
 await p.locator('.uo .iam-realtime-voice__summon').click();await p.locator('[data-text]').click();await p.waitForTimeout(1200);const restored=await p.evaluate(()=>document.querySelector('video').volume);assert(Math.abs(restored-.8)<.01);assert.equal(await p.evaluate(()=>UniversalOneResearch.getState().microphone),false);assert(await p.locator('.uo-compose').isVisible());report.checks.push({textReturn:true,restoredVolume:restored,microphoneClosed:true});
 assert.equal(report.errors.length,0);
}catch(e){report.failure=e.stack;throw e;}
finally{report.events=await p.evaluate(()=>window.__events).catch(()=>[]);report.finalState=await p.evaluate(()=>window.UniversalOneResearch?.getState()).catch(()=>null);fs.writeFileSync(`${output}/live-verification.json`,JSON.stringify(report,null,2));await p.evaluate(()=>{UniversalOneResearch?.close();document.querySelectorAll('video,audio').forEach(m=>m.pause());window.__audios?.forEach(m=>m.pause());}).catch(()=>{});await b.close();}
console.log(JSON.stringify(report.checks,null,2));
