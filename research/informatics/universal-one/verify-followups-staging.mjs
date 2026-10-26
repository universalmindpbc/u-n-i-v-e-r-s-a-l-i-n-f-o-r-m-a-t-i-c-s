import fs from 'node:fs';
import assert from 'node:assert/strict';
import {installQAAuth,readBudget,endpoint} from './qa-auth.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const output=process.env.UO_QA_OUTPUT||'/private/tmp/uo-followup-qa';
fs.mkdirSync(output,{recursive:true});
const questions=JSON.parse(fs.readFileSync(output+'/questions.json','utf8'));
const report={processAudioDisabled:true,syntheticMicrophoneOnly:true,nativeFullscreenInvoked:false,turns:[],requests:[],errors:[],budgetBefore:await readBudget()};
const browser=await chromium.launch({headless:true,args:['--mute-audio','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--autoplay-policy=no-user-gesture-required']});
const p=await browser.newPage({viewport:{width:1470,height:1000}});
await installQAAuth(p);
await p.addInitScript(()=>{
 window.__events=[];window.__sent=[];window.__audios=[];window.__fixtureContexts=[];
 navigator.mediaDevices.getUserMedia=async()=>{
  const ctx=new AudioContext();await ctx.resume();const destination=ctx.createMediaStreamDestination();
  const silence=ctx.createConstantSource();silence.offset.value=0;silence.connect(destination);silence.start();
  window.__fixtureContexts.push(ctx);window.__fixtureContext=ctx;window.__fixtureDestination=destination;
  return destination.stream;
 };
 const PC=RTCPeerConnection;
 window.RTCPeerConnection=class extends PC{createDataChannel(...args){const c=super.createDataChannel(...args);const send=c.send.bind(c);c.send=v=>{const e=JSON.parse(v);window.__sent.push({type:e.type,toolChoice:e.response?.tool_choice});send(v);};c.addEventListener('message',m=>{const e=JSON.parse(m.data);window.__events.push({type:e.type,transcript:e.transcript,status:e.response?.status,error:e.error});});return c;}};
 const A=Audio;window.Audio=class extends A{constructor(...args){super(...args);window.__audios.push(this);}};
});
p.on('pageerror',e=>report.errors.push(e.message));
p.on('response',r=>{if(r.url().startsWith(endpoint)||r.url().includes('/realtime/calls')){
 const body=r.request().postData()||'';let parsed={};try{parsed=JSON.parse(body);}catch{}
 const entry={path:new URL(r.url()).pathname,status:r.status(),bytes:Buffer.byteLength(body),historyMessages:parsed.history?.length,previousReferences:parsed.references?.length,question:parsed.question,area:parsed.context?.area};
 report.requests.push(entry);
}});
const checkpoint=()=>fs.writeFileSync(output+'/verification.json',JSON.stringify(report,null,2));
const idle=()=>p.waitForFunction(()=>{const s=UniversalOneResearch.getState().live;return s.active&&!s.gated&&!s.pending&&!s.playing;},undefined,{timeout:240000});
async function snapshot(){return p.evaluate(()=>({rawHistoryBytes:new TextEncoder().encode(JSON.stringify(UniversalOneConversation)).length,forced:__sent.filter(e=>e.type==='response.create'&&e.toolChoice?.name==='ask_research').length,users:UniversalOneConversation.filter(m=>m.role==='user').length,answers:UniversalOneConversation.filter(m=>m.role==='assistant'&&m.sources?.length).length}));}
async function voiceTurn(i){
 await idle();const before=await snapshot(),requestCount=report.requests.length;
 const fixture=fs.readFileSync(`${output}/question-${i}.mp3`).toString('base64');
 await p.evaluate(async base64=>{const bytes=Uint8Array.from(atob(base64),c=>c.charCodeAt(0));const buffer=await __fixtureContext.decodeAudioData(bytes.buffer);const node=__fixtureContext.createBufferSource();node.buffer=buffer;node.connect(__fixtureDestination);node.start();},fixture);
 await p.waitForFunction(n=>UniversalOneConversation.filter(m=>m.role==='user').length>n,before.users,{timeout:45000});
 await p.waitForFunction(n=>UniversalOneConversation.filter(m=>m.role==='assistant'&&m.sources?.length).length>n,before.answers,{timeout:240000});
 await idle();const after=await snapshot();
 assert(after.forced>before.forced,'Each spoken follow-up forces research');
 const exchange=await p.evaluate(()=>({question:UniversalOneConversation.findLast(m=>m.role==='user').content,answer:UniversalOneConversation.findLast(m=>m.role==='assistant').content,sources:UniversalOneConversation.findLast(m=>m.role==='assistant').sources?.map(s=>({id:s.id,title:s.title,url:s.url}))}));
 const requests=report.requests.slice(requestCount).filter(r=>r.path==='/research');
 assert(requests.length>0);assert(requests.every(r=>r.status===200&&r.bytes<=44000));assert(exchange.sources.length>0);
 assert.doesNotMatch(exchange.answer,/request is too large|daily allowance|fun mash.up/i);
 report.turns.push({mode:'voice',fixture:questions[i],...exchange,...after,requests});checkpoint();
 console.log(JSON.stringify({turn:report.turns.length,mode:'voice',sources:exchange.sources.length,requestBytes:requests.map(r=>r.bytes),rawHistoryBytes:after.rawHistoryBytes}));
}
try{
 await p.goto('https://staging.universalmind.tv/informatics/?area=vision',{waitUntil:'domcontentloaded'});
 await p.waitForFunction(()=>window.UniversalOneResearch&&window.UniversalOneContext);await p.locator('#launch:not([disabled])').waitFor({timeout:60000});
 await p.evaluate(()=>document.dispatchEvent(new CustomEvent('atlas:navigate',{detail:'vision'})));await p.locator('#experience').waitFor({state:'visible'});
 await p.locator('.uo .iam-realtime-voice__summon').click();
 assert.equal(await p.locator('.uo-privacy').count(),0);assert(!await p.locator('.uo-compose').innerText().then(t=>t.includes('messages and recordings')));report.footerAbsent=true;
 await p.screenshot({path:output+'/composer-desktop.png'});
 await p.locator('[data-mode=voice]').click();await p.waitForFunction(()=>UniversalOneResearch.getState().live.connected,undefined,{timeout:45000});await idle();
 for(let i=0;i<questions.length;i++)await voiceTurn(i);
 assert(await p.evaluate(()=>__audios.some(a=>a.srcObject&&a.currentTime>1)));report.remoteAudioDecoded=true;
 await p.locator('.uo .iam-realtime-voice__summon').click();await p.locator('[data-text]').click();
 const before=await snapshot(),requestCount=report.requests.length;
 const question='Which paper supports the younger-participant finding, and how many participants were included? Answer briefly.';
 await p.locator('#uo-question').fill(question);await p.locator('[data-send]').click();
 await p.waitForFunction(n=>UniversalOneConversation.filter(m=>m.role==='assistant'&&m.sources?.length).length>n,before.answers,{timeout:240000});
 const answer=await p.locator('.uo-message[data-role=assistant]').last().innerText();const after=await snapshot();
 const requests=report.requests.slice(requestCount).filter(r=>r.path==='/research');assert(requests.every(r=>r.status===200&&r.bytes<=44000));assert(requests.length);assert.match(answer,/\d/);
 report.turns.push({mode:'text',question,answer,...after,requests});checkpoint();console.log(JSON.stringify({turn:report.turns.length,mode:'text',requestBytes:requests.map(r=>r.bytes),rawHistoryBytes:after.rawHistoryBytes}));
 await p.screenshot({path:output+'/followup-transcript-desktop.png'});await p.setViewportSize({width:390,height:844});await p.screenshot({path:output+'/followup-transcript-mobile.png'});
 const box=await p.locator('.uo-panel').boundingBox();assert(box.x>=0&&box.x+box.width<=391);assert.equal(await p.locator('.uo-privacy').count(),0);
 await p.setViewportSize({width:1470,height:1000});await p.locator('[data-mode=voice]').click();await p.waitForFunction(()=>UniversalOneResearch.getState().live.connected,undefined,{timeout:45000});await idle();
 const sessions=report.requests.filter(r=>r.path==='/voice-session');assert.equal(sessions.length,2);assert(sessions.every(r=>r.status===200&&r.bytes<100));report.reconnectedWithRetainedTranscript=true;
 await voiceTurn(5);
 await p.locator('[data-end]').click();await p.waitForTimeout(1000);assert(!await p.evaluate(()=>UniversalOneResearch.getState().microphone));report.microphoneReleased=true;
 report.budgetAfter=await readBudget();assert.deepEqual(report.budgetAfter.visitors.counts,report.budgetBefore.visitors.counts);assert(!report.requests.some(r=>r.status>=400));assert.deepEqual(report.errors,[]);
 report.success=true;
}catch(error){report.failure=error.stack;throw error;}finally{
 report.finalState=await p.evaluate(()=>UniversalOneResearch?.getState()).catch(()=>null);report.events=await p.evaluate(()=>__events).catch(()=>[]);
 await p.evaluate(async()=>{UniversalOneResearch?.close();document.querySelectorAll('video,audio').forEach(m=>m.pause());window.__audios?.forEach(a=>a.pause());await Promise.all(window.__fixtureContexts?.map(c=>c.close())||[]);}).catch(()=>{});
 await browser.close();report.browserClosed=true;checkpoint();
}
console.log(JSON.stringify({passed:report.turns.length,footerAbsent:report.footerAbsent,reconnected:true,quotaIsolated:true}));
