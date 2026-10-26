import fs from 'node:fs';
import assert from 'node:assert/strict';
import {installQAAuth,readBudget,endpoint,origin} from './qa-auth.mjs';
const {chromium}=await import('/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const output=process.env.UO_QA_OUT||'/private/tmp/uo-research-access-qa';fs.mkdirSync(output,{recursive:true});
const report={processAudioDisabled:true,syntheticMicrophoneOnly:true,nativeFullscreenInvoked:false,layouts:[],requests:[],errors:[],budgetBefore:await readBudget()};
const browser=await chromium.launch({headless:true,args:['--mute-audio','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--autoplay-policy=no-user-gesture-required']});
const p=await browser.newPage({viewport:{width:1470,height:1000}});
await installQAAuth(p);
await p.addInitScript(()=>{
 window.__audios=[];window.__fixtureContexts=[];
 navigator.mediaDevices.getUserMedia=async()=>{const ctx=new AudioContext();await ctx.resume();const dest=ctx.createMediaStreamDestination();const silence=ctx.createConstantSource();silence.offset.value=0;silence.connect(dest);silence.start();window.__fixtureContexts.push(ctx);return dest.stream;};
 const A=Audio;window.Audio=class extends A{constructor(...args){super(...args);window.__audios.push(this);}};
});
p.on('pageerror',e=>report.errors.push(e.message));
p.on('response',r=>{if(r.url().startsWith(endpoint)||r.url().includes('/realtime/calls')){let data={};try{data=JSON.parse(r.request().postData()||'{}');}catch{}report.requests.push({path:new URL(r.url()).pathname,status:r.status(),question:data.question,context:data.context});checkpoint();}});
const checkpoint=()=>fs.writeFileSync(output+'/verification.json',JSON.stringify(report,null,2));
const summon=()=>p.locator('.uo .iam-realtime-voice__summon');
async function bounds(label){
 const result=await p.evaluate(()=>{const root=document.querySelector('.uo'),host=document.querySelector('.library-overlay[open]');const box=e=>e?.getBoundingClientRect().toJSON();const planet=root.querySelector('button'),b=planet.getBoundingClientRect();const panel=root.querySelector('.uo-panel'),dock=root.querySelector('.uo-dock');return {viewport:{width:innerWidth,height:innerHeight},parent:root.parentElement.className,host:box(host),planet:box(planet),panel:panel.hidden?null:box(panel),dock:dock.hidden?null:box(dock),hit:planet.contains(document.elementFromPoint(b.x+b.width/2,b.y+b.height/2)),count:document.querySelectorAll('.uo').length,panelBackground:getComputedStyle(panel).backgroundColor};});
 const area=result.host||{left:0,top:0,right:result.viewport.width,bottom:result.viewport.height};
 for(const key of ['planet','panel','dock']){const b=result[key];if(!b)continue;assert(b.left>=area.left-1&&b.right<=area.right+1&&b.top>=area.top-1&&b.bottom<=area.bottom+1,`${label} ${key} fits its host: ${JSON.stringify(result)}`);}
 assert(result.hit,`${label}: planet receives clicks`);assert.equal(result.count,1);report.layouts.push({label,...result});checkpoint();
}
try{
 await p.goto(origin+'/informatics/?area=sound',{waitUntil:'domcontentloaded'});await p.waitForFunction(()=>window.UniversalOneResearch);await p.locator('#launch:not([disabled])').waitFor({timeout:60000});
 await p.evaluate(()=>document.dispatchEvent(new CustomEvent('atlas:navigate',{detail:'sound'})));await p.locator('#experience').waitFor({state:'visible'});
 await p.locator('[data-library="sound"]').filter({visible:true}).first().click();
 const frame=p.frameLocator('.library-overlay iframe');await frame.locator('#collection-statistics:not([hidden])').waitFor({timeout:60000});await bounds('research-desktop');
 assert.equal(await frame.locator('.uo').count(),0);report.parentOwnedSingleton=true;
 await summon().hover();await p.screenshot({path:output+'/planet-desktop.png'});await summon().click();await bounds('research-text-desktop');
 for(const theme of ['black','white','video']){
  await p.locator('.uo-minimize').click();await frame.locator(`[data-reading-background="${theme}"]`).filter({visible:true}).first().click();await p.waitForFunction(t=>document.querySelector('.library-overlay').dataset.readingBackground===t,theme);await summon().click();await bounds('research-'+theme);
 }
 const question='Which studies in our Sound collection report gene-expression changes after recreational music making? Give a brief answer with sources.';
 await p.locator('#uo-question').fill(question);const outgoing=p.waitForRequest(r=>r.url()===endpoint+'/research',{timeout:15000});await p.locator('[data-send]').click();await outgoing;
 await p.waitForFunction(()=>UniversalOneConversation.some(m=>m.role==='assistant'&&m.sources?.length),undefined,{timeout:240000});
 report.answer=await p.evaluate(()=>UniversalOneConversation.findLast(m=>m.role==='assistant'));assert(report.answer.sources.length>0);assert.match(report.answer.content,/music|IL6|interleukin/i);
 const research=report.requests.filter(r=>r.path==='/research');assert(research.some(r=>r.status===200&&r.context?.library));report.textWorkflowVerified=true;checkpoint();console.log('Cited text answer displayed in Research.');
 await p.waitForTimeout(900);await p.screenshot({path:output+'/text-answer-desktop.png'});
 for(const size of [{width:1024,height:768},{width:390,height:844},{width:320,height:700}]){await p.setViewportSize(size);await p.waitForTimeout(300);await bounds('research-text-'+size.width);await p.screenshot({path:output+`/text-${size.width}.png`});}
 await p.setViewportSize({width:1470,height:1000});await p.locator('[data-mode=voice]').click();
 await p.waitForFunction(()=>UniversalOneResearch.getState().live.connected,undefined,{timeout:45000});
 await p.waitForFunction(()=>UniversalOneConversation.some(m=>m.role==='assistant'&&/^Hi[.!]?\s+What research question can I help you with\?/i.test(m.content)),undefined,{timeout:45000});
 await p.waitForFunction(()=>{const s=UniversalOneResearch.getState().live;return s.active&&!s.gated&&!s.pending&&!s.playing;},undefined,{timeout:45000});
 report.greeting=await p.evaluate(()=>UniversalOneConversation.findLast(m=>m.role==='assistant').content);assert.match(report.greeting,/^Hi[.!]?\s/);assert(!/Namaste/i.test(report.greeting));
 assert(await p.evaluate(()=>__audios.some(a=>a.srcObject&&a.currentTime>1)));report.remoteAudioDecoded=true;await bounds('research-voice-desktop');await p.screenshot({path:output+'/voice-desktop.png'});
 for(const width of [390,320]){await p.setViewportSize({width,height:844});await p.waitForTimeout(300);await bounds('research-voice-'+width);}
 await p.setViewportSize({width:1470,height:1000});
 const turns=await p.evaluate(()=>UniversalOneConversation.length);
 await p.locator('.library-overlay-bar button[aria-label="Close Universal Research"]').click();await p.waitForTimeout(200);await bounds('atlas-after-research');
 assert(await p.evaluate(()=>UniversalOneResearch.getState().live.connected));
 await p.locator('[data-library="sound"]').filter({visible:true}).first().click();await frame.locator('#collection-statistics:not([hidden])').waitFor();await bounds('research-reopened-during-voice');assert.equal(await p.evaluate(()=>UniversalOneConversation.length),turns);assert(await p.evaluate(()=>UniversalOneResearch.getState().live.connected));report.conversationAndVoiceRetained=true;
 await p.locator('[data-end]').click();assert(!await p.evaluate(()=>UniversalOneResearch.getState().microphone));report.microphoneReleased=true;
 await p.locator('.library-overlay-bar button[aria-label="Close Universal Research"]').click();
 await p.locator('[data-about]').filter({visible:true}).first().click();await p.locator('[data-network]').click();await p.locator('.network-overlay [data-library-gene="OXTR"]').first().click();await frame.locator('#collection-statistics:not([hidden])').waitFor();await bounds('research-from-network');assert.equal(await p.locator('.uo').evaluate(e=>e.parentElement.className),'library-overlay');report.topmostResearchHostVerified=true;
 await summon().click();assert(await p.locator('.uo-message[data-role=assistant]').count());await p.locator('.uo-minimize').click();
 if(await frame.locator('[data-view="search"]').isVisible())await frame.locator('[data-view="search"]').click();await frame.locator('#results [data-record]').first().click();await frame.locator('#article[open]').waitFor();await bounds('research-paper-open');await summon().click();await bounds('research-paper-with-conversation');report.paperOverlayAccess=true;
 await p.locator('.uo-close').click();await p.locator('.library-overlay-bar button[aria-label="Close Universal Research"]').click();assert.equal(await p.locator('.uo').evaluate(e=>e.parentElement.className),'network-overlay');
 await p.locator('[data-network-close]').click();await p.locator('#about-close').click();
 await p.goto(origin+'/informatics/library.html?collection=sound',{waitUntil:'domcontentloaded'});await p.waitForFunction(()=>window.UniversalOneResearch);await summon().click();await bounds('standalone-library');report.standaloneAccessible=true;
 report.budgetAfter=await readBudget();assert.deepEqual(report.budgetAfter.visitors.counts,report.budgetBefore.visitors.counts);assert(report.requests.every(r=>r.status<400));assert.deepEqual(report.errors,[]);report.success=true;
} catch(error){report.failure=error.stack;report.failureState=await p.evaluate(()=>({state:window.UniversalOneResearch?.getState(),status:document.querySelector('.uo-status')?.textContent,history:window.UniversalOneConversation})).catch(()=>null);throw error;} finally{
 for(const frame of p.frames())await frame.evaluate(()=>{document.querySelectorAll('video,audio').forEach(m=>m.pause());}).catch(()=>{});
 await p.evaluate(async()=>{window.UniversalOneResearch?.close();window.__audios?.forEach(a=>a.pause());await Promise.all(window.__fixtureContexts?.map(c=>c.close())||[]);}).catch(()=>{});await browser.close();report.browserClosed=true;checkpoint();
}
console.log(JSON.stringify({success:report.success,layouts:report.layouts.length,greeting:report.greeting,textWorkflowVerified:report.textWorkflowVerified}));
