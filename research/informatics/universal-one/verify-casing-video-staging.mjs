import fs from 'node:fs';import assert from 'node:assert/strict';import {installQAAuth} from './qa-auth.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const output=process.env.UO_QA_OUTPUT||'/private/tmp/uo-specialty-qa';
const browser=await chromium.launch({headless:true,args:['--mute-audio','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});const p=await browser.newPage({viewport:{width:1470,height:1000}});await installQAAuth(p);
const report={processAudioDisabled:true,nativeFullscreenInvoked:false,checks:[]};
try{
 await p.goto('https://staging.universalmind.tv/informatics/',{waitUntil:'domcontentloaded'});await p.waitForFunction(()=>window.UniversalOneResearch);await p.locator('#launch:not([disabled])').waitFor({timeout:60000});
 await p.evaluate(async()=>{document.dispatchEvent(new CustomEvent('atlas:navigate',{detail:'geometry'}));document.dispatchEvent(new CustomEvent('atlas:choose-record',{detail:{id:'fibonacci-m9-p24',mode:'animation'}}));const v=document.querySelector('video');v.muted=true;v.volume=.8;if(!v.src)v.src=v.dataset.src;await v.play();document.querySelector('[data-stage="2"]').click();});
 await p.locator('#experience').waitFor({state:'visible'});assert(await p.locator('#landing').isHidden());await p.waitForFunction(()=>document.querySelector('video').currentTime>1,undefined,{timeout:45000});
 await p.locator('.uo .iam-realtime-voice__summon').hover();await p.waitForTimeout(400);await p.screenshot({path:`${output}/rollover-video-1470.png`});
 await p.locator('.uo .iam-realtime-voice__summon').click();await p.locator('[data-mode=voice]').click();await p.waitForFunction(()=>UniversalOneResearch.getState().live.active,undefined,{timeout:45000});await p.locator('[data-mute]').click();await p.waitForFunction(()=>!UniversalOneResearch.getState().live.gated,undefined,{timeout:45000});
 const mix=await p.evaluate(()=>document.querySelector('video').volume);assert(Math.abs(mix-.12)<.01);report.checks.push({liveVideoAdvancing:true,duckedVolume:mix});
 await p.mouse.move(500,100);await p.waitForTimeout(250);for(const width of [1470,320]){await p.setViewportSize({width,height:1000});await p.screenshot({path:`${output}/casing-video-${width}.png`});const r=await p.locator('.uo-dock').boundingBox();assert(r.x>=0&&r.x+r.width<=width);report.checks.push({width,liveVoiceCasingFits:true});}
 await p.locator('[data-end]').click();await p.waitForTimeout(1200);const restored=await p.evaluate(()=>document.querySelector('video').volume);assert(Math.abs(restored-.8)<.01);assert(!await p.evaluate(()=>UniversalOneResearch.getState().microphone));report.checks.push({restoredVolume:restored,microphoneClosed:true});
}catch(e){report.failure=e.stack;throw e;}finally{fs.writeFileSync(`${output}/video-casing-verification.json`,JSON.stringify(report,null,2));await p.evaluate(()=>{UniversalOneResearch?.close();document.querySelectorAll('video,audio').forEach(m=>m.pause());}).catch(()=>{});await browser.close();}
console.log(JSON.stringify(report));
