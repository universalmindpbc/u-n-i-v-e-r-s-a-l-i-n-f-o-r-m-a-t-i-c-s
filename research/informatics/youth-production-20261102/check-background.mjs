import fs from 'node:fs/promises';import path from 'node:path';import assert from 'node:assert/strict';import {createRequire} from 'node:module';import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url),{chromium}=require('/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=process.cwd(),out=path.join(root,'research/informatics/youth-production-20261102'),phase='production';assert.ok(['production'].includes(phase));
const css=await fs.readFile(path.join(root,'Informatics/library.css'),'utf8'),server=await chromium.launchServer({headless:true,args:['--mute-audio','--disable-background-networking']});assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),context=await browser.newContext(),page=await context.newPage();page.setDefaultTimeout(30000);const errors=[],checks=[];page.on('pageerror',e=>errors.push(e.message));
await context.route('**/*',r=>{const u=new URL(r.request().url());if(phase==='candidate'&&u.pathname==='/informatics/library.css')return r.fulfill({contentType:'text/css',body:css});return ['universalmind.tv','rmu.universalmind.tv'].includes(u.hostname)?r.continue():r.abort();});
let reader;async function open(query,reload=false){
 if(!reader||reload){await page.goto('https://universalmind.tv/informatics/research/',{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>document.querySelector('#research-frame')?.contentDocument?.querySelector('[data-stat="papers"]'),{},{timeout:60000});reader=await (await page.locator('#research-frame').elementHandle()).contentFrame();}
 await reader.goto('https://universalmind.tv/informatics/library.html?embedded=1&'+query,{waitUntil:'domcontentloaded'});await reader.locator('[data-stat="papers"]').waitFor({state:'attached'});
}
const luminance=rgb=>rgb.slice(0,3).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;}).reduce((a,v,i)=>a+v*[.2126,.7152,.0722][i],0);
async function state(mode){const s=await reader.evaluate(()=>({mode:document.documentElement.dataset.readingBackground,html:getComputedStyle(document.documentElement).backgroundColor,body:getComputedStyle(document.body).backgroundColor,ink:getComputedStyle(document.body).color,stored:localStorage.getItem('informatics-library-background'),pressed:[...document.querySelectorAll('.reading-toolbar [aria-pressed=true]')].map(n=>n.dataset.readingBackground),overflow:document.documentElement.scrollWidth>innerWidth+1}));assert.equal(s.mode,mode);assert.equal(s.stored,mode);assert.deepEqual(s.pressed,[mode]);assert.equal(s.overflow,false);const expected={black:'rgb(0, 0, 0)',white:'rgb(255, 255, 255)',video:'rgba(0, 0, 0, 0)'}[mode];assert.equal(s.html,expected);assert.equal(s.body,expected);if(mode!=='video'){const a=luminance(s.ink.match(/[\d.]+/g).map(Number)),b=mode==='black'?0:1;assert.ok((Math.max(a,b)+.05)/(Math.min(a,b)+.05)>4.5);}return s;}
try{
 for(const width of [1440,390,320]){
  await page.setViewportSize({width,height:1000});
  for(const query of ['collection=all&view=search','collection=sound','collection=sound&chapter=8','collection=vision','collection=vision&chapter=8','collection=language','collection=language&chapter=6']){
   await open(query);const modes=[];
   for(const mode of ['black','white','video','black','video','white','black']){await reader.locator('.reading-toolbar [data-reading-background="'+mode+'"]').click();modes.push(await state(mode));}
   checks.push({width,query,modes,passed:true});
  }
  // Refresh each persisted mode through the real shell.
  for(const mode of ['black','white','video']){await reader.locator('.reading-toolbar [data-reading-background="'+mode+'"]').click();await open('collection=language&chapter=6',true);await state(mode);await page.screenshot({path:path.join(out,phase+'-'+mode+'-'+width+'.png')});}
  // Article controls share the same state and must repaint the reading surface.
  await open('collection=all&view=search&record=youth-pmid-34002511');await reader.locator('#article[open]').waitFor();
  for(const mode of ['white','black','video']){await reader.locator('#article [data-reading-background="'+mode+'"]').click();await state(mode);}
  await reader.locator('#article [data-close="article"]').click();
 }
 assert.deepEqual(errors,[]);const report={passed:true,phase,processAudioMuted:true,paidRequestsBlocked:true,views:checks.length,transitions:checks.reduce((n,c)=>n+c.modes.length,0),viewports:[1440,390,320],reloadPersistenceAllModes:true,articleControls:true,contrastAtLeast4_5:true,checks,errors};await fs.writeFile(path.join(out,phase+'-background.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({...report,checks:undefined}));
}catch(error){await fs.writeFile(path.join(out,phase+'-failure.json'),JSON.stringify({error:String(error),page:page.url(),frame:reader?.url(),body:await reader?.locator('body').innerText().catch(()=>''),errors,checksCompleted:checks.length},null,2));throw error;}finally{await context.close();await browser.close();await server.close();}
