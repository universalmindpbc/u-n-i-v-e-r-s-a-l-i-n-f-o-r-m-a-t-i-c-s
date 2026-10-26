import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const baseline='309367a9515024f2f2eeb72d519f0cbc4a5c6747',source=execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();
const files=execFileSync('git',['diff','--name-only',baseline,source,'--','Informatics'],{encoding:'utf8'}).trim().split('\n');
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint());
async function remote(page,path){return page.evaluate(async path=>{const r=await fetch('/informatics/'+path,{cache:'reload'});if(!r.ok)throw new Error(path+': '+r.status);const data=await r.arrayBuffer();return {bytes:data.byteLength,sha256:[...new Uint8Array(await crypto.subtle.digest('SHA-256',data))].map(n=>n.toString(16).padStart(2,'0')).join('')};},path);}
try{
 const page=await browser.newPage();await page.goto('https://staging.universalmind.tv/informatics/library.html',{waitUntil:'domcontentloaded'});
 const assets=[];
 for(const path of files){const local=hash(await fs.readFile(path));const relative=path.slice('Informatics/'.length);const version=relative==='library.js'?'14':relative==='library-model.js'?'6':relative.endsWith('.json')?'20261023h':source.slice(0,8);const actual=await remote(page,relative+'?v='+version);assert.equal(actual.sha256,local,path);assets.push({path,version,...actual,matches:true});}
 await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(e=>e.pause()));
 await page.goto('https://universalmind.tv/informatics/library.html',{waitUntil:'domcontentloaded'});
 const production=await remote(page,'library.js?v='+source.slice(0,8));assert.equal(production.sha256,hash(execFileSync('git',['show',baseline+':Informatics/library.js'])));
 await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(e=>e.pause()));
 const result={checkedAt:new Date().toISOString(),sourceCommit:source,assets,productionResearchSourceUnchanged:true,productionCommit:baseline};
 await fs.writeFile(new URL('assets.json',import.meta.url),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({assets:assets.length,productionResearchSourceUnchanged:true}));
}finally{for(const c of browser.contexts())for(const p of c.pages())await p.evaluate(()=>document.querySelectorAll('video,audio').forEach(e=>e.pause())).catch(()=>{});await browser.close();await server.close();}
