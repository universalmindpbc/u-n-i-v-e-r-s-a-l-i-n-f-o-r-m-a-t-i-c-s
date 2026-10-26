import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const {chromium}=await import('/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const paths=execFileSync('git',['diff','--name-only','7f1de4ff','HEAD'],{encoding:'utf8'}).trim().split('\n').filter(p=>p.startsWith('Informatics/'));
const browser=await chromium.launch({headless:true,args:['--mute-audio']});const page=await browser.newPage();
const report={checkedAt:new Date().toISOString(),sourceCommit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),stagingAssets:[]};
try{
 await page.goto('https://staging.universalmind.tv/informatics/connect.html',{waitUntil:'domcontentloaded'});
 for(const p of paths){
  const url='/informatics/'+p.split('/').slice(1).join('/');
  const r=await page.evaluate(async url=>{const r=await fetch(url,{cache:'reload'});return {status:r.status,text:await r.text()};},url);
  assert.equal(r.status,200,p);assert.equal(r.text,await fs.readFile(p,'utf8'),p);
  report.stagingAssets.push({path:p,sha256:crypto.createHash('sha256').update(r.text).digest('hex'),bytes:Buffer.byteLength(r.text),matches:true});
 }
 await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause()));
 await page.goto('https://universalmind.tv/informatics/?entry=research',{waitUntil:'domcontentloaded'});
 const prod=await page.evaluate(async()=>{const r=await fetch('/informatics/index.html',{cache:'reload'});return {status:r.status,text:await r.text()};});
 assert.equal(prod.status,200);assert.equal(prod.text,execFileSync('git',['show','7f1de4ff:Informatics/index.html'],{encoding:'utf8'}));report.productionEntryMatchesBaseline=true;
 report.passed=true;
}finally{await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause())).catch(()=>{});await browser.close();report.browserClosed=true;await fs.writeFile(new URL('./verification/staging-assets.json',import.meta.url),JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({passed:report.passed,assets:report.stagingAssets.length,productionEntryMatchesBaseline:report.productionEntryMatchesBaseline}));
