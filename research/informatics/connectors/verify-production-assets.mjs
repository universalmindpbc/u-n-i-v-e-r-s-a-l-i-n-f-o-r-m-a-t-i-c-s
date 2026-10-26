import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const {chromium}=await import('/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const paths=execFileSync('git',['diff','--name-only','7f1de4ff','HEAD'],{encoding:'utf8'}).trim().split('\n').filter(p=>p.startsWith('Informatics/'));
const out=new URL('./production-verification/',import.meta.url);await fs.mkdir(out,{recursive:true});
const report={checkedAt:new Date().toISOString(),sourceCommit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),approvedStagingCommit:'bd7fce6a69f7210b45c69f045bae7e13f5c7e346',processAudioDisabled:true,assets:[]};
const browser=await chromium.launch({headless:true,args:['--mute-audio']});const page=await browser.newPage();
try{
 for(const site of ['https://staging.universalmind.tv','https://universalmind.tv']){
  await page.goto(site+'/informatics/connect.html',{waitUntil:'domcontentloaded'});
  for(const p of paths){
   const route='/informatics/'+p.split('/').slice(1).join('/');
   const r=await page.evaluate(async route=>{const r=await fetch(route,{cache:'reload'});return {status:r.status,text:await r.text()};},route);
   assert.equal(r.status,200,site+route);
   const approved=execFileSync('git',['show','bd7fce6a:'+p],{encoding:'utf8'});
   assert.equal(r.text,approved,site+route);assert.equal(r.text,await fs.readFile(p,'utf8'),p);
   report.assets.push({site,path:p,sha256:crypto.createHash('sha256').update(r.text).digest('hex'),bytes:Buffer.byteLength(r.text),matchesApproved:true});
  }
  await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause()));
 }
 report.passed=true;
}finally{await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(m=>m.pause())).catch(()=>{});await browser.close();report.browserClosed=true;await fs.writeFile(new URL('assets.json',out),JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({passed:report.passed,comparisons:report.assets.length}));
