import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const commit='ede9babf',files=execFileSync('git',['diff-tree','--no-commit-id','--name-only','-r',commit],{encoding:'utf8'}).trim().split('\n').filter(f=>f.startsWith('Informatics/')),checks=[];
for(const file of files){
 const url='https://staging.universalmind.tv/informatics/'+file.slice('Informatics/'.length);
 const r=await fetch(url+'?verify='+Date.now(),{cache:'no-store'});assert.equal(r.status,200,file);
 const actual=Buffer.from(await r.arrayBuffer()),expected=await fs.readFile(file);assert.ok(actual.equals(expected),file+' byte equality');
 checks.push({file,url,http:r.status,content_type:r.headers.get('content-type'),sha256:createHash('sha256').update(actual).digest('hex'),bytes:actual.length});
}
await fs.writeFile('research/informatics/quantum-lab-20261025/http-verification.json',JSON.stringify({sourceCommit:commit,verifiedAt:new Date().toISOString(),all_match:true,checked_files:checks.length,checks},null,2)+'\n');
console.log(JSON.stringify({passed:true,files:checks.length}));
