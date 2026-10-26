import fs from 'node:fs/promises';import assert from 'node:assert/strict';
const production='https://universalmind.tv',staging='https://staging.universalmind.tv';
const api='https://informatics-universal-one.universalmind.workers.dev',stageAPI='https://informatics-universal-one-staging.universalmind.workers.dev';
const proof={checkedAt:new Date().toISOString(),checks:[]};
async function token(origin){const r=await fetch(origin+'/informatics/api/astra-session.php',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json'},body:'{}'});assert.equal(r.status,200);const d=await r.json();assert(d.token);return d.token;}
const prodToken=await token(production),stageToken=await token(staging);
for(const [endpoint,origin,t,expected] of [[api,production,prodToken,200],[stageAPI,staging,stageToken,200],[api,production,stageToken,401],[stageAPI,staging,prodToken,401]]){
 const r=await fetch(endpoint+'/agent/access',{headers:{Origin:origin,Authorization:'Bearer '+t}});assert.equal(r.status,expected);if(expected===200){const d=await r.json();assert.equal(d.newAquilaAllowed,false);assert.equal(d.newPaidGPUAllowed,false);}proof.checks.push({endpoint,origin,expected,actual:r.status});
}
for(const path of ['/vogue/iam/api/.runtime-private/informatics-astra.json','/informatics/api/astra-session.php']){const r=await fetch(production+path);assert([403,404].includes(r.status));proof.checks.push({path,status:r.status});}
proof.passed=true;await fs.writeFile(new URL('./production/access-proof.json',import.meta.url),JSON.stringify(proof,null,2)+'\n');console.log(JSON.stringify(proof));
