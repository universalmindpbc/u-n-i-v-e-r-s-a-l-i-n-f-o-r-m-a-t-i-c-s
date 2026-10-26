import path from 'node:path';
import {fileURLToPath} from 'node:url';
const modules = process.env.PHP_WASM_MODULE_DIR;
if (!modules) throw new Error('Set PHP_WASM_MODULE_DIR to a temporary @php-wasm installation.');
const {PHP} = await import(path.join(modules, '@php-wasm/universal/index.js'));
const {loadNodeRuntime} = await import(path.join(modules, '@php-wasm/node/index.js'));
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const php=new PHP(await loadNodeRuntime('8.3',{emscriptenOptions:{processId:1}}));
const report={runtime:'PHP 8.3 WASM',checks:[]};
php.mkdir('/app/api');
for(const name of ['quantum-tree-settings.php','quantum-tree-owner-password.php','owner-state-history.php']) {
 let code=await fs.readFile(root+'/i  a m/api/'+name,'utf8');
 if(name==='quantum-tree-settings.php') code=code.replace(/const QUANTUM_TREE_OWNER_KEY_HASH = '[a-f0-9]+';/,`const QUANTUM_TREE_OWNER_KEY_HASH = '${createHash('sha256').update('legacy-test-fixture').digest('hex')}';`);
 php.writeFile('/app/api/'+name,code);
}
let cookie='';
async function request(payload, options={}) {
 const r=await php.run({scriptPath:'/app/api/quantum-tree-settings.php',relativeUri:'/vogue/iam/api/quantum-tree-settings.php',protocol:'https',method:payload?'POST':'GET',
 headers:{Host:'staging.universalmind.tv',Origin:options.origin||'https://staging.universalmind.tv',Cookie:options.cookie??cookie,'Content-Type':'application/json'},
 $_SERVER:{HTTPS:'on'}, ...(payload?{body:new TextEncoder().encode(JSON.stringify(payload))}:{})});
 assert.equal(r.errors,'',r.errors);
 const set=r.headers['set-cookie'];if(set?.length && !options.keepCookie)cookie=set.at(-1).split(';')[0];
 return {status:r.httpStatusCode,body:JSON.parse(r.text)};
}
try {
 assert.equal((await request()).body.authenticated,false);
 assert.equal((await request({action:'change_password',new_password:'not-used-fixture',confirm_password:'not-used-fixture'})).status,401);
 const login=await request({action:'login',password:'legacy-test-fixture'});
 assert.equal(login.body.authenticated,true);let csrf=login.body.csrf_token;
 const firstCookie=cookie;
 await request({action:'login',password:'legacy-test-fixture'},{cookie:''});
 const otherCookie=cookie;
 cookie=firstCookie;
 const oldCookie=cookie;
 assert.equal((await request({action:'change_password',csrf_token:'wrong',new_password:'not-used-fixture',confirm_password:'not-used-fixture'})).status,403);
 assert.equal((await request({action:'change_password',csrf_token:csrf,new_password:'not-used-fixture',confirm_password:'not-used-fixture'},{origin:'https://other.test'})).status,403);
 assert.equal((await request({action:'change_password',csrf_token:csrf,new_password:'tooShort',confirm_password:'mismatch'})).status,422);
 for(const invalid of ['short','a'.repeat(73),'abcd\u0000efgh']) assert.equal((await request({action:'change_password',csrf_token:csrf,new_password:invalid,confirm_password:invalid})).status,422);
 report.checks.push('Unauthenticated, CSRF, origin, mismatch, short, overlong and null-byte passwords rejected');
 const password='  Owner-fixture Case Sensitive  ';
 const changed=await request({action:'change_password',csrf_token:csrf,new_password:password,confirm_password:password});
 assert.equal(changed.status,200);assert.ok(changed.body.csrf_token!==csrf);csrf=changed.body.csrf_token;
 const credentials=JSON.parse(php.readFileAsText('/app/api/.runtime-private/quantum-tree-owner.json'));
 assert.ok(credentials.password_hash.startsWith('$2y$'));
 assert.ok(!JSON.stringify(credentials).includes(password));
 assert.equal((await request()).body.authenticated,true);
 assert.equal((await request(null,{cookie:oldCookie,keepCookie:true})).body.authenticated,false);
 assert.equal((await request(null,{cookie:otherCookie,keepCookie:true})).body.authenticated,false);
 report.checks.push('Custom password stored only as an adaptive hash; CSRF/session rotated; prior session invalidated');
 await request({action:'logout'});
 for(const wrong of ['legacy-test-fixture',password.trim(),password.toLowerCase()]) assert.equal((await request({action:'login',password:wrong})).status,401);
 assert.equal((await request({action:'login',password})).body.authenticated,true);
 report.checks.push('New password signs in exactly; previous key, changed case and missing spaces rejected');
 const before=(await request()).body;
 const saved=await request({action:'save_mantra',csrf_token:before.csrf_token,mantra_enabled:false});assert.equal(saved.body.mantra_enabled,false);assert.equal(saved.body.peak_volume,before.peak_volume);
 const peakSaved=await request({action:'save',csrf_token:before.csrf_token,peak_volume:0.055});assert.equal(peakSaved.body.mantra_enabled,false);
 report.checks.push('Authenticated mantra and peak updates preserve each other');
 const token='a'.repeat(64);
 const resetPath='/app/api/.runtime-private/quantum-tree-owner-reset.json';
 const seedReset=(changes={})=>{
  const current=JSON.parse(php.readFileAsText('/app/api/.runtime-private/quantum-tree-owner.json'));
  const record={token_hash:createHash('sha256').update(token).digest('hex'),expires_at:Math.floor(Date.now()/1000)+3600,credential_revision:current.revision,...changes};
  php.writeFile(resetPath,JSON.stringify(record));return record;
 };
 seedReset({expires_at:1});
 assert.equal((await request({action:'verify_reset',reset_token:token})).status,400);
 seedReset({credential_revision:'old-revision'});
 assert.equal((await request({action:'verify_reset',reset_token:token})).status,400);
 const issued=seedReset();
 assert.equal((await request({action:'verify_reset',reset_token:'b'.repeat(64)})).status,400);
 assert.equal((await request({action:'verify_reset',reset_token:token},{origin:'https://other.test'})).status,403);
 for(let i=0;i<2;i++)assert.equal((await request({action:'verify_reset',reset_token:token})).body.valid,true);
 const resetPassword='  New Owner Reset Case  ';
 const payload={action:'reset_password',reset_token:token,new_password:resetPassword,confirm_password:resetPassword};
 assert.equal((await request({...payload,confirm_password:'mismatch'})).status,422);
 assert.equal((await request({...payload,new_password:'short',confirm_password:'short'})).status,422);
 const beforeResetCookie=cookie;
 const reset=await request(payload,{cookie:''});
 assert.equal(reset.body.authenticated,true);
 assert.ok(reset.body.csrf_token);
 assert.equal((await request(null,{cookie:beforeResetCookie,keepCookie:true})).body.authenticated,false);
 assert.equal((await request({action:'verify_reset',reset_token:token})).status,400);
 // Even if the old reset file survived deletion, the atomic revision change
 // makes concurrent/replayed redemption fail.
 php.writeFile(resetPath,JSON.stringify(issued));
 assert.equal((await request(payload)).status,400);
 await request({action:'logout'});
 assert.equal((await request({action:'login',password})).status,401);
 assert.equal((await request({action:'login',password:resetPassword.trim()})).status,401);
 assert.equal((await request({action:'login',password:resetPassword})).body.authenticated,true);
 report.checks.push('One-use recovery requires the secret token; expired, wrong, stale-revision and cross-origin tokens rejected; verification does not consume it; invalid passwords leave it usable');
 report.checks.push('Recovery without the old key signs in with the exact new password, invalidates prior sessions and rejects replay even if the reset file survives cleanup');
 php.writeFile('/app/api/.runtime-private/quantum-tree-owner.json','invalid');
 assert.equal((await request()).body.authenticated,false);
 assert.equal((await request({action:'login',password:'legacy-test-fixture'})).status,401);
 report.checks.push('Corrupt credential fails closed and never restores the legacy key');
 report.passed=true;
 await fs.writeFile('/private/tmp/um-mantra-password-test.json',JSON.stringify(report,null,2));
 console.log(JSON.stringify(report,null,2));
} finally { php.exit(); }
process.exit(0);
