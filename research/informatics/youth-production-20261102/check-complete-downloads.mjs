import fs from 'node:fs';import crypto from 'node:crypto';import assert from 'node:assert/strict';import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);const {chromium}=require('/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='/Users/benkorbel/Documents/Codex/2026-11-02/task/youth-research-20261102',out=root+'/research/informatics/youth-production-20261102';
const expected=JSON.parse(fs.readFileSync(root+'/Informatics/exports/manifest.json'));
const server=await chromium.launchServer({headless:true,args:['--mute-audio','--disable-background-networking']});assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),context=await browser.newContext({acceptDownloads:true}),page=await context.newPage();const origin='https://universalmind.tv';
await context.route('**/*',r=>r.request().url().startsWith(origin+'/')||r.request().url().startsWith('blob:')?r.continue():r.abort());
const results=[];
try{
 await page.goto(origin+'/informatics/connect/',{waitUntil:'domcontentloaded'});const frame=page.locator('#connect-frame');await frame.waitFor({state:'attached'});const reader=await (await frame.elementHandle()).contentFrame();
 await reader.locator('[data-complete-download="json"]').waitFor({timeout:30000});
 for(const format of ['json','csv']){
  const wait=page.waitForEvent('download',{timeout:180000});await reader.locator('[data-complete-download="'+format+'"]').click();const d=await wait;const filename='/tmp/youth-staging-complete-'+format+'-20261102.gz';await d.saveAs(filename);
  const h=crypto.createHash('sha256');for await(const bytes of fs.createReadStream(filename))h.update(bytes);const digest=h.digest('hex');assert.equal(digest,expected.downloads[format].sha256);assert.equal(fs.statSync(filename).size,expected.downloads[format].bytes);
  const status=await reader.locator('#complete-download-status').innerText();assert.match(status,/1,195 data files/);assert.ok(status.includes(expected.dataset_version.slice(0,12)));results.push({format,bytes:fs.statSync(filename).size,sha256:digest,status,passed:true});fs.unlinkSync(filename);
 }
 const report={passed:true,origin,processAudioMuted:true,noPaidRequests:true,version:expected.dataset_version,downloads:results,fullRoundtrip:'Live downloaded archive hashes equal the locally fully reconstructed JSON and CSV archives, each containing all 1,195 sources.'};fs.writeFileSync(out+'/production-complete-downloads.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}finally{await context.close();await browser.close();await server.close();}
