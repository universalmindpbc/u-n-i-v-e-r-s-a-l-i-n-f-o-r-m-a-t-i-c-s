import fs from 'node:fs';import path from 'node:path';import assert from 'node:assert/strict';import {createRequire} from 'node:module';import {execFileSync} from 'node:child_process';import {fileURLToPath} from 'node:url';import {youthRecords,youthVersion} from '../cloudflare/informatics-youth-knowledge.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),out=path.join(root,'research/informatics/research-background-20261102'),api='https://informatics-universal-one-staging.universalmind.workers.dev',origin='https://staging.universalmind.tv';
const {chromium}=createRequire(import.meta.url)('/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const server=await chromium.launchServer({headless:true,args:['--mute-audio','--disable-background-networking']});assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint()),context=await browser.newContext({acceptDownloads:true}),page=await context.newPage();
await context.route('**/*',r=>{const u=new URL(r.request().url());return (u.origin===origin||u.protocol==='blob:'||u.origin===api&&(/^\/api\/v1\//.test(u.pathname)||u.pathname==='/mcp'||u.pathname==='/health'))?r.continue():r.abort();});
async function get(p){const r=await context.request.get(api+p,{headers:{Origin:origin}});assert.equal(r.status(),200,p);return r.json();}
function verify(rows){assert.equal(new Set(rows.map(r=>r.id)).size,rows.length);for(const r of youthRecords)assert.deepEqual(rows.find(a=>a.id===r.id),JSON.parse(JSON.stringify(r)));assert.ok(!rows.some(r=>r.pmid==='40421183'));}
try{
 const inventory=await get('/api/v1/inventory');assert.equal(inventory.sourceVersions.approvedYouth.sourceSHA256,youthVersion.sourceSHA256);assert.equal(inventory.coverage.approvedYouthPublications,105);assert.equal(inventory.counts.publications,inventory.coverage.databasePublications-inventory.coverage.replacedDatabaseRows+105);assert.equal(inventory.counts.public_full_texts,146);
 const health=await get('/health');assert.deepEqual(health.knowledgeVersions.approvedYouth,youthVersion);
 const all=[];let offset=0;do{const r=await get('/api/v1/export/publications.json?limit=100&offset='+offset);all.push(...r.records);offset=r.page.nextOffset;}while(offset!==null);verify(all);assert.equal(all.length,inventory.counts.publications);
 const jsonFile=path.join(out,'live-publications.json');fs.writeFileSync(jsonFile,JSON.stringify(all));
 const csvPages=[];let csvOffset=0;do{const r=await context.request.get(api+'/api/v1/export/publications.csv?limit=100&offset='+csvOffset);assert.equal(r.status(),200);csvPages.push(await r.text());const n=r.headers()['x-next-offset'];csvOffset=n===''?null:Number(n);}while(csvOffset!==null);
 const apiCSV=path.join(out,'api-publication-pages.json');fs.writeFileSync(apiCSV,JSON.stringify(csvPages));execFileSync('python3',['-c',`import csv,json,sys,io\ncsv.field_size_limit(10000000)\na=json.load(open(sys.argv[1]));b=[json.loads(r['record_json']) for page in json.load(open(sys.argv[2])) for r in csv.DictReader(io.StringIO(page))];assert a==b`,jsonFile,apiCSV]);fs.unlinkSync(apiCSV);

 for(const id of ['34002511','36620186','19798037']){const result=await get('/api/v1/publications/'+id);assert.equal(result.record.pmid,id);}
 const retracted=await context.request.get(api+'/api/v1/publications/40421183');assert.equal(retracted.status(),404);
 const searched=await get('/api/v1/publications?query=Horyzons&limit=100');assert.ok(searched.records.some(r=>r.pmid==='34002511'));
 const mcp=await context.request.post(api+'/mcp',{data:{jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'get_publication',arguments:{id:'34002511'}}},headers:{'Content-Type':'application/json'}});assert.equal(mcp.status(),200);const mcpData=await mcp.json();assert.equal(mcpData.result.structuredContent.record.pmid,'34002511');
 await page.goto(origin+'/informatics/connect/',{waitUntil:'domcontentloaded'});await page.locator('#connect-frame').waitFor({state:'attached'});const reader=await (await page.locator('#connect-frame').elementHandle()).contentFrame();await reader.locator('[data-download]').first().waitFor();await reader.waitForFunction(()=>typeof document.querySelector('[data-download]')?.onclick==='function');
 const downloads=[];
 for(const format of ['json','csv']){
  const waiting=page.waitForEvent('download',{timeout:60000});await reader.locator('[data-download$=".'+format+'"]').click();const d=await waiting,file=path.join(out,'connect-publications.'+format);await d.saveAs(file);
  if(format==='json'){const parsed=JSON.parse(fs.readFileSync(file));verify(parsed.records);assert.deepEqual(parsed.records,all);}
  else{execFileSync('python3',['-c',`import csv,json,sys\ncsv.field_size_limit(10000000)\na=json.load(open(sys.argv[1]));b=[json.loads(r['record_json']) for r in csv.DictReader(open(sys.argv[2],newline=''))];assert a==b`,jsonFile,file]);}
  downloads.push({format,records:all.length,bytes:fs.statSync(file).size,status:await reader.locator('#download-status').innerText()});fs.unlinkSync(file);
 }
 fs.unlinkSync(jsonFile);
 const report={passed:true,paidRequests:false,processAudioMuted:true,inventory,health,all105ApprovedRecordsExact:true,allPublicationPages:true,RESTCSVEquivalent:true,connectJSONCSVEquivalent:true,withheldRetractionAbsent:true,MCPPublicationLookup:true,RESTIdentityLookup:true,downloads};fs.writeFileSync(path.join(out,'staging-public-api.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:true,total:all.length,coverage:inventory.coverage,downloads}));
}finally{await context.close();await browser.close();await server.close();}
