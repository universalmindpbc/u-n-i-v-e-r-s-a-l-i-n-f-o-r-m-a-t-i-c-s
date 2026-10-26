import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {Client} from '/private/tmp/informatics-connector-qa/node_modules/@modelcontextprotocol/sdk/dist/esm/client/index.js';
import {StreamableHTTPClientTransport} from '/private/tmp/informatics-connector-qa/node_modules/@modelcontextprotocol/sdk/dist/esm/client/streamableHttp.js';
const base=process.env.UO_ENDPOINT||'https://informatics-universal-one-staging.universalmind.workers.dev';
const out=new URL(process.env.UO_PROOF_DIR||'./verification/',import.meta.url);
await fs.mkdir(out,{recursive:true});
const report={checkedAt:new Date().toISOString(),endpoint:base};
const client=new Client({name:'informatics-proof',version:'1.0.0'});
try{
 await client.connect(new StreamableHTTPClientTransport(new URL(base+'/mcp')));
 report.server=client.getServerVersion();report.tools=(await client.listTools()).tools.map(x=>x.name);assert.equal(report.tools.length,9);
 const call=async(name,args={})=>{const d=await client.callTool({name,arguments:args});assert.ok(!d.isError,JSON.stringify(d));return d.structuredContent??JSON.parse(d.content[0].text);};
 report.inventory=await call('get_inventory');assert.equal(report.inventory.counts.atlas,1001);assert.equal(report.inventory.counts.publications,811);
 const index=await call('list_atlas_records',{family:'m9',limit:100});assert.equal(index.page.total,111);assert.equal(index.page.nextOffset,100);
 report.m9=await call('get_atlas_record',{id:'M9',section:'summary'});assert.ok(report.m9.data.availableSections.includes('probability'));assert.ok(report.m9.data.availableSections.includes('research'));
 const prob=await call('get_atlas_record',{id:'M9',section:'probability'});report.probabilityKeys=Object.keys(prob.data??prob.fields);assert.ok(prob.data||prob.fields);
 const result=await call('search_publications',{query:'music gene expression',limit:10});report.publicationSearch={total:result.page.total,ids:result.records.map(x=>x.id)};assert.ok(result.page.total>0);
 report.genericSearch=await call('search',{query:'M9'});assert.ok(report.genericSearch.results.some(r=>r.id==='atlas:fibonacci-m9-p24'));
 const pub=result.records[0];assert.equal((await call('fetch',{id:'publication:'+pub.id})).id,'publication:'+pub.id);
 report.resources=(await client.listResources()).resources;
 const passages=await call('get_passages',{id:pub.id,limit:2});report.passageCheck={id:pub.id,total:passages.page.total,rights:passages.passages.map(p=>p.rights.license)};
 const full=await(await fetch(base+'/api/v1/atlas/M9')).json();const original=JSON.parse(await fs.readFile(new URL('../../../Informatics/pilot-v1/m9.json',import.meta.url),'utf8'));assert.deepEqual(full.data,original);report.fullM9MatchesOriginal=true;
 const card=await(await fetch(base+'/.well-known/agent-card.json')).json();assert.equal(card.supportedInterfaces[0].protocolVersion,'1.0');report.agentCard=card;
 const error=await(await fetch(base+'/a2a/v1',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({jsonrpc:'2.0',id:1,method:'SendMessage',params:{}})})).json();assert.equal(error.error.code,-32009);report.a2aVersionGuard=true;
 report.passed=true;
}finally{await client.close();await fs.writeFile(new URL('protocols.json',out),JSON.stringify(report,null,2)+'\n');}
console.log(JSON.stringify({passed:report.passed,counts:report.inventory.counts,tools:report.tools,search:report.publicationSearch,fullM9MatchesOriginal:report.fullM9MatchesOriginal}));
