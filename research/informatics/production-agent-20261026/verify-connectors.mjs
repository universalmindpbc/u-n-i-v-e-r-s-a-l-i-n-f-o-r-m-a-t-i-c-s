import fs from 'node:fs/promises';import assert from 'node:assert/strict';
const base='https://informatics-universal-one.universalmind.workers.dev';
const rpc=async(method,params={})=>{const r=await fetch(base+'/mcp',{method:'POST',headers:{'Content-Type':'application/json',Accept:'application/json, text/event-stream'},body:JSON.stringify({jsonrpc:'2.0',id:1,method,params})});assert.equal(r.status,200);const d=await r.json();assert(!d.error,JSON.stringify(d.error));return d.result;};
const tools=(await rpc('tools/list')).tools.map(t=>t.name);assert(tools.includes('get_inventory'));assert(tools.includes('get_quantum_study'));const result=await rpc('tools/call',{name:'get_inventory',arguments:{}});const inventory=result.structuredContent||JSON.parse(result.content[0].text);assert.equal(inventory.counts.publications,891);
const publication=await(await fetch(base+'/api/v1/publications/pmid-32412916')).json();assert.match(JSON.stringify(publication),/5.75/);
const study=await(await fetch(base+'/api/v1/quantum/oxtr-qrc-01')).json();assert.match(JSON.stringify(study),/OXTR/);
const card=await(await fetch(base+'/.well-known/agent-card.json')).json();assert(card.supportedInterfaces.length);
await fs.writeFile(new URL('./production/connector-proof.json',import.meta.url),JSON.stringify({passed:true,endpoint:base,tools,inventory,publication_id:'pmid-32412916',study_id:'oxtr-qrc-01',agentCard:card},null,2)+'\n');console.log(JSON.stringify({passed:true,tools:tools.length,publications:inventory.counts.publications}));
