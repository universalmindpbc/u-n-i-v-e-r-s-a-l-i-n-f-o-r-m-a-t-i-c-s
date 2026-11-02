import assert from 'node:assert/strict';import fs from 'node:fs';import {repository} from '../cloudflare/informatics-repository.js';import {youthRecords,youthReplaces} from '../cloudflare/informatics-youth-knowledge.js';import {qaSpendLedger,reserveQACall,reconcileQACall} from '../cloudflare/informatics-qa-spend.js';
const old=youthRecords.filter(r=>['19798037','29422189','42324567','41340540','40243164'].includes(r.pmid)).map((r,i)=>({id:i===0?'legacy-identity':r.id,data:{...r,summary:'Old summary'}}));
const base=[...old,...Array.from({length:10},(_,i)=>({id:'unrelated-'+i,data:{id:'unrelated-'+i,title:'Other approved research '+i,collections:['general'],targets:[],conditions:[],journal:'Other journal'}}))];
const db={async query(sql,v=[]){
 if(sql.includes('(SELECT count(*)::int FROM research.publications)'))return [{publications:base.length,atlas:1001,full_texts:149,public_full_texts:146,m9_family:111}];
 if(sql.includes('SELECT id,data FROM research.publications WHERE'))return base.filter(r=>r.id===v[0]||r.data.doi===v[0]||r.data.pmid===v[0]);
 if(sql.includes('research.passages'))return sql.includes('count(*)')?[{total:0}]:[];
 let rows;
 if(v.length===3)rows=base.filter(youthReplaces);
 else{assert.ok(sql.includes('AND NOT'));assert.equal(v.length>=8,true);rows=base.filter(r=>!youthReplaces(r)).filter(r=>(!v[0].length||v[0].some(x=>r.data.collections.includes(x)))&&(!v[4]||r.data.title.toLowerCase().includes(v[4].toLowerCase())));}
 if(sql.includes('count(*)'))return [{total:rows.length}];
 assert.match(sql,/LIMIT \$9 OFFSET \$10/);return rows.slice(v[9],v[9]+v[8]);
}};
const api=repository(db,{DATASET_VERSION:'fixture-v1',ALLOWED_ORIGIN:'https://staging.universalmind.tv'},'https://example.invalid');const inventory=await api.inventory();assert.equal(inventory.counts.publications,115);assert.equal(inventory.coverage.replacedDatabaseRows,5);assert.equal(inventory.counts.public_full_texts,146);
const all=[];let offset=0;do{const r=await api.publications({offset,limit:100});all.push(...r.records);offset=r.page.nextOffset;assert.ok(r.datasetVersion.includes('youth-'));}while(offset!==null);
assert.equal(all.length,115);assert.equal(new Set(all.map(r=>r.id)).size,115);assert.equal(all.filter(r=>r.publicationApproval).length,105);for(const r of youthRecords)assert.deepEqual(all.find(a=>a.id===r.id),r);
for(const r of youthRecords){assert.deepEqual((await api.publication(r.pmid)).record,r);assert.deepEqual((await api.publication(r.doi)).record,r);}
assert.equal((await api.publications({offset:105,limit:5})).records.length,5);assert.equal((await api.publications({offset:115,limit:5})).records.length,0);
assert.equal((await api.publications({filters:{collections:['general']},limit:100})).page.total,10);
assert.equal((await api.publication('19798037')).record.citationCount,197);await assert.rejects(api.publication('40421183'),/not found/);
const ledger=qaSpendLedger(.2);const body={model:'gpt-6-astra',input:'Test',max_output_tokens:100};const reserved=reserveQACall(ledger,'/responses',body);reconcileQACall(ledger,reserved,{input_tokens:100,output_tokens:50});assert.ok(Math.abs(ledger.chargedUpperUSD-.00375)<1e-9);assert.throws(()=>reserveQACall(ledger,'/responses',{...body,input:'a'.repeat(50000)}),/spending guard/);assert.equal(ledger.calls.length,1);assert.throws(()=>reserveQACall(ledger,'/responses',{...body,model:'other'}),/outside/);assert.equal(qaSpendLedger(2),null);assert.equal(reserveQACall(null,'/responses',body),null);
const report={passed:true,canonicalIdentityMerge:true,fiveMatchedRecordsReplaced:true,all105QualifiedFieldsPreserved:true,paginationBoundaries:true,unrelatedRecordsPreserved:true,aliases:true,sourceVersions:true,publicFulltextLimitsPreserved:true,withheldNotAdded:true,spendingGuardBeforeProviderCall:true,networkRequests:0};fs.writeFileSync(new URL('../research/informatics/research-background-20261102/public-api-local.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
