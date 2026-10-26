// Import reviewed evidence only. Private task drafts never enter shared RAG here.
import fs from 'node:fs';import path from 'node:path';import {createHash} from 'node:crypto';import {createRequire} from 'node:module';
const require=createRequire('/private/tmp/uo-runtime/package.json'),{Client}=require('pg');
const root=path.resolve(import.meta.dirname,'../../../..'),site=path.join(root,'Informatics');
const secret=JSON.parse(fs.readFileSync(process.argv[2]||'/private/tmp/uo-secrets/import.json'));
const u=new URL(secret.DATABASE_URL);u.hostname=u.hostname.replace('-pooler.','.');u.searchParams.set('sslmode','verify-full');
const db=new Client({connectionString:u.href});await db.connect();
const sha=b=>createHash('sha256').update(b).digest('hex'),read=p=>JSON.parse(fs.readFileSync(p));
const entries=read(path.join(site,'quantum/experiments/index.json')).experiments;
const memory=[];const experiments=[];
for(const e of entries){
 const jb=fs.readFileSync(path.join(site,e.json)),cb=fs.readFileSync(path.join(site,e.csv));
 if(sha(jb)!==e.json_sha256||sha(cb)!==e.csv_sha256)throw Error('Archive hash mismatch: '+e.id);
 const data=JSON.parse(jb);experiments.push({...e,data,json_text:jb.toString(),csv_text:cb.toString()});
 const details={question:data.question,phase:data.phase,interpretation:data.interpretation||null,biological_evidence:data.biological_evidence||null,methods:data.methods,field_guide:data.field_guide};
 const document=JSON.stringify(details).slice(0,18000);
 memory.push({id:e.id,revision:e.revision,kind:'experiment',data:{experiment_id:e.id,revision:e.revision,question:e.title,json_path:e.json,csv_path:e.csv,phase:data.phase,interpretation:data.interpretation||null,review:data.review||{status:'recorded_qualification'}},document,source_sha256:e.json_sha256});
}
for(const [file,id,kind] of [['../intent.json','universal-mind-intent','mission'],['research-directions.json','research-directions','research_direction']]){
 const raw=fs.readFileSync(path.resolve(import.meta.dirname,file)),data=JSON.parse(raw);memory.push({id,revision:data.revision,kind,data,document:JSON.stringify(data),source_sha256:sha(raw)});
}
const api=JSON.parse(fs.readFileSync('/private/tmp/uo-secrets/worker.json')).OPENAI_API_KEY;
const cacheFile='/private/tmp/astra-memory-embeddings.json',cache=fs.existsSync(cacheFile)?read(cacheFile):{};let tokens=0;
for(const m of memory){if(cache[m.source_sha256])continue;
 const r=await fetch('https://api.openai.com/v1/embeddings',{method:'POST',headers:{Authorization:'Bearer '+api,'Content-Type':'application/json'},body:JSON.stringify({model:'text-embedding-3-small',dimensions:512,input:m.document})});
 if(!r.ok)throw Error('Embedding failed '+r.status);const d=await r.json();cache[m.source_sha256]=d.data[0].embedding;tokens+=d.usage.total_tokens;
 fs.writeFileSync(cacheFile,JSON.stringify(cache),{mode:0o600});
}
try{
 await db.query('BEGIN');await db.query(fs.readFileSync(new URL('./schema.sql',import.meta.url),'utf8'));
 for(const e of experiments){
  const old=(await db.query('SELECT json_sha256,csv_sha256 FROM research.agent_experiments WHERE id=$1 AND revision=$2',[e.id,e.revision])).rows[0];
  if(old&&(old.json_sha256!==e.json_sha256||old.csv_sha256!==e.csv_sha256))throw Error('Immutable experiment revision differs: '+e.id);
  await db.query('INSERT INTO research.agent_experiments(id,revision,question,data,json_text,csv_text,json_sha256,csv_sha256,json_path,csv_path) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) ON CONFLICT DO NOTHING',[e.id,e.revision,e.title,e.data,e.json_text,e.csv_text,e.json_sha256,e.csv_sha256,e.json,e.csv]);
 }
 for(const m of memory){
  const old=(await db.query('SELECT source_sha256 FROM research.agent_memory WHERE id=$1 AND revision=$2',[m.id,m.revision])).rows[0];
  if(old&&old.source_sha256!==m.source_sha256)throw Error('Memory revision differs; record a new revision: '+m.id);
  await db.query('INSERT INTO research.agent_memory(id,revision,kind,data,document,source_sha256,embedding) VALUES($1,$2,$3,$4,$5,$6,$7::vector) ON CONFLICT DO NOTHING',[m.id,m.revision,m.kind,m.data,m.document,m.source_sha256,JSON.stringify(cache[m.source_sha256])]);
 }
 const copies=(await db.query('SELECT id,json_text,csv_text,json_sha256,csv_sha256 FROM research.agent_experiments')).rows;
 for(const r of copies)if(sha(r.json_text)!==r.json_sha256||sha(r.csv_text)!==r.csv_sha256)throw Error('Database archive verification failed');
 const nature=(await db.query("SELECT p.id,count(s.embedding)::int embedded FROM research.publications p JOIN research.passages s ON s.publication_id=p.id WHERE lower(p.data->>'doi')='10.1038/s41586-022-04853-4' GROUP BY p.id")).rows;
 if(nature[0]?.embedded<1||!nature.length)throw Error('Nature stimulus missing from research RAG');
 await db.query('COMMIT');console.log(JSON.stringify({experiments:copies.length,memory:memory.length,archiveHashesMatch:true,nature,embeddingTokens:tokens}));
}catch(e){await db.query('ROLLBACK');throw e;}finally{await db.end();}
