// Add reviewed records to an explicit isolated research branch. No DDL or role changes.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const require=createRequire('/private/tmp/uo-runtime/package.json'),{Client}=require('pg');
const [credentialFile,output]=process.argv.slice(2);
if(!credentialFile||!output)throw Error('Explicit branch credential file and report path are required.');
const secret=JSON.parse(fs.readFileSync(credentialFile));
const u=new URL(secret.DATABASE_URL);u.hostname=u.hostname.replace('-pooler.','.');u.searchParams.set('sslmode','verify-full');
const db=new Client({connectionString:u.href});await db.connect();
const records=JSON.parse(fs.readFileSync(new URL('./records.json',import.meta.url))).records;
const version='research-20261026-geometry-v1';
const hash=s=>createHash('sha256').update(s).digest('hex');
const passages=records.map(r=>{
 const text=[r.title,r.summary,r.limitation,JSON.stringify(r.evidence),JSON.stringify(r.studyDesignAssessment)].join('\n\n');
 const sha=hash(text);
 return {id:r.id+':approved-record:'+sha.slice(0,20),publication_id:r.id,text,sha256:sha};
});
const cachePath='/private/tmp/geometry-literature-embeddings.json';
const cache=fs.existsSync(cachePath)?JSON.parse(fs.readFileSync(cachePath)):{};
const key=JSON.parse(fs.readFileSync('/private/tmp/uo-secrets/worker.json')).OPENAI_API_KEY;
const missing=passages.filter(p=>!cache[p.sha256]);let tokens=0;
if(missing.length){
 const response=await fetch('https://api.openai.com/v1/embeddings',{method:'POST',headers:{Authorization:'Bearer '+key,'Content-Type':'application/json'},body:JSON.stringify({model:'text-embedding-3-small',dimensions:512,input:missing.map(p=>p.text)})});
 if(!response.ok)throw Error('Embedding HTTP '+response.status);
 const data=await response.json();tokens=data.usage.total_tokens;
 for(const row of data.data)cache[missing[row.index].sha256]=row.embedding;
 fs.writeFileSync(cachePath,JSON.stringify(cache),{mode:0o600});
}
try{
 await db.query('BEGIN');
 for(const [i,r] of records.entries()){
  const p=passages[i];
  const document=[r.title,r.summary,r.limitation,r.journal,r.design,r.doi,...r.categories,...r.collections,JSON.stringify(r.evidence)].join('\n');
  await db.query('INSERT INTO research.publications(id,data,document,version) VALUES($1,$2,$3,$4) ON CONFLICT(id) DO UPDATE SET data=EXCLUDED.data,document=EXCLUDED.document,version=EXCLUDED.version',[r.id,r,document,version]);
  await db.query(`INSERT INTO research.passages(id,publication_id,kind,section,text,source_url,rights,sha256,embedding,version) VALUES($1,$2,'approved-record','Verified geometry research summary',$3,$4,$5,$6,$7::vector,$8) ON CONFLICT(id) DO NOTHING`,[p.id,r.id,p.text,r.urls[0],{basis:'Original curated summary and verified bibliographic metadata; no full article text reproduced.',reviewedAt:'2026-10-26'},p.sha256,JSON.stringify(cache[p.sha256]),version]);
 }
 const counts=(await db.query(`SELECT (SELECT count(*)::int FROM research.publications) publications,(SELECT count(*)::int FROM research.passages) passages,(SELECT count(DISTINCT publication_id)::int FROM research.passages WHERE kind='full-text') "fullTextArticles",(SELECT count(*)::int FROM research.atlas) "atlasRecords"`)).rows[0];
 const old=(await db.query("SELECT manifest FROM research.releases WHERE version='research-20261024-v2'")).rows[0].manifest;
 const manifest={...old,...counts,version,geometryAddition:{newRecords:10,vision:3,quantum:7,indexedContent:'Original evidence summaries; full texts unchanged'},recordsSHA256:hash(JSON.stringify(records)),updatedAt:'2026-10-26'};
 await db.query('INSERT INTO research.releases(version,manifest) VALUES($1,$2) ON CONFLICT(version) DO UPDATE SET manifest=EXCLUDED.manifest',[version,manifest]);
 const verified=(await db.query('SELECT p.id,p.data,s.sha256,s.embedding IS NOT NULL AS embedded FROM research.publications p JOIN research.passages s ON s.publication_id=p.id WHERE s.id=ANY($1)',[passages.map(p=>p.id)])).rows;
 if(verified.length!==10||verified.some(r=>!r.embedded||r.sha256!==passages.find(p=>p.publication_id===r.id).sha256))throw Error('Import verification failed');
 await db.query('COMMIT');
 const proof={passed:true,version,added:10,counts,embeddingTokens:tokens,sourceHashes:verified.map(r=>({id:r.id,sha256:r.sha256,embedded:r.embedded}))};
 fs.writeFileSync(output,JSON.stringify(proof,null,2)+'\n');console.log(JSON.stringify({passed:true,added:10,counts,tokens}));
}catch(e){await db.query('ROLLBACK');throw e;}finally{await db.end();}
