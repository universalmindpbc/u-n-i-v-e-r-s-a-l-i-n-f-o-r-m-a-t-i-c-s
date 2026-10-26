// Run with UO_RUNTIME pointing to a task-local node_modules directory.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
const require=createRequire(path.resolve(process.env.UO_RUNTIME,'package.json'));
const {Client}=require('pg');
const {S3Client,PutObjectCommand,HeadObjectCommand}=require('@aws-sdk/client-s3');
const [secretsPath,corpusPath]=process.argv.slice(2);
const secrets=JSON.parse(fs.readFileSync(secretsPath));
const read=name=>JSON.parse(fs.readFileSync(path.join(corpusPath,name+'.json')));
const records=read('publications'),passages=read('passages'),manifest=read('manifest');
const db=new Client({connectionString:secrets.DATABASE_URL});await db.connect();
await db.query(fs.readFileSync(new URL('./schema.sql',import.meta.url),'utf8'));
const apiKey=process.env.OPENAI_API_KEY;
if(!apiKey)throw Error('OPENAI_API_KEY required');
const embedCache=path.join(corpusPath,'embeddings.json');
const embeddings=fs.existsSync(embedCache)?JSON.parse(fs.readFileSync(embedCache)):{};
let tokenUsage=0;
for(let start=0;start<passages.length;start+=48){
 const batch=passages.slice(start,start+48).filter(p=>!embeddings[p.sha256]);
 if(!batch.length)continue;
 const response=await fetch('https://api.openai.com/v1/embeddings',{method:'POST',headers:{Authorization:`Bearer ${apiKey}`,'Content-Type':'application/json'},body:JSON.stringify({model:'text-embedding-3-small',dimensions:512,input:batch.map(p=>p.text.slice(0,22000))})});
 if(!response.ok)throw Error(`Embedding HTTP ${response.status}: ${await response.text()}`);
 const data=await response.json();tokenUsage+=data.usage.total_tokens;
 for(const item of data.data)embeddings[batch[item.index].sha256]=item.embedding;
 fs.writeFileSync(embedCache,JSON.stringify(embeddings));
 if(start%240===0)console.log('Embedded',start+batch.length,'/',passages.length);
}
for(const r of records){
 const document=[r.title,r.summary,r.limitation,r.journal,r.design,r.doi,r.pmid,...r.targets,...r.categories,...r.collections,...(r.neurotransmitters||[]),JSON.stringify(r.conditions||[]),...r.quotes.map(q=>typeof q==='string'?q:JSON.stringify(q))].filter(Boolean).join('\n');
 await db.query('INSERT INTO research.publications(id,data,document,version) VALUES($1,$2,$3,$4) ON CONFLICT(id) DO UPDATE SET data=EXCLUDED.data,document=EXCLUDED.document,version=EXCLUDED.version',[r.id,r,document,manifest.version]);
}
for(let i=0;i<passages.length;i+=100){
 const batch=passages.slice(i,i+100).map(p=>({...p,embedding:JSON.stringify(embeddings[p.sha256])}));
 await db.query(`INSERT INTO research.passages(id,publication_id,kind,section,text,source_url,rights,sha256,embedding,version)
 SELECT id,publication_id,kind,section,text,source_url,rights,sha256,embedding::vector,version FROM jsonb_to_recordset($1::jsonb)
 AS x(id text,publication_id text,kind text,section text,text text,source_url text,rights jsonb,sha256 text,embedding text,version text)
 ON CONFLICT(id) DO UPDATE SET embedding=EXCLUDED.embedding`,[JSON.stringify(batch)]);
}
const atlasHashes=new Map((await db.query('SELECT id,sha256 FROM research.atlas')).rows.map(r=>[r.id,r.sha256]));
for(const r of read('atlas'))if(atlasHashes.get(r.id)!==r.sha256)await db.query('INSERT INTO research.atlas(id,modulus,data,sha256) VALUES($1,$2,$3,$4) ON CONFLICT(id) DO UPDATE SET data=EXCLUDED.data,sha256=EXCLUDED.sha256',[r.id,r.modulus,r.data,r.sha256]);
await db.query('INSERT INTO research.releases VALUES($1,$2) ON CONFLICT(version) DO UPDATE SET manifest=EXCLUDED.manifest',[manifest.version,manifest]);
const password=crypto.randomBytes(32).toString('hex');
const roles=await db.query("SELECT 1 FROM pg_roles WHERE rolname='informatics_reader'");
if(!roles.rowCount){
 await db.query(`CREATE ROLE informatics_reader LOGIN PASSWORD '${password}'`);
 await db.query('GRANT CONNECT ON DATABASE informatics TO informatics_reader');
 await db.query('GRANT USAGE ON SCHEMA research TO informatics_reader');
 await db.query('GRANT SELECT ON ALL TABLES IN SCHEMA research TO informatics_reader');
 await db.query("ALTER ROLE informatics_reader SET statement_timeout='10s'");
 await db.query('ALTER ROLE informatics_reader SET default_transaction_read_only=on');
 const u=new URL(secrets.DATABASE_URL);u.username='informatics_reader';u.password=password;
 fs.writeFileSync(path.join(path.dirname(secretsPath),'worker.json'),JSON.stringify({DATABASE_URL:u.href,OPENAI_API_KEY:apiKey,ACCESS_SECRET:crypto.randomBytes(32).toString('hex')}),{mode:0o600});
}
const count=await db.query('SELECT (SELECT count(*) FROM research.publications)::int publications,(SELECT count(*) FROM research.passages)::int passages,(SELECT count(*) FROM research.atlas)::int atlas,pg_database_size(current_database()) bytes');
await db.end();
const storage=new S3Client({endpoint:secrets.endpoint,region:'ap-southeast-1',forcePathStyle:true,credentials:{accessKeyId:secrets.storage.token_id,secretAccessKey:secrets.storage.s3_secret_access_key}});
const uploaded=[];const objects=read('objects');
// Include immutable metadata and extraction manifest with the originals.
for(const name of ['publications','manifest'])objects.push({path:path.join(corpusPath,name+'.json'),key:`${manifest.version}/${name}.json`});
for(let i=0;i<objects.length;i+=5){
 await Promise.all(objects.slice(i,i+5).map(async o=>{
  const body=fs.readFileSync(o.path), sha=crypto.createHash('sha256').update(body).digest('hex');
  await storage.send(new PutObjectCommand({Bucket:'research-evidence',Key:o.key,Body:body,ContentType:o.path.endsWith('.pdf')?'application/pdf':o.path.endsWith('.xml')?'application/xml':o.path.endsWith('.html')?'text/html':'application/json',Metadata:{sha256:sha}}));
  const check=await storage.send(new HeadObjectCommand({Bucket:'research-evidence',Key:o.key}));
  if(check.ContentLength!==body.length||check.Metadata.sha256!==sha)throw Error('Object verification failed');
  uploaded.push({key:o.key,sha256:sha,bytes:body.length});
 }));
 if(i%50===0)console.log('Archived',Math.min(i+5,objects.length),'/',objects.length);
}
fs.writeFileSync(path.join(corpusPath,'import-report.json'),JSON.stringify({version:manifest.version,database:count.rows[0],embeddingTokens:tokenUsage,objects:uploaded},null,2));
console.log(JSON.stringify({database:count.rows[0],archivedObjects:uploaded.length,embeddingTokens:tokenUsage}));
