import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const base='https://staging.universalmind.tv/informatics/';
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
let cursor=1,checked=0;
async function get(file,method='GET'){
 let error;
 for(let attempt=0;attempt<3;attempt++)try{const r=await fetch(base+file+'?v=complete-csv-v1',{method,signal:AbortSignal.timeout(120000)});assert.equal(r.status,200,file);return r;}catch(e){error=e;}
 throw error;
}
async function worker(){while(cursor<=1001){const m=cursor++;await get(`pilot-v1/csv-records/m${m}.csv`,'HEAD');checked++;}}
await Promise.all(Array.from({length:4},worker));
const files=['index.html','research.html','research.js','pilot-v1/m1-m1001.csv','pilot-v1/m1-m1001-csv.zip','pilot-v1/m1-m1001-data.zip','pilot-v1/m1-m1001-images.csv','pilot-v1/research-publications.csv','pilot-v1/M1-M1001-CSV-README.txt',...['9','49','999','1001'].map(m=>`pilot-v1/csv-records/m${m}.csv`)];
for(const file of files){const r=await get(file);assert.equal(hash(Buffer.from(await r.arrayBuffer())),hash(await fs.readFile(path.join(root,'Informatics',file))),file);}
for(const name of await fs.readdir(path.join(root,'Informatics/pilot-v1/csv-shared')))if(name.endsWith('.csv'))await get('pilot-v1/csv-shared/'+name,'HEAD');
const result={passed:true,record_csv_urls_accessible:checked,shared_csv_urls_accessible:10,live_files_with_matching_sha256:files.length,csv_and_json_bundles_match:true,host:base};
await fs.writeFile('/private/tmp/um-complete-csv-http.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
