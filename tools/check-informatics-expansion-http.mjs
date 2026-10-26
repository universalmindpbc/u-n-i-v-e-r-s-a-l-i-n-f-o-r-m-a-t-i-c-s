import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const index=JSON.parse(await fs.readFile(path.join(root,'Informatics/pilot-v1/index.json'),'utf8'));
const base='https://staging.universalmind.tv/informatics/';
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const files=index.records.map(r=>({file:r.file,hash:r.sha256,version:'m1001-v1'}));
for(const file of ['pilot-v1/index.json','pilot-v1/dataset.json','pilot-v1/m1-m1001.json','pilot-v1/m1-m1001.csv','pilot-v1/m1-m1001-data.zip','index.html','research.html','research.js','science.js','science-studies.js','pilot-v1/family.js']){
 files.push({file,hash:hash(await fs.readFile(path.join(root,'Informatics',file))),version:file.endsWith('.zip')?'m1001-v3':'complete-csv-v1'});
}
let cursor=0,checked=0;
async function worker(){while(cursor<files.length){const item=files[cursor++];let error;
 for(let attempt=0;attempt<3;attempt++){try{const r=await fetch(base+item.file+'?v='+item.version,{signal:AbortSignal.timeout(45000)});assert.equal(r.status,200,item.file);assert.equal(hash(Buffer.from(await r.arrayBuffer())),item.hash,item.file);error=null;break;}catch(e){error=e;}}
 if(error)throw error;checked++;if(checked%200===0)console.log('Verified live files:',checked);
}}
await Promise.all(Array.from({length:4},worker));
const result={passed:true,records:1001,live_files_with_matching_sha256:checked,full_json_csv_bundle_verified:true,host:base};
await fs.writeFile('/private/tmp/um-atlas-m1001-http.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
