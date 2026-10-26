// Node.js 20+. No dependency or credential needed.
// node export-research.mjs --endpoint HTTPS_ORIGIN --out NEW_DIRECTORY --atlas all
// --atlas accepts all, m9 or none. All preserves every M1–M1001 record in full.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
const args=Object.fromEntries(process.argv.slice(2).reduce((a,x,i,v)=>x.startsWith('--')?[...a,[x.slice(2),v[i+1]]]:a,[]));
if(!args.endpoint?.startsWith('https://')||!args.out||!['all','m9','none'].includes(args.atlas||'all'))throw Error('Provide --endpoint HTTPS_ORIGIN --out NEW_DIRECTORY [--atlas all|m9|none].');
const endpoint=new URL(args.endpoint).origin,out=path.resolve(args.out),family=args.atlas||'all';
await fs.mkdir(out);await fs.mkdir(path.join(out,'atlas'));
const files=[];let datasetVersion;
async function get(route){
 for(let retry=0;retry<4;retry++){
  const response=await fetch(endpoint+route,{signal:AbortSignal.timeout(120000)});
  if(response.status===429){await new Promise(r=>setTimeout(r,61000));continue;}
  if(!response.ok)throw Error(`Read failed (${response.status}): ${route}`);
  const data=await response.json();
  if(datasetVersion&&data.datasetVersion!==datasetVersion)throw Error('Dataset changed during export. Start a new export.');
  datasetVersion=data.datasetVersion;return data;
 }
 throw Error('Rate limit persists. Retry the export later.');
}
async function save(name,data){
 const bytes=Buffer.from(typeof data==='string'?data:JSON.stringify(data,null,2)+'\n');
 await fs.writeFile(path.join(out,name),bytes,{flag:'wx'});
 files.push({path:name,bytes:bytes.length,sha256:crypto.createHash('sha256').update(bytes).digest('hex')});
}
const inventory=await get('/api/v1/inventory');await save('inventory.json',inventory);
let offset=0;const publications=[];
do{const r=await get('/api/v1/publications?limit=100&offset='+offset);publications.push(...r.records);offset=r.page.nextOffset;}while(offset!==null);
if(new Set(publications.map(r=>r.id)).size!==inventory.counts.publications)throw Error('Publication count mismatch.');
await save('publications.json',{datasetVersion,records:publications});
const cell=value=>'"'+String(value??'').replaceAll('"','""')+'"';
// A JSON field retains every nested record attribute without lossy flattening.
await save('publications.csv','id,record_json\r\n'+publications.map(r=>cell(r.id)+','+cell(JSON.stringify(r))).join('\r\n')+'\r\n');
let atlasCount=0;
if(family!=='none'){
 offset=0;
 do{
  const result=await get('/api/v1/atlas?limit=100&family='+family+'&offset='+offset);
  for(const r of result.records){
   const record=await get('/api/v1/atlas/'+encodeURIComponent(r.id));
   await save('atlas/M'+r.modulus+'.json',record);
   atlasCount++;process.stdout.write(`\rAtlas records exported: ${atlasCount}`);
   await new Promise(resolve=>setTimeout(resolve,550));
  }
  offset=result.page.nextOffset;
 }while(offset!==null);
 const expected=family==='all'?inventory.counts.atlas:inventory.counts.m9_family;
 if(atlasCount!==expected)throw Error('Atlas count mismatch.');
}
const manifest={schemaVersion:'1.0',datasetVersion,createdAt:new Date().toISOString(),source:endpoint,scope:{publications:publications.length,atlasRecords:atlasCount,atlasFamily:family,fullTextIncluded:false},reuse:'Article metadata and approved quotations retain their attribution. Obtain original article reuse rights before redistribution beyond their terms. Atlas hypotheses remain hypotheses.',files};
await fs.writeFile(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
const manifestBytes=await fs.readFile(path.join(out,'manifest.json'));
await fs.writeFile(path.join(out,'integrity.json'),JSON.stringify({algorithm:'SHA-256',manifest:'manifest.json',sha256:crypto.createHash('sha256').update(manifestBytes).digest('hex'),publication:{provider:'DeSci Publish',status:'not-submitted',receipt:null}},null,2)+'\n',{flag:'wx'});
console.log(`\nExport complete: ${publications.length} publications; ${atlasCount} complete Atlas records. No external upload or registration performed.`);
