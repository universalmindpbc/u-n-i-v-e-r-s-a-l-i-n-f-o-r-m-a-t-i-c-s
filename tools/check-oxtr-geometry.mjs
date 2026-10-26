import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
import {projectInterval,atlasGuide,countRadius,positionCounts} from '../Informatics/tfbs-geometry.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const source=path.join(root,'Informatics/pilot-v1/oxtr-tfbs-v1'),dir=path.join(source,'experiment-v1');
const read=async f=>JSON.parse(await fs.readFile(path.join(dir,f),'utf8'));
const manifest=await read('manifest.json'),p=await read('python-validation.json'),w=await read('wolfram-validation.json');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
assert.equal(p.status,'passed');assert.equal(p.wolfram_coordinate_table_sha256,w.coordinate_table_sha256);
let table='';
for(let d=0;d<39248;d++){
 const r=d%9;table+=[d,r,r||9,Math.round(1e9*Math.sin(2*Math.PI*d/39248)),Math.round(-1e9*Math.cos(2*Math.PI*d/39248)),Math.round(1e9*Math.sin(2*Math.PI*r/9)),Math.round(-1e9*Math.cos(2*Math.PI*r/9))].join(',')+'\n';
}
assert.equal(hash(table),w.coordinate_table_sha256,'All browser coordinate results agree with Python and Wolfram');
for(const [file,sha] of Object.entries(manifest.source_sha256))assert.equal(hash(await fs.readFile(path.join(source,file))),sha,file);
assert.equal(hash(await fs.readFile(path.join(source,'../m9.json'))),manifest.atlas.sha256);
let total=0;
for(const [collection,{file,sha256}] of Object.entries(manifest.files)){
 const bytes=await fs.readFile(path.join(dir,file));assert.equal(hash(bytes),sha256);
 const rows=JSON.parse(file.endsWith('.gz')?gunzipSync(bytes):bytes),counts=Array(9).fill(0);
 assert.equal(rows.length,manifest.summaries[collection].count);
 for(const row of rows){
  const m=projectInterval(row.start_0based,row.end_exclusive);
  for(const [k,v] of Object.entries(m))assert.equal(v,row.mapping[k],row.id+': '+k);
  assert.ok(row.mapping.anchor_genomic_0based>=row.mapping.visible_start_0based&&row.mapping.anchor_genomic_0based<row.mapping.visible_end_exclusive);
  counts[m.m9_residue]++;total++;
 }
 assert.deepEqual(counts,manifest.summaries[collection].residue_counts_0_to_8);
 const positions=positionCounts(rows);assert.equal(positions.reduce((n,p)=>n+p.count,0),rows.length);
 assert.equal(positions.length,new Set(rows.map(r=>r.mapping.anchor_offset_bp)).size);
 if(collection==='studies'){
  assert.equal(positions.length,81);assert.equal(Math.max(...positions.map(p=>p.count)),6);
  assert.deepEqual(positions.find(p=>p.offset===10092),{offset:10092,count:6});
  const region=positionCounts(rows,9812,10358);assert.equal(region.length,66);assert.equal(region.reduce((n,p)=>n+p.count,0),113);
 }

}
for(const [s,e,g,d,r,z] of w.interval_cases){const m=projectInterval(s,e);assert.deepEqual([m.anchor_genomic_0based,m.anchor_offset_bp,m.m9_residue,m.m9_display],[g,d,r,z]);}
for(const pair of [[0,0],[1,0],[8740000,8740001],[8779628,8779630],[NaN,5]])assert.throws(()=>projectInterval(...pair));
assert.equal(total,9547);
const guide=pattern=>atlasGuide(manifest.atlas,pattern);
assert.deepEqual(guide('alpha').turns,Array.from({length:24},(_,i)=>i/24));
assert.deepEqual(guide('beta').turns,[4,8,7,5,1,2].map(i=>i/9));
assert.deepEqual(guide('beta').paths,[[4,8,7,5,1,2,4]]);
assert.deepEqual(guide('gamma').turns,[0,4,8,12,16,20].map(i=>i/24));
assert.deepEqual(guide('gamma').paths,[[0,8,16,0],[4,12,20,4]]);
assert.deepEqual(guide('epsilon').paths,[[0,8,20,12,4,16,0]]);
assert.equal(countRadius(0,39),0);
for(const counts of [[22,39],[909,1097],[1,10]])assert.ok(Math.abs((countRadius(counts[0],counts[1])/countRadius(counts[1],counts[1]))**2-counts[0]/counts[1])<1e-12,'Dot area preserves the count ratio');
const overlay=await read('overlay-v1.json');
assert.equal(overlay.coordinate_experiment,manifest.version);
assert.equal(overlay.atlas_registration.rotation_turns,0);
assert.equal(overlay.atlas_registration.fit_to_observations,false);
console.log('PASS: 9,547 interval mappings; all 39,248 coordinate positions agree across JavaScript, Python and Wolfram; source hashes, edge cases, native Atlas overlay angles and proportional dot areas verified.');
