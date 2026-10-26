import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';

const changed=['molecular/index.html','molecular/atlas.js','molecular/interaction-ui.js','molecular/interactions.css','molecular/vendor/engine.js'];
const preserved=['index.html','universal-one-live.js','universal-one-routing.js','universal-one.css','molecular/atlas.css','molecular/data/interactions/contact.json','molecular/data/interactions/quantum.json','molecular/data/interactions/fragment.pdb','molecular/data/interactions/fragment.xyz','molecular/data/interactions/homo.cube','molecular/data/interactions/homo-minus-1.cube','molecular/data/interactions/lumo.cube','molecular/data/interactions/density.cube'];
const results=[];
for(const file of [...changed,...preserved]){
 const url='https://staging.universalmind.tv/informatics/'+file+'?verify='+Date.now();
 const response=await fetch(url,{cache:'no-store'});
 assert.equal(response.status,200,file);
 const actual=Buffer.from(await response.arrayBuffer()),expected=await fs.readFile('Informatics/'+file);
 assert.ok(actual.equals(expected),'HTTP bytes match committed source: '+file);
 results.push({file,group:changed.includes(file)?'changed':'preserved',status:200,bytes:actual.length,sha256:crypto.createHash('sha256').update(actual).digest('hex')});
}
const proof={sourceCommit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),verifiedAt:new Date().toISOString(),passed:true,results};
await fs.writeFile('/private/tmp/quantum-teaching-http.json',JSON.stringify(proof,null,2)+'\n');
console.log(JSON.stringify({passed:true,assets:results.length,sourceCommit:proof.sourceCommit}));
