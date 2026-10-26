import fs from 'node:fs';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {verifiedSolve,BASE,validate} from '../Informatics/quantum/model.mjs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),cases=[BASE];
for(const d of [4.5,6.8,9])for(const T of [2,6])for(const o of [1,10])for(const delta of [10,50])cases.push({distance_um:d,duration_us:T,omega_rad_us:o,detuning_rad_us:delta});
const results=[];
for(const spec of cases){
 fs.writeFileSync('/private/tmp/quantum-numeric-case.json',JSON.stringify({model_version:'rydberg-pair-v1',spec}));
 const independent=JSON.parse(execFileSync('/private/tmp/quantum-gpu-venv/bin/python',[root+'/Informatics/quantum/reproduce.py','/private/tmp/quantum-numeric-case.json'],{encoding:'utf8'}));
 const r=verifiedSolve(spec),difference=Math.max(...r.probabilities.map((v,i)=>Math.abs(v-independent.probabilities[['gg','gr','rg','rr'][i]])));
 assert.ok(difference<1e-7,JSON.stringify({spec,difference}));assert.ok(r.normError<1e-6);assert.ok(r.convergence<1e-6);
 results.push({spec,difference,convergence:r.convergence,normError:r.normError});
}
for(const bad of [null,{}, {...BASE,distance_um:0},{...BASE,omega_rad_us:Infinity},{...BASE,duration_us:'4'}])assert.throws(()=>validate(bad));
fs.writeFileSync(root+'/research/informatics/quantum-lab-20261025/numerical-verification.json',JSON.stringify({independent:'SciPy DOP853, rtol=1e-11, atol=1e-13',cases:results.length,tolerance:1e-7,maxDifference:Math.max(...results.map(x=>x.difference)),results},null,2));
console.log(JSON.stringify({passed:true,cases:results.length,maxDifference:Math.max(...results.map(x=>x.difference))}));
