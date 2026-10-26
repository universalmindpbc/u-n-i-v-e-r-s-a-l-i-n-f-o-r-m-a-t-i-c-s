import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {build} from '/private/tmp/uo-runtime/node_modules/esbuild/lib/main.js';
const study=JSON.parse(await fs.readFile('Informatics/quantum/data/oxtr-qrc-study.json'));
const state=JSON.parse(await fs.readFile('Informatics/quantum/data/oxtr-qrc-state.json'));
const manifest=JSON.parse(await fs.readFile('Informatics/pilot-v1/oxtr-tfbs-v1/experiment-v1/manifest.json'));
assert.equal(study.gene,'OXTR');assert.equal(study.status,'design_review');assert.equal(state.result,null);assert.deepEqual(state.jobs,[]);assert.equal(study.execution.submission_enabled,false);
assert.deepEqual(study.atlas,manifest.atlas);
const t=study.template,r=t.radius_m,coords=Array.from({length:6},(_,i)=>[Math.round(r*Math.sin(i*Math.PI/3)/1e-8)*1e-8,Math.round(r*Math.cos(i*Math.PI/3)/1e-8)*1e-8]);
for(let i=0;i<6;i++)for(let j=i+1;j<6;j++){assert(Math.hypot(coords[i][0]-coords[j][0],coords[i][1]-coords[j][1])>=4e-6);const dy=Math.abs(coords[i][1]-coords[j][1]);assert(dy<1e-12||dy>=4e-6);}
assert(t.omega_rad_s<=15.8e6);assert(t.duration_s<=20e-6);assert(t.omega_rad_s/t.edge_ramp_s<=4e14);
assert(2*t.detuning_bound_rad_s/(t.duration_s/t.input_slots)<=6e15);
await build({entryPoints:['cloudflare/informatics-quantum.js'],bundle:true,format:'esm',platform:'node',outfile:'/private/tmp/quantum-qrc-handler-test.mjs',logLevel:'silent'});
const {quantumPlan,planRequest}=await import('/private/tmp/quantum-qrc-handler-test.mjs');
assert.throws(()=>planRequest({question:'x'}));
const env={RESEARCH_MODEL:'gpt-6-astra',BUDGET:{idFromName:x=>x,get:()=>({fetch:async()=>Response.json({allowed:true,periods:{}})})}};
let instructions='';const valid={title:'OXTR study',answer:'Prepare the matched experiment.',approach:'qrc',method:'Compare held-out occupancy predictions.',needed_data:[],next_question:'',limitations:[],evidence_ids:[],reads:[]};
async function mock(_env,_path,body){instructions=body.instructions;assert.equal(body.text.format.strict,true);return Response.json({status:'completed',output:[{type:'message',content:[{type:'output_text',text:JSON.stringify(valid)}]}]});}
const result=await(await quantumPlan(env,new Request('https://test/quantum/plan',{method:'POST',body:JSON.stringify({question:study.question})}),mock)).json();
assert.equal(result.study.id,study.id);assert.equal(result.execution,'Study brief');assert.equal(result.calculation,null);assert.match(instructions,/Gamma and Epsilon need an order-sensitive encoding/);assert.match(instructions,/never ask a visitor to define M9 P24/);assert.match(instructions,/"residues"/);assert.match(instructions,/CUDA-Q/);assert.match(instructions,/Aquila/);
const text=await fs.readFile('Informatics/quantum/index.html','utf8');assert(text.indexOf('id="workflow"')<text.indexOf('id="ask"'));assert(!/\b(should|could|would)\b/i.test(text));
const output={passed:true,at:new Date().toISOString(),canonical_atlas_exact:true,coordinate_and_drive_checks:true,current_device_revalidation_required:true,strict_ai_context:true,source_only_no_compute:true};
await fs.writeFile('research/informatics/quantum-qrc-design-20261025/source-validation.json',JSON.stringify(output,null,2)+'\n');console.log(JSON.stringify(output));
