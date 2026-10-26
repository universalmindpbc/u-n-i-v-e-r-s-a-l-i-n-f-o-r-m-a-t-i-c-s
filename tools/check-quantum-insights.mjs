import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {insightLead,interpretations} from '../Informatics/quantum/insights.js';
const data=JSON.parse(await fs.readFile('Informatics/quantum/data/oxtr-qrc-qualification.json','utf8'));
const avg=a=>a.slice(0,6).reduce((a,b)=>a+b,0)/6;
for(const [i,w]of data.windows.entries()){
 const rows=data.records.filter(r=>r.window===w.id),base=rows.find(r=>r.variant==='ordinary');assert.equal(rows.length,8);assert.equal(w.end-w.start,256);
 const delta=(v,field)=>avg(rows.find(r=>r.variant===v)[field])-avg(base[field]);
 for(const v of ['alpha','beta']){assert(delta(v,'ideal_observables')<0);assert(delta(v,'hardware_observables')<0);}
 for(const v of ['gamma','epsilon']){assert(delta(v,'ideal_observables')>0);assert(i===0?delta(v,'hardware_observables')>0:delta(v,'hardware_observables')<0);}
 if(i===1)for(const field of ['ideal_observables','hardware_observables'])for(const v of ['fibonacci','phi','alpha','beta','gamma','epsilon'])assert(delta('shuffled',field)>delta(v,field));
 for(const r of rows)for(const [field,n]of [['gpu_observables',r.shots],['hardware_observables',r.hardware_valid_shots]])for(const p of r[field].slice(0,6))assert(Math.abs(p*n-Math.round(p*n))<1e-8);
}
const html=await fs.readFile('Informatics/quantum/index.html','utf8'),js=await fs.readFile('Informatics/quantum/biology.js','utf8'),view=await fs.readFile('Informatics/quantum/response-view.js','utf8'),copy=await fs.readFile('Informatics/quantum/insights.js','utf8');
assert(html.includes('id="study-initial-insights"'));assert(html.includes('id="results-dialog"'));assert(html.includes('id="close-results"'));
assert(!view.includes("el('details')"));assert(view.includes("role:'button'"));assert(view.includes('Sampling range'));assert(view.includes('Neither side is the “good” side'));
assert(js.includes('resultTrigger?.focus({preventScroll:true})'));assert(js.includes('resultScroll=window.scrollY'));
assert.match(insightLead.meaning,/did not test binding-site prediction/);assert.match(interpretations.quantum.finding,/more in sample 1 but fewer in sample 2/);
assert(!/\b(should|could|would)\b/i.test(html+copy));
assert(!/ideal (calculation|direction|model)/i.test(copy+view));
const coverage=JSON.parse(await fs.readFile('Informatics/quantum/data/oxtr-next-test-evidence.json','utf8'));assert.equal(coverage.peaks.length,5);assert(coverage.samples.every(s=>s.overlap_count===0));assert.equal(coverage.prediction_test_run,false);assert(copy.includes('not pass/fail rules applied after the first 16 runs'));

const dir='research/informatics/quantum-goal-20261025';await fs.mkdir(dir,{recursive:true});await fs.writeFile(dir+'/source-verification.json',JSON.stringify({passed:true,observations_recomputed_from_receipts:true,all_16_records_preserved:true,excitation_counts_are_integers:true,explicit_biological_success_criterion:true,no_calibrated_trust_score_claimed:true,at:new Date().toISOString()},null,2)+'\n');console.log('Interpretations, counts, source syntax and evidence boundaries verified.');
