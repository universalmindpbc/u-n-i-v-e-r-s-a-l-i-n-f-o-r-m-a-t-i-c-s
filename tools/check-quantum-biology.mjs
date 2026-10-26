import assert from 'node:assert/strict';
import {compileAssembly,overlap} from '../cloudflare/informatics-quantum-model.js';
import {pooledPlanning,windows,PLAN_LIMITS} from '../cloudflare/informatics-quantum-budget.js';
class Storage{constructor(){this.data=new Map();this.tail=Promise.resolve();}transaction(fn){const run=this.tail.then(()=>fn({get:async k=>structuredClone(this.data.get(k)),put:async(k,v)=>this.data.set(k,structuredClone(v))}));this.tail=run.catch(()=>{});return run;}}
const reads=['ATGCCTA','CCTAGGA','AGGATTC'],c=compileAssembly(reads);assert.equal(c.sequence,'ATGCCTAGGATTC');assert.equal(c.tested,6);assert.equal(c.optimalOrders,1);assert.equal(c.overlapScore,8);
const Q=x=>c.qubo.constant+c.qubo.linear.reduce((s,v,i)=>s+v*x[i],0)+c.qubo.quadratic.reduce((s,t)=>s+t.value*x[t.i]*x[t.j],0);
const H=x=>{const z=x.map(t=>1-2*t);return c.ising.offset+c.ising.h.reduce((s,v,i)=>s+v*z[i],0)+c.ising.J.reduce((s,t)=>s+t.value*z[t.i]*z[t.j],0);};
let min=Infinity,best=[];for(let b=0;b<512;b++){const x=Array.from({length:9},(_,i)=>(b>>i)&1);assert.equal(Q(x),H(x),'Ising preserves every energy');
 let penalty=0;for(let i=0;i<3;i++){penalty+=(1-x.slice(i*3,i*3+3).reduce((a,b)=>a+b,0))**2;penalty+=(1-x[i]-x[i+3]-x[i+6])**2;}
 let reward=0;for(let p=0;p<2;p++)for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(i!==j)reward+=overlap(reads[i],reads[j])*x[i*3+p]*x[j*3+p+1];assert.equal(Q(x),c.qubo.penalty*penalty-reward,'QUBO matches independent constraints');
 if(Q(x)<min){min=Q(x);best=[x];}else if(Q(x)===min)best.push(x);
}assert.equal(min,-c.overlapScore);assert.equal(best.length,1);assert.deepEqual(best[0],[1,0,0,0,1,0,0,0,1]);
assert.throws(()=>compileAssembly(['ACGT','ACGT','TGCA']));assert.throws(()=>compileAssembly(['ACGT','NCTG','TGCA']));assert.throws(()=>compileAssembly(['ACGT','AACGTA','GGGG']));assert.throws(()=>compileAssembly(Array(7).fill('ACGT')));
assert.equal(compileAssembly(['AAAA','CCCC','GGGG']).optimalOrders,6);assert.equal(compileAssembly(['AAAA','CCCC','GGGG']).disconnected,true);
const s=new Storage(),date=new Date('2026-10-25T23:59:59Z');let calls=await Promise.all(Array.from({length:45},()=>pooledPlanning(s,false,date)));assert.equal(calls.filter(x=>x.allowed).length,30);assert.equal((await pooledPlanning(s,true,date)).periods.day.used,30);
const next=await pooledPlanning(s,false,new Date('2026-10-26T00:00:00Z'));assert.equal(next.periods.day.used,1);assert.equal(next.periods.week.used,1);assert.equal(next.periods.month.used,31);
const weekly=new Storage();for(let d=26;d<30;d++)for(let i=0;i<30;i++)assert.equal((await pooledPlanning(weekly,false,new Date(`2026-10-${d}T12:00:00Z`))).allowed,true);const blocked=await pooledPlanning(weekly,false,new Date('2026-10-30T12:00:00Z'));assert.equal(blocked.allowed,false);assert.equal(blocked.periods.day.used,0);assert.equal(blocked.resetAt,'2026-11-02T00:00:00.000Z');
const monthly=new Storage();for(let d=1;d<=30;d++)for(let i=0;i<10;i++)assert.equal((await pooledPlanning(monthly,false,new Date(`2026-10-${String(d).padStart(2,'0')}T12:00:00Z`))).allowed,true);assert.equal((await pooledPlanning(monthly,false,new Date('2026-10-31T12:00:00Z'))).allowed,false);assert.equal((await pooledPlanning(monthly,false,new Date('2026-11-01T12:00:00Z'))).allowed,true);
assert.equal(windows(new Date('2026-12-31T23:59:59Z')).month[1].toISOString(),'2027-01-01T00:00:00.000Z');
console.log(JSON.stringify({passed:true,energy_states_checked:512,exact_dna_result:c.sequence,concurrent_admission_limit:PLAN_LIMITS.day,weekly_and_monthly_boundaries:true,validation:true}));
