import {astraStorage} from './informatics-astra.js';
// A question is reserved once. Its bounded retrieval/model loop is included.
// Automated verification uses a separately authenticated, equally bounded pool.
import {pooledPlanning} from './informatics-quantum-budget.js';
const limits={research_question:120,transcription:60,speech:60,voice:20};
export class BudgetError extends Error {
 constructor(kind,resetAt){super(`The daily ${kind==='research_question'?'research':kind} allowance is used. It resets at 00:00 UTC. Please try again after the reset.`);this.status=429;this.code='daily_allowance';this.kind=kind;this.resetAt=resetAt;}
}
export class ResearchBudget {
 constructor(state){this.state=state;}
 async fetch(request){
  const {kind,inspect,scope,quantum,astra}=await request.json();
  if(astra){const r=await astraStorage(this.state.storage,astra);return Response.json(r.data,{status:r.status});}
  if(quantum)return Response.json(await pooledPlanning(this.state.storage,inspect));
  const scopedLimits=scope==='connectors'?{...limits,research_question:30}:limits;
  if(!inspect&&!limits[kind])return new Response('{}',{status:400});
  const date=new Date().toISOString().slice(0,10),resetAt=new Date(Date.parse(date)+86400000).toISOString();
  return this.state.storage.transaction(async txn=>{
   const stored=await txn.get('quota');
   const state=stored?.date===date?stored:{date,counts:{}};
   if(inspect)return Response.json({date,counts:state.counts,limits:scopedLimits,resetAt});
   const used=state.counts[kind]||0;
   if(used>=scopedLimits[kind])return Response.json({allowed:false,kind,resetAt},{status:429});
   state.counts[kind]=used+1;await txn.put('quota',state);
   return Response.json({allowed:true,remaining:scopedLimits[kind]-used-1,resetAt});
  });
 }
}
export async function budget(env,kind){
 const scope=env.QUOTA_SCOPE==='connectors'?'connectors':env.QUOTA_SCOPE==='qa'?'qa':'visitors';
 const response=await env.BUDGET.get(env.BUDGET.idFromName(scope==='qa'?'staging-qa':scope==='connectors'?'connectors':'staging')).fetch('https://budget/check',{method:'POST',body:JSON.stringify({kind,scope})});
 if(!response.ok){const detail=await response.json();throw new BudgetError(kind,detail.resetAt);}
}
export async function budgetStatus(env){
 const result={};
 for(const [label,id]of [['visitors','staging'],['verification','staging-qa'],['connectors','connectors']])result[label]=await(await env.BUDGET.get(env.BUDGET.idFromName(id)).fetch('https://budget/status',{method:'POST',body:JSON.stringify({inspect:true,scope:label})})).json();
 return result;
}
