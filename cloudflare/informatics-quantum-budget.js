// Separate shared planning pool; no researcher grants and no hardware dispatch.
export const PLAN_LIMITS = {day:30, week:120, month:300};
export function windows(now = new Date()) {
 const day = new Date(Date.UTC(now.getUTCFullYear(),now.getUTCMonth(),now.getUTCDate()));
 const week = new Date(day); week.setUTCDate(week.getUTCDate() - (week.getUTCDay()+6)%7);
 const month = new Date(Date.UTC(now.getUTCFullYear(),now.getUTCMonth(),1));
 return {day:[day,new Date(+day+86400000)],week:[week,new Date(+week+604800000)],month:[month,new Date(Date.UTC(now.getUTCFullYear(),now.getUTCMonth()+1,1))]};
}
export async function pooledPlanning(storage,inspect=false,now=new Date()) {
 return storage.transaction(async tx=>{
  const old=await tx.get('quantum-planning')||{},periods={};
  for(const [period,[start,end]] of Object.entries(windows(now))){
   const key=start.toISOString();
   periods[period]={key,used:old[period]?.key===key?old[period].used:0,limit:PLAN_LIMITS[period],resetAt:end.toISOString()};
  }
  const blocked=Object.values(periods).filter(p=>p.used>=p.limit);
  const allowed=!blocked.length;
  if(!inspect&&allowed){for(const p of Object.values(periods))p.used++;await tx.put('quantum-planning',periods);}
  return {allowed,unit:'AI questions',periods:Object.fromEntries(Object.entries(periods).map(([k,v])=>[k,{...v,remaining:v.limit-v.used}])),resetAt:blocked.length?blocked.map(p=>p.resetAt).sort().at(-1):null,hardware:{available:false,reason:'The approved Study 01 batch is recorded separately. New hardware submissions are not enabled through this question endpoint.'}};
 });
}
export async function quantumAllowance(env,inspect=false){
 const id=env.BUDGET.idFromName(env.QUOTA_SCOPE==='qa'?'quantum-planning-qa':'quantum-planning-public');
 const response=await env.BUDGET.get(id).fetch('https://budget/quantum',{method:'POST',body:JSON.stringify({quantum:true,inspect})});
 const result=await response.json();
 if(!inspect&&!result.allowed){const e=new Error('The shared planning allowance is full. Please return after '+new Date(result.resetAt).toUTCString()+'.');Object.assign(e,{status:429,code:'shared_allowance',resetAt:result.resetAt,publicMessage:true});throw e;}
 return result;
}
