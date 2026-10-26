// Bounded research sessions; no page-level login. Provider credentials never reach the browser.
import {retrieveAstraMemory} from './informatics-astra-memory.js';
const json=(data,status=200)=>Response.json(data,{status,headers:{'Cache-Control':'no-store'}});
const terminal=new Set(['complete','failed','cancelled','needs_attention']);
const fail=(message,status=400)=>Object.assign(new Error(message),{status,publicMessage:true});
const decode=s=>Uint8Array.from(atob(s.replace(/-/g,'+').replace(/_/g,'/')),c=>c.charCodeAt(0));
export async function researchIdentity(request,env){
 try{
  if(!env.ASTRA_SESSION_SECRET)return false;
  const raw=(request.headers.get('Authorization')||'').replace(/^Bearer /,''),[body,signature,extra]=raw.split('.');if(extra||!body||!signature)return false;
  const key=await crypto.subtle.importKey('raw',new TextEncoder().encode(env.ASTRA_SESSION_SECRET),{name:'HMAC',hash:'SHA-256'},false,['verify']);
  if(!await crypto.subtle.verify('HMAC',key,decode(signature),new TextEncoder().encode(body)))return false;
  const claims=JSON.parse(new TextDecoder().decode(decode(body))),now=Date.now()/1000;
  if(claims.aud!==env.ALLOWED_ORIGIN||!Number.isInteger(claims.exp)||claims.exp<=now||claims.exp>now+86400)return false;
  if(claims.scope==='astra-owner')return 'legacy-owner';
  return claims.scope==='astra-research'&&/^[a-f0-9]{32}$/.test(claims.sid||'')?claims.sid:false;
 }catch{return false;}
}
export async function ownerAllowed(request,env){return Boolean(await researchIdentity(request,env));}
async function broker(env,payload){
 const r=await env.BUDGET.get(env.BUDGET.idFromName('astra-owner-v1')).fetch('https://budget/astra',{method:'POST',body:JSON.stringify({astra:payload})});
 if(!r.ok){const d=await r.json();throw fail(d.error||'Study service unavailable.',r.status);}return r.json();
}
export async function createStudy(env,request,receipt,requestKey){
 if(env.ASTRA_ENABLED!=='true')return receipt;
 const allowed=await researchIdentity(request,env);
 if(request.headers.has('Authorization')&&!allowed)throw fail('Your research session expired. Reopen the question to reconnect.',401);
 if(!allowed)return receipt;
 if(!/^[a-zA-Z0-9-]{16,80}$/.test(requestKey||''))throw fail('A study request identifier is required.');
 const task=await broker(env,{op:'create',requestKey,question:receipt.question,clientId:allowed});
 return {...receipt,study:{id:task.id,version:task.version},status:'agent_running',message:task.message,agent:task};
}
export async function astraRoute(request,env){
 if(env.ASTRA_ENABLED!=='true')return json({error:'Agent execution is unavailable.'},404);
 const path=new URL(request.url).pathname;
 if(path==='/agent/memory'){
  if(request.method!=='POST')return json({error:'Use POST for research retrieval.'},405);
  if(!env.ASTRA_WORKER_TOKEN||request.headers.get('Authorization')!=='Bearer '+env.ASTRA_WORKER_TOKEN)return json({error:'Worker authentication required.'},401);
  const raw=await request.text();if(raw.length>3000)return json({error:'Query too large.'},413);
  return json(await retrieveAstraMemory(env,JSON.parse(raw).query));
 }
 if(path==='/agent/control'){
  if(!env.ASTRA_WORKER_TOKEN||request.headers.get('Authorization')!=='Bearer '+env.ASTRA_WORKER_TOKEN)return json({error:'Worker authentication required.'},401);
  const text=await request.text();if(text.length>500000)return json({error:'Update too large.'},413);
  const body=JSON.parse(text);if(!['pending','claim','update','heartbeat'].includes(body.op))return json({error:'Unknown worker action.'},400);
  return json(await broker(env,body));
 }
 const clientId=await researchIdentity(request,env);
 if(!clientId)return json({error:'Research session required. Reopen Quantum Atlas to reconnect.'},401);
 if(path==='/agent/access')return json({access:'research-session',newAquilaAllowed:false,newPaidGPUAllowed:false});
 const match=path.match(/^\/agent\/studies\/([a-f0-9-]{36})(?:\/(cancel|resume))?$/);
 if(!match)return json({error:'Study not found.'},404);
 if(match[2]&&request.method!=='POST')return json({error:'Use POST to cancel.'},405);
 return json(await broker(env,{op:match[2]||'get',id:match[1],clientId}));
}
export async function astraStorage(storage,b){
 return storage.transaction(async txn=>{
  const now=Date.now(),date=new Date(now).toISOString().slice(0,10),month=date.slice(0,7);
  let state=await txn.get('astra')||{tasks:[],daily:{},monthly:{}};
  state.tasks=await Promise.all(state.tasks.map(async t=>await txn.get('study:'+t.id)||t));
  const safe=t=>{const {lease,...v}=t;return v;};
  const answer=(d,status=200)=>({data:d,status});
  let result;
  if(b.op==='pending')return answer({pending:state.tasks.some(t=>!terminal.has(t.status)&&(!t.lease||t.lease.until<now))});
  if(b.op==='create'){
   const existing=state.tasks.find(t=>t.requestKey===b.requestKey&&(t.clientId||'legacy-owner')===(b.clientId||'legacy-owner'));
   if(existing){if(existing.question!==b.question)return answer({error:'Request identifier already belongs to another question.'},409);return answer(safe(existing));}
   if(typeof b.question!=='string'||b.question.length<5||b.question.length>3000)return answer({error:'Invalid question.'},400);
   if((state.daily[date]||0)>=6||(state.monthly[month]||0)>=40)return answer({error:'The shared analysis allowance is used. Saved studies remain available.'},429);
   if(state.tasks.filter(t=>!terminal.has(t.status)).length>=2)return answer({error:'Two studies are already in progress. Reopen one or wait for completion.'},429);
   const task={clientId:b.clientId||'legacy-owner',id:crypto.randomUUID(),version:'oxtr-follow-through-v1',requestKey:b.requestKey,question:b.question,status:'queued',stage:'question',message:'Astra will analyse the saved measurements and public DNA evidence. No new Aquila run will be submitted.',createdAt:new Date(now).toISOString(),updatedAt:new Date(now).toISOString(),events:[],policy:{newAquilaUSD:0,newPaidGPUUSD:0,maxModelCalls:4,requireGeminiReview:true,maxModelTokens:24000},report:null};
   state.tasks.push(task);state.daily[date]=(state.daily[date]||0)+1;state.monthly[month]=(state.monthly[month]||0)+1;result=answer(safe(task));
  }else if(b.op==='claim'){
   const task=state.tasks.find(t=>!terminal.has(t.status)&&(!t.lease||t.lease.until<now));
   if(!task)return answer({task:null});
   task.lease={token:crypto.randomUUID(),until:now+180000};task.status='running';task.updatedAt=new Date(now).toISOString();result=answer({task:safe(task),lease:task.lease.token});
  }else{
   const task=state.tasks.find(t=>t.id===b.id);if(!task)return answer({error:'Study not found.'},404);
   if(b.clientId!==undefined&&(task.clientId||'legacy-owner')!==b.clientId)return answer({error:'Study not found in this research session.'},404);
   if(b.op==='get')return answer(safe(task));
   if(b.op==='resume'){
    if(task.status!=='needs_attention')return answer({error:'Only a paused study can resume.'},409);task.status='queued';task.message='Saved analysis queued to resume within its existing allowance.';delete task.lease;result=answer(safe(task));
   }else if(b.op==='cancel'){
    if(!terminal.has(task.status)){task.status='cancelled';task.message='Study cancelled. Completed evidence remains saved.';task.updatedAt=new Date(now).toISOString();delete task.lease;}
    result=answer(safe(task));
   }else{
    if(terminal.has(task.status)||task.lease?.token!==b.lease||task.lease.until<now)return answer({error:'Study lease ended; stop work and reconcile.'},409);
    task.lease.until=now+180000;
    if(b.op==='update'){
     if(!['question','evidence','model','classical','simulation','hardware','design_review','final_review','result'].includes(b.stage)||!['running','complete','failed','needs_attention'].includes(b.status))return answer({error:'Invalid study transition.'},400);
     if(b.reviews){if(!Array.isArray(b.reviews)||b.reviews.length>4||JSON.stringify(b.reviews).length>30000||b.reviews.some(r=>!['design','final'].includes(r.phase)||!['approve','revise','human'].includes(r.decision)||!r.provider_response_id||!r.evidence_sha256||!r.reviewed_at))return answer({error:'Invalid scientific review receipt.'},422);task.reviews=b.reviews;}
     task.stage=b.stage;task.status=b.status;task.message=String(b.message||'').slice(0,700);task.updatedAt=new Date(now).toISOString();
     task.events.push({stage:b.stage,status:b.status,message:task.message,at:task.updatedAt});task.events=task.events.slice(-40);
     if(b.report){if(JSON.stringify(b.report).length>200000)return answer({error:'Report exceeds the bounded study size.'},413);if(b.report.version!=='oxtr-follow-through-v1'||!b.report.analysis||!b.report.prior_quantum||!b.report.narrative)return answer({error:'Report is incomplete.'},422);task.report=b.report;}
     if(task.status==='complete'&&task.policy.requireGeminiReview&&(!task.report||task.report.review?.status!=='model_review_complete'||['design','final'].some(phase=>task.report.reviews?.filter(r=>r.phase===phase).at(-1)?.decision!=='approve')))return answer({error:'Both Gemini scientific reviews are required.'},422);
     if(task.status==='complete'&&!task.report)return answer({error:'A completed task needs a report.'},422);
     if(terminal.has(task.status))delete task.lease;
    }
    result=answer(safe(task));
   }
  }
  for(const task of state.tasks)await txn.put('study:'+task.id,task);
  await txn.put('astra',{...state,tasks:state.tasks.map(({id,requestKey,status})=>({id,requestKey,status}))});return result;
 });
}
