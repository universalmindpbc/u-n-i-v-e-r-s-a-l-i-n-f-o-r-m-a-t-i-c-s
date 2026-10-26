const endpoint=location.hostname==='staging.universalmind.tv'?'https://informatics-universal-one-staging.universalmind.workers.dev':'https://informatics-universal-one.universalmind.workers.dev';
const $=s=>document.querySelector(s),el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const tokenKey='astra-research-session-v1';let timer,task=null,changed=()=>{};
function token(){try{return localStorage.getItem(tokenKey)||'';}catch{return '';}}
export function agentHeaders(){return token()?{Authorization:'Bearer '+token()}:{};}
let connecting;
export async function ensureAgentSession(){
 if(!['staging.universalmind.tv','universalmind.tv'].includes(location.hostname))return;
 try{const claims=JSON.parse(atob(token().split('.')[0].replace(/-/g,'+').replace(/_/g,'/')));if(claims.exp>Date.now()/1000+60)return;}catch{}
 if(!connecting)connecting=fetch('../api/astra-session.php',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({previous:token()})}).then(async r=>{const d=await r.json();if(!r.ok)throw Error(d.error||'The research session did not connect.');localStorage.setItem(tokenKey,d.token);}).finally(()=>{connecting=null;});
 return connecting;
}
function save(value,name,type='application/json'){const text=typeof value==='string'?value:JSON.stringify(value,null,2),url=URL.createObjectURL(new Blob([text],{type})),a=el('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);}
function csv(value){const rows=[['field','type','value','escaped']];function walk(v,path){const type=v===null?'null':Array.isArray(v)?'array':typeof v;let text=type==='object'||type==='array'?'':String(v??'');const escaped=/^[=+@\-\t\r]/.test(text);if(escaped)text="'"+text;rows.push([path,type,text,escaped?'1':'0']);if(v!==null&&typeof v==='object')for(const [k,x]of Object.entries(v))walk(x,path+'/'+k.replace(/~/g,'~0').replace(/\//g,'~1'));}walk(value,'');return rows.map(r=>r.map(v=>'"'+v.replace(/"/g,'""')+'"').join(',')).join('\r\n');}

function show(t){task=t;const box=$('#agent-progress');box.hidden=false;$('#agent-status').textContent=t.status==='complete'?'Your analysis is ready':t.status==='queued'?'Astra is waiting for the research worker':t.status==='needs_attention'?'Progress saved · service attention needed':t.status==='cancelled'?'Study cancelled':'Astra is working on your question';$('#agent-message').textContent=t.message;$('#agent-open').hidden=!t.report;$('#agent-cancel').hidden=['complete','failed','cancelled','needs_attention'].includes(t.status);$('#agent-history').replaceChildren();for(const e of t.events||[])$('#agent-history').append(el('li',e.message));window.dispatchEvent(new CustomEvent('quantum:agent',{detail:t}));changed(t);}
export function followAgent(t,onChange){clearTimeout(timer);changed=onChange;show(t);if(!['complete','failed','cancelled','needs_attention'].includes(t.status))timer=setTimeout(poll,4000);}
async function poll(){try{await ensureAgentSession();const r=await fetch(endpoint+'/agent/studies/'+task.id,{headers:agentHeaders(),signal:AbortSignal.timeout(20000)});if(!r.ok)throw Error(r.status===401?'Reconnect to reopen the saved agent study.':'Progress is temporarily unavailable. Your task remains saved.');followAgent(await r.json(),changed);}catch(e){$('#agent-message').textContent=e.message;timer=setTimeout(poll,15000);}}
export function resetAgent(){clearTimeout(timer);task=null;$('#agent-progress').hidden=true;window.dispatchEvent(new CustomEvent('quantum:agent',{detail:null}));}
export function renderAgentReport(slot,t,{integrated=false}={}){
 const r=t.report;if(!r){slot.append(el('h3','Your study is still in progress'),el('p',t.message));return;}
 const reviewed=r.review?.status==='model_review_complete';
 slot.dataset.state='complete';slot.append(el('span',reviewed?'ASTRA · GEMINI REVIEW COMPLETE':'ASTRA · PRELIMINARY INSIGHTS','result-status'),el('h3','What we found'));
 const answer=reviewed?r.narrative.answer.replace(/^Draft for Gemini scientific review:\s*/i,''):r.narrative.answer;
 const callout=el('div',undefined,'agent-finding');callout.append(el('p',answer));slot.append(callout);
 for(const [key,title]of[['what_we_did','What the agent completed'],['meaning','What this means for your question'],...(!integrated?[['next_dependency','Where this run stops']]:[])]){slot.append(el('h4',title),el('p',r.narrative[key]));}
 if(r.reviews?.length){const review=r.reviews.filter(x=>x.phase==='final').at(-1);if(review){const stamp=el('div',undefined,'recorded-review-stamp');stamp.append(el('strong','Reviewed by Gemini 4 Argon'),el('time',new Date(review.reviewed_at).toLocaleString()),el('p',review.summary));slot.append(stamp);}}
 const a=r.analysis;slot.append(el('h4','The comparison, in plain numbers'),el('p','CTCF is one protein that binds DNA. We checked whether each method ranked the eight OXTR windows overlapping reported CTCF regions above the other 144 windows. The laboratory map contains five peak regions; some span more than one window.'));
 const table=el('table',undefined,'agent-table');table.append(el('caption','Average precision is a ranking score from 0 to 1. Higher is better; it is not percentage accuracy.'));
 const head=el('tr');for(const h of ['Method','OXTR ranking score'])head.append(el('th',h));table.append(head);
 for(const [name,v]of Object.entries(a.models)){const row=el('tr');row.append(el('td',name==='ordinary motif and spacing'?'DNA motif and spacing features':'Same features + Fibonacci spacing and Phi Ratio'),el('td',v.OXTR_average_precision.toFixed(4)));table.append(row);}slot.append(table);
 slot.append(el('p',`Change: ${a.difference_average_precision.toFixed(4)}. The 95% interval from resampling DNA blocks runs from ${a.block_bootstrap_95_interval[0].toFixed(4)} to ${a.block_bootstrap_95_interval[1].toFixed(4)}. An interval spanning zero leaves the benefit unresolved.`));
 if(!integrated)slot.append(el('h4','How the saved quantum work fits'),el('p','The agent reused all 16 CUDA-Q simulations and 16 Aquila measurements, covering two DNA samples prepared with ordinary input, Fibonacci, Phi Ratio, Alpha, Beta, Gamma, Epsilon and a shuffled control. Those two samples contain no reported CTCF peak. They establish measured quantum responses, but this set alone cannot compare binding regions with non-binding regions. The new prediction scores above come from classical calculations.'));
 if(r.prepared_quantum&&!integrated){slot.append(el('h4','The next quantum test is already prepared'),el('p','The agent selected one OXTR DNA window overlapping a reported CTCF region and one nearby comparison window. It compiled all eight representations for each and checked all 16 programs with CUDA-Q on the local CPU. These are new local simulations; the qBraid GPU and Aquila readings above are the saved earlier measurements. No new cloud quantum job was submitted. Two windows check the programs; a larger set is necessary to assess prediction.'));}
 const methods=el('details');methods.append(el('summary','Data, quality checks and exact methods'));for(const text of [...r.cohort.limits,'ENCODE assay '+r.cohort.assay+' · '+r.cohort.cell,...r.cohort.qc])methods.append(el('p',text));for(const s of r.cohort.sources){const p=el('p'),a=el('a',s.kind+' ↗');a.href=s.source;a.target='_blank';a.rel='noopener noreferrer';p.append(a);methods.append(p);}slot.append(methods);
 const downloads=el('div',undefined,'completed-study-actions');for(const format of ['JSON','CSV']){const b=el('button','Download complete study · '+format+' ↓','text-button');b.onclick=()=>save(format==='JSON'?r:csv(r),'astra-'+t.id+'.'+format.toLowerCase(),format==='CSV'?'text/csv':'application/json');downloads.append(b);}slot.append(downloads);
}
export function setupAgent(open){
 if(!['staging.universalmind.tv','universalmind.tv'].includes(location.hostname))return;
 $('#agent-open').onclick=()=>open(task);$('#agent-cancel').onclick=async()=>{try{await ensureAgentSession();const r=await fetch(endpoint+'/agent/studies/'+task.id+'/cancel',{method:'POST',headers:agentHeaders()});if(!r.ok)throw Error('Cancellation is not confirmed. Retry.');followAgent(await r.json(),changed);}catch(e){$('#agent-message').textContent=e.message;}};
 window.addEventListener('pagehide',()=>clearTimeout(timer));
}
