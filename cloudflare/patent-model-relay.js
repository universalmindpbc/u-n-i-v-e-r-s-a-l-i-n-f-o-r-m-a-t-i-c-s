// Server-to-server access for the protected patent reader. No browser credential.
const json=(value,status=200)=>Response.json(value,{status,headers:{'Cache-Control':'no-store'}});
const fail=(message,status=403)=>json({error:message},status);
export async function patentModelRelay(request,env){
 if(!['https://staging.universalmind.tv','https://universalmind.tv'].includes(env.ALLOWED_ORIGIN)||!env.PATENT_RELAY_SECRET)return fail('Unavailable.',404);
 if(request.method!=='POST')return fail('Method not allowed.',405);
 const timestamp=request.headers.get('X-Patent-Time')||'',signature=request.headers.get('X-Patent-Signature')||'';
 if(!/^\d{10}$/.test(timestamp)||Math.abs(Date.now()/1000-Number(timestamp))>90||!/^[a-f0-9]{64}$/.test(signature))return fail('Private service access is required.');
 if(Number(request.headers.get('Content-Length'))>12000000)return fail('Request too large.',413);
 const bytes=await request.arrayBuffer();if(bytes.byteLength>12000000)return fail('Request too large.',413);
 const body=new TextDecoder().decode(bytes),key=await crypto.subtle.importKey('raw',new TextEncoder().encode(env.PATENT_RELAY_SECRET),{name:'HMAC',hash:'SHA-256'},false,['verify']);
 const valid=await crypto.subtle.verify('HMAC',key,Uint8Array.from(signature.match(/../g),x=>parseInt(x,16)),new TextEncoder().encode(timestamp+'.'+body));
 if(!valid)return fail('Private service access is required.');
 const limited=await env.REQUEST_LIMIT.limit({key:'patent:'+request.headers.get('CF-Connecting-IP')});
 if(!limited.success)return fail('Please wait a minute before the next patent question.',429);
 let data;try{data=JSON.parse(body);}catch{return fail('Invalid request.',400);}
 const {operation,payload}=data;
 if(!['responses','realtime/client_secrets'].includes(operation)||!payload||typeof payload!=='object')return fail('Invalid operation.',400);
 let kind;
 if(operation==='responses'){
  if(payload.model!==env.RESEARCH_MODEL||!Array.isArray(payload.input)||!Array.isArray(payload.tools)||payload.tools.some(t=>t.type!=='function'||!['search_patent','read_patent_pages'].includes(t.name)))return fail('Invalid patent request.',400);
  payload.store=false;payload.stream=false;payload.background=false;payload.max_output_tokens=Math.min(2600,Number(payload.max_output_tokens)||2600);kind='research_question';
 }else{
  if(payload.session?.model!=='gpt-realtime-2.1'||payload.session?.tools?.some(t=>t.type!=='function'||t.name!=='ask_research'))return fail('Invalid patent voice request.',400);
  payload.expires_after={anchor:'created_at',seconds:120};kind='voice';
 }
 const quota=await env.BUDGET.get(env.BUDGET.idFromName('patent')).fetch('https://budget/check',{method:'POST',body:JSON.stringify({kind,scope:'patent'})});
 if(!quota.ok)return fail('The patent research allowance is used for today. Please return after 00:00 UTC.',429);
 const response=await fetch('https://api.openai.com/v1/'+operation,{method:'POST',headers:{Authorization:'Bearer '+env.OPENAI_API_KEY,'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(90000)});
 if(!response.ok){console.error('Patent model request failed',operation,response.status);return fail('The AI service did not complete this request. Please retry.',503);}
 return new Response(response.body,{status:200,headers:{'Content-Type':'application/json','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
}
