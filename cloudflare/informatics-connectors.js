import {repository,RepositoryError,recordsCSV} from './informatics-repository.js';
const str={type:'string',maxLength:800},id={type:'string',maxLength:250};
const page={offset:{type:'integer',minimum:0,maximum:10000},limit:{type:'integer',minimum:1,maximum:100}};
const filterSchema={type:'object',additionalProperties:false,properties:Object.fromEntries(['collections','targets','conditions','journals'].map(k=>[k,{type:'array',maxItems:12,items:{type:'string',maxLength:200}}]))};
const definition=(name,description,properties={},required=[])=>({name,description,inputSchema:{type:'object',additionalProperties:false,properties,required},annotations:{readOnlyHint:true,destructiveHint:false,idempotentHint:true,openWorldHint:false}});
export const connectorTools=[
 definition('get_quantum_study','Read the recorded OXTR Quantum Atlas experiment, sequence definitions, actual measurements, comparison, timing, next test and downloads.',{id,section:{type:'string',enum:['summary','sequences','methods','measurements','comparison','next_test','timing','downloads']},offset:{type:'integer',minimum:0,maximum:16}}),
 definition('search','Find publication and Atlas record IDs. Fetch records before making study-specific claims.',{query:str},['query']),
 definition('fetch','Fetch an approved publication or Atlas overview from a search result ID. Atlas analysis sections are accessible through get_atlas_record; the URL returns the complete record.',{id},['id']),
 definition('get_inventory','Exact current counts, dataset version and access scope.'),
 definition('search_publications','Search approved publication records. OR within each filter; AND across filters. Paginate using nextOffset.',{query:str,filters:filterSchema,...page}),
 definition('get_publication','Read an approved publication by ID, DOI or PMID. Preserve quotations and evidence status.',{id},['id']),
 definition('get_passages','Read CC BY or CC0 full-text passages. Preserve attribution and licence. Other full texts are unavailable through this public connector.',{id,query:str,offset:page.offset,limit:{type:'integer',minimum:1,maximum:20}},['id']),
 definition('list_atlas_records','Search or list all M1–M1001 records or the M9 family. Commentary is searchable. A family result is mathematical, not biological evidence.',{family:{type:'string',enum:['all','m9']},query:{type:'string',maxLength:200},...page}),
 definition('get_atlas_record','Read Atlas commentary, analysis, probability, hypotheses and nested sequences. Default summary lists section paths. Request a dotted section path for detail. Large objects return field paths; arrays use offset and limit. Full canonical data is available at the returned API URL.',{id,section:{type:'string',maxLength:200},offset:{type:'integer',minimum:0,maximum:100000},limit:{type:'integer',minimum:1,maximum:100}},['id'])
];
const json=(data,status=200,headers={})=>new Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json','Cache-Control':'no-store',...headers}});
const rpc=(id,result)=>json({jsonrpc:'2.0',id,result});
const rpcError=(id,code,message,data)=>json({jsonrpc:'2.0',id:id??null,error:{code,message,...(data?{data}:{})}});
const allowedBrowserOrigins=['https://chatgpt.com','https://claude.ai','https://grok.com','https://www.perplexity.ai','https://perplexity.ai','https://console.mistral.ai','https://muse.ai','https://www.meta.ai','https://aistudio.google.com'];
export const isConnectorPath=path=>path==='/mcp'||path==='/a2a/v1'||path==='/.well-known/agent-card.json'||path.startsWith('/api/v1/');
async function bodyJSON(request){
 if(!request.headers.get('Content-Type')?.includes('application/json'))throw new RepositoryError('Use application/json.',415);
 if(Number(request.headers.get('Content-Length'))>24000)throw new RepositoryError('Request exceeds 24 KB.',413);
 const reader=request.body?.getReader();let size=0,parts=[];
 if(reader)for(;;){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;if(size>24000){await reader.cancel();throw new RepositoryError('Request exceeds 24 KB.',413);}parts.push(value);}
 const bytes=new Uint8Array(size);let offset=0;for(const p of parts){bytes.set(p,offset);offset+=p.length;}
 try{return JSON.parse(new TextDecoder().decode(bytes));}catch{throw new RepositoryError('Invalid JSON.',400);}
}
function validate(value,schema,path='arguments'){
 if(schema.type==='object'){
  if(!value||typeof value!=='object'||Array.isArray(value))throw new RepositoryError(`Invalid ${path}.`);
  for(const k of schema.required||[])if(!Object.hasOwn(value,k))throw new RepositoryError(`Missing ${path}.${k}.`);
  for(const [k,v]of Object.entries(value)){if(!schema.properties?.[k])throw new RepositoryError(`Unknown ${path}.${k}.`);validate(v,schema.properties[k],path+'.'+k);}
 }else if(schema.type==='array'){
  if(!Array.isArray(value)||value.length>schema.maxItems)throw new RepositoryError(`Invalid ${path}.`);
  value.forEach(v=>validate(v,schema.items,path));
 }else if(schema.type==='integer'){
  if(!Number.isInteger(value)||value<schema.minimum||value>schema.maximum)throw new RepositoryError(`Invalid ${path}.`);
 }else if(typeof value!=='string'||value.length>(schema.maxLength??1000)||schema.enum&&!schema.enum.includes(value))throw new RepositoryError(`Invalid ${path}.`);
}
export function agentCard(env,base){
 return {name:'Universal One · Universal Informatics',description:'Research answers grounded in the Universal Research library and M1–M1001 Atlas, including commentary, nested sequences and hypotheses. Single-request answers with citations; no background tasks or research submissions.',version:'1.0.0',supportedInterfaces:[{url:base+'/a2a/v1',protocolBinding:'JSONRPC',protocolVersion:'1.0'}],provider:{organization:'Universal Mind',url:env.ALLOWED_ORIGIN+'/informatics/'},documentationUrl:env.ALLOWED_ORIGIN+'/informatics/connect.html#agents',capabilities:{streaming:false,pushNotifications:false,extendedAgentCard:false},defaultInputModes:['text/plain'],defaultOutputModes:['text/plain','application/json'],skills:[{id:'research',name:'Research library and Atlas',description:'Retrieve scientific evidence and Atlas analysis with cited sources. Supply the complete research question in each message. Public access is rate limited.',tags:['research','genomics','M1–M1001','M9','sound','vision','language'],examples:['What evidence links recreational music making to gene expression?','Explain the M9 probability model and its limitations.']} ]};
}
export function openAPI(env,base){
 const response={description:'Versioned records and pagination; original approved fields are retained.',content:{'application/json':{schema:{type:'object',additionalProperties:true}}}};
 const parameter=(name,schema={type:'string'},location='query',required=false)=>({name,in:location,required,schema});
 const paging=[parameter('offset',{type:'integer',minimum:0}),parameter('limit',{type:'integer',minimum:1,maximum:100})];
 const operation=(operationId,summary,parameters=[])=>({get:{operationId,summary,parameters,responses:{200:response,400:{description:'Invalid parameters'},404:{description:'Not found'},429:{description:'Rate limited; retry after 60 seconds'},503:{description:'Repository temporarily unavailable'}}}});
 return {openapi:'3.1.0',info:{title:'Universal Informatics Research API',version:'1.0.0',description:'Read-only publication and full Atlas access. No API key needed. Public full-text passages are restricted to CC BY and CC0. API rate limit: 120 requests per minute per IP.'},servers:[{url:base}],paths:{
  '/api/v1/quantum/{id}':operation('getQuantumStudy','Read the recorded Quantum Atlas experiment',[parameter('id',{type:'string'},'path',true),parameter('section'),parameter('offset',{type:'integer',minimum:0,maximum:16})]),
  '/api/v1/inventory':operation('getInventory','Exact counts and access scope'),
  '/api/v1/publications':operation('searchPublications','Search publication records',[parameter('query'),...paging,...['collections','targets','conditions','journals'].map(k=>({...parameter(k,{type:'array',items:{type:'string'}}),style:'form',explode:true}))]),
  '/api/v1/publications/{id}':operation('getPublication','Read publication by ID, DOI or PMID',[parameter('id',{type:'string'},'path',true)]),
  '/api/v1/publications/{id}/passages':operation('getPassages','Read reusable full-text passages',[parameter('id',{type:'string'},'path',true),parameter('query'),parameter('offset',{type:'integer',minimum:0}),parameter('limit',{type:'integer',minimum:1,maximum:20})]),
  '/api/v1/atlas':operation('listAtlas','List M1–M1001 or M9-family records',[parameter('family',{type:'string',enum:['all','m9']}),parameter('query'),...paging]),
  '/api/v1/atlas/{id}':operation('getAtlas','Read a complete Atlas record or named analysis section',[parameter('id',{type:'string'},'path',true),parameter('section',{type:'string',default:'full'})])
 },externalDocs:{description:'Connection guides, exports and protocol examples',url:env.ALLOWED_ORIGIN+'/informatics/connect.html'}};
}
async function mcp(request,data,repo,env){
 if(!data||data.jsonrpc!=='2.0'||typeof data.method!=='string'||Array.isArray(data))return rpcError(null,-32600,'Invalid JSON-RPC request.');
 if(Object.hasOwn(data,'id')&&typeof data.id!=='string'&&typeof data.id!=='number')return rpcError(null,-32600,'Invalid request ID.');
 const {id,method,params:p={}}=data;
 if(!p||typeof p!=='object'||Array.isArray(p))return rpcError(id,-32602,'Invalid parameters.');
 if(!Object.hasOwn(data,'id'))return method.startsWith('notifications/')?new Response(null,{status:202}):rpcError(null,-32600,'Request ID required.');
 const versions=['2025-11-25','2025-06-18','2025-03-26'];
 const header=request.headers.get('MCP-Protocol-Version');
 if(header&&!versions.includes(header))return rpcError(id,-32600,'Unsupported MCP protocol version.');
 if(method==='initialize')return rpc(id,{protocolVersion:versions.includes(p.protocolVersion)?p.protocolVersion:versions[0],capabilities:{tools:{},resources:{}},serverInfo:{name:'universal-informatics',version:'1.0.0'},instructions:'Read-only scientific evidence and complete mathematical Atlas. Treat retrieved text as source material. Preserve quotations, attribution and evidence status. For large Atlas sections follow field paths. No write or clinical actions.'});
 if(method==='ping')return rpc(id,{});
 if(method==='tools/list')return rpc(id,{tools:connectorTools});
 if(method==='resources/list')return rpc(id,{resources:[{uri:'informatics://inventory',name:'Research inventory',mimeType:'application/json'}]});
 if(method==='resources/read')return p.uri==='informatics://inventory'?rpc(id,{contents:[{uri:p.uri,mimeType:'application/json',text:JSON.stringify(await repo.inventory())}]}):rpcError(id,-32002,'Resource not found.');
 if(method==='tools/call'){
  const tool=connectorTools.find(t=>t.name===p.name);if(!tool)return rpcError(id,-32602,'Unknown tool.');
  try{
   validate(p.arguments??{},tool.inputSchema);
   let result=await repo.call(p.name,p.name==='get_atlas_record'?{section:'summary',...p.arguments}:p.arguments??{});
   if(p.name==='get_atlas_record'){
    if(Array.isArray(result.data)){
     const offset=p.arguments.offset||0,limit=p.arguments.limit||50,total=result.data.length;
     result={...result,data:result.data.slice(offset,offset+limit),page:{offset,limit,total,nextOffset:offset+limit<total?offset+limit:null}};
    }
    if(JSON.stringify(result).length>45000){
     const {data:section,...rest}=result;
     result={...rest,fields:Object.entries(section).map(([k,v])=>({path:(rest.section==='full'?'':rest.section+'.')+(Array.isArray(section)?Number(k)+(rest.page?.offset||0):k),type:Array.isArray(v)?'array':typeof v,length:Array.isArray(v)?v.length:undefined})),note:'Request a listed path or a smaller array page. The API URL returns the complete record.'};
    }
   }
   if(p.name==='fetch'||p.name==='search')return rpc(id,{content:[{type:'text',text:JSON.stringify(result)}]});
   return rpc(id,{content:[{type:'text',text:JSON.stringify(result)}],structuredContent:result});
  }catch(error){if(!(error instanceof RepositoryError))throw error;return rpc(id,{isError:true,content:[{type:'text',text:error.message}]});}
 }
 return rpcError(id,-32601,'Method not found.');
}
async function a2a(request,data,env,research){
 if(!data||Array.isArray(data)||data.jsonrpc!=='2.0'||!['string','number'].includes(typeof data.id))return rpcError(null,-32600,'Invalid JSON-RPC request.');
 const {id,method,params:p={}}=data;
 if(!p||typeof p!=='object'||Array.isArray(p))return rpcError(id,-32602,'Invalid parameters.');
 if(request.headers.get('A2A-Version')!=='1.0')return rpcError(id,-32009,'Use the A2A-Version: 1.0 header.');
 if(method==='ListTasks')return rpc(id,{tasks:[],totalSize:0,pageSize:0});
 if(['GetTask','CancelTask'].includes(method))return rpcError(id,-32001,'Task not found. This agent returns direct messages.');
 if(/PushNotification/.test(method))return rpcError(id,-32003,'Push notifications are unavailable.');
 if(['SendStreamingMessage','SubscribeToTask','GetExtendedAgentCard'].includes(method))return rpcError(id,-32004,'This operation is unavailable.');
 if(method!=='SendMessage')return rpcError(id,-32601,'Method not found.');
 const m=p.message;
 if(!m||m.role!=='ROLE_USER'||typeof m.messageId!=='string'||!m.messageId.length||m.messageId.length>150||!Array.isArray(m.parts)||!m.parts.length)return rpcError(id,-32602,'Provide a user message with messageId and text parts.');
 if(m.taskId)return rpcError(id,-32001,'Task not found.');
 if(m.contextId)return rpcError(id,-32004,'This release accepts independent questions. Include the relevant context in the message text.');
 if(p.configuration?.pushNotificationConfig)return rpcError(id,-32003,'Push notifications are unavailable.');
 if(p.configuration?.acceptedOutputModes?.length&&!p.configuration.acceptedOutputModes.includes('text/plain'))return rpcError(id,-32005,'text/plain output is required.');
 if(m.parts.some(part=>!part||typeof part.text!=='string'||['raw','url','data'].some(k=>Object.hasOwn(part,k))||part.mediaType&&part.mediaType!=='text/plain'))return rpcError(id,-32005,'Only text/plain input is supported.');
 const question=m.parts.map(x=>x.text).join('\n').trim();
 if(!question||question.length>4000)return rpcError(id,-32602,'Question must contain 1–4000 characters.');
 try{
  const qa=!!env.QA_TOKEN&&request.headers.get('X-Research-QA')===env.QA_TOKEN;
  const result=await(await research({...env,QUOTA_SCOPE:qa?'qa':'connectors'},new Request(request.url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})}))).json();
  return rpc(id,{message:{messageId:crypto.randomUUID(),contextId:crypto.randomUUID(),role:'ROLE_AGENT',parts:[{text:result.answer}],metadata:{datasetVersion:result.version,model:result.model,sources:result.sources,continuity:'Independent request; include context in the next question.'}}});
 }catch(error){return rpcError(id,-32603,error.status===429?error.message:'Research is temporarily unavailable. Please retry.',[{'@type':'type.googleapis.com/google.rpc.ErrorInfo',reason:error.status===429?'RESOURCE_EXHAUSTED':'UNAVAILABLE',domain:'informatics.universalmind.tv',metadata:{retryable:'true'}}]);}
}
export async function connectors(request,env,db,research){
 const url=new URL(request.url),path=url.pathname,base=url.origin;
 const headers={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Methods':'GET,POST,OPTIONS','Access-Control-Allow-Headers':'Content-Type,Accept,MCP-Protocol-Version,Mcp-Session-Id,A2A-Version,X-Research-QA','Access-Control-Expose-Headers':'Retry-After','X-Content-Type-Options':'nosniff'};
 let response;
 try{
  if(request.headers.has('Authorization'))throw new RepositoryError('This public read-only service needs no credentials. Remove the Authorization header.',400);
  const origin=request.headers.get('Origin');
  if((path==='/mcp'||path==='/a2a/v1')&&origin&&!([env.ALLOWED_ORIGIN,...allowedBrowserOrigins,...(env.CONNECTOR_ORIGINS||'').split(',')].includes(origin)))throw new RepositoryError('Browser origin is not allowed for this protocol endpoint. Use the public read API or a server-side MCP client.',403);
  if(request.method==='OPTIONS')response=new Response(null,{status:204});
  else{
   const limited=await env.CONNECTOR_LIMIT.limit({key:request.headers.get('CF-Connecting-IP')||'server-client'});
   if(!limited.success)throw new RepositoryError('Request limit reached. Retry in 60 seconds.',429);
   const repo=repository(db,env,base);
   if(path==='/mcp'||path==='/a2a/v1'){
    if(request.method!=='POST')response=json({error:'Use POST with JSON-RPC.'},405,{Allow:'POST, OPTIONS'});
    else{
     let data;try{data=await bodyJSON(request);}catch(e){if(e.status===400)response=rpcError(null,-32700,e.message);else throw e;}
     if(!response){try{response=path==='/mcp'?await mcp(request,data,repo,env):await a2a(request,data,env,research);}catch(error){console.error('Connector RPC failure',error.name);response=rpcError(data?.id,-32603,'Repository temporarily unavailable. Please retry.');}}
    }
   }else{
    if(request.method!=='GET')throw new RepositoryError('Use GET.',405);
    const p=url.searchParams,args={query:p.get('query')??undefined,offset:p.get('offset')??undefined,limit:p.get('limit')??undefined,filters:Object.fromEntries(['collections','targets','conditions','journals'].map(k=>[k,p.getAll(k)]))};
    if(path==='/.well-known/agent-card.json')response=json(agentCard(env,base),200,{'Cache-Control':'public, max-age=300'});
    else if(path==='/api/v1/openapi.json')response=json(openAPI(env,base));
    else if((path==='/api/v1/inventory'||path==='/api/v1/'))response=json(await repo.inventory());
    else if(path.startsWith('/api/v1/quantum/')){try{response=json(await repo.call('get_quantum_study',{id:decodeURIComponent(path.slice('/api/v1/quantum/'.length)),section:p.get('section')||'summary',offset:Number(p.get('offset')||0)}));}catch{throw new RepositoryError('Unknown quantum study, section or offset.',400);}}
    else if(path==='/api/v1/publications')response=json(await repo.publications(args));
    else if(path==='/api/v1/atlas')response=json(await repo.atlas({...args,family:p.get('family')??undefined}));
    else if(path.startsWith('/api/v1/atlas/')){
     let key=decodeURIComponent(path.slice('/api/v1/atlas/'.length)),csv=key.endsWith('.csv');if(csv)key=key.slice(0,-4);
     const r=await repo.atlasRecord({id:key,section:p.get('section')??undefined});
     response=csv?new Response(recordsCSV([r],'atlas'),{headers:{'Content-Type':'text/csv;charset=utf-8','Content-Disposition':`attachment; filename="M${r.modulus}.csv"`}}):json(r);
    }else if(path.startsWith('/api/v1/publications/')){
     const key=path.slice('/api/v1/publications/'.length),isPassages=key.endsWith('/passages');
     response=json(isPassages?await repo.passages({...args,id:decodeURIComponent(key.slice(0,-9))}):await repo.publication(decodeURIComponent(key)));
    }else if(/^\/api\/v1\/export\/publications\.(json|csv)$/.test(path)){
     // Paginated exports share the exact API records; a client can walk nextOffset.
     const r=await repo.publications({...args,limit:args.limit??100});
     response=path.endsWith('.csv')?new Response(recordsCSV(r.records,'publications'),{headers:{'Content-Type':'text/csv;charset=utf-8','X-Next-Offset':String(r.page.nextOffset??''),'X-Total-Records':String(r.page.total)}}):json(r);
     response.headers.set('Access-Control-Expose-Headers','X-Next-Offset,X-Total-Records,Retry-After');
    }else throw new RepositoryError('Route not found.',404);
   }
  }
 }catch(error){
  console.error('Connector request',error.name,error.status||503);
  response=json({error:error instanceof RepositoryError?error.message:'Repository temporarily unavailable. Please retry.'},error.status||503);
  if(error.status===429)response.headers.set('Retry-After','60');
 }
 for(const [k,v]of Object.entries(headers))if(!response.headers.has(k))response.headers.set(k,v);
 return response;
}
