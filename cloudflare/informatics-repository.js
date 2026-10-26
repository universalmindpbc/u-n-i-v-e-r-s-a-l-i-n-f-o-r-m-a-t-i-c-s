import {quantumKnowledge,quantumRelevant} from './informatics-quantum-knowledge.js';
// Shared, read-only data service for REST, MCP and export clients.
// Approved fields remain intact. No model generation or write credentials here.
export class RepositoryError extends Error {
 constructor(message,status=400){super(message);this.status=status;}
}
export function textArg(value,name,max=800,fallback=''){
 if(value===undefined)return fallback;
 if(typeof value!=='string'||value.length>max)throw new RepositoryError(`${name} must be text of at most ${max} characters.`);
 return value.trim();
}
export function integerArg(value,name,max=10000,fallback=0){
 if(value===undefined||value==='')return fallback;
 if(!/^\d+$/.test(String(value))||Number(value)>max)throw new RepositoryError(`Invalid ${name}.`);
 return Number(value);
}
function filters(value={}){
 if(!value||typeof value!=='object'||Array.isArray(value))throw new RepositoryError('Invalid filters.');
 const out={};
 for(const k of ['collections','targets','conditions','journals']){
  const a=value[k]??[];
  if(!Array.isArray(a)||a.length>12)throw new RepositoryError(`Invalid ${k} filter.`);
  out[k]=a.map(x=>textArg(x,k,200));
 }
 if(out.collections.some(x=>!['sound','vision','language','general'].includes(x)))throw new RepositoryError('Unknown collection.');
 return out;
}
const where=`(cardinality($1::text[])=0 OR p.data->'collections' ?| $1::text[])
 AND (cardinality($2::text[])=0 OR p.data->'targets' ?| $2::text[])
 AND (cardinality($3::text[])=0 OR p.data->'conditions' ?| $3::text[])
 AND (cardinality($4::text[])=0 OR p.data->>'journal'=ANY($4::text[]))`;
// Public passage access is narrower than AI-use permission. Keep licence and attribution.
const reusable=`(s.rights->>'license' ~* '^https?://creativecommons.org/(licenses/by/[0-9.]+/?|publicdomain/zero/[0-9.]+/?)$'
 OR s.rights->>'license' ~* '^CC BY( [0-9.]+| \\(version not specified in article\\))?$'
 OR s.rights->>'license' ~* '^CC0( [0-9.]+)?$')`;
export const csvCell=value=>{
 let s=typeof value==='object'&&value!==null?JSON.stringify(value):String(value??'');
 if(/^[\s]*[=+@-]/.test(s))s="'"+s;
 return '"'+s.replaceAll('"','""')+'"';
};
export function recordsCSV(records,kind){
 const headers=kind==='atlas'?['id','modulus','sourceSHA256','record_json']:['id','title','journal','year','doi','record_json'];
 return headers.join(',')+'\r\n'+records.map(r=>headers.map(k=>csvCell(k==='record_json'?r.data??r:r[k]??r.data?.[k])).join(',')).join('\r\n')+'\r\n';
}
export function repository(db,env,base){
 const version=env.DATASET_VERSION,site=env.ALLOWED_ORIGIN+'/informatics/';
 const wrap=data=>({schemaVersion:'1.0',datasetVersion:version,...data});
 const page=(offset,limit,total)=>({offset,limit,total,nextOffset:offset+limit<total?offset+limit:null});
 const recordURL=(kind,id)=>`${base}/api/v1/${kind}/${encodeURIComponent(id)}`;
 async function inventory(){
  const [r]=await db.query(`SELECT
   (SELECT count(*)::int FROM research.publications) publications,
   (SELECT count(*)::int FROM research.atlas) atlas,
   (SELECT count(DISTINCT publication_id)::int FROM research.passages WHERE kind='full-text') full_texts,
   (SELECT count(DISTINCT s.publication_id)::int FROM research.passages s WHERE s.kind='full-text' AND ${reusable}) public_full_texts,
   (SELECT count(*)::int FROM research.atlas WHERE data->'family'->>'m9'='true') m9_family`,[]);
  return wrap({counts:r,access:{records:'Public publication metadata, approved summaries and quotations; complete Atlas records and commentary.',passages:'CC BY and CC0 passages only, with source, attribution and licence. Other full texts remain outside public exports.',writing:false},links:{publications:base+'/api/v1/publications',atlas:base+'/api/v1/atlas',openapi:base+'/api/v1/openapi.json',mcp:base+'/mcp',agentCard:base+'/.well-known/agent-card.json',guide:site+'connect.html',quantum:base+'/api/v1/quantum/oxtr-qrc-01',completeDataset:site+'exports/manifest.json'}});
 }
 async function publications(args={}){
  const f=filters(args.filters),q=textArg(args.query,'query'),offset=integerArg(args.offset,'offset'),limit=integerArg(args.limit,'limit',100,20);
  if(!limit)throw new RepositoryError('limit must be positive.');
  const values=[f.collections,f.targets,f.conditions,f.journals,q];
  const clause=where+` AND ($5='' OR p.search @@ websearch_to_tsquery('english',$5) OR p.data->>'title' ILIKE '%'||$5||'%')`;
  const [count]=await db.query(`SELECT count(*)::int total FROM research.publications p WHERE ${clause}`,values);
  const rows=await db.query(`SELECT p.id,p.data FROM research.publications p WHERE ${clause} ORDER BY ts_rank(p.search,websearch_to_tsquery('english',$5)) DESC,p.id LIMIT $6 OFFSET $7`,[...values,limit,offset]);
  return wrap({records:rows.map(r=>({...r.data,id:r.id})),page:page(offset,limit,count.total),filters:f,query:q});
 }
 async function publication(id){
  id=textArg(id,'id',250);
  const rows=await db.query(`SELECT id,data FROM research.publications WHERE id=$1 OR lower(data->>'doi')=lower($1) OR data->>'pmid'=$1 LIMIT 1`,[id]);
  if(!rows.length)throw new RepositoryError('Publication not found.',404);
  return wrap({record:{...rows[0].data,id:rows[0].id},url:recordURL('publications',rows[0].id)});
 }
 async function passages(args){
  const p=await publication(args.id),id=p.record.id,q=textArg(args.query,'query'),offset=integerArg(args.offset,'offset',100000),limit=integerArg(args.limit,'limit',20,5);
  if(!limit)throw new RepositoryError('limit must be positive.');
  const [count]=await db.query(`SELECT count(*)::int total FROM research.passages s WHERE publication_id=$1 AND kind='full-text' AND ${reusable}`,[id]);
  const rows=await db.query(`SELECT s.id,s.section,s.text,s.source_url,s.rights,s.sha256 FROM research.passages s WHERE publication_id=$1 AND kind='full-text' AND ${reusable} ORDER BY ts_rank(s.search,websearch_to_tsquery('english',$2)) DESC,s.id LIMIT $3 OFFSET $4`,[id,q,limit,offset]);
  return wrap({publication:{id,title:p.record.title,authors:p.record.authors,journal:p.record.journal,year:p.record.year,doi:p.record.doi},passages:rows,page:page(offset,limit,count.total),availability:count.total?'Reusable full-text passages':'No publicly reusable full text indexed. Consult the publication record and publisher.',reuse:'Retain attribution, source URLs, licence and copyright statements with every reused passage.'});
 }
 async function atlas(args={}){
  const family=textArg(args.family,'family',10,'all'),q=textArg(args.query,'query',200),offset=integerArg(args.offset,'offset',1001),limit=integerArg(args.limit,'limit',100,30);
  if(!['all','m9'].includes(family)||!limit)throw new RepositoryError('Invalid Atlas page.');
  const numeric=/^m?\d+$/i.test(q)?Number(q.replace(/^m/i,'')):0;
  const clause=`($1='all' OR data->'family'->>'m9'='true') AND ($2='' OR ($3>0 AND modulus=$3) OR ($3=0 AND data::text ILIKE '%'||$2||'%'))`;
  const vals=[family,q,numeric];
  const [count]=await db.query(`SELECT count(*)::int total FROM research.atlas WHERE ${clause}`,vals);
  const rows=await db.query(`SELECT id,modulus,data->>'name' name,data->'pisano_period' period,data->'family' family,sha256 FROM research.atlas WHERE ${clause} ORDER BY modulus LIMIT $4 OFFSET $5`,[...vals,limit,offset]);
  return wrap({records:rows.map(r=>({...r,url:recordURL('atlas',r.id)})),page:page(offset,limit,count.total),family,query:q});
 }
 async function atlasRecord(args){
  const id=textArg(args.id,'id',100),section=textArg(args.section,'section',200,'full');
  const modulus=Number(id.match(/^(?:m)?(\d+)$/i)?.[1])||0;
  const rows=await db.query('SELECT id,modulus,data,sha256 FROM research.atlas WHERE id=$1 OR modulus=$2 LIMIT 1',[id,modulus]);
  if(!rows.length)throw new RepositoryError('Atlas record not found.',404);
  const r=rows[0];let data=r.data;
  if(section==='summary')data={...Object.fromEntries(['id','name','modulus','pisano_period','family','conventions','geometry_summary','verification'].map(k=>[k,r.data[k]])),availableSections:Object.keys(r.data),analysisSections:['research','research_register','probability','m9_sequence_views','reduction_to_m9'].filter(k=>k in r.data)};
  else if(section!=='full')for(const key of section.split('.')){
   if(!data||typeof data!=='object'||['__proto__','constructor','prototype'].includes(key)||!Object.hasOwn(data,key))throw new RepositoryError('Atlas section not found.',404);
   data=data[key];
  }
  return wrap({id:r.id,modulus:r.modulus,section,data,sourceSHA256:r.sha256,url:recordURL('atlas',r.id),interpretation:'Preserve native and reduced-to-M9 representations. Computed geometry, conditional shuffle probabilities, commentary and biological hypotheses have distinct evidence status.'});
 }
 async function search(args){
  const query=textArg(args.query,'query');if(!query)throw new RepositoryError('Enter a search query.');
  const [p,a]=await Promise.all([publications({query,limit:8}),atlas({query,limit:5})]);
  return {results:[...(quantumRelevant(query)?[{id:'quantum:oxtr-qrc-01',title:'OXTR Fibonacci and M9 P24 quantum experiment',url:recordURL('quantum','oxtr-qrc-01')}]:[]),...p.records.map(r=>({id:'publication:'+r.id,title:r.title,url:recordURL('publications',r.id)})),...a.records.map(r=>({id:'atlas:'+r.id,title:r.name||'M'+r.modulus,url:r.url}))],datasetVersion:version};
 }
 async function fetchRecord(args){
  const id=textArg(args.id,'id',300);let result,title;
  if(id.startsWith('quantum:')){result=quantumKnowledge({id:id.slice(8)},env.ALLOWED_ORIGIN);title=result.data.question;}
  else if(id.startsWith('atlas:')){result=await atlasRecord({id:id.slice(6),section:'summary'});title=result.data.name;}
  else{result=await publication(id.replace(/^publication:/,''));title=result.record.title;}
  return {id,title,text:JSON.stringify(result),url:result.url,metadata:{datasetVersion:version}};
 }
 async function call(name,args={}){
  if(name==='get_quantum_study')return quantumKnowledge(args,env.ALLOWED_ORIGIN);
  if(name==='get_inventory')return inventory();
  if(name==='search_publications')return publications(args);
  if(name==='get_publication')return publication(args.id);
  if(name==='get_passages')return passages(args);
  if(name==='list_atlas_records')return atlas(args);
  if(name==='get_atlas_record')return atlasRecord(args);
  if(name==='search')return search(args);
  if(name==='fetch')return fetchRecord(args);
  throw new RepositoryError('Unknown operation.',404);
 }
 return {inventory,publications,publication,passages,atlas,atlasRecord,search,fetchRecord,call};
}
