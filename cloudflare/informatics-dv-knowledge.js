// Versioned approved summaries travel with the worker release. No shared database
// mutation, fine-tuning, new embeddings, or third-party full-text upload occurs.
import bundle from '../Informatics/library-data/domestic-violence-expansion.json' with {type:'json'};
export const dvRelevant=q=>/domestic|family violence|intimate.partner|\bipv\b|battered|AI.guided recommendations/i.test(q);
const fields=['id','aliases','title','authors','journal','year','doi','pmid','urls','collections','categories','targets','conditions','ageGroups','design','summary','limitation','evidence','studyDesignAssessment','paperTrustAssessment','jadad','sourceAccess','publicationApproval','studyFamilyId','familyLinkageUncertain','approaches'];
export const dvRecords=bundle.records.map(r=>Object.fromEntries(fields.map(k=>[k,r[k]])));
const normal=value=>String(value||'').toLowerCase().replace(/https:\/\/doi.org\//,'');
export const dvRecord=id=>dvRecords.find(r=>[r.id,r.doi,r.pmid,...r.aliases].some(v=>v&&normal(v)===normal(id)));
export const dvReplaces=row=>!!dvRecord(row.id)||!!dvRecord(row.data?.doi)||!!dvRecord(row.data?.pmid);
export const dvMatches=(r,f={})=>[['collections','collections'],['targets','targets'],['conditions','conditions'],['journals','journal']].every(([facet,key])=>!f[facet]?.length||f[facet].some(v=>Array.isArray(r[key])?r[key].includes(v):r[key]===v));
export function dvSearch(query,filters={},limit=5){
 const stop=new Set('the a an and or of for in to on with from study studies research evidence what is are does do this that tell me about domestic violence intimate partner'.split(' '));
 const terms=[...new Set(normal(query).match(/[\p{L}\p{N}-]+/gu)||[])].filter(t=>!stop.has(t)&&t.length>2);
 const rows=dvRecords.filter(r=>dvMatches(r,filters)).map(r=>{
   const text=normal([r.title,r.summary,r.limitation,r.design,r.evidence.intervention,r.evidence.measurement,...r.approaches,r.id,r.pmid].join(' '));
   return {r,score:terms.reduce((n,t)=>n+(text.includes(t)?1:0),0)};
 }).filter(x=>x.score>0||dvRelevant(query)).sort((a,b)=>b.score-a.score||b.r.year-a.r.year);
 return rows.slice(0,limit).map(x=>x.r);
}
export function dvKnowledge(args={},origin='https://staging.universalmind.tv'){
 let rows=args.id?[dvRecord(args.id)].filter(Boolean):dvRecords.filter(r=>(!args.collection||args.collection==='all'||r.collections.includes(args.collection))&&(!args.approach||r.approaches.includes(args.approach)));
 if(args.query&&!args.id){const matches=new Set(dvSearch(args.query,{},117).map(r=>r.id));rows=rows.filter(r=>matches.has(r.id));}
 const offset=Math.max(0,Number(args.offset)||0),limit=args.id?1:4;
 return {id:'approved-domestic-violence-20261101',title:'Domestic & Family Violence · approved research summaries',version:bundle.sourceBatch,approvedAt:bundle.approvedAt,overview:bundle.overview,limitations:bundle.limitations,counts:bundle.counts,paperTrustFactorPolicy:bundle.paperTrustFactorPolicy,chapterTrustFactorPolicy:{1:'At least three distinct peer-reviewed studies',2:'At least three distinct randomized controlled trials',3:'At least three distinct verified double-blind randomized controlled trials'},chapterTrustFactors:Object.fromEntries(Object.entries(bundle.chapters).map(([c,v])=>[c,v.topics[0].trustScore])),approaches:bundle.approaches.map(({id,label,count})=>({id,label,count})),chapters:Object.fromEntries(Object.entries(bundle.chapters).map(([c,v])=>[c,`${origin}/informatics/library.html?collection=${c}&chapter=${v.number}#topic-domestic-family-violence`])),matchedRecords:rows.length,offset,nextOffset:offset+limit<rows.length?offset+limit:null,records:rows.slice(offset,offset+limit),knowledgeMethod:'Versioned retrieval of human-approved AI-assessed summaries; no model fine-tuning. Source access is appraisal access, not a promise of indexed article full text.'};
}
