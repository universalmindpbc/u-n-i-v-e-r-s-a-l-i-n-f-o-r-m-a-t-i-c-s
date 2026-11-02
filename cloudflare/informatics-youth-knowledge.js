// Approved summaries are bundled locally; no embeddings, full-text upload or DB writes.
import bundle from '../Informatics/library-data/youth-expansion.json' with {type:'json'};
export const youthVersion={sourceBatch:bundle.sourceBatch,sourceSHA256:bundle.sourceSHA256,approvedAt:bundle.approvedAt,publications:bundle.records.length};
export const youthRelevant=q=>/\byouth\b|young (?:people|adults)|black dog|orygen|universal mind (?:mobile )?lab|\b16\s*[–-]\s*24\b/i.test(q);
const fields=['id','aliases','title','authors','journal','year','doi','pmid','urls','collections','categories','targets','conditions','ageGroups','agePopulation','design','summary','limitation','evidence','studyDesignAssessment','paperTrustAssessment','jadad','sourceAccess','publicationApproval','studyFamilyId','randomizedStudyUnitId','institutions','institutionEvidence','researchTopics','labRelevance','publicationStatus','noticeSummary','citationCount','citationSnapshot'];
export const youthRecords=bundle.records.map(r=>Object.fromEntries(fields.map(k=>[k,r[k]])));
const normal=v=>String(v||'').toLowerCase().replace(/^https:\/\/(?:dx\.)?doi.org\//,'').replace(/[‐‑–—]/g,'-');
export const youthRecord=id=>youthRecords.find(r=>[r.id,r.doi,r.pmid,...r.aliases].some(v=>v&&normal(v)===normal(id)));
export const youthReplaces=row=>!!youthRecord(row.id)||!!youthRecord(row.data?.doi)||!!youthRecord(row.data?.pmid);
export const youthMatches=(r,f={})=>[['collections','collections'],['targets','targets'],['conditions','conditions'],['journals','journal']].every(([facet,key])=>!f[facet]?.length||f[facet].some(v=>Array.isArray(r[key])?r[key].includes(v):r[key]===v));
export function youthSearch(query,filters={},limit=5){
 const stop=new Set('the a an and or of for in to on with from study studies research evidence what is are does do this that tell me about'.split(' '));
 const terms=[...new Set(normal(query).match(/[\p{L}\p{N}-]+/gu)||[])].filter(t=>!stop.has(t)&&t.length>2);
 return youthRecords.filter(r=>youthMatches(r,filters)).map(r=>({r,score:terms.reduce((n,t)=>n+(normal([r.title,r.summary,r.limitation,r.design,r.evidence.intervention,...r.researchTopics,...r.institutions,r.labRelevance.implication,r.id,r.pmid].join(' ')).includes(t)?1:0),0)}))
  .filter(x=>x.score>0).sort((a,b)=>b.score-a.score||b.r.year-a.r.year).slice(0,limit).map(x=>x.r);
}
export function youthKnowledge(args={},origin='https://staging.universalmind.tv'){
 let rows=args.id?[youthRecord(args.id)].filter(Boolean):youthRecords.filter(r=>(!args.collection||args.collection==='all'||r.collections.includes(args.collection))&&(!args.institution||r.institutions.includes(args.institution))&&(!args.relationship||r.labRelevance.relationship===args.relationship));
 if(args.query&&!args.id){const matches=new Set(youthSearch(args.query,{},105).map(r=>r.id));rows=rows.filter(r=>matches.has(r.id));}
 const offset=Math.max(0,Number(args.offset)||0),limit=args.id?1:4;
 return {id:'approved-youth-20261102',title:'Youth research and Universal Mind Mobile Lab · Approved summaries',version:bundle.sourceBatch,sourceSHA256:bundle.sourceSHA256,approvedAt:bundle.approvedAt,
  overview:bundle.overview,limitations:bundle.limitations,labFraming:bundle.labFraming,counts:bundle.counts,paperTrustFactorPolicy:bundle.paperTrustFactorPolicy,
  chapterTrustFactorPolicy:{1:'At least three distinct peer-reviewed study families',2:'At least three distinct randomized controlled trial units',3:'At least three distinct verified double-blind randomized controlled trial units'},
  chapters:Object.fromEntries(Object.entries(bundle.chapters).map(([c,v])=>[c,{url:`${origin}/informatics/library.html?collection=${c}&chapter=${v.number}#topic-universal-mind-mobile-lab`,topics:v.topics.map(t=>({id:t.id,title:t.title,publications:t.papers.length,trustScore:t.trustScore,trustAssessment:t.trustAssessment}))}])),
  matchedRecords:rows.length,offset,nextOffset:offset+limit<rows.length?offset+limit:null,records:rows.slice(offset,offset+limit),
  knowledgeMethod:'Versioned retrieval of human-approved AI-assessed summaries. Source access describes selected appraisal evidence, not indexed full text or independent clinical peer review. No model fine-tuning.'};
}
