import {youthKnowledge} from './informatics-youth-knowledge.js';
import {dvKnowledge} from './informatics-dv-knowledge.js';
import communities from '../Informatics/library-data/community-index.json' with {type:'json'};
import ages from '../Informatics/library-data/age-index.json' with {type:'json'};

const communityById=new Map(communities.records.map(r=>[r.recordId,r.matches]));
const allowedCollections=['all','sound','vision','language','general'];
export function researchNavigation(args={},origin='https://staging.universalmind.tv'){
 const collection=allowedCollections.includes(args.collection)?args.collection:'all';
 const community=communities.taxonomy.some(t=>t.id===args.community)?args.community:'';
 const age=ages.taxonomy.some(t=>t.id===args.age)?args.age:'';
 const includeRelated=args.includeRelated===true;
 const hasCommunity=(r,id,related)=>communityById.get(r.recordId).some(m=>m.categoryId===id&&(related||m.relation!=='related'));
 const collectionRows=ages.records.filter(r=>collection==='all'||r.collections.includes(collection));
 const selected=collectionRows.filter(r=>(!community||hasCommunity(r,community,includeRelated))&&(!age||r.groups.includes(age)));
 const url=new URL('/informatics/library.html',origin);url.searchParams.set('collection',collection);url.searchParams.set('view','search');
 if(community)url.searchParams.set('community',community);if(age)url.searchParams.set('age',age);if(includeRelated)url.searchParams.set('related','1');
 return {
  id:'universal-research-search-guide',title:'Universal Research search and population filters',updatedAt:ages.updatedAt,
  pageURL:url.href,collection,filters:{community,age,includeRelated},matchedPapers:selected.length,
  totalVisiblePapers:ages.recordCount,collectionPapers:collectionRows.length,
  communities:communities.taxonomy.map(t=>({id:t.id,label:t.label,directAndContext:collectionRows.filter(r=>hasCommunity(r,t.id,false)).length,includingRelated:collectionRows.filter(r=>hasCommunity(r,t.id,true)).length})),
  ageGroups:ages.taxonomy.map(t=>({id:t.id,label:t.label,count:collectionRows.filter(r=>r.groups.includes(t.id)).length})),
  youth:{covered:collectionRows.filter(r=>r.groups.includes('youth')).length,focus:collectionRows.filter(r=>r.youthScope==='focus').length,includedWithoutEstablishedFocus:collectionRows.filter(r=>r.youthScope==='included').length},
  approvedYouth:youthKnowledge({collection:'all'},origin).counts,
  approvedDomesticViolence:dvKnowledge({collection:'all'},origin).counts,
  guidance:[
   'The approved youth addition contains 105 publications, including 100 additions and five existing-library matches. Black Dog Institute, Orygen, Universal Mind Mobile Lab and youth-intervention topics are appended to the relevant Sound, Vision and Language contents. Cross-listing does not increase unique-paper counts. Read reported ages: the Youth filter also includes adolescents and mixed-age evidence.',
   'Open Universal Research, choose Search papers, then Communities and Contexts and Age group. Sound, Vision and Language retain their existing human-curated contents.',
   'Selections within one filter use OR. Different filters combine with AND. Multiple categories can cover the same paper, so category totals are not additive.',
   'Include related research adds links labelled Related · inferred. A shared intervention or outcome is not proof that the selected community was studied or that treatment effects transfer.',
   'Age groups describe populations covered and can overlap. Youth & young adults includes adolescents and source-defined young people. Distinguish a youth focus from youth inclusion in a broader or incompletely described sample. Reported age ranges remain source-specific.',
   'Age not specified means no supported age classification from the checked sources. It does not establish that a publication contains no age information.',
   'JSON and CSV buttons download the current filtered results. Counts here come from the deployed visible library classifications; publication/full-text retrieval coverage is a separate inventory.',
   'These tags aid discovery. They do not establish clinical efficacy, safety, individual suitability or a higher evidence grade.',
   'AI Guided Recommendations is now appended to Sound, Vision and Language, with Domestic & Family Violence summaries: 8 Sound, 10 Vision, 99 Language. All 117 were AI assessed and human approved for publication; this does not claim clinical peer review. Universal One retrieves these approved summaries. A continuous researcher is not active.'
  ],
  matchedRecordIds:(community||age)?selected.slice(0,20).map(r=>r.recordId):[],
  recordLimit:20,
  evidenceLimit:'This guide describes site navigation and classification. Retrieve each publication before making clinical or study-specific claims. Do not infer participant membership from related tags.'
 };
}
export const navigationQuestion=q=>/communities\s+(?:and|&)\s+contexts|include related|related research|age[ -]groups?|(?:how|where).{0,60}(?:filter|search papers|find.{0,30}(?:youth|refugee|domestic violence))|(?:filter|search).{0,40}(?:communities|contexts)/i.test(q);
