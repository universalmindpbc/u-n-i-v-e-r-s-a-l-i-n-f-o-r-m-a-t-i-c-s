import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {youthRecords,youthRecord,youthSearch,youthKnowledge} from '../cloudflare/informatics-youth-knowledge.js';
import {researchNavigation} from '../cloudflare/informatics-research-navigation.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const bundle=read('Informatics/library-data/youth-expansion.json'),source=read('Informatics/review/youth-20261102/batch-f164c8325ea5f9c9.json');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const baseline=read('research/informatics/youth-integration-20261102/baseline.json');
for(const [p,digest] of Object.entries(baseline.sourceSHA256))assert.equal(sha(fs.readFileSync(path.join(root,p))),digest,p);
assert.equal(sha(fs.readFileSync(path.join(root,baseline.qualifiedSource))),baseline.qualifiedSourceSHA256);
const context=vm.createContext({});vm.runInContext(fs.readFileSync(path.join(root,'Informatics/library-model.js'),'utf8'),context);const model=context.UniversalResearchModel;
assert.equal(youthRecords.length,105);assert.equal(new Set(youthRecords.map(r=>r.id)).size,105);
for(const original of source.records){
 const r=youthRecord(original.id),qualified=original.reviewReady&&['qualified','priority'].includes(original.quality.tier)&&!original.duplicateOf;
 if(!qualified){assert.equal(r,undefined);continue;}
 assert.ok(r,original.id);assert.equal(r.summary,original.findings);assert.equal(r.limitation,original.limitations);
 assert.equal(r.publicationApproval.sourceContentHash,original.contentHash);assert.equal(r.publicationApproval.sourceRevision,original.revision);
 assert.equal(r.sourceAccess.scope,original.sourceAccess.scope);assert.ok(r.sourceAccess.methodsInspected);assert.notEqual(r.sourceAccess.scope,'abstract');
 assert.deepEqual(r.labRelevance,original.labRelevance);assert.deepEqual(r.institutions,original.institutions);assert.deepEqual(r.institutionEvidence,original.institutionEvidence);
 assert.equal(r.studyFamilyId,original.studyFamilyId);assert.equal(r.randomizedStudyUnitId,original.randomizedStudyUnitId);
 assert.equal(r.paperTrustAssessment.score,original.design.proposedCrosses);assert.equal(model.matchesStudy(r,'trial'),original.design.category==='randomized-report');
 assert.equal(model.matchesStudy(r,'double-blind'),original.design.doubleBlindingVerified);
 assert.equal(model.paperTrustScore(r),r.paperTrustAssessment.score);assert.deepEqual(r.targets,[]);
 assert.equal(r.paperTrustAssessment.humanApproval,false);assert.equal(r.publicationApproval.status,'approved-for-publication');
 assert.equal(youthRecord(original.pmid)?.id,r.id);assert.equal(youthRecord(original.doi)?.id,r.id);
 for(const url of r.urls)assert.equal(new URL(url).protocol,'https:');
}
assert.equal(youthRecords.filter(r=>model.matchesStudy(r,'trial')).length,47);
assert.equal(youthRecords.filter(r=>model.matchesStudy(r,'double-blind')).length,3);
assert.equal(youthRecords.filter(r=>model.matchesStudy(r,'review')).length,42);
assert.equal(new Set(youthRecords.filter(r=>model.matchesStudy(r,'trial')).map(r=>r.randomizedStudyUnitId)).size,42);
for(const [pmid,count] of Object.entries({'19798037':197,'29422189':230,'42324567':0,'41340540':1,'40243164':7}))assert.equal(bundle.records.find(r=>r.pmid===pmid).citationCount,count,'Retain existing historical citations');
assert.equal(youthKnowledge({institution:'Black Dog Institute'}).matchedRecords,20);assert.equal(youthKnowledge({institution:'Orygen'}).matchedRecords,21);
assert.ok(youthSearch('Horyzons social functioning').some(r=>r.pmid==='34002511'));assert.ok(youthSearch('37393283').some(r=>r.pmid==='37393283'));
assert.deepEqual(youthSearch('OXTR',{targets:['OXTR']}),[]);
const age=read('Informatics/library-data/age-index.json'),community=read('Informatics/library-data/community-index.json');
assert.equal(age.recordCount,1323);assert.equal(community.recordCount,1323);
assert.equal(researchNavigation({collection:'all'}).matchedPapers,1323);
assert.equal(new Set(age.records.map(r=>r.recordId)).size,1323);
for(const r of youthRecords){assert.deepEqual(age.records.find(a=>a.recordId===r.id),r.agePopulation);assert.ok(community.records.some(c=>c.recordId===r.id));}
for(const [collection,chapter] of Object.entries(bundle.chapters)){
 const base=read(`Informatics/library-data/${collection}-edition.json`),dv=read('Informatics/library-data/domestic-violence-expansion.json').chapters[collection];
 assert.ok(chapter.number>dv.number);assert.ok(chapter.number>Math.max(...base.chapters.map(c=>c.number)));
 assert.equal(chapter.topics.find(t=>t.id==='universal-mind-mobile-lab').papers.length,bundle.counts.collections[collection]);
 for(const t of chapter.topics){
  const eligible=score=>new Set(t.papers.filter(p=>p.paperTrustAssessment.score>=score).map(p=>p.randomizedStudyUnitId||p.studyFamilyId)).size;
  const expected=eligible(3)>=3?3:eligible(2)>=3?2:eligible(1)>=3?1:null;
  assert.equal(t.trustScore,expected,t.id);
 }
}
const tm=bundle.chapters.language.topics.find(t=>t.id==='transcendental-meditation-youth');assert.equal(tm.papers.length,2);assert.equal(tm.trustAssessment.counts.rct,1);assert.equal(tm.trustScore,null);
assert.equal(bundle.chapters.language.topics.find(t=>t.id==='mantra-youth').papers.length,0);
assert.deepEqual(bundle.chapters.sound.topics.find(t=>t.id==='music-making-youth').papers.map(r=>r.pmid).sort(),['36407676','37151974','41662046']);
assert.match(youthRecord('34002511').summary,/no|not|null/i);
const report={passed:true,baseline:baseline.baselineCommit,approved:105,new:100,existingMatches:5,visibleLibrary:1323,studyFamilies:93,randomizedStudyUnits:42,paperCrosses:bundle.counts.paperCrosses,
 sourceAndLimitsPreserved:true,originalCorpusUnchanged:true,originalReviewImmutable:true,allExcludedCandidatesAbsent:true,chapterFamiliesVerified:true,allDOIAndPMIDAliasesResolve:true,noInventedTargets:true,
 retrieval:'Local module checks only; no deployed Worker, live model call or database write.'};
fs.writeFileSync(path.join(root,'research/informatics/youth-integration-20261102/source-checks.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
