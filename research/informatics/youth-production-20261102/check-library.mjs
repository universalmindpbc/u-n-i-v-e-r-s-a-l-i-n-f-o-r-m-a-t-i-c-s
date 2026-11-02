import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=process.cwd(),site=path.join(root,'Informatics'),out=path.join(root,'research/informatics/youth-production-20261102');
const approved=JSON.parse(await fs.readFile(path.join(site,'library-data/youth-expansion.json'),'utf8'));
const mime={'.html':'text/html','.js':'text/javascript','.json':'application/json','.css':'text/css','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg'};
const server=http.createServer(async(req,res)=>{
 const u=new URL(req.url,'http://localhost'),pathname=decodeURIComponent(u.pathname);
 if(pathname==='/__youth-reader-test'){res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'});return res.end('<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}iframe{width:100%;height:100vh;border:0}</style><iframe id="research-frame" src="/informatics/library.html?embedded=1&amp;'+u.searchParams.toString().replaceAll('&','&amp;')+'"></iframe>');}
 const rel=pathname.startsWith('/informatics/')?'Informatics/'+pathname.slice(13):pathname.slice(1),file=path.resolve(root,rel);
 if(!file.startsWith(root+'/')||/\.(mp3|mp4|wav|m3u8)$/.test(file)){res.writeHead(403);return res.end();}
 try{const bytes=await fs.readFile(file);res.writeHead(200,{'Content-Type':(mime[path.extname(file)]||'application/octet-stream')+'; charset=utf-8','Cache-Control':'no-store'});res.end(bytes);}catch{res.writeHead(404);res.end();}
});await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const live=process.env.YOUTH_QA_ORIGIN;
if(live)assert.equal(live,'https://universalmind.tv');
const base=live||'http://127.0.0.1:'+server.address().port;
const browserServer=await chromium.launchServer({headless:true,args:['--mute-audio','--disable-background-networking']});assert.ok(browserServer.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(browserServer.wsEndpoint()),context=await browser.newContext({acceptDownloads:true}),page=await context.newPage();
await context.addInitScript(()=>{localStorage.setItem('informatics-library-background','black');localStorage.setItem('informatics-library-video-paused','true');});
await context.route('**/*',route=>route.request().url().startsWith(base+'/')||route.request().url().startsWith('blob:')?route.continue():route.abort());
let reader;
const errors=[],checks=[];page.on('pageerror',e=>errors.push(e.message));
async function go(params){
 if(live){
  if(!page.url().startsWith(base+'/informatics/research/'))await page.goto(base+'/informatics/research/',{waitUntil:'networkidle'});
  const frame=page.locator('#research-frame');await frame.waitFor({state:'attached'});
  reader=await (await frame.elementHandle()).contentFrame();await reader.locator('[data-stat="papers"]').waitFor({timeout:30000});await reader.goto(base+'/informatics/library.html?embedded=1&'+params,{waitUntil:'networkidle'});
 }else{await page.goto(base+'/__youth-reader-test?'+params,{waitUntil:'networkidle'});reader=await (await page.locator('#research-frame').elementHandle()).contentFrame();}
 await reader.locator('[data-stat="papers"]').waitFor({timeout:15000}).catch(async error=>{console.error(JSON.stringify({url:page.url(),errors,body:(await reader.locator('body').innerText()).slice(0,600)}));throw error;});assert.ok(await reader.locator('#load-error').isHidden());}
async function count(n){const text=await reader.locator('#result-count').innerText();assert.match(text.replaceAll(',',''),new RegExp('^'+n+' papers?'),JSON.stringify(await reader.evaluate(()=>({url:location.href,context:window.UniversalResearchContext()}))));}
async function download(kind){const p=page.waitForEvent('download');await reader.locator('#export-'+kind).click();const d=await p,file=path.join(out,'candidate-filtered.'+kind);await d.saveAs(file);return file;}
const allTopics=['sound','vision','language'].map(c=>'topic='+c+':universal-mind-mobile-lab').join('&');
let exported;
try{
 for(const width of [1440,390,320]){
  await page.setViewportSize({width,height:1000});
  for(const [collection,chapter] of Object.entries(approved.chapters)){
   await go('collection='+collection);assert.ok(await reader.getByText('AI Guided Youth Research',{exact:true}).count());
   await reader.locator(`[data-chapter="${chapter.number}"]`).first().click();
   await reader.waitForFunction(n=>document.querySelectorAll('.paper').length===n,chapter.topics.reduce((n,t)=>n+t.papers.length,0),{timeout:15000});
   const actual=await reader.locator('.paper').evaluateAll(cards=>cards.map(c=>({id:c.dataset.paper,crosses:c.querySelectorAll('.trust-score svg').length})));
   assert.equal(actual.length,chapter.topics.reduce((n,t)=>n+t.papers.length,0));
   for(const a of actual)assert.equal(a.crosses,approved.records.find(r=>r.id===a.id).paperTrustAssessment.score,a.id);
   for(const t of chapter.topics){
    assert.equal(await reader.locator('#topic-'+t.id+' > .topic-heading .trust-score svg').count(),t.trustScore||0,t.id);
    assert.equal(await reader.locator('#topic-'+t.id+' .paper').count(),t.papers.length);
   }
   assert.ok(await reader.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),collection+' '+width+' overflow');
   await reader.locator('#topic-universal-mind-mobile-lab > .topic-heading').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,(live?'production':'candidate')+'-'+collection+'-'+width+'.png')});
   assert.equal(await reader.getByText('[object Object]',{exact:false}).count(),0);
  }
  await go('collection=all&view=search&'+allTopics);await count(105);assert.ok(await reader.locator('#youth-overview').isVisible());
  await reader.locator('#next').click();assert.match(await reader.locator('#page-status').innerText(),/2/);await reader.locator('#previous').click();
  // Exercise a real dropdown checkbox and the selected-filter removal control.
  await reader.locator('#age-toggle').click();await reader.locator('#age-options input[value="youth"]').check();await count(105);
  await reader.getByRole('button',{name:/Remove Age group: Youth/}).click();await count(105);
  assert.ok(await reader.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Search overflow '+width);
  checks.push({width,chapters:true,allPaperCrosses:true,chapterCrosses:true,dropdownCheckboxAndRemoval:true,pagination:true,noHorizontalOverflow:true});
  if(width===1440){const file=await download('json');exported=JSON.parse(await fs.readFile(file,'utf8')).records;await download('csv');}
 }
 assert.equal(exported.length,105);assert.deepEqual(new Set(exported.map(r=>r.id)),new Set(approved.records.map(r=>r.id)));
 for(const [pmid,n] of Object.entries({'19798037':197,'29422189':230,'42324567':0,'41340540':1,'40243164':7}))assert.equal(exported.find(r=>r.pmid===pmid).citationCount,n,'Historical citation count '+pmid);
 for(const r of exported){const a=approved.records.find(a=>a.id===r.id);assert.equal(r.summary,a.summary);assert.equal(r.limitation,a.limitation);assert.deepEqual(r.labRelevance,a.labRelevance);assert.deepEqual(r.agePopulation,a.agePopulation);assert.ok(r.publicationApproval);assert.ok(r.sourceAccess.methodsInspected);}
 await page.setViewportSize({width:1440,height:1000});
 for(const [kind,n] of [['trial',47],['double-blind',3],['review',42],['meta-analysis',exported.filter(r=>/meta.analy/i.test(r.title+' '+r.design)).length]]){await go('collection=all&view=search&'+allTopics+'&kind='+kind);await count(n);}
 const cases=[['condition','ptsd',r=>r.conditions.includes('ptsd')],['age','children',r=>r.ageGroups.includes('children')],['community','indigenous',r=>r.communityMatches.some(m=>m.categoryId==='indigenous')],['community','defence',r=>r.communityMatches.some(m=>m.categoryId==='defence')],['category','Gene expression',r=>r.categories.includes('Gene expression')],['physiology','Clinical & self-report',r=>r.categories.includes('Clinical & self-report')],['target','OXTR',r=>r.targets.includes('OXTR')],['journal',exported[0].journalKey,r=>r.journalKey===exported[0].journalKey],['trust','3',r=>r.paperTrustAssessment.score===3],['jadad','unassessed',r=>r.jadad.status==='unassessed'],['period','recent',r=>r.year>=2022&&r.year<=2026]];
 for(const [facet,value,predicate] of cases){await go('collection=all&view=search&'+allTopics+'&'+facet+'='+encodeURIComponent(value));await count(exported.filter(predicate).length);}
 await go('collection=all&view=search&'+allTopics+'&trust=1&trust=3&kind=trial');await count(3);
 for(const [topic,n] of [['black-dog-youth',20],['orygen-youth',21]]){await go('collection=language&view=search&topic=language:'+topic);await count(n);}
 for(const pmid of ['19798037','42324567','40243164','41340540','29422189','29622047','39720610','36995752']){
  const record=approved.records.find(r=>r.pmid===pmid);await go('collection=all&view=search&record='+record.id);
  await reader.locator('#article[open]').waitFor();assert.equal(await reader.locator('#article .paper .trust-score svg').count(),record.paperTrustAssessment.score);
  assert.match(await reader.locator('#article-body').innerText(),/Reported ages:/);assert.match(await reader.locator('#article-body').innerText(),/Fit to ages 16–24:/);
  await reader.locator('#article [data-close="article"]').click();
 }
 await go('collection=language&chapter='+approved.chapters.language.number+'#topic-mantra-youth');assert.match(await reader.locator('#topic-mantra-youth').innerText(),/No new mantra paper met/);
 await go('collection=all&view=search');await count(1323);
 await go('collection=language&chapter='+(approved.chapters.language.number-1));assert.equal(await reader.locator('#topic-domestic-family-violence .paper').count(),99);
 assert.deepEqual(errors,[]);
 const roundtrip=JSON.parse(execFileSync('python3',['-c',`import csv,json,sys
j=json.load(open(sys.argv[1]))['records']; c=list(csv.DictReader(open(sys.argv[2],encoding='utf-8-sig',newline='')))
assert len(j)==len(c)==105
for a,b in zip(j,c):
 for k,v in a.items():
  actual=b[k]
  if actual.startswith("'") and len(actual)>1 and actual[1] in '=+-@\\t\\r':actual=actual[1:]
  if isinstance(v,(list,dict)):assert json.loads(actual)==v,(a['id'],k)
  elif v is None:assert actual==''
  elif isinstance(v,bool):assert actual==str(v).lower()
  else:assert actual==str(v),(a['id'],k)
print(json.dumps({'records':len(j),'allExportedFieldsEquivalent':True}))`,path.join(out,'candidate-filtered.json'),path.join(out,'candidate-filtered.csv')],{encoding:'utf8'}));
 checks.push({filteredDownloads:roundtrip,all105FindingsAndLimits:true,allFiveExistingMatches:true,allDropdownFacets:true,withinFacetORAndAcrossFacetsAND:true,institutionCounts:true,emptyMantraHonest:true,originalDVChapterPreserved:true,visibleLibrary:1323});
 const report={passed:true,phase:live?'live production research route':'local embedded reader',origin:base,processAudioMuted:true,externalRequestsBlocked:true,checks,errors};await fs.writeFile(path.join(out,(live?'production':'candidate')+'-browser.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 await fs.unlink(path.join(out,'candidate-filtered.json'));await fs.unlink(path.join(out,'candidate-filtered.csv'));
}finally{await context.close();await browser.close();await browserServer.close();await new Promise(resolve=>server.close(resolve));}
