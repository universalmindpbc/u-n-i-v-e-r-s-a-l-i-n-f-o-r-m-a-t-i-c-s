import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';

const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../../..');
const output=path.join(here,'staging-qa'),base='https://staging.universalmind.tv';
const additions=JSON.parse(await fs.readFile(path.join(here,'additions.json'),'utf8')).records;
await fs.mkdir(output,{recursive:true});
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const server=await chromium.launchServer({executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',headless:true,args:['--mute-audio']});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser=await chromium.connect(server.wsEndpoint());
const report={checkedAt:new Date().toISOString(),sourceCommit:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),processMuted:true,views:[],files:[]};
try {
  for(const viewport of [{width:1440,height:1000},{width:390,height:844}]) {
    const context=await browser.newContext({viewport,reducedMotion:'reduce'}),page=await context.newPage(),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(base+'/informatics/read/',{waitUntil:'networkidle'});
    assert.match(await page.locator('main').innerText(),/more than 800 publications/);
    assert.match(await page.locator('main').innerText(),/891 records/);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    const icon=await page.locator('link[rel="icon"]').getAttribute('href');
    assert.equal((await context.request.get(base+icon)).status(),200);
    await page.screenshot({path:path.join(output,`reading-${viewport.width}.png`),fullPage:true});
    await page.goto(base+'/informatics/library.html?collection=all&view=search',{waitUntil:'domcontentloaded'});
    const library=page.frameLocator('#research-frame');
    await library.locator('#result-count').filter({hasText:/^891 papers$/}).waitFor();
    assert.ok(await library.locator('html').evaluate(e=>e.scrollWidth<=e.clientWidth));
    assert.ok(await library.locator('script[src="library.js?v=19"]').count());
    if(viewport.width===1440) {
      for(const format of ['json','csv']) {
        const pending=page.waitForEvent('download');await library.locator('#export-'+format).click();
        await (await pending).saveAs(path.join(output,'library-download.'+format));
      }
      const records=JSON.parse(await fs.readFile(path.join(output,'library-download.json'),'utf8')).records;
      assert.equal(records.length,891);
      for(const added of additions) {
        const live=records.find(r=>r.id===added.id);assert.ok(live,added.id);
        assert.equal(live.summary,added.finding);assert.equal(live.limitation,added.limitation);
        assert.deepEqual(live.evidence,added.evidence);
      }
      report.exportedRecords=records.length;report.newRecordsMatched=additions.length;
    }
    await library.locator('#query').fill('32412916');
    await library.locator('#result-count').filter({hasText:/^1 paper$/}).waitFor();
    await library.locator('[data-record="pmid-32412916"]').first().click();
    await library.locator('#article[open]').waitFor();
    assert.match(await library.locator('#article-body').innerText(),/5\.75/);
    assert.match(await library.locator('#article-body').innerText(),/11\.85/);
    await page.screenshot({path:path.join(output,`sound-paper-${viewport.width}.png`)});
    await page.keyboard.press('Escape');await library.locator('#reset').click();
    await library.locator('#category-toggle').click();
    await library.locator('#category-options input[value="Gene expression"]').check();
    await library.locator('#category-toggle').click();
    await library.locator('#query').fill('29900257');
    await library.locator('#result-count').filter({hasText:/^1 paper$/}).waitFor();
    assert.equal(await library.locator('.search-paper[data-id="pmid-29900257"]').count(),1);
    await library.locator('#reset').click();await library.locator('#kind-toggle').click();
    await library.locator('#kind-options input[value="single-blind"]').check();
    await library.locator('#kind-toggle').click();
    await library.locator('#query').fill('29921143');
    await library.locator('#result-count').filter({hasText:/^1 paper$/}).waitFor();
    await library.locator('#reset').click();await library.locator('#target-toggle').click();
    await library.locator('#target-options input[value="DRD2"]').check();
    await library.locator('#target-toggle').click();
    await library.locator('#query').fill('39413359');
    await library.locator('#result-count').filter({hasText:/^1 paper$/}).waitFor();
    await library.locator('[data-record="pmid-39413359"]').first().click();
    assert.match(await library.locator('#article-body').innerText(),/genotype/i);
    await page.keyboard.press('Escape');
    await page.goto(base+'/informatics/library.html?collection=all&view=search&record=pmid-30279666',{waitUntil:'domcontentloaded'});
    await library.locator('#article[open]').waitFor();
    const correction=await library.locator('#article-body').innerText();
    assert.match(correction,/increased FKBP5/);assert.match(correction,/0\.052/);
    assert.ok(await library.locator('#article-body a[href="https://doi.org/10.3389/fpsyt.2021.642245"]').count());
    assert.deepEqual(errors,[]);
    report.views.push({viewport,overflow:false,pageErrors:errors,search:true,geneExpressionFilter:true,singleBlindFilter:true,genotypeFilter:true,correction:true,favicon:true});
    await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(media=>media.pause()));
    await context.close();
  }
  const context=await browser.newContext();
  const files=execFileSync('git',['diff','--name-only','3ce760b2203db363078fac573aa8f46769e41d26','HEAD','--','Informatics','llms.txt','sitemap.xml'],{cwd:root,encoding:'utf8'}).trim().split('\n').filter(p=>p&&!p.endsWith('.bin'));
  for(let i=0;i<files.length;i+=6) {
    await Promise.all(files.slice(i,i+6).map(async file=>{
      let route=file.replace(/^Informatics\//,'informatics/');
      if(file==='Informatics/library.js')route+='?v=19';
      else if(file.startsWith('Informatics/library-data/'))route+='?v=daytab-20261026';
      const response=await context.request.get(base+'/'+route,{timeout:60000});assert.equal(response.status(),200,file);
      const digest=sha(await response.body());assert.equal(digest,sha(await fs.readFile(path.join(root,file))),file);
      report.files.push({path:file,sha256:digest});
    }));
  }
  const guide=await (await context.request.get(base+'/informatics/llms.txt')).text();
  const rootGuide=await (await context.request.get(base+'/llms.txt')).text();
  for(const text of [guide,rootGuide])for(const term of ['more than 800','OXTR','oxytocin','Fibonacci'])assert.ok(text.toLowerCase().includes(term.toLowerCase()),term);
  report.passed=true;
  await fs.writeFile(path.join(output,'verification.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({passed:true,views:report.views.length,exportedRecords:891,newRecordsMatched:70,liveFileHashes:report.files.length,sourceCommit:report.sourceCommit}));
} finally {
  for(const context of browser.contexts()){for(const page of context.pages())await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(media=>media.pause())).catch(()=>{});await context.close().catch(()=>{});}
  await browser.close().catch(()=>{});await server.close().catch(()=>{});
}
